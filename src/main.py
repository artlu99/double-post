"""CLI entry point for Double Post.

Provides CSV reconciliation with dry-run mode and interactive TUI.
"""

from pathlib import Path

import pandas as pd
import typer

from src.aliases import AliasDatabase, seed_defaults
from src.csv_loader import load_csv
from src.matcher import MatchConfig, find_matches, normalize_sign_conventions
from src.models import ConfidenceTier


def reconcile(
    source: Path = typer.Argument(..., help="Source CSV file (bank statement)"),
    target: Path = typer.Argument(..., help="Target CSV file (personal records)"),
    min_confidence: float = typer.Option(
        0.1, "--min-confidence", "-c", help="Minimum confidence to include (default: 0.1)"
    ),
    date_window: int = typer.Option(
        3, "--date-window", "-d", help="Days to consider for date matching"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print results without launching TUI"),
) -> None:
    """Reconcile personal CSV records against bank statement.

    Uses 4-tier confidence categorization:
    - HIGH (≥0.9): Auto-accepted
    - MEDIUM (0.5-0.9): Manual review
    - LOW (0.1-0.5): Weak suggestions
    - NONE (<0.1): Excluded

    Source (bank) is treated as source of truth.
    """
    if not source.exists():
        typer.echo(f"Error: Source file not found: {source}", err=True)
        raise typer.Exit(1)

    if not target.exists():
        typer.echo(f"Error: Target file not found: {target}", err=True)
        raise typer.Exit(1)

    # Load CSVs
    typer.echo(f"Loading source: {source.name}")
    source_df, source_mapping, source_convention = load_csv(source)

    typer.echo(f"Loading target: {target.name}")
    target_df, target_mapping, target_convention = load_csv(target)

    # Report detected formats
    typer.echo(f"\nDetected {source_mapping.format_type.upper()} format for source")
    typer.echo(f"Detected {target_mapping.format_type.upper()} format for target")

    # Report sign convention
    typer.echo(
        f"\nSource: DEBIT sign is '{source_convention['debit_sign']}' ({source_convention['debit_count'] if 'debit_count' in source_convention else source_convention['negative_count']} records)"
    )
    typer.echo(
        f"Target: DEBIT sign is '{target_convention['debit_sign']}' ({target_convention['debit_count'] if 'debit_count' in target_convention else target_convention['negative_count']} records)"
    )

    # Report date format hints
    if source_mapping.date:
        typer.echo(f"  Using date column: {source_mapping.date}")
    if target_mapping.date:
        typer.echo(f"  Using date column: {target_mapping.date}")

    # Report record counts
    typer.echo(f"\nLoaded {len(source_df)} source records, {len(target_df)} target records")

    # Normalize sign conventions if they differ
    source_debit_sign = source_convention.get("debit_sign", "negative")
    target_debit_sign = target_convention.get("debit_sign", "negative")

    if (
        source_debit_sign != target_debit_sign
        and source_debit_sign != "debit_col"
        and target_debit_sign != "debit_col"
    ):
        typer.echo(
            "\nNormalizing signs: Target uses opposite sign convention, flipping amounts to match source"
        )
        source_df, target_df = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

    # Filter target records to only include dates up to latest source date + 1 day cushion
    # This removes personal records that couldn't possibly be in the bank statement yet
    if "date_clean" in source_df.columns and "date_clean" in target_df.columns:
        latest_source_date = source_df["date_clean"].max()
        if pd.notna(latest_source_date):
            from datetime import timedelta

            cutoff_date = latest_source_date + timedelta(days=1)
            original_target_count = len(target_df)

            # Filter target_df to only include records with dates <= cutoff
            target_df = target_df[target_df["date_clean"] <= cutoff_date].copy()

            # Reset index after filtering
            target_df.reset_index(drop=True, inplace=True)

            filtered_count = original_target_count - len(target_df)
            if filtered_count > 0:
                typer.echo(
                    f"\nFiltered {filtered_count} target records dated after {cutoff_date.strftime('%Y-%m-%d')} (latest source date + 1 day)"
                )

    # Filter out already-reconciled target records
    # This removes personal records that have already been matched in previous runs
    if "reconciled" in target_df.columns:
        original_target_count = len(target_df)

        # Filter to only include records where reconciled is not True (case-insensitive)
        # Accept: false, False, FALSE, 0, empty string, NaN
        # Reject: true, True, TRUE, 1
        target_df = target_df[
            target_df["reconciled"].astype(str).str.lower().isin(["false", "0", ""])
            | target_df["reconciled"].isna()
        ].copy()

        # Reset index after filtering
        target_df.reset_index(drop=True, inplace=True)

        filtered_count = original_target_count - len(target_df)
        if filtered_count > 0:
            typer.echo(f"\nFiltered {filtered_count} already-reconciled target record(s)")

    # Run matching
    config = MatchConfig(threshold=0.7, date_window_days=date_window)

    # Initialize alias database with defaults
    alias_db_path = Path("data/aliases.db")
    alias_db_path.parent.mkdir(parents=True, exist_ok=True)
    alias_db = AliasDatabase(alias_db_path)
    seed_defaults(alias_db)

    result = find_matches(
        source_df, target_df, config, min_confidence=min_confidence, alias_db=alias_db
    )

    # Count tiers
    high_tier = sum(1 for m in result.matches if m.tier == ConfidenceTier.HIGH)
    medium_tier = sum(1 for m in result.matches if m.tier == ConfidenceTier.MEDIUM)
    low_tier = sum(1 for m in result.matches if m.tier == ConfidenceTier.LOW)

    # Print results
    typer.echo("\n" + "=" * 50)
    typer.echo("MATCHING RESULTS")
    typer.echo("=" * 50)
    typer.echo(f"  ⭐ High confidence (≥0.9): {high_tier} [Auto-accepted]")
    typer.echo(f"  ○ Medium confidence (0.5-0.9): {medium_tier}")
    typer.echo(f"  ○ Low confidence (0.1-0.5): {low_tier}")
    typer.echo(f"  - Missing in target: {len(result.missing_in_target)}")
    typer.echo(f"  + Unmatched targets: {len(result.missing_in_source)}")
    typer.echo(f"\n  Total matches: {len(result.matches)}")
    if result.matches:
        typer.echo(
            f"  Accept rate: {high_tier}/{len(result.matches)} ({high_tier / len(result.matches) * 100:.1f}%)"
        )

    if dry_run:
        # Show all matches with amounts
        if result.matches:
            typer.echo("\n" + "-" * 50)
            typer.echo("MATCHES (Source Amt → Target Amt)")
            typer.echo("-" * 50)
            for match in result.matches:
                source_rec = source_df.iloc[match.source_idx]
                source_amt = f"${source_rec['amount_clean']:.2f}"
                source_desc = source_rec["description_clean"][:40]

                if match.target_idx is not None:
                    target_rec = target_df.iloc[match.target_idx]
                    target_amt = f"${target_rec['amount_clean']:.2f}"
                    target_desc = target_rec["description_clean"][:40]
                    typer.echo(
                        f"  [{match.tier.value}] {match.confidence:.2f} {source_amt} → {target_amt}"
                    )
                    typer.echo(f"      {source_desc} → {target_desc}")
                else:
                    typer.echo(
                        f"  [{match.tier.value}] {match.confidence:.2f} {source_amt} → (no match)"
                    )
                    typer.echo(f"      {source_desc}")

        # Show missing source records
        if result.missing_in_target:
            typer.echo("\n" + "-" * 50)
            typer.echo(f"MISSING IN TARGET ({len(result.missing_in_target)} records)")
            typer.echo("-" * 50)
            for idx in result.missing_in_target[:10]:  # Show first 10
                rec = source_df.iloc[idx]
                typer.echo(
                    f"  {rec['date_clean'].strftime('%Y-%m-%d')} | ${rec['amount_clean']:.2f} | {rec['description_clean'][:60]}"
                )
            if len(result.missing_in_target) > 10:
                typer.echo(f"  ... and {len(result.missing_in_target) - 10} more")

        # Show unmatched target records
        if result.missing_in_source:
            typer.echo("\n" + "-" * 50)
            typer.echo(f"UNMATCHED TARGETS ({len(result.missing_in_source)} records)")
            typer.echo("-" * 50)
            for idx in result.missing_in_source[:10]:  # Show first 10
                rec = target_df.iloc[idx]
                typer.echo(
                    f"  {rec['date_clean'].strftime('%Y-%m-%d')} | ${rec['amount_clean']:.2f} | {rec['description_clean'][:60]}"
                )
            if len(result.missing_in_source) > 10:
                typer.echo(f"  ... and {len(result.missing_in_source) - 10} more")

    if dry_run:
        typer.echo("\nDry run complete. Use --min-confidence to adjust minimum threshold.")
        typer.echo("Run without --dry-run to launch TUI for interactive review.")
    else:
        # Import TUI and launch it
        from src.tui.app import run_tui

        typer.echo("\nLaunching TUI...")
        run_tui(source_df, target_df, result, source, target)


if __name__ == "__main__":
    typer.run(reconcile)


# Entry point for CLI
cli = reconcile
app = typer.Typer(add_completion=False)
app.command()(reconcile)
