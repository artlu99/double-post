"""TUI screens for Double Post."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import pandas as pd
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from src.models import ConfidenceTier, Match, MatchDecision, MatchResult
from src.tui import display_utils


@dataclass
class MatchState:
    """Tracks the state of matches during TUI session.

    Attributes:
        match_result: The original match result
        filter_mode: Current filter mode for displaying matches
        sort_mode: Current sort mode for ordering matches
        selected_match_idx: Currently selected match index
        source_df: Source DataFrame (for date sorting)
    """

    match_result: MatchResult
    filter_mode: Literal["all", "pending", "accepted", "rejected"] = "all"
    sort_mode: Literal["status", "confidence", "date"] = "status"
    selected_match_idx: int = -1
    source_df: pd.DataFrame | None = None

    def cycle_sort_mode(self) -> None:
        """Cycle to the next sort mode."""
        modes: list[Literal["status", "confidence", "date"]] = ["status", "confidence", "date"]
        current_idx = modes.index(self.sort_mode)
        self.sort_mode = modes[(current_idx + 1) % len(modes)]

    def create_missing_match(self, source_idx: int) -> Match:
        """Create a Match object for a missing source record.

        Args:
            source_idx: Index of source record with no match

        Returns:
            Match object representing the missing record
        """
        from src.matcher import Match

        return Match(
            source_idx=source_idx,
            target_idx=None,
            confidence=0.0,
            reason="No match found",
            decision=MatchDecision.PENDING,
            tier=ConfidenceTier.NONE,
        )

    def promote_missing_to_match(self, match: Match) -> None:
        """Promote a missing match to the formal matches list.

        When a missing item is accepted/rejected, it needs to be added to
        the matches list and removed from missing_in_target.

        Args:
            match: The missing match to promote
        """
        if match.target_idx is None and match.source_idx in self.match_result.missing_in_target:
            self.match_result.matches.append(match)
            self.match_result.missing_in_target.remove(match.source_idx)

    def _apply_sorting(self, matches: list[Match]) -> list[Match]:
        """Apply current sort mode to matches list.

        Args:
            matches: List of matches to sort

        Returns:
            Sorted list of matches (in-place sort)
        """
        if self.sort_mode == "status":
            decision_order = {
                MatchDecision.PENDING: 0,
                MatchDecision.REJECTED: 1,
                MatchDecision.ACCEPTED: 2,
            }
            matches.sort(key=lambda m: (decision_order.get(m.decision, 3), m.confidence))
        elif self.sort_mode == "confidence":
            matches.sort(key=lambda m: m.confidence)
        elif self.sort_mode == "date" and self.source_df is not None:
            matches.sort(
                key=lambda m: self.source_df.iloc[m.source_idx]["date_clean"],
                reverse=True,
            )
        return matches

    def get_sorted_matches(self) -> list:
        """Get matches sorted by current sort mode."""
        matches = self.match_result.matches.copy()
        return self._apply_sorting(matches)

    def get_filtered_and_sorted_matches(self) -> list:
        """Get matches filtered by current filter mode and sorted.

        For 'all' filter mode, includes missing items as special no-match entries.
        """
        # Start with regular matches
        if self.filter_mode == "all":
            filtered = self.match_result.matches.copy()
            # Add missing items as special no-match entries
            for source_idx in self.match_result.missing_in_target:
                filtered.append(self.create_missing_match(source_idx))
        elif self.filter_mode == "pending":
            filtered = [m for m in self.match_result.matches if m.decision == MatchDecision.PENDING]
            # Include missing items (they're implicitly pending)
            for source_idx in self.match_result.missing_in_target:
                filtered.append(self.create_missing_match(source_idx))
        elif self.filter_mode == "accepted":
            filtered = [
                m for m in self.match_result.matches if m.decision == MatchDecision.ACCEPTED
            ]
        elif self.filter_mode == "rejected":
            filtered = [
                m for m in self.match_result.matches if m.decision == MatchDecision.REJECTED
            ]
        else:
            filtered = []

        # Then sort
        return self._apply_sorting(filtered)


class MatchReviewScreen(Screen):
    """Screen for reviewing matches between source and target records."""

    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("s", "show_summary", "Summary"),
        ("i", "show_missing", "Missing Items"),
        ("u", "show_unmatched_targets", "Unmatched Targets"),
        ("m", "manual_match", "Manual Match"),
        ("a", "accept_match", "Accept Match"),
        ("r", "reject_match", "Reject Match"),
        ("f", "toggle_filter", "Filter"),
        ("t", "cycle_sort", "Sort (Status/Conf/Date)"),
        ("down", "move_down", "Next"),
        ("up", "move_up", "Prev"),
    ]

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        match_result: MatchResult,
        match_state: MatchState | None = None,
    ) -> None:
        """Initialize the match review screen.

        Args:
            source_df: Normalized source DataFrame
            target_df: Normalized target DataFrame
            match_result: Results from matching operation
            match_state: Shared match state (creates new if None)
        """
        super().__init__()
        self.source_df = source_df
        self.target_df = target_df
        if match_state:
            self.match_state = match_state
        else:
            self.match_state = MatchState(match_result=match_result, source_df=source_df)

    def compose(self):
        """Compose the match review screen."""
        yield Header()
        yield self._get_title_static()
        yield Static(
            "[dim]↑↓: Navigate | a: Accept | r: Reject | m: Manual Match | f: Filter | t: Sort | i: Missing | u: Unmatched | s: Summary | q: Quit[/]",
            id="help_text",
        )
        yield DataTable(id="matches_table")
        yield Footer()

    def _get_title_static(self) -> Static:
        """Generate title static with tier breakdown."""
        matches = self.match_state.match_result.matches
        missing = len(self.match_state.match_result.missing_in_target)
        unmatched_targets = len(self.match_state.match_result.missing_in_source)
        total_source = len(matches) + missing

        # Count tiers
        high = sum(1 for m in matches if m.tier == ConfidenceTier.HIGH)
        medium = sum(1 for m in matches if m.tier == ConfidenceTier.MEDIUM)
        low = sum(1 for m in matches if m.tier == ConfidenceTier.LOW)

        # Count decisions
        pending = sum(1 for m in matches if m.decision == MatchDecision.PENDING) + missing
        accepted = sum(1 for m in matches if m.decision == MatchDecision.ACCEPTED)
        rejected = sum(1 for m in matches if m.decision == MatchDecision.REJECTED)

        title = (
            f"Review: [bold]{total_source}[/] source records | "
            f"[dim]...{pending}[/] [green]✓{accepted}[/] [red]✗{rejected}[/] | "
            f"([bold green]⭐{high}[/] [yellow]○{medium}[/] [dim cyan]○{low}[/] [dim red]—{missing}[/] [yellow]+{unmatched_targets}[/]) | "
            f"Filter: [bold]{self.match_state.filter_mode.upper()}[/] | "
            f"Sort: [bold]{self.match_state.sort_mode.upper()}[/]"
        )
        return Static(title, id="review_title")

    def on_mount(self) -> None:
        """Populate tables when mounted."""
        self._refresh_tables()
        # Auto-select first row if matches exist
        matches = self.match_state.get_filtered_and_sorted_matches()
        if matches and self.match_state.selected_match_idx == -1:
            self.match_state.selected_match_idx = 0
            self._refresh_tables()

    def _refresh_tables(self) -> None:
        """Refresh the single table with current data."""
        table = self.query_one("#matches_table", DataTable)

        # Clear table
        table.clear()

        # Configure columns
        columns = self._get_table_columns()
        table.add_columns(*columns)
        table.zebra_striping = True
        table.cursor_type = "row"

        # Get filtered and sorted matches
        matches = self.match_state.get_filtered_and_sorted_matches()

        # Add matched rows
        for i, match in enumerate(matches):
            source_row = self.source_df.iloc[match.source_idx]

            # Get target description and amount if matched
            if match.target_idx is not None:
                target_row = self.target_df.iloc[match.target_idx]
                target_desc = str(target_row["description_clean"])
                target_amount = display_utils.format_amount(target_row["amount_clean"])
            else:
                target_desc = "—"
                target_amount = "—"

            table.add_row(
                self._get_status_text(match.decision, match.tier),
                self._get_tier_text_from_display(match.tier),
                display_utils.format_date(source_row["date_clean"]),
                display_utils.format_amount(source_row["amount_clean"]),
                target_amount,
                display_utils.truncate_string(str(source_row["description_clean"]), 30),
                display_utils.truncate_string(target_desc, 30),
                self._get_match_info_text(match),
            )

        # Add separator row if there are unmatched targets
        unmatched_targets = self.match_state.match_result.missing_in_source
        if unmatched_targets:
            table.add_row(
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )
            # Add unmatched target records
            for target_idx in unmatched_targets:
                target_row = self.target_df.iloc[target_idx]

                table.add_row(
                    "[bold yellow]+ UNMATCHED[/]",  # Status
                    "[dim]—[/]",  # Tier
                    display_utils.format_date(target_row["date_clean"]),
                    "[dim]—[/]",  # Source amount (no source)
                    display_utils.format_amount(target_row["amount_clean"]),
                    "[dim]—[/]",  # Source description (no source)
                    display_utils.truncate_string(str(target_row["description_clean"]), 30),
                    "[dim italic]Unreconciled target record[/]",
                )

        # Move cursor to selected row after all rows are added
        if 0 <= self.match_state.selected_match_idx < len(matches):
            table.move_cursor(row=self.match_state.selected_match_idx)

    def _get_tier_text_from_display(self, tier: ConfidenceTier) -> str:
        """Get tier text for display using centralized utility."""
        text, _, color = display_utils.get_tier_display(tier)
        return f"[{color}]{text}[/]"

    def _sync_cursor_to_selected_idx(self) -> None:
        """Sync the table cursor position to selected_match_idx.

        This ensures that if the user navigates with arrow keys (which move
        the DataTable cursor directly), we update selected_match_idx to match.
        """
        table = self.query_one("#matches_table", DataTable)
        cursor_row = table.cursor_row
        matches = self.match_state.get_filtered_and_sorted_matches()

        # Only update if cursor is within the matches range (not in unmatched section)
        if 0 <= cursor_row < len(matches):
            if self.match_state.selected_match_idx != cursor_row:
                self.match_state.selected_match_idx = cursor_row

    def _get_decision_icon(self, decision: MatchDecision) -> str:
        """Get icon for decision status."""
        if decision == MatchDecision.ACCEPTED:
            return "[green]✓ Accepted[/]"
        elif decision == MatchDecision.REJECTED:
            return "[red]✗ Rejected[/]"
        else:
            return "[dim]...Pending[/]"

    def action_accept_match(self) -> None:
        """Accept the currently selected match."""
        self._sync_cursor_to_selected_idx()
        filtered_matches = self.match_state.get_filtered_and_sorted_matches()
        if 0 <= self.match_state.selected_match_idx < len(filtered_matches):
            match = filtered_matches[self.match_state.selected_match_idx]

            # For missing items (dynamically created), add to matches first
            self.match_state.promote_missing_to_match(match)

            # Now update the decision
            match.decision = MatchDecision.ACCEPTED
            self._refresh_tables()
        else:
            self.app.notify("No match selected to accept", severity="warning")

    def action_reject_match(self) -> None:
        """Reject the currently selected match."""
        self._sync_cursor_to_selected_idx()
        filtered_matches = self.match_state.get_filtered_and_sorted_matches()
        if 0 <= self.match_state.selected_match_idx < len(filtered_matches):
            match = filtered_matches[self.match_state.selected_match_idx]

            # For missing items (dynamically created), add to matches first
            self.match_state.promote_missing_to_match(match)

            # Now update the decision
            match.decision = MatchDecision.REJECTED
            self._refresh_tables()
        else:
            self.app.notify("No match selected to reject", severity="warning")

    def action_toggle_filter(self) -> None:
        """Toggle between filter modes."""
        modes = ["all", "pending", "accepted", "rejected"]
        current_idx = modes.index(self.match_state.filter_mode)
        self.match_state.filter_mode = modes[(current_idx + 1) % len(modes)]
        self.match_state.selected_match_idx = 0
        self._refresh_tables()
        # Update title with tier breakdown
        title = self.query_one("#review_title", Static)
        new_title = self._get_title_static()
        title.update(new_title.render())

    def action_move_down(self) -> None:
        """Move selection down."""
        filtered_matches = self.match_state.get_filtered_and_sorted_matches()
        if self.match_state.selected_match_idx < len(filtered_matches) - 1:
            self.match_state.selected_match_idx += 1
            self._refresh_tables()

    def action_move_up(self) -> None:
        """Move selection up."""
        if self.match_state.selected_match_idx > 0:
            self.match_state.selected_match_idx -= 1
            self._refresh_tables()

    def action_cycle_sort(self) -> None:
        """Cycle through sort modes (status -> confidence -> date)."""
        self.match_state.cycle_sort_mode()
        self.match_state.selected_match_idx = 0
        self._refresh_tables()
        # Update title with new sort mode
        title = self.query_one("#review_title", Static)
        new_title = self._get_title_static()
        title.update(new_title.render())

    def action_manual_match(self) -> None:
        """Open manual match screen for the selected or missing source record.

        If a match is selected, offers to rematch it to a different target.
        If no match is selected, opens manual match for the first missing record.
        """
        from src.tui.manual_match_screen import ManualMatchScreen

        # Sync cursor position before using selected_match_idx
        self._sync_cursor_to_selected_idx()

        # Determine which source index to manually match
        # Priority: selected match, or first missing record
        source_idx: int | None = None

        filtered_matches = self.match_state.get_filtered_and_sorted_matches()
        if 0 <= self.match_state.selected_match_idx < len(filtered_matches):
            # Use the selected match's source index
            source_idx = filtered_matches[self.match_state.selected_match_idx].source_idx
        elif self.match_state.match_result.missing_in_target:
            # Use the first missing record
            source_idx = self.match_state.match_result.missing_in_target[0]

        if source_idx is None:
            self.app.notify("No source record available for manual matching", severity="warning")
            return

        # Push the manual match screen
        self.app.push_screen(
            ManualMatchScreen(
                self.source_df,
                self.target_df,
                source_idx,
                self.match_state.match_result,
            )
        )

    def action_show_summary(self) -> None:
        """Show the summary screen."""
        self.app.push_screen(
            SummaryScreen(
                self.source_df,
                self.target_df,
                self.match_state.match_result,
                str(self.app.source_path) if hasattr(self.app, "source_path") else "source.csv",
                str(self.app.target_path) if hasattr(self.app, "target_path") else "target.csv",
            )
        )

    def action_show_missing(self) -> None:
        """Show the missing items screen."""
        self.app.push_screen(MissingItemsScreen(self.source_df, self.match_state.match_result))

    def action_show_unmatched_targets(self) -> None:
        """Show the unmatched targets screen."""
        self.app.push_screen(UnmatchedTargetsScreen(self.target_df, self.match_state.match_result))

    def _get_table_columns(self) -> list[str]:
        """Get column names for the single-table view."""
        return [
            "Status",
            "Tier",
            "Date",
            "Source Amt",
            "Target Amt",
            "Source Desc",
            "Target Desc",
            "Match Info",
        ]

    def _get_status_text(self, decision: MatchDecision, tier: ConfidenceTier) -> str:
        """Get status text with decision and tier indicators."""
        decision_part = {
            MatchDecision.ACCEPTED: "[green]✓[/]",
            MatchDecision.REJECTED: "[red]✗[/]",
            MatchDecision.PENDING: "[dim]...[/]",
        }.get(decision, "?")

        tier_part = {
            ConfidenceTier.HIGH: "⭐",
            ConfidenceTier.MEDIUM: "○",
            ConfidenceTier.LOW: "○",
            ConfidenceTier.NONE: "—",
        }.get(tier, "?")

        return f"{decision_part} {tier_part}"

    def _get_match_info_text(self, match: Match) -> str:
        """Get match info text showing target and confidence."""
        if match.target_idx is None:
            return "[dim]No match found[/]"

        target_row = self.target_df.iloc[match.target_idx]
        target_desc = str(target_row["description_clean"])

        # Use centralized color mapping
        _, _, conf_color = display_utils.get_tier_display(match.tier)

        confidence_str = f"[{conf_color}]{match.confidence:.2f}[/]"
        reason_short = match.reason.split()[0] if match.reason else ""

        return (
            f"{display_utils.truncate_string(target_desc, 25)} • {confidence_str} ({reason_short})"
        )


class MissingItemsScreen(Screen):
    """Screen displaying source records missing from target."""

    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("escape", "pop_screen", "Back to Review"),
        ("s", "show_summary", "Summary"),
    ]

    def __init__(self, source_df: pd.DataFrame, match_result: MatchResult) -> None:
        """Initialize the missing items screen.

        Args:
            source_df: Normalized source DataFrame
            match_result: Results from matching operation
        """
        super().__init__()
        self.source_df = source_df
        self.match_result = match_result

    def compose(self):
        """Compose the missing items screen."""
        yield Header()
        yield Static(
            f"[bold red]Missing Items ({len(self.match_result.missing_in_target)})[/]",
            id="missing_title",
        )
        yield DataTable(id="missing_table")
        yield Footer()

    def on_mount(self) -> None:
        """Populate table when mounted."""
        table = self.query_one("#missing_table", DataTable)
        table.add_columns("Index", "Date", "Amount", "Description")
        table.zebra_striping = True

        for idx in self.match_result.missing_in_target:
            row = self.source_df.iloc[idx]
            table.add_row(
                str(idx),
                display_utils.format_date(row["date_clean"]),
                display_utils.format_amount(row["amount_clean"]),
                display_utils.truncate_string(str(row["description_clean"]), 50),
            )

    def action_show_summary(self) -> None:
        """Show the summary screen."""
        self.app.push_screen(
            SummaryScreen(
                self.source_df,
                None,  # target_df not needed for summary
                self.match_result,
                "source.csv",  # Placeholder - summary screen doesn't heavily use paths
                "target.csv",
            )
        )

    def action_pop_screen(self) -> None:
        """Pop the screen and return to the previous screen."""
        self.app.pop_screen()


class UnmatchedTargetsScreen(Screen):
    """Screen displaying target records not matched to source (unreconciled items)."""

    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("escape", "pop_screen", "Back to Review"),
        ("s", "show_summary", "Summary"),
    ]

    def __init__(self, target_df: pd.DataFrame, match_result: MatchResult) -> None:
        """Initialize the unmatched targets screen.

        Args:
            target_df: Normalized target DataFrame
            match_result: Results from matching operation
        """
        super().__init__()
        self.target_df = target_df
        self.match_result = match_result

    def compose(self):
        """Compose the unmatched targets screen."""
        yield Header()
        yield Static(
            f"[bold yellow]Unmatched Targets ({len(self.match_result.missing_in_source)})[/]",
            id="unmatched_title",
        )
        yield DataTable(id="unmatched_table")
        yield Footer()

    def on_mount(self) -> None:
        """Populate table when mounted."""
        table = self.query_one("#unmatched_table", DataTable)
        table.add_columns("Index", "Date", "Amount", "Description")
        table.zebra_striping = True

        for idx in self.match_result.missing_in_source:
            row = self.target_df.iloc[idx]
            table.add_row(
                str(idx),
                display_utils.format_date(row["date_clean"]),
                display_utils.format_amount(row["amount_clean"]),
                display_utils.truncate_string(str(row["description_clean"]), 50),
            )

    def action_show_summary(self) -> None:
        """Show the summary screen."""
        self.app.push_screen(
            SummaryScreen(
                None,  # source_df not needed for summary
                self.target_df,
                self.match_result,
                "source.csv",
                "target.csv",
            )
        )

    def action_pop_screen(self) -> None:
        """Pop the screen and return to the previous screen."""
        self.app.pop_screen()


class SummaryScreen(Screen):
    """Screen showing reconciliation summary statistics."""

    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("escape", "pop_screen", "Back to Review"),
    ]

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        match_result: MatchResult,
        source_path: str,
        target_path: str,
    ) -> None:
        """Initialize the summary screen.

        Args:
            source_df: Normalized source DataFrame
            target_df: Normalized target DataFrame
            match_result: Results from matching operation
            source_path: Source file path
            target_path: Target file path
        """
        super().__init__()
        self.source_df = source_df
        self.target_df = target_df
        self.match_result = match_result
        self.source_path = source_path
        self.target_path = target_path

    def compose(self):
        """Compose the summary screen."""
        # Calculate statistics
        total_source = len(self.source_df)
        total_target = len(self.target_df)
        total_matches = len(self.match_result.matches)
        missing_in_target = len(self.match_result.missing_in_target)
        missing_in_source = len(self.match_result.missing_in_source)
        match_rate = (total_matches / total_source * 100) if total_source > 0 else 0

        # Count decisions
        accepted = sum(1 for m in self.match_result.matches if m.decision == MatchDecision.ACCEPTED)
        rejected = sum(1 for m in self.match_result.matches if m.decision == MatchDecision.REJECTED)
        pending = sum(1 for m in self.match_result.matches if m.decision == MatchDecision.PENDING)

        # High confidence matches
        high_conf = sum(1 for m in self.match_result.matches if m.confidence >= 0.8)
        medium_conf = sum(1 for m in self.match_result.matches if 0.6 <= m.confidence < 0.8)
        low_conf = sum(1 for m in self.match_result.matches if m.confidence < 0.6)

        summary = f"""[bold]
╭─────────────────────────────────────────────────────────╮
│           Double Post - Reconciliation Summary          │
╰─────────────────────────────────────────────────────────╯
[/bold]

[bold]Files[/bold]
  Source: {self.source_path}
  Target: {self.target_path}

[bold]Statistics[/bold]
  ┌─────────────────────────────────────────────────────┐
  │ Total Source Records:     {total_matches:>6} / {total_source:<6}│
  │ Total Target Records:     {total_target:>6}           │
  │ Successful Matches:       {total_matches:>6}           │
  │ Missing in Target:        {missing_in_target:>6}           │
  │ Missing in Source:        {missing_in_source:>6}           │
  │ Match Rate:               {match_rate:>5.1f}%           │
  └─────────────────────────────────────────────────────┘

[bold]Your Decisions[/bold]
  ┌─────────────────────────────────────────────────────┐
  │ [green]Accepted:       {accepted:>6}[/]                              │
  │ [red]Rejected:       {rejected:>6}[/]                              │
  │ [dim]Pending:        {pending:>6}[/]                              │
  └─────────────────────────────────────────────────────┘

[bold]Confidence Distribution[/bold]
  ┌─────────────────────────────────────────────────────┐
  │ [green]High (≥0.8):     {high_conf:>6}[/]                              │
  │ [yellow]Medium (0.6-0.8):{medium_conf:>6}[/]                              │
  │ [red]Low (<0.6):      {low_conf:>6}[/]                              │
  └─────────────────────────────────────────────────────┘

[dim]Press ESC to go back, 'q' to quit[/dim]
"""
        yield Header()
        yield Static(summary, id="summary_text")
        yield Footer()

    def action_pop_screen(self) -> None:
        """Pop the screen and return to the previous screen."""
        self.app.pop_screen()


__all__ = [
    "MatchReviewScreen",
    "MissingItemsScreen",
    "UnmatchedTargetsScreen",
    "SummaryScreen",
    "MatchState",
]
