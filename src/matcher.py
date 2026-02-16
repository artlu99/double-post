"""Matching engine with duplicate prevention.

Mitigation #1: Greedy matching algorithm that prevents one target from matching
to multiple sources by tracking matched indices and processing highest confidence first.

Description matching: When an AliasDatabase is provided (e.g. from main.reconcile),
all description logic uses the same canonical form via _description_for_matching:
intelligent match (first-two-words) and confidence (fuzzy) both resolve aliases
through the DB first. The alias DB is the only source of merchant equivalence.
"""

from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.models import ConfidenceTier, Match, MatchConfig, MatchDecision, MatchResult


def _get_row_field(row: Any, field: str) -> Any:
    """Safely get a field value from a pandas Series or itertuples namedtuple.

    Args:
        row: A pandas Series or namedtuple from itertuples()
        field: Field name to retrieve

    Returns:
        Field value or None if not found
    """
    # Try dictionary-style access first (Series)
    try:
        return row[field]
    except (KeyError, TypeError):
        pass
    # Try attribute access (namedtuple from itertuples)
    return getattr(row, field, None)


def _normalize_for_intelligent_match(text: str) -> str:
    """Normalize text for intelligent matching.

    Removes apostrophes and converts to lowercase for comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized text string
    """
    return text.lower().replace("'", "")


def _get_first_two_words(text: str) -> str:
    """Extract first two words from text for intelligent matching.

    Args:
        text: Text to extract words from

    Returns:
        First two words (normalized, joined by space) or single word if only one
    """
    words = text.lower().split()
    if len(words) >= 2:
        return " ".join(words[:2])
    elif len(words) == 1:
        return words[0]
    return ""


def _description_for_matching(description: str, alias_db: Any | None) -> str:
    """Single canonical description used for all matcher logic.

    When alias_db is provided, resolves to primary name if the description
    is a known alias; otherwise uses normalized description. When alias_db
    is None, returns normalized description only. This ensures intelligent
    match (first-two-words) and confidence (fuzzy) both use the same
    canonical form and the alias DB is the only source of merchant equivalence.

    Args:
        description: Raw description from record
        alias_db: Optional AliasDatabase for resolution

    Returns:
        Normalized string to use for comparison (lower, apostrophes removed)
    """
    s = str(description).strip()
    if not s:
        return ""
    if alias_db is not None:
        primary = alias_db.get_primary_name(s)
        if primary:
            s = primary
    return _normalize_for_intelligent_match(s)


def _check_intelligent_match(
    source: Any,
    target: Any,
    config: MatchConfig,  # noqa: ARG001 - unused but kept for API compatibility
    alias_db: Any | None = None,
) -> float | None:
    """Check for intelligent match with 0.90 confidence.

    Uses the same canonical description as the rest of the matcher
    (_description_for_matching with alias_db). Criteria:
    1. Amounts match EXACTLY
    2. First TWO words of canonical descriptions match.

    Args:
        source: Source record (Series or namedtuple from itertuples)
        target: Target record (Series or namedtuple from itertuples)
        config: Matching configuration
        alias_db: Optional AliasDatabase; when provided, descriptions are
            resolved to primary name before first-two-words comparison.

    Returns:
        0.90 if intelligent match criteria met, None otherwise
    """
    source_amount = _get_row_field(source, "amount_clean")
    target_amount = _get_row_field(target, "amount_clean")

    if pd.isna(source_amount) or pd.isna(target_amount):
        return None

    if source_amount != target_amount:
        return None

    source_desc = _get_row_field(source, "description_clean")
    target_desc = _get_row_field(target, "description_clean")

    if pd.isna(source_desc) or pd.isna(target_desc):
        return None

    source_canonical = _description_for_matching(str(source_desc), alias_db)
    target_canonical = _description_for_matching(str(target_desc), alias_db)

    source_first_two = _get_first_two_words(source_canonical)
    target_first_two = _get_first_two_words(target_canonical)

    if len(source_canonical.split()) < 2 or len(target_canonical.split()) < 2:
        return None

    if source_first_two == target_first_two:
        return 0.90

    return None


def normalize_sign_conventions(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_convention: dict[str, str | int],
    target_convention: dict[str, str | int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize amount signs between source and target DataFrames.

    Ensures both DataFrames use the same sign convention for debits/credits.
    If conventions differ, flips the signs in the target DataFrame to match
    the source convention.

    Args:
        source_df: Source DataFrame (will be modified in place)
        target_df: Target DataFrame (will be modified in place)
        source_convention: Sign convention dict from detect_sign_convention() for source
        target_convention: Sign convention dict from detect_sign_convention() for target

    Returns:
        Tuple of (source_df, target_df) with normalized signs
    """

    # Get the debit sign for each file
    source_debit_sign = source_convention.get("debit_sign", "negative")
    target_debit_sign = target_convention.get("debit_sign", "negative")

    # If both use the same convention, no changes needed
    if source_debit_sign == target_debit_sign:
        return source_df, target_df

    # Handle Chase format (uses separate Debit/Credit columns)
    if source_debit_sign == "debit_col" or target_debit_sign == "debit_col":
        # Chase format: debit and credit are in separate columns, already normalized
        # during CSV loading. No need to flip signs here.
        return source_df, target_df

    # If conventions differ, flip target signs to match source
    # Example: source uses "-" for debits, target uses "+" for debits
    # We'll flip target so both use the same sign for the same transaction type
    if source_debit_sign != target_debit_sign:
        # Flip the signs in target_df
        target_df = target_df.copy()
        target_df["amount_clean"] = target_df["amount_clean"].apply(
            lambda x: -x if pd.notna(x) else x
        )

    return source_df, target_df


def calculate_confidence(
    source: Any,
    target: Any,
    config: MatchConfig,
    alias_db: Any | None = None,
) -> float:
    """Calculate confidence score for a potential match.

    Combines amount match, date proximity, and description similarity.
    Description comparison uses the same canonical form as intelligent match
    (_description_for_matching with alias_db when provided).

    Args:
        source: Source record (Series or namedtuple from itertuples)
        target: Target record (Series or namedtuple from itertuples)
        config: Matching configuration
        alias_db: Optional AliasDatabase; when provided, descriptions are
            resolved to primary name before comparison (same as rest of matcher).

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Amount match: 1.0 if equal, 0.0 otherwise
    amount_score: float = 0.0
    source_amount = _get_row_field(source, "amount_clean")
    target_amount = _get_row_field(target, "amount_clean")
    if (
        pd.notna(source_amount)
        and pd.notna(target_amount)
        and abs(source_amount - target_amount) <= config.amount_tolerance
    ):
        amount_score = 1.0

    # Date proximity: 1.0 if same date, decreases with distance
    date_score: float = 0.0
    source_date = _get_row_field(source, "date_clean")
    target_date = _get_row_field(target, "date_clean")
    if pd.notna(source_date) and pd.notna(target_date):
        days_diff = abs((source_date - target_date).days)

        if days_diff == 0:
            date_score = 1.0
        elif days_diff <= config.date_window_days:
            date_score = 1.0 - (days_diff / config.date_window_days)

    # Description similarity: same canonical form as intelligent match (alias DB when provided)
    desc_score: float = 0.0
    source_desc = _get_row_field(source, "description_clean")
    target_desc = _get_row_field(target, "description_clean")
    if pd.notna(source_desc) and pd.notna(target_desc):
        source_canonical = _description_for_matching(str(source_desc), alias_db)
        target_canonical = _description_for_matching(str(target_desc), alias_db)
        if source_canonical == target_canonical:
            desc_score = 1.0
        else:
            desc_score = fuzz.ratio(source_canonical, target_canonical) / 100.0

    # Weighted combination
    confidence = (amount_score * 0.3) + (date_score * 0.3) + (desc_score * 0.4)

    return round(confidence, 4)


def classify_confidence_tier(confidence: float) -> ConfidenceTier:
    """Classify confidence score into tier.

    Args:
        confidence: Confidence score from 0.0 to 1.0

    Returns:
        ConfidenceTier classification (HIGH/MEDIUM/LOW/NONE)
    """
    if confidence >= 0.9:
        return ConfidenceTier.HIGH
    elif confidence >= 0.5:
        return ConfidenceTier.MEDIUM
    elif confidence >= 0.1:
        return ConfidenceTier.LOW
    else:
        return ConfidenceTier.NONE


def calculate_reason(source: Any, target: Any) -> str:
    """Generate human-readable explanation of match quality.

    Args:
        source: Source record (Series or namedtuple from itertuples)
        target: Target record (Series or namedtuple from itertuples)

    Returns:
        Human-readable reason string
    """
    reasons = []

    # Amount match
    source_amount = _get_row_field(source, "amount_clean")
    target_amount = _get_row_field(target, "amount_clean")
    if pd.notna(source_amount) and pd.notna(target_amount):
        if abs(source_amount - target_amount) == 0:
            reasons.append("exact amount")
        else:
            reasons.append("different amount")

    # Date match
    source_date = _get_row_field(source, "date_clean")
    target_date = _get_row_field(target, "date_clean")
    if pd.notna(source_date) and pd.notna(target_date):
        days_diff = abs((source_date - target_date).days)
        if days_diff == 0:
            reasons.append("same date")
        elif days_diff <= 3:
            reasons.append(f"{days_diff} days apart")
        else:
            reasons.append(f"{days_diff} days apart")

    # Description match
    source_desc = _get_row_field(source, "description_clean")
    target_desc = _get_row_field(target, "description_clean")
    if pd.notna(source_desc) and pd.notna(target_desc):
        source_desc_str = str(source_desc)
        target_desc_str = str(target_desc)
        similarity = fuzz.ratio(source_desc_str, target_desc_str)
        if similarity >= 95:
            reasons.append("nearly identical description")
        elif similarity >= 80:
            reasons.append("similar description")
        else:
            reasons.append("different description")

    return ", ".join(reasons)


def find_matches(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: MatchConfig,
    min_confidence: float = 0.1,
    alias_db: Any | None = None,
) -> MatchResult:
    """Find matches between source and target DataFrames.

    Uses 4-tier confidence categorization system:
    - HIGH (≥0.9): Auto-accepted
    - MEDIUM (0.5-0.9): Pending review
    - LOW (0.1-0.5): Weak suggestion
    - NONE (<0.1): Not included in results

    Mitigation #1: Uses greedy matching to prevent duplicate false matches.
    - Collects ALL (source, target) pairs with confidence >= min_confidence
    - Sorts by confidence descending
    - Processes in order: add match if both source and target still unmatched
    - Tracks matched sources and targets to prevent reuse (handles duplicate transactions)

    Performance optimization: Pre-calculates vectorized amount bounds (±amount_tolerance)
    to skip expensive fuzzy matching for pairs where amount difference exceeds tolerance.

    Args:
        source_df: Normalized source DataFrame
        target_df: Normalized target DataFrame
        config: Matching configuration
        min_confidence: Minimum confidence to include (default 0.1)
        alias_db: Optional AliasDatabase for merchant name lookups (same as calculate_confidence)

    Returns:
        MatchResult with matches categorized by tier
    """
    matches: list[Match] = []
    matched_sources: set[int] = set()
    matched_targets: set[int] = set()

    # Early return for empty DataFrames or missing columns
    if len(source_df) == 0 or len(target_df) == 0:
        return MatchResult(matches=[], missing_in_target=[], missing_in_source=[])

    if "amount_clean" not in source_df.columns or "amount_clean" not in target_df.columns:
        return MatchResult(matches=[], missing_in_target=[], missing_in_source=[])

    # Pre-calculate vectorized amount bounds for early-exit optimization
    # This avoids expensive fuzzy matching for pairs with wildly different amounts
    # Convert Decimal to float for numpy vectorized operations
    source_amounts = source_df["amount_clean"].astype(float).values
    tolerance = float(config.amount_tolerance)
    # Calculate bounds and ensure lower <= upper (handles negative amounts correctly)
    lower_bound = source_amounts * (1 - tolerance)
    upper_bound = source_amounts * (1 + tolerance)
    source_amount_lower = np.minimum(lower_bound, upper_bound)
    source_amount_upper = np.maximum(lower_bound, upper_bound)

    # Vectorized filtering: Pre-filter target DataFrame to only include rows
    # that could potentially match based on amount tolerance
    # This significantly reduces the inner loop size
    global_source_min = float(source_amounts.min())
    global_source_max = float(source_amounts.max())
    # Calculate global bounds (accounting for tolerance)
    global_lower = min(global_source_min * (1 - tolerance), global_source_min * (1 + tolerance))
    global_upper = max(global_source_max * (1 - tolerance), global_source_max * (1 + tolerance))

    # Filter targets to only those within the global amount range
    target_amounts = target_df["amount_clean"].astype(float).values
    target_mask = (target_amounts >= global_lower) & (target_amounts <= global_upper)
    filtered_target_df = target_df[target_mask].copy()
    filtered_to_original_indices = np.where(target_mask)[0].tolist()

    # If no targets pass the amount filter, return early
    if len(filtered_target_df) == 0:
        return MatchResult(
            matches=[],
            missing_in_target=list(range(len(source_df))),
            missing_in_source=list(range(len(target_df))),
        )

    # OPTIMIZATION: Pre-compute canonical descriptions to avoid repeated alias DB queries
    # and string normalization in the nested loop
    source_canonical_descs = [
        _description_for_matching(
            str(row.description_clean) if pd.notna(row.description_clean) else "", alias_db
        )
        for row in source_df.itertuples(index=False)
    ]
    target_canonical_descs = [
        _description_for_matching(
            str(row.description_clean) if pd.notna(row.description_clean) else "", alias_db
        )
        for row in filtered_target_df.itertuples(index=False)
    ]

    # Collect ALL (source, target) pairs with confidence >= min_confidence
    candidate_pairs: list[tuple[float, int, int]] = []

    # Use itertuples() for faster iteration (returns namedtuples instead of Series)
    for source_idx, source_row in enumerate(source_df.itertuples(index=False)):
        source_lower = source_amount_lower[source_idx]
        source_upper = source_amount_upper[source_idx]
        source_canonical = source_canonical_descs[source_idx]
        source_first_two = (
            _get_first_two_words(source_canonical) if len(source_canonical.split()) >= 2 else None
        )

        for filtered_idx, target_row in enumerate(filtered_target_df.itertuples(index=False)):
            target_amount = target_row.amount_clean
            if pd.notna(target_amount) and (
                target_amount < source_lower or target_amount > source_upper
            ):
                continue

            intelligent_confidence = None
            source_amt = _get_row_field(source_row, "amount_clean")
            target_amt = _get_row_field(target_row, "amount_clean")

            if (
                pd.notna(source_amt)
                and pd.notna(target_amt)
                and source_amt == target_amt
                and source_first_two is not None
            ):
                target_canonical = target_canonical_descs[filtered_idx]
                if len(target_canonical.split()) >= 2:
                    target_first_two = _get_first_two_words(target_canonical)
                    if source_first_two == target_first_two:
                        intelligent_confidence = 0.90

            if intelligent_confidence is not None:
                confidence = intelligent_confidence
            else:
                confidence = calculate_confidence(source_row, target_row, config, alias_db=alias_db)

            if confidence >= min_confidence:
                target_idx = int(filtered_to_original_indices[filtered_idx])
                candidate_pairs.append((confidence, source_idx, target_idx))

    # Sort by confidence descending (greedy: highest confidence first)
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    # Process in order: add match if both source and target still unmatched
    for confidence, source_idx, target_idx in candidate_pairs:
        if source_idx in matched_sources or target_idx in matched_targets:
            continue

        source_row = source_df.iloc[source_idx]
        target_row = target_df.iloc[target_idx]

        tier = classify_confidence_tier(confidence)
        decision = MatchDecision.ACCEPTED if tier == ConfidenceTier.HIGH else MatchDecision.PENDING

        matches.append(
            Match(
                source_idx=source_idx,
                target_idx=target_idx,
                confidence=confidence,
                reason=calculate_reason(source_row, target_row),
                tier=tier,
                decision=decision,
            )
        )

        matched_sources.add(source_idx)
        matched_targets.add(target_idx)

    # Find source records that weren't matched
    all_source_indices = set(source_df.index)
    missing_in_target = sorted(all_source_indices - matched_sources)

    # Find target records that weren't matched
    all_target_indices = set(target_df.index)
    missing_in_source = sorted(all_target_indices - matched_targets)

    return MatchResult(
        matches=matches,
        missing_in_target=missing_in_target,
        missing_in_source=missing_in_source,
        duplicate_matches=[],
    )


def create_manual_match(
    source_idx: int,
    target_idx: int,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Match:
    """Create a manual match between a source and target record.

    This allows users to manually link records that the auto-matching
    algorithm missed. Confidence is still calculated based on the data.

    Args:
        source_idx: Index in source DataFrame
        target_idx: Index in target DataFrame
        source_df: Source DataFrame
        target_df: Target DataFrame

    Returns:
        Match object with calculated confidence and manual flag set to True

    Raises:
        IndexError: If source_idx or target_idx is out of range
    """
    # Validate indices
    if source_idx < 0 or source_idx >= len(source_df):
        raise IndexError(
            f"Source index {source_idx} out of range for DataFrame with {len(source_df)} rows"
        )
    if target_idx < 0 or target_idx >= len(target_df):
        raise IndexError(
            f"Target index {target_idx} out of range for DataFrame with {len(target_df)} rows"
        )

    # Get the records
    source_row = source_df.iloc[source_idx]
    target_row = target_df.iloc[target_idx]

    # Calculate confidence using existing logic
    config = MatchConfig()
    confidence = calculate_confidence(source_row, target_row, config)

    # Classify tier and set decision
    tier = classify_confidence_tier(confidence)

    # Generate reason
    reason = calculate_reason(source_row, target_row)
    reason = f"Manual match: {reason}"

    # Create match with manual flag and tier
    return Match(
        source_idx=source_idx,
        target_idx=target_idx,
        confidence=confidence,
        reason=reason,
        manual=True,
        tier=tier,
    )


__all__ = [
    "Match",
    "MatchResult",
    "MatchConfig",
    "calculate_confidence",
    "calculate_reason",
    "classify_confidence_tier",
    "find_matches",
    "create_manual_match",
    "normalize_sign_conventions",
]
