"""Matching engine with duplicate prevention.

Mitigation #1: Greedy matching algorithm that prevents one target from matching
to multiple sources by tracking matched indices and processing highest confidence first.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.models import ConfidenceTier, Match, MatchConfig, MatchDecision, MatchResult


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


def _check_intelligent_match(
    source: pd.Series, target: pd.Series, config: MatchConfig
) -> float | None:
    """Check for intelligent match with 0.90 confidence.

    Intelligent match criteria:
    1. Amounts must match EXACTLY (zero tolerance for intelligent matching)
    2. First TWO words of normalized descriptions must match

    The first-two-words match handles cases like:
    - "McDonald's #1234" vs "McDonalds Restaurant" (apostrophes removed, but still won't match)
    - "Trader Joe's Market" vs "Trader Joes Downtown" (first two: "trader joe's" vs "trader joes")
    - "Simply Noodles 00-08" vs "Simply Noodles 267" (first two: "simply noodles" match)

    Args:
        source: Source record
        target: Target record
        config: Match configuration

    Returns:
        0.90 if intelligent match criteria met, None otherwise
    """
    from decimal import Decimal

    # Check amount match first (EXACT match required for intelligent matching)
    if pd.isna(source["amount_clean"]) or pd.isna(target["amount_clean"]):
        return None

    # Require EXACT amount match (not just within tolerance)
    if source["amount_clean"] != target["amount_clean"]:
        return None

    # Check description match (first TWO normalized words)
    if pd.isna(source["description_clean"]) or pd.isna(target["description_clean"]):
        return None

    source_desc = str(source["description_clean"])
    target_desc = str(target["description_clean"])

    # Normalize and extract first two words
    source_normalized = _normalize_for_intelligent_match(source_desc)
    target_normalized = _normalize_for_intelligent_match(target_desc)

    source_words = source_normalized.split()
    target_words = target_normalized.split()

    # Need at least two words in both descriptions
    if len(source_words) < 2 or len(target_words) < 2:
        return None

    # Check if first TWO words match
    source_first_two = " ".join(source_words[:2])
    target_first_two = " ".join(target_words[:2])

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
    from decimal import Decimal

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
    source: pd.Series,
    target: pd.Series,
    config: MatchConfig,
    alias_db: Any | None = None,
) -> float:
    """Calculate confidence score for a potential match.

    Combines amount match, date proximity, and description similarity.
    Optionally boosts confidence when merchant aliases are found.

    Args:
        source: Source record (from DataFrame row)
        target: Target record (from DataFrame row)
        config: Matching configuration
        alias_db: Optional AliasDatabase for merchant name lookups

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Amount match: 1.0 if equal, 0.0 otherwise
    amount_score: float = 0.0
    if pd.notna(source["amount_clean"]) and pd.notna(target["amount_clean"]):
        if abs(source["amount_clean"] - target["amount_clean"]) <= config.amount_tolerance:
            amount_score = 1.0

    # Date proximity: 1.0 if same date, decreases with distance
    date_score: float = 0.0
    if pd.notna(source["date_clean"]) and pd.notna(target["date_clean"]):
        source_date: datetime = source["date_clean"]
        target_date: datetime = target["date_clean"]
        days_diff = abs((source_date - target_date).days)

        if days_diff == 0:
            date_score = 1.0
        elif days_diff <= config.date_window_days:
            date_score = 1.0 - (days_diff / config.date_window_days)

    # Description similarity: RapidFuzz ratio
    desc_score: float = 0.0
    if pd.notna(source["description_clean"]) and pd.notna(target["description_clean"]):
        source_desc: str = str(source["description_clean"])
        target_desc: str = str(target["description_clean"])
        similarity = fuzz.ratio(source_desc, target_desc) / 100.0

        # Check for merchant alias match if database provided
        if alias_db is not None:
            # Check if target is an alias for source, or vice versa
            primary_for_target = alias_db.get_primary_name(target_desc)
            primary_for_source = alias_db.get_primary_name(source_desc)

            # If source description matches target's alias, or vice versa
            if primary_for_target == source_desc or primary_for_source == target_desc:
                # Perfect match due to alias!
                similarity = 1.0
            elif primary_for_target and primary_for_target == primary_for_source:
                # Both are aliases for same primary name
                similarity = max(similarity, 0.9)

        desc_score = similarity

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


def calculate_reason(source: pd.Series, target: pd.Series) -> str:
    """Generate human-readable explanation of match quality.

    Args:
        source: Source record
        target: Target record

    Returns:
        Human-readable reason string
    """
    reasons = []

    # Amount match
    if pd.notna(source["amount_clean"]) and pd.notna(target["amount_clean"]):
        if abs(source["amount_clean"] - target["amount_clean"]) == 0:
            reasons.append("exact amount")
        else:
            reasons.append("different amount")

    # Date match
    if pd.notna(source["date_clean"]) and pd.notna(target["date_clean"]):
        days_diff = abs((source["date_clean"] - target["date_clean"]).days)
        if days_diff == 0:
            reasons.append("same date")
        elif days_diff <= 3:
            reasons.append(f"{days_diff} days apart")
        else:
            reasons.append(f"{days_diff} days apart")

    # Description match
    if pd.notna(source["description_clean"]) and pd.notna(target["description_clean"]):
        source_desc = str(source["description_clean"])
        target_desc = str(target["description_clean"])
        similarity = fuzz.ratio(source_desc, target_desc)
        if similarity >= 95:
            reasons.append("nearly identical description")
        elif similarity >= 80:
            reasons.append("similar description")
        else:
            reasons.append("different description")

    return ", ".join(reasons)


def find_matches(
    source_df: pd.DataFrame, target_df: pd.DataFrame, config: MatchConfig, min_confidence: float = 0.1
) -> MatchResult:
    """Find matches between source and target DataFrames.

    Uses 4-tier confidence categorization system:
    - HIGH (≥0.9): Auto-accepted
    - MEDIUM (0.5-0.9): Pending review
    - LOW (0.1-0.5): Weak suggestion
    - NONE (<0.1): Not included in results

    Mitigation #1: Uses greedy matching to prevent duplicate false matches.
    - Finds best match for ALL source rows
    - Sorts by confidence descending
    - Processes highest confidence matches first
    - Tracks matched target indices to prevent reuse

    Args:
        source_df: Normalized source DataFrame
        target_df: Normalized target DataFrame
        config: Matching configuration
        min_confidence: Minimum confidence to include (default 0.1)

    Returns:
        MatchResult with matches categorized by tier
    """
    matches: list[Match] = []
    matched_targets: set[int] = set()  # Critical: track used targets

    # Find best match for each source row
    best_matches_for_source: dict[int, tuple[float, int]] = {}

    for source_idx, source_row in source_df.iterrows():
        best_confidence = 0.0
        best_target_idx = -1

        # Find the best target match for this source
        for target_idx, target_row in target_df.iterrows():
            # Calculate general confidence
            confidence = calculate_confidence(source_row, target_row, config)

            # Check for intelligent match (0.90 confidence) - use if higher
            intelligent_confidence = _check_intelligent_match(source_row, target_row, config)
            if intelligent_confidence is not None and intelligent_confidence > confidence:
                confidence = intelligent_confidence

            if confidence > best_confidence:
                best_confidence = confidence
                best_target_idx = int(target_idx)

        # Store if above minimum threshold
        if best_confidence >= min_confidence and best_target_idx >= 0:
            best_matches_for_source[int(source_idx)] = (best_confidence, best_target_idx)

    # Sort by confidence descending (greedy: highest confidence first)
    sorted_matches = sorted(
        best_matches_for_source.items(),
        key=lambda x: x[1][0],
        reverse=True
    )

    # Process matches in confidence order, preventing duplicate target usage
    for source_idx, (confidence, target_idx) in sorted_matches:
        # Skip if target already matched
        if target_idx in matched_targets:
            continue

        source_row = source_df.iloc[source_idx]
        target_row = target_df.iloc[target_idx]

        # Classify confidence tier
        tier = classify_confidence_tier(confidence)

        # Auto-accept HIGH tier matches
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

        matched_targets.add(target_idx)

    # Find source records that weren't matched (none had confidence ≥ min_confidence)
    all_source_indices = set(source_df.index)
    matched_source_indices = {m.source_idx for m in matches}
    missing_in_target = sorted(all_source_indices - matched_source_indices)

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
        raise IndexError(f"Source index {source_idx} out of range for DataFrame with {len(source_df)} rows")
    if target_idx < 0 or target_idx >= len(target_df):
        raise IndexError(f"Target index {target_idx} out of range for DataFrame with {len(target_df)} rows")

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
