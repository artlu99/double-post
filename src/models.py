"""Data structures for Double Post."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal


@dataclass
class ColumnMapping:
    """Detected column mappings for a CSV.

    Attributes:
        date: Name of the date column
        amount: Name of the amount column
        description: Name of the description column
        debit: Name of the debit column (Chase format)
        credit: Name of the credit column (Chase format)
        type: Name of the type column (Amex format)
        format_type: Detected bank format type
    """

    date: str | None
    amount: str | None
    description: str | None
    debit: str | None
    credit: str | None
    type: str | None
    format_type: Literal["chase", "generic"]


class MatchDecision(Enum):
    """Decision status for a match."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ConfidenceTier(str, Enum):
    """Confidence tier classification for matches."""

    HIGH = "high"  # ≥ 0.9
    MEDIUM = "medium"  # 0.5 - 0.9
    LOW = "low"  # 0.1 - 0.5
    NONE = "none"  # < 0.1


@dataclass
class NormalizedRecord:
    """Standardized transaction record.

    Attributes:
        date: Parsed transaction date
        amount: Normalized amount (negative = expense, positive = income)
        description: Cleaned description text
        original_idx: Original row index in source DataFrame
    """

    date: datetime
    amount: Decimal
    description: str
    original_idx: int


@dataclass
class Match:
    """Represents a match between source and target records.

    Attributes:
        source_idx: Index in source DataFrame
        target_idx: Index in target DataFrame (None if unmatched)
        confidence: Confidence score from 0.0 to 1.0
        reason: Human-readable explanation of match quality
        decision: User decision on this match (pending/accepted/rejected)
        manual: True if this match was created manually by user
        tier: Confidence tier classification (auto-accepted if HIGH)
    """

    source_idx: int
    target_idx: int | None
    confidence: float
    reason: str
    decision: MatchDecision = field(default_factory=lambda: MatchDecision.PENDING)
    manual: bool = False
    tier: ConfidenceTier = field(default_factory=lambda: ConfidenceTier.MEDIUM)


@dataclass
class MatchResult:
    """Result of matching operation.

    Attributes:
        matches: List of successful matches
        missing_in_target: Source indices not found in target (BOOKS_AND_RECORDS items not matched to BANK)
        missing_in_source: Target indices not matched to source (BANK items not matched to BOOKS_AND_RECORDS)
        duplicate_matches: Low confidence matches due to duplicates
    """

    matches: list[Match]
    missing_in_target: list[int]
    missing_in_source: list[int] = field(default_factory=list)
    duplicate_matches: list[Match] = field(default_factory=list)


@dataclass
class MatchConfig:
    """Configuration for matching algorithm.

    Attributes:
        threshold: Minimum confidence score for auto-accept
        date_window_days: Maximum days apart for date matching
        amount_tolerance: Tolerance for amount comparison
    """

    threshold: float = 0.7
    date_window_days: int = 3
    amount_tolerance: Decimal = Decimal("0.01")


@dataclass
class RecordEdit:
    """Tracks edits made to a record.

    Attributes:
        source_idx: Index in source DataFrame (None if editing target)
        target_idx: Index in target DataFrame (None if editing source)
        field: Field being edited (date, amount, description)
        original_value: Original value before edit
        new_value: New value after edit
    """

    source_idx: int | None
    target_idx: int | None
    field: Literal["date", "amount", "description"]
    original_value: Any
    new_value: Any
