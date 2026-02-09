"""Test data factories for Double Post.

Provides factory methods to create test data for various scenarios
without depending on external fixture files.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd

from src.models import Match, MatchDecision, MatchResult


class TestDataFactory:
    """Factory for creating test data across different scenarios."""

    @staticmethod
    def create_normalized_record(
        date: datetime,
        amount: Decimal,
        description: str,
        original_idx: int = 0,
    ) -> pd.Series:
        """Create a normalized record as a pandas Series.

        Args:
            date: Transaction date
            amount: Transaction amount
            description: Transaction description
            original_idx: Original row index

        Returns:
            pandas Series with normalized record data
        """
        return pd.Series({
            "date_clean": date,
            "amount_clean": amount,
            "description_clean": description,
            "original_idx": original_idx,
        })

    @staticmethod
    def create_source_dataframe(records: list[dict] | None = None) -> pd.DataFrame:
        """Create a source DataFrame for testing.

        Args:
            records: Optional list of record dicts. If None, creates default test data.

        Returns:
            DataFrame with source data
        """
        if records is None:
            records = [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "payment",
                    "original_idx": 1,
                },
                {
                    "date_clean": datetime(2024, 1, 17),
                    "amount_clean": Decimal("-25.50"),
                    "description_clean": "grocery store",
                    "original_idx": 2,
                },
            ]
        return pd.DataFrame(records)

    @staticmethod
    def create_target_dataframe(records: list[dict] | None = None) -> pd.DataFrame:
        """Create a target DataFrame for testing.

        Args:
            records: Optional list of record dicts. If None, creates default test data.

        Returns:
            DataFrame with target data
        """
        if records is None:
            records = [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-15.99"),
                    "description_clean": "netflix",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 1, 17),
                    "amount_clean": Decimal("-25.50"),
                    "description_clean": "grocery",
                    "original_idx": 1,
                },
            ]
        return pd.DataFrame(records)

    @staticmethod
    def create_match(
        source_idx: int = 0,
        target_idx: int | None = 0,
        confidence: float = 0.85,
        reason: str = "test match",
        decision: MatchDecision = MatchDecision.PENDING,
    ) -> Match:
        """Create a Match object for testing.

        Args:
            source_idx: Source DataFrame index
            target_idx: Target DataFrame index (None if unmatched)
            confidence: Confidence score 0-1
            reason: Human-readable reason
            decision: Match decision status

        Returns:
            Match object
        """
        return Match(
            source_idx=source_idx,
            target_idx=target_idx,
            confidence=confidence,
            reason=reason,
            decision=decision,
        )

    @staticmethod
    def create_match_result(
        matches: list[Match] | None = None,
        missing_in_target: list[int] | None = None,
    ) -> MatchResult:
        """Create a MatchResult for testing.

        Args:
            matches: List of Match objects
            missing_in_target: List of source indices not in target

        Returns:
            MatchResult object
        """
        if matches is None:
            matches = [
                TestDataFactory.create_match(0, 0, 0.95, "exact match"),
                TestDataFactory.create_match(1, 1, 0.75, "similar description"),
            ]
        if missing_in_target is None:
            missing_in_target = [2, 3]
        return MatchResult(matches=matches, missing_in_target=missing_in_target, duplicate_matches=[])

    @staticmethod
    def create_chase_csv_row(
        date: str = "01/15/2024",
        description: str = "Test Transaction",
        debit: str | None = None,
        credit: str | None = "100.00",
    ) -> dict:
        """Create a Chase format CSV row.

        Args:
            date: Transaction date
            description: Transaction description
            debit: Debit amount (expense)
            credit: Credit amount (income)

        Returns:
            Dictionary representing a Chase CSV row
        """
        return {
            "Transaction Date": date,
            "Post Date": date,
            "Description": description,
            "Type": "Sale",
            "Debit": debit,
            "Credit": credit,
        }

    @staticmethod
    def create_amex_csv_row(
        date: str = "01/15/2024",
        description: str = "Test Transaction",
        amount: str = "100.00",
        type_val: str = "Purchase",
    ) -> dict:
        """Create an Amex format CSV row.

        Args:
            date: Transaction date
            description: Transaction description
            amount: Transaction amount
            type_val: Transaction type

        Returns:
            Dictionary representing an Amex CSV row
        """
        return {
            "Date": date,
            "Description": description,
            "Amount": amount,
            "Type": type_val,
        }

    @staticmethod
    def create_generic_csv_row(
        date: str = "2024-01-15",
        amount: str = "-100.00",
        description: str = "Test Transaction",
    ) -> dict:
        """Create a generic format CSV row.

        Args:
            date: Transaction date (ISO format)
            amount: Transaction amount (signed)
            description: Transaction description

        Returns:
            Dictionary representing a generic CSV row
        """
        return {
            "date": date,
            "amount": amount,
            "description": description,
        }


class PropertyTestData:
    """Test data generators for property-based testing with hypothesis."""

    @staticmethod
    def valid_date_strings() -> list[str]:
        """Return list of valid date string formats for property testing."""
        return [
            "2024-01-15",  # ISO
            "01/15/2024",  # US MDY
            "15/01/2024",  # EU DMY
            "15-Jan-2024",  # Abbreviated month
            "January 15, 2024",  # Full month name
        ]

    @staticmethod
    def valid_amount_strings() -> list[str]:
        """Return list of valid amount string formats for property testing."""
        return [
            "100.00",
            "-100.00",
            "$100.00",
            "-$100.00",
            "€100,00",  # European format
            "100",  # No decimal
            "0.99",  # Small amount
        ]

    @staticmethod
    def edge_case_dates() -> list[str]:
        """Return edge case date strings."""
        return [
            "",  # Empty
            "N/A",  # Not applicable
            "   ",  # Whitespace
            "invalid",  # Invalid date
            "13/13/2024",  # Invalid month/day
        ]
