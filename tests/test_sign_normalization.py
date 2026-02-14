"""Tests for sign convention normalization."""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.matcher import normalize_sign_conventions


class TestSignNormalization:
    """Tests for sign convention normalization between source and target DataFrames."""

    def test_no_normalization_when_conventions_match(self):
        """Test that no changes are made when both use same convention."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "coffee",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("-50.00"),
                    "description_clean": "lunch",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "coffee",
                },
            ]
        )

        source_convention = {"debit_sign": "negative", "negative_count": 2}
        target_convention = {"debit_sign": "negative", "negative_count": 1}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # No changes expected
        assert result_source["amount_clean"].tolist() == source_df["amount_clean"].tolist()
        assert result_target["amount_clean"].tolist() == target_df["amount_clean"].tolist()

    def test_normalize_opposite_conventions(self):
        """Test that target signs are flipped when conventions differ."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "coffee",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee",
                },
            ]
        )

        # Source uses "-" for debits, target uses "+" for debits
        source_convention = {"debit_sign": "negative", "negative_count": 1}
        target_convention = {"debit_sign": "positive", "positive_count": 1}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # Source unchanged
        assert result_source["amount_clean"].iloc[0] == Decimal("-100.00")
        # Target flipped to match source
        assert result_target["amount_clean"].iloc[0] == Decimal("-100.00")

    def test_normalize_with_positive_debits_to_negative(self):
        """Test normalizing target with positive debits to negative convention."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-62.50"),
                    "description_clean": "noodles",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-62.50"),
                    "description_clean": "noodles",
                },
            ]
        )

        # Source: negative = debits, Target: positive = debits (need to flip target)
        source_convention = {"debit_sign": "negative", "negative_count": 1}
        target_convention = {"debit_sign": "positive", "positive_count": 1}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # Target should be flipped
        assert result_target["amount_clean"].iloc[0] == Decimal("62.50")

    def test_normalize_with_negative_debits_to_positive(self):
        """Test normalizing target with negative debits to positive convention."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("62.50"),
                    "description_clean": "noodles",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("62.50"),
                    "description_clean": "noodles",
                },
            ]
        )

        # Source: positive = debits, Target: negative = debits (need to flip target)
        source_convention = {"debit_sign": "positive", "positive_count": 1}
        target_convention = {"debit_sign": "negative", "negative_count": 1}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # Target should be flipped
        assert result_target["amount_clean"].iloc[0] == Decimal("-62.50")

    def test_chase_format_no_normalization(self):
        """Test that Chase format (separate debit/credit columns) is not normalized."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee",
                },
            ]
        )

        # Chase format uses debit_col
        source_convention = {"debit_sign": "negative", "negative_count": 1}
        target_convention = {"debit_sign": "debit_col", "debit_count": 1}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # No changes expected for Chase format
        assert result_target["amount_clean"].iloc[0] == Decimal("100.00")

    def test_normalize_preserves_nan_values(self):
        """Test that NaN values are preserved during normalization."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "coffee",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": None,
                    "description_clean": "unknown",
                },
            ]
        )

        source_convention = {"debit_sign": "negative", "negative_count": 1}
        target_convention = {"debit_sign": "positive", "positive_count": 0}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # NaN should remain NaN
        assert pd.isna(result_target["amount_clean"].iloc[0])

    def test_normalize_multiple_records(self):
        """Test normalization with multiple records."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "coffee",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("-50.00"),
                    "description_clean": "lunch",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "lunch",
                },
            ]
        )

        source_convention = {"debit_sign": "negative", "negative_count": 2}
        target_convention = {"debit_sign": "positive", "positive_count": 2}

        result_source, result_target = normalize_sign_conventions(
            source_df, target_df, source_convention, target_convention
        )

        # All target amounts should be flipped
        assert result_target["amount_clean"].iloc[0] == Decimal("-100.00")
        assert result_target["amount_clean"].iloc[1] == Decimal("-50.00")
