"""Tests for matcher edge cases.

Tests for calculate_reason and find_matches edge cases.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.matcher import calculate_confidence, calculate_reason, find_matches
from src.models import MatchConfig, MatchResult
from tests.factories import TestDataFactory


class TestCalculateReason:
    """Test calculate_reason function."""

    def test_exact_match_reason(self) -> None:
        """Test reason for exact match."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee shop",
        })
        target = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee shop",
        })

        reason = calculate_reason(source, target)

        # Should indicate exact match on all fields
        assert "exact amount" in reason
        assert "same date" in reason
        assert "identical" in reason or "similar" in reason

    def test_amount_mismatch_reason(self) -> None:
        """Test reason when amounts differ."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee",
        })
        target = pd.Series({
            "amount_clean": Decimal("50.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee",
        })

        reason = calculate_reason(source, target)

        assert "different amount" in reason
        assert "same date" in reason

    def test_date_difference_reason(self) -> None:
        """Test reason when dates differ."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee",
        })
        target = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 18),
            "description_clean": "coffee",
        })

        reason = calculate_reason(source, target)

        assert "exact amount" in reason
        assert "days apart" in reason

    def test_description_difference_reason(self) -> None:
        """Test reason when descriptions differ."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "netflix",
        })
        target = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "netflix.com",
        })

        reason = calculate_reason(source, target)

        assert "exact amount" in reason
        assert "same date" in reason
        # Similar descriptions show as "nearly identical description" or "different description"
        assert "description" in reason

    def test_similar_description_reason(self) -> None:
        """Test reason when descriptions are similar but not nearly identical (80-94% match)."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee",
        })
        target = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffees",  # Similar but not identical (92% match)
        })

        reason = calculate_reason(source, target)

        assert "exact amount" in reason
        assert "same date" in reason
        assert "similar description" in reason

    def test_all_different_reason(self) -> None:
        """Test reason when everything differs."""
        source = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 1),
            "description_clean": "coffee",
        })
        target = pd.Series({
            "amount_clean": Decimal("200.00"),
            "date_clean": datetime(2024, 2, 1),
            "description_clean": "lunch",
        })

        reason = calculate_reason(source, target)

        assert "different amount" in reason
        assert "days apart" in reason
        assert "different description" in reason

    def test_nan_values_handling(self) -> None:
        """Test reason calculation with NaN values."""
        source = pd.Series({
            "amount_clean": None,
            "date_clean": None,
            "description_clean": None,
        })
        target = pd.Series({
            "amount_clean": Decimal("100.00"),
            "date_clean": datetime(2024, 1, 15),
            "description_clean": "coffee",
        })

        reason = calculate_reason(source, target)

        # Should handle NaN gracefully
        assert reason is not None


class TestFindMatchesEdgeCases:
    """Test find_matches edge cases."""

    def test_empty_dataframes(self) -> None:
        """Test matching with empty DataFrames."""
        source_df = pd.DataFrame([])
        target_df = pd.DataFrame([])
        config = MatchConfig(threshold=0.7, date_window_days=3)

        result = find_matches(source_df, target_df, config)

        assert isinstance(result, MatchResult)
        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 0

    def test_no_matches_above_threshold(self) -> None:
        """Test when no records match above threshold."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 1),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 2, 1),
            "amount_clean": Decimal("200.00"),
            "description_clean": "lunch",
        }])
        config = MatchConfig(threshold=0.9, date_window_days=1)

        result = find_matches(source_df, target_df, config)

        # No matches should be found with high threshold
        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 1

    def test_all_records_missing(self) -> None:
        """Test when all source records are missing from target."""
        source_df = pd.DataFrame([
            {
                "date_clean": datetime(2024, 1, 1),
                "amount_clean": Decimal("100.00"),
                "description_clean": "coffee",
            },
            {
                "date_clean": datetime(2024, 1, 2),
                "amount_clean": Decimal("50.00"),
                "description_clean": "lunch",
            },
        ])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 2, 1),
            "amount_clean": Decimal("200.00"),
            "description_clean": "dinner",
        }])
        config = MatchConfig(threshold=0.9)

        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 2

    def test_custom_threshold(self) -> None:
        """Test with custom confidence threshold."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee shop",
        }])

        # High threshold - might not match
        config_high = MatchConfig(threshold=0.99)
        result_high = find_matches(source_df, target_df, config_high)

        # Low threshold - should match
        config_low = MatchConfig(threshold=0.5)
        result_low = find_matches(source_df, target_df, config_low)

        assert len(result_low.matches) >= len(result_high.matches)

    def test_custom_date_window(self) -> None:
        """Test with custom date window."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 10),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 20),  # 10 days apart
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])

        # Narrow date window - might not match
        config_narrow = MatchConfig(threshold=0.7, date_window_days=3)
        result_narrow = find_matches(source_df, target_df, config_narrow)

        # Wide date window - should match
        config_wide = MatchConfig(threshold=0.7, date_window_days=15)
        result_wide = find_matches(source_df, target_df, config_wide)

        assert len(result_wide.matches) >= len(result_narrow.matches)

    def test_exact_duplicate_records(self) -> None:
        """Test handling of exact duplicate source records.

        Greedy algorithm: only one source can match a given target.
        The duplicate source record will be marked as missing.
        """
        source_df = pd.DataFrame([
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix",
            },
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix",
            },
        ])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix",
        }])
        config = MatchConfig(threshold=0.7)

        result = find_matches(source_df, target_df, config)

        # Greedy algorithm: first source matches target, second is missing
        assert len(result.matches) == 1
        assert len(result.missing_in_target) == 1

    def test_one_to_many_prevention(self) -> None:
        """Test that one target doesn't match multiple sources (greedy algorithm)."""
        # Create scenario where one target could match multiple sources
        source_df = pd.DataFrame([
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix subscription",
            },
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix",
            },
        ])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix.com",
        }])
        config = MatchConfig(threshold=0.7)

        result = find_matches(source_df, target_df, config)

        # Only one source should match the target (greedy prevents double-matching)
        matched_target_indices = [m.target_idx for m in result.matches if m.target_idx is not None]
        assert len(matched_target_indices) == 1  # Only one match to the target

    def test_all_missing_from_target(self) -> None:
        """Test when all source records are missing from target (below minimum confidence)."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 3, 1),
            "amount_clean": Decimal("999.99"),
            "description_clean": "something else",
        }])
        config = MatchConfig(threshold=0.9)

        # Use high min_confidence to ensure no matches are found
        result = find_matches(source_df, target_df, config, min_confidence=0.9)

        # All source records should be missing (no matches ≥ 0.9)
        assert len(result.missing_in_target) == len(source_df)
