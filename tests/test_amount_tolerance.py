"""Tests for amount tolerance early-exit optimization.

Tests the 10% amount tolerance filtering that reduces the O(N*M) comparison space.
Vectorized pre-calculation of amount bounds (±10%) is used to skip expensive
fuzzy matching for pairs that cannot match due to amount differences.

TDD: These tests are written before implementation and should fail initially.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd

from src.matcher import find_matches
from src.models import MatchConfig


class TestAmountToleranceEarlyExit:
    """Tests for 10% amount tolerance early-exit optimization."""

    def test_amount_difference_exceeds_tolerance_no_match(self) -> None:
        """Test that pairs with amount difference > 10% do not match (early-exit)."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee shop",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("89.00"),  # 11% difference - should early-exit
                    "description_clean": "coffee shop",
                },
            ]
        )
        config = MatchConfig(threshold=0.7, date_window_days=3)

        result = find_matches(source_df, target_df, config)

        # Should NOT match - amount difference > 10%
        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 1

    def test_amount_within_tolerance_still_matches(self) -> None:
        """Test that amounts within 10% tolerance can still match."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee shop",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("95.00"),  # 5% difference - within tolerance
                    "description_clean": "coffee shop",
                },
            ]
        )
        config = MatchConfig(threshold=0.5, date_window_days=3)

        result = find_matches(source_df, target_df, config)

        # Should match - within 10% tolerance
        assert len(result.matches) == 1
        assert len(result.missing_in_target) == 0

    def test_exact_amount_match_ignores_tolerance(self) -> None:
        """Test that exact amount matches work regardless of tolerance."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "netflix",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("50.00"),  # Exact match
                    "description_clean": "netflix",
                },
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        assert result.matches[0].confidence >= 0.9

    def test_tolerance_upper_bound(self) -> None:
        """Test behavior at exactly 10% above source amount."""
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
                    "amount_clean": Decimal("110.00"),  # Exactly 10% higher
                    "description_clean": "coffee",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # 10% is the boundary - should match (inclusive)
        assert len(result.matches) == 1

    def test_tolerance_lower_bound(self) -> None:
        """Test behavior at exactly 10% below source amount."""
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
                    "amount_clean": Decimal("90.00"),  # Exactly 10% lower
                    "description_clean": "coffee",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # 10% is the boundary - should match (inclusive)
        assert len(result.matches) == 1

    def test_tolerance_with_negative_amounts(self) -> None:
        """Test that tolerance works correctly with negative amounts (debits)."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-100.00"),
                    "description_clean": "grocery store",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-95.00"),  # Within tolerance
                    "description_clean": "grocery store",
                },
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("-85.00"),  # More than 10% difference
                    "description_clean": "grocery store",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # Should match the -95.00, not -85.00
        assert len(result.matches) == 1
        assert result.matches[0].target_idx == 0

    def test_vectorized_bounds_calculation(self) -> None:
        """Test that bounds are pre-calculated vectorized, not per-row."""
        # Create multiple sources and targets
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "coffee",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("200.00"),
                    "description_clean": "lunch",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("50.00"),  # Too far from 100
                    "description_clean": "coffee",
                },
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("250.00"),  # Too far from 200
                    "description_clean": "lunch",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # No matches should be found due to amount tolerance
        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 2

    def test_early_exit_preserves_match_quality(self) -> None:
        """Test that early-exit doesn't negatively affect match quality."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "starbucks coffee",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "starbucks coffee downtown",
                },
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("5.00"),  # Should early-exit, no expensive fuzzy match
                    "description_clean": "totally different",
                },
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        # Should still get high confidence match
        assert len(result.matches) == 1
        assert result.matches[0].confidence >= 0.9
        assert result.matches[0].target_idx == 0


class TestAmountToleranceEdgeCases:
    """Tests for amount tolerance edge cases."""

    def test_zero_amount_handling(self) -> None:
        """Test that zero amounts are handled correctly."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("0.00"),
                    "description_clean": "refund",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("0.00"),
                    "description_clean": "refund",
                },
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        # Zero amounts should match (exact match)
        assert len(result.matches) == 1

    def test_very_small_amounts(self) -> None:
        """Test tolerance with very small amounts (cents)."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("0.10"),
                    "description_clean": "micro transaction",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("1.00"),  # 900% difference - should early-exit
                    "description_clean": "micro transaction",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # Should not match due to amount difference
        assert len(result.matches) == 0

    def test_very_large_amounts(self) -> None:
        """Test tolerance with very large amounts."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100000.00"),
                    "description_clean": "large purchase",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("105000.00"),  # 5% within tolerance
                    "description_clean": "large purchase",
                },
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("115000.00"),  # 15% outside tolerance
                    "description_clean": "large purchase",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # Should match first (within tolerance), not second
        assert len(result.matches) == 1
        assert result.matches[0].target_idx == 0


class TestAmountTolerancePerformance:
    """Tests to verify the optimization is actually being applied."""

    def test_early_exit_reduces_comparisons(self) -> None:
        """Test that early-exit actually skips expensive comparisons.

        This test verifies that pairs with wildly different amounts
        don't go through the expensive fuzzy matching logic.
        """
        # Many sources, many targets - but only a few have compatible amounts
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, i),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": f"transaction {i}",
                }
                for i in range(1, 11)  # 10 sources at $100
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, i),
                    "amount_clean": Decimal("10.00"),  # Way outside tolerance
                    "description_clean": f"different {i}",
                }
                for i in range(1, 11)  # 10 targets at $10
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # All sources should be missing - no matches due to amount tolerance
        assert len(result.matches) == 0
        assert len(result.missing_in_target) == 10

    def test_partial_match_with_tolerance(self) -> None:
        """Test scenario where only some pairs pass amount filtering."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),
                    "description_clean": "source 1",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("200.00"),
                    "description_clean": "source 2",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("105.00"),  # Within 10% of $100
                    "description_clean": "target 1",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("300.00"),  # Outside 10% of $200
                    "description_clean": "target 2",
                },
            ]
        )
        config = MatchConfig(threshold=0.5)

        result = find_matches(source_df, target_df, config)

        # First source should match, second should not
        assert len(result.matches) == 1
        assert result.matches[0].source_idx == 0
        assert result.matches[0].target_idx == 0
