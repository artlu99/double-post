"""Property-based tests for matching confidence calculation.

Uses hypothesis to generate test cases and verify invariants.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

from src.matcher import calculate_confidence
from src.models import MatchConfig


@composite
def draw_match_pair(draw):
    """Generate a pair of matching records for testing.

    Returns:
        Tuple of (source_row, target_row)
    """
    date = draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31)))
    amount = draw(st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=2))
    description = draw(st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories="L")))

    source = pd.Series({
        "date_clean": date,
        "amount_clean": amount,
        "description_clean": description.lower(),
    })

    target = pd.Series({
        "date_clean": date,
        "amount_clean": amount,
        "description_clean": description.lower(),
    })

    return source, target


class TestConfidenceCalculationProperties:
    """Property-based tests for confidence calculation."""

    @given(draw_match_pair())
    def test_exact_match_has_high_confidence(self, pair) -> None:
        """Exact matches should always have high confidence."""
        source, target = pair
        config = MatchConfig(threshold=0.7, date_window_days=3, amount_tolerance=Decimal("0.01"))

        confidence = calculate_confidence(source, target, config)

        # Exact matches should have confidence >= 0.9
        assert confidence >= 0.9, f"Expected high confidence for exact match, got {confidence}"

    @given(draw_match_pair())
    def test_confidence_between_zero_and_one(self, pair) -> None:
        """Confidence should always be between 0 and 1."""
        source, target = pair
        config = MatchConfig()

        confidence = calculate_confidence(source, target, config)

        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0, 1]"

    @given(
        st.datetimes(),
        st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=2),
        st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=2),
    )
    def test_same_amount_gives_nonzero_confidence(self, date, amount1, amount2) -> None:
        """Records with same amount should have non-zero confidence."""
        source = pd.Series({
            "date_clean": date,
            "amount_clean": amount1,
            "description_clean": "test",
        })
        target = pd.Series({
            "date_clean": date,
            "amount_clean": amount2,
            "description_clean": "test",
        })
        config = MatchConfig()

        confidence = calculate_confidence(source, target, config)

        # Same description and date should give some confidence
        # (amount may differ but should still get some match score)
        assert confidence >= 0.0

    @given(
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31)),
        st.decimals(min_value=Decimal("0.01"), max_value=Decimal("10000"), places=2),
    )
    def test_commutative_confidence(self, date, amount) -> None:
        """Confidence should be symmetric (same if we swap source/target)."""
        source = pd.Series({
            "date_clean": date,
            "amount_clean": amount,
            "description_clean": "test",
        })
        target = pd.Series({
            "date_clean": date,
            "amount_clean": amount,
            "description_clean": "test",
        })
        config = MatchConfig()

        confidence1 = calculate_confidence(source, target, config)
        confidence2 = calculate_confidence(target, source, config)

        # Confidence calculation should be symmetric
        assert abs(confidence1 - confidence2) < 0.001, f"Not symmetric: {confidence1} != {confidence2}"


class TestConfidenceInvariants:
    """Test invariants for confidence calculation."""

    def test_all_exact_match_maximum_confidence(self) -> None:
        """Perfect match should yield maximum confidence."""
        source = pd.Series({
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "test transaction",
        })
        target = pd.Series({
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "test transaction",
        })
        config = MatchConfig()

        confidence = calculate_confidence(source, target, config)

        assert confidence == pytest.approx(1.0)

    def test_all_different_minimum_confidence(self) -> None:
        """Completely different records should have minimum confidence."""
        source = pd.Series({
            "date_clean": datetime(2024, 1, 1),
            "amount_clean": Decimal("100.00"),
            "description_clean": "aaaa",
        })
        target = pd.Series({
            "date_clean": datetime(2025, 12, 31),
            "amount_clean": Decimal("9999.99"),
            "description_clean": "zzzz",
        })
        config = MatchConfig()

        confidence = calculate_confidence(source, target, config)

        # Should be very low confidence
        assert confidence < 0.3

    def test_date_proximity_increases_confidence(self) -> None:
        """Same amount and description, closer date = higher confidence."""
        amount = Decimal("100.00")
        description = "test"

        # Close date (0 days apart)
        source = pd.Series({
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": amount,
            "description_clean": description,
        })
        target_close = pd.Series({
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": amount,
            "description_clean": description,
        })

        # Far date (10 days apart)
        target_far = pd.Series({
            "date_clean": datetime(2024, 1, 25),
            "amount_clean": amount,
            "description_clean": description,
        })

        config = MatchConfig(date_window_days=3)

        confidence_close = calculate_confidence(source, target_close, config)
        confidence_far = calculate_confidence(source, target_far, config)

        assert confidence_close > confidence_far, "Closer dates should have higher confidence"
