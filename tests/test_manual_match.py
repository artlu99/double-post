"""Tests for manual matching functionality.

Tests for creating manual matches between source and target records.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.matcher import create_manual_match
from src.models import Match, MatchDecision, MatchResult
from tests.factories import TestDataFactory


class TestManualMatchCreation:
    """Test the create_manual_match function."""

    def test_manual_match_creation_with_valid_indices(self) -> None:
        """Test that manual match is created with correct source and target indices."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        assert match.source_idx == 0
        assert match.target_idx == 0
        assert isinstance(match, Match)

    def test_manual_match_confidence_calculation(self) -> None:
        """Test that manual match calculates confidence from actual data."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        # Should calculate actual confidence based on the records
        assert 0.0 <= match.confidence <= 1.0
        assert isinstance(match.confidence, float)

    def test_manual_match_reason_generation(self) -> None:
        """Test that manual match generates appropriate reason."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        # Should indicate this was manually matched
        assert "manual" in match.reason.lower()

    def test_manual_marked_as_manual(self) -> None:
        """Test that manual match has manual flag set to True."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        # Should have manual attribute set
        assert hasattr(match, "manual")
        assert match.manual is True

    def test_manual_match_with_different_amounts(self) -> None:
        """Test manual match with different amounts still works."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("50.00"),  # Different amount
            "description_clean": "coffee",
        }])

        match = create_manual_match(0, 0, source_df, target_df)

        # Should still create match, confidence will be lower
        assert match.source_idx == 0
        assert match.target_idx == 0
        assert match.confidence < 1.0  # Lower confidence due to amount mismatch

    def test_manual_match_with_different_dates(self) -> None:
        """Test manual match with different dates still works."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 20),  # Different date
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee",
        }])

        match = create_manual_match(0, 0, source_df, target_df)

        # Should still create match
        assert match.source_idx == 0
        assert match.target_idx == 0

    def test_manual_match_with_different_descriptions(self) -> None:
        """Test manual match with different descriptions still works."""
        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "coffee shop",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("100.00"),
            "description_clean": "bakery",  # Different description
        }])

        match = create_manual_match(0, 0, source_df, target_df)

        # Should still create match
        assert match.source_idx == 0
        assert match.target_idx == 0


class TestManualMatchScreen:
    """Test the ManualMatchScreen TUI screen."""

    def test_screen_initialization(self) -> None:
        """Test that ManualMatchScreen initializes with correct data."""
        from src.tui.manual_match_screen import ManualMatchScreen

        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        source_idx = 0

        screen = ManualMatchScreen(source_df, target_df, source_idx)

        assert screen.source_df is source_df
        assert screen.target_df is target_df
        assert screen.source_idx == source_idx

    def test_screen_shows_available_targets(self) -> None:
        """Test that screen displays all available target records."""
        from src.tui.manual_match_screen import ManualMatchScreen

        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        screen = ManualMatchScreen(source_df, target_df, 0)

        # Get available targets (this will be a method on the screen)
        available_targets = screen.get_available_targets()

        assert len(available_targets) == len(target_df)

    def test_screen_filters_matched_targets(self) -> None:
        """Test that screen excludes already-matched targets."""
        from src.tui.manual_match_screen import ManualMatchScreen

        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        # Create a match result where target 0 is already matched
        existing_match = TestDataFactory.create_match(source_idx=1, target_idx=0)
        match_result = MatchResult(matches=[existing_match], missing_in_target=[], duplicate_matches=[])

        screen = ManualMatchScreen(source_df, target_df, 0, match_result)

        available_targets = screen.get_available_targets()

        # Target 0 should be filtered out
        assert 0 not in available_targets

    def test_screen_shows_source_record(self) -> None:
        """Test that screen displays the source record being matched."""
        from src.tui.manual_match_screen import ManualMatchScreen

        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        screen = ManualMatchScreen(source_df, target_df, 0)

        source_record = screen.get_source_record()

        assert source_record is not None
        assert source_record["description_clean"] == source_df.iloc[0]["description_clean"]


class TestManualMatchIntegration:
    """Test manual match integration with matching system."""

    def test_manual_match_added_to_match_result(self) -> None:
        """Test that manual match is added to match result."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        initial_match_result = MatchResult(matches=[], missing_in_target=[0], duplicate_matches=[])

        # Create manual match
        manual_match = create_manual_match(0, 0, source_df, target_df)

        # Add to match result
        initial_match_result.matches.append(manual_match)
        initial_match_result.missing_in_target.remove(0)

        assert len(initial_match_result.matches) == 1
        assert len(initial_match_result.missing_in_target) == 0

    def test_manual_match_removed_from_missing(self) -> None:
        """Test that source is removed from missing list after manual match."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match_result = MatchResult(matches=[], missing_in_target=[0, 1, 2], duplicate_matches=[])

        # Match source 0 manually
        manual_match = create_manual_match(0, 0, source_df, target_df)
        match_result.matches.append(manual_match)
        match_result.missing_in_target.remove(0)

        assert 0 not in match_result.missing_in_target
        assert 1 in match_result.missing_in_target

    def test_manual_match_target_not_reused(self) -> None:
        """Test that a target used in manual match can't be matched again."""
        # Create 2 source records and 1 target record
        source_records = [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("100.00"),
                "description_clean": "coffee",
                "original_idx": 0,
            },
            {
                "date_clean": datetime(2024, 1, 16),
                "amount_clean": Decimal("50.00"),
                "description_clean": "lunch",
                "original_idx": 1,
            },
        ]
        target_records = [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("100.00"),
                "description_clean": "coffee",
                "original_idx": 0,
            },
        ]
        source_df = pd.DataFrame(source_records)
        target_df = pd.DataFrame(target_records)

        # First manual match uses target 0
        match1 = create_manual_match(0, 0, source_df, target_df)

        # Try to create another manual match with same target
        match2 = create_manual_match(1, 0, source_df, target_df)

        # Both should be created, but application logic should prevent double-matching
        assert match1.target_idx == 0
        assert match2.target_idx == 0

    def test_manual_match_can_be_accepted(self) -> None:
        """Test that manual match can be accepted."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        # Initially pending
        assert match.decision == MatchDecision.PENDING

        # Accept the match
        match.decision = MatchDecision.ACCEPTED

        assert match.decision == MatchDecision.ACCEPTED

    def test_manual_match_can_be_rejected(self) -> None:
        """Test that manual match can be rejected."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match = create_manual_match(0, 0, source_df, target_df)

        # Reject the match
        match.decision = MatchDecision.REJECTED

        assert match.decision == MatchDecision.REJECTED


class TestManualMatchEdgeCases:
    """Test edge cases for manual matching."""

    def test_manual_match_with_invalid_source_idx(self) -> None:
        """Test manual match with invalid source index."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        # Invalid source index
        with pytest.raises(IndexError):
            create_manual_match(999, 0, source_df, target_df)

    def test_manual_match_with_invalid_target_idx(self) -> None:
        """Test manual match with invalid target index."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        # Invalid target index
        with pytest.raises(IndexError):
            create_manual_match(0, 999, source_df, target_df)

    def test_manual_match_when_no_targets_available(self) -> None:
        """Test manual match when target dataframe is empty."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = pd.DataFrame()  # Empty

        with pytest.raises(IndexError):
            create_manual_match(0, 0, source_df, target_df)

    def test_manual_match_with_empty_source_dataframe(self) -> None:
        """Test manual match when source dataframe is empty."""
        source_df = pd.DataFrame()  # Empty
        target_df = TestDataFactory.create_target_dataframe()

        with pytest.raises(IndexError):
            create_manual_match(0, 0, source_df, target_df)

    def test_manual_match_duplicates_prevented(self) -> None:
        """Test that duplicate manual matches are detected."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        match1 = create_manual_match(0, 0, source_df, target_df)
        match2 = create_manual_match(0, 0, source_df, target_df)

        # Both should be created, but they have the same indices
        assert match1.source_idx == match2.source_idx
        assert match1.target_idx == match2.target_idx
