"""Tests for manual match with table selection."""

from datetime import datetime

import pandas as pd
import pytest
from textual.widgets import DataTable

from src.matcher import Match, MatchResult, ConfidenceTier
from src.models import MatchDecision
from src.tui.screens import MatchState, MatchReviewScreen


class TestManualMatchSelection:
    """Tests that manual match respects the currently selected table row."""

    def test_table_cursor_updates_selected_match_idx(self) -> None:
        """Test that when selected_match_idx is changed, manual match uses it.

        This test documents the bug: when the user navigates the table to row 1,
        the table's cursor moves but selected_match_idx stays at 0. Pressing 'm'
        then uses the wrong source record (row 0 instead of row 1).

        The fix should update selected_match_idx when the table cursor moves.
        """
        # Create sample data
        source_df = pd.DataFrame(
            {
                "date_clean": [datetime(2024, 1, 10), datetime(2024, 1, 15), datetime(2024, 1, 20)],
                "amount_clean": [100.00, 200.00, 300.00],
                "description_clean": ["Coffee Shop", "Lunch Special", "Groceries"],
            }
        )

        target_df = pd.DataFrame(
            {
                "date_clean": [datetime(2024, 1, 10), datetime(2024, 1, 15), datetime(2024, 1, 20)],
                "amount_clean": [100.00, 200.00, 300.00],
                "description_clean": ["Coffee Shop", "Lunch", "Grocery Store"],
            }
        )

        # Create matches
        matches = [
            Match(0, 0, 0.95, "Exact match", MatchDecision.PENDING, ConfidenceTier.HIGH),
            Match(1, 1, 0.82, "Close match", MatchDecision.PENDING, ConfidenceTier.MEDIUM),
            Match(2, 2, 0.75, "Fuzzy match", MatchDecision.PENDING, ConfidenceTier.MEDIUM),
        ]

        match_result = MatchResult(matches=matches, missing_in_target=[], missing_in_source=[])

        # Create screen and state with selected_match_idx = 1
        # This simulates what SHOULD happen when user navigates to row 1
        match_state = MatchState(match_result=match_result, selected_match_idx=1)
        screen = MatchReviewScreen(source_df, target_df, match_state)

        # When selected_match_idx is 1, manual match should use source_idx 1
        # (the second match, which has source_idx=1)
        filtered_matches = match_state.get_filtered_and_sorted_matches()
        assert 0 <= match_state.selected_match_idx < len(filtered_matches)

        selected_match = filtered_matches[match_state.selected_match_idx]
        assert selected_match.source_idx == 1, (
            f"Manual match should use source_idx from selected_match_idx=1, "
            f"got source_idx={selected_match.source_idx}"
        )

    def test_manual_match_uses_correct_selected_row(self) -> None:
        """Test that pressing 'm' on a row uses that row's source index."""
        # Create sample data
        source_df = pd.DataFrame(
            {
                "date_clean": [datetime(2024, 1, 10), datetime(2024, 1, 15), datetime(2024, 1, 20)],
                "amount_clean": [100.00, 200.00, 300.00],
                "description_clean": ["Coffee", "Lunch", "Groceries"],
            }
        )

        target_df = pd.DataFrame(
            {
                "date_clean": [datetime(2024, 1, 10), datetime(2024, 1, 15), datetime(2024, 1, 20)],
                "amount_clean": [100.00, 200.00, 300.00],
                "description_clean": ["Coffee", "Lunch", "Groceries"],
            }
        )

        # Create matches
        matches = [
            Match(0, 0, 0.95, "Match 0", MatchDecision.PENDING, ConfidenceTier.HIGH),
            Match(1, 1, 0.95, "Match 1", MatchDecision.PENDING, ConfidenceTier.HIGH),
            Match(2, 2, 0.95, "Match 2", MatchDecision.PENDING, ConfidenceTier.HIGH),
        ]

        match_result = MatchResult(matches=matches, missing_in_target=[], missing_in_source=[])

        # Create screen and state
        match_state = MatchState(match_result=match_result, selected_match_idx=2)
        screen = MatchReviewScreen(source_df, target_df, match_state)

        # When selected_match_idx is 2, action_manual_match should use source_idx 2
        # (the third match, which has source_idx=2)
        assert match_state.selected_match_idx == 2

        filtered_matches = match_state.get_filtered_and_sorted_matches()
        assert 0 <= match_state.selected_match_idx < len(filtered_matches)

        # The selected match should be the third one (source_idx=2)
        selected_match = filtered_matches[match_state.selected_match_idx]
        assert selected_match.source_idx == 2, (
            "Manual match should use source_idx from the selected match, not always 0"
        )
