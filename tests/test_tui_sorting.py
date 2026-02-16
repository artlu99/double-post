"""Tests for TUI single-table sort functionality."""

from datetime import datetime

import pandas as pd

from src.models import ConfidenceTier, MatchDecision
from src.tui.screens import MatchState


class TestSortModes:
    """Tests for sort mode cycling and functionality."""

    def test_sort_status_pending_first(self):
        """Test sorting by status puts pending matches first."""
        from src.matcher import Match, MatchResult

        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.95,
                reason="Exact",
                decision=MatchDecision.ACCEPTED,
                tier=ConfidenceTier.HIGH,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.6,
                reason="Partial",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.MEDIUM,
            ),
            Match(
                source_idx=2,
                target_idx=2,
                confidence=0.3,
                reason="Weak",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.LOW,
            ),
            Match(
                source_idx=3,
                target_idx=None,
                confidence=0.0,
                reason="No match",
                decision=MatchDecision.REJECTED,
                tier=ConfidenceTier.NONE,
            ),
        ]

        result = MatchResult(matches=matches, missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="status")

        # Sort by status: pending first, then rejected, then accepted
        sorted_matches = state.get_sorted_matches()

        assert len(sorted_matches) == 4
        # First two should be pending
        assert sorted_matches[0].decision == MatchDecision.PENDING
        assert sorted_matches[1].decision == MatchDecision.PENDING
        # Then rejected
        assert sorted_matches[2].decision == MatchDecision.REJECTED
        # Then accepted
        assert sorted_matches[3].decision == MatchDecision.ACCEPTED

    def test_sort_confidence_low_to_high(self):
        """Test sorting by confidence puts lowest confidence first."""
        from src.matcher import Match, MatchResult

        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.95,
                reason="Exact",
                decision=MatchDecision.ACCEPTED,
                tier=ConfidenceTier.HIGH,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.6,
                reason="Partial",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.MEDIUM,
            ),
            Match(
                source_idx=2,
                target_idx=2,
                confidence=0.3,
                reason="Weak",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.LOW,
            ),
        ]

        result = MatchResult(matches=matches, missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="confidence")

        sorted_matches = state.get_sorted_matches()

        # Should be sorted by confidence ascending (low to high)
        assert sorted_matches[0].confidence == 0.3
        assert sorted_matches[1].confidence == 0.6
        assert sorted_matches[2].confidence == 0.95

    def test_sort_date_newest_first(self):
        """Test sorting by date puts newest records first."""
        from src.matcher import Match, MatchResult

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Old",
                },
                {
                    "date_clean": datetime(2024, 2, 1),
                    "amount_clean": 200,
                    "description_clean": "New",
                },
                {
                    "date_clean": datetime(2024, 1, 20),
                    "amount_clean": 150,
                    "description_clean": "Middle",
                },
            ]
        )

        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.9,
                reason="Exact",
                decision=MatchDecision.ACCEPTED,
                tier=ConfidenceTier.HIGH,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.6,
                reason="Partial",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.MEDIUM,
            ),
            Match(
                source_idx=2,
                target_idx=2,
                confidence=0.3,
                reason="Weak",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.LOW,
            ),
        ]

        result = MatchResult(matches=matches, missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="date", source_df=source_df)

        sorted_matches = state.get_sorted_matches()

        # Should be sorted by date descending (newest first)
        assert sorted_matches[0].source_idx == 1  # Feb 1
        assert sorted_matches[1].source_idx == 2  # Jan 20
        assert sorted_matches[2].source_idx == 0  # Jan 15

    def test_sort_mode_cycling(self):
        """Test that sort mode cycles through available modes."""
        from src.matcher import MatchResult

        result = MatchResult(matches=[], missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="status")

        # Cycle through sort modes
        modes = ["status", "confidence", "date"]
        for expected_mode in modes:
            assert state.sort_mode == expected_mode
            state.cycle_sort_mode()

        # Should cycle back to first mode
        assert state.sort_mode == "status"

    def test_sort_within_status_group_by_confidence(self):
        """Test that within status group, items are sorted by confidence."""
        from src.matcher import Match, MatchResult

        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.9,
                reason="Good",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.HIGH,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.5,
                reason="Okay",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.MEDIUM,
            ),
            Match(
                source_idx=2,
                target_idx=2,
                confidence=0.3,
                reason="Weak",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.LOW,
            ),
        ]

        result = MatchResult(matches=matches, missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="status")

        sorted_matches = state.get_sorted_matches()

        # All pending, should be sorted by confidence ascending
        assert sorted_matches[0].confidence == 0.3
        assert sorted_matches[1].confidence == 0.5
        assert sorted_matches[2].confidence == 0.9

    def test_filter_and_sort_combined(self):
        """Test that filtering and sorting work together."""
        from src.matcher import Match, MatchResult

        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.9,
                reason="Good",
                decision=MatchDecision.ACCEPTED,
                tier=ConfidenceTier.HIGH,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.5,
                reason="Okay",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.MEDIUM,
            ),
            Match(
                source_idx=2,
                target_idx=2,
                confidence=0.3,
                reason="Weak",
                decision=MatchDecision.PENDING,
                tier=ConfidenceTier.LOW,
            ),
        ]

        result = MatchResult(matches=matches, missing_in_target=[], duplicate_matches=[])
        state = MatchState(match_result=result, sort_mode="confidence", filter_mode="pending")

        # Get filtered and sorted matches
        filtered_and_sorted = state.get_filtered_and_sorted_matches()

        # Should only show pending matches, sorted by confidence
        assert len(filtered_and_sorted) == 2
        assert filtered_and_sorted[0].confidence == 0.3
        assert filtered_and_sorted[1].confidence == 0.5


class TestSingleTableDisplay:
    """Tests for single-table display layout."""

    def test_table_columns_include_status(self):
        """Test that table includes status column."""
        from src.tui.screens import MatchReviewScreen

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Test",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Test",
                }
            ]
        )
        from src.matcher import Match, MatchResult

        result = MatchResult(
            matches=[
                Match(
                    source_idx=0,
                    target_idx=0,
                    confidence=0.9,
                    reason="Exact",
                    tier=ConfidenceTier.HIGH,
                )
            ],
            missing_in_target=[],
            duplicate_matches=[],
        )

        screen = MatchReviewScreen(source_df, target_df, result)
        columns = screen._get_table_columns()

        # Should include Status column
        assert "Status" in columns
        # Should include key columns
        assert "Date" in columns
        assert "Source Amt" in columns
        assert "Target Amt" in columns
        assert "Source Desc" in columns or "Description" in columns
        assert "Match Info" in columns

    def test_status_column_content(self):
        """Test that status column shows appropriate icons."""
        from src.tui.screens import MatchReviewScreen

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Test",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Test",
                }
            ]
        )
        from src.matcher import Match, MatchResult

        result = MatchResult(
            matches=[
                Match(
                    source_idx=0,
                    target_idx=0,
                    confidence=0.9,
                    reason="Exact",
                    decision=MatchDecision.ACCEPTED,
                    tier=ConfidenceTier.HIGH,
                )
            ],
            missing_in_target=[],
            duplicate_matches=[],
        )

        screen = MatchReviewScreen(source_df, target_df, result)
        status_text = screen._get_status_text(MatchDecision.ACCEPTED, ConfidenceTier.HIGH)

        # Should show accepted status with tier
        assert "✓" in status_text or "⭐" in status_text

    def test_match_info_column_content(self):
        """Test that match info column shows target and confidence."""
        from src.matcher import Match, MatchResult
        from src.tui.screens import MatchReviewScreen

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Coffee",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Coffee Shop",
                }
            ]
        )

        result = MatchResult(
            matches=[
                Match(
                    source_idx=0,
                    target_idx=0,
                    confidence=0.85,
                    reason="Fuzzy match",
                    decision=MatchDecision.PENDING,
                    tier=ConfidenceTier.MEDIUM,
                )
            ],
            missing_in_target=[],
            duplicate_matches=[],
        )

        screen = MatchReviewScreen(source_df, target_df, result)
        match_info = screen._get_match_info_text(result.matches[0])

        # Should show target description
        assert "Coffee Shop" in match_info
        # Should show confidence
        assert "0.85" in match_info or "85%" in match_info

    def test_match_info_for_unmatched(self):
        """Test that match info shows appropriate message for unmatched records."""
        from src.matcher import Match, MatchResult
        from src.tui.screens import MatchReviewScreen

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": 100,
                    "description_clean": "Missing",
                }
            ]
        )
        target_df = pd.DataFrame([])

        result = MatchResult(
            matches=[
                Match(
                    source_idx=0,
                    target_idx=None,
                    confidence=0.0,
                    reason="No match found",
                    decision=MatchDecision.PENDING,
                    tier=ConfidenceTier.NONE,
                )
            ],
            missing_in_target=[0],
            duplicate_matches=[],
        )

        screen = MatchReviewScreen(source_df, target_df, result)
        match_info = screen._get_match_info_text(result.matches[0])

        # Should indicate no match
        assert "No match" in match_info or "Missing" in match_info or "—" in match_info
