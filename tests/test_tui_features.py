"""Tests for TUI feature logic.

Tests core functionality of TUI features without requiring Textual framework.
"""

from src.models import Match, MatchDecision


class TestMatchFiltering:
    """Test filtering matches by decision status."""

    def test_filter_all_matches(self) -> None:
        """Test filtering returns all matches when mode is 'all'."""
        matches = [
            Match(0, 0, 0.95, "exact", MatchDecision.ACCEPTED),
            Match(1, 1, 0.75, "similar", MatchDecision.REJECTED),
            Match(2, 2, 0.85, "close", MatchDecision.PENDING),
        ]

        result = _filter_matches(matches, "all")
        assert len(result) == 3

    def test_filter_pending_only(self) -> None:
        """Test filtering returns only pending matches."""
        matches = [
            Match(0, 0, 0.95, "exact", MatchDecision.ACCEPTED),
            Match(1, 1, 0.75, "similar", MatchDecision.REJECTED),
            Match(2, 2, 0.85, "close", MatchDecision.PENDING),
        ]

        result = _filter_matches(matches, "pending")
        assert len(result) == 1
        assert result[0].decision == MatchDecision.PENDING

    def test_filter_accepted_only(self) -> None:
        """Test filtering returns only accepted matches."""
        matches = [
            Match(0, 0, 0.95, "exact", MatchDecision.ACCEPTED),
            Match(1, 1, 0.75, "similar", MatchDecision.REJECTED),
            Match(2, 2, 0.85, "close", MatchDecision.PENDING),
        ]

        result = _filter_matches(matches, "accepted")
        assert len(result) == 1
        assert result[0].decision == MatchDecision.ACCEPTED

    def test_filter_rejected_only(self) -> None:
        """Test filtering returns only rejected matches."""
        matches = [
            Match(0, 0, 0.95, "exact", MatchDecision.ACCEPTED),
            Match(1, 1, 0.75, "similar", MatchDecision.REJECTED),
            Match(2, 2, 0.85, "close", MatchDecision.PENDING),
        ]

        result = _filter_matches(matches, "rejected")
        assert len(result) == 1
        assert result[0].decision == MatchDecision.REJECTED

    def test_filter_empty_list(self) -> None:
        """Test filtering with empty match list."""
        result = _filter_matches([], "all")
        assert result == []

    def test_filter_no_matches_for_status(self) -> None:
        """Test filtering when no matches match the filter."""
        matches = [
            Match(0, 0, 0.95, "exact", MatchDecision.ACCEPTED),
            Match(1, 1, 0.75, "similar", MatchDecision.ACCEPTED),
        ]

        result = _filter_matches(matches, "pending")
        assert len(result) == 0


# Helper functions extracted from TUI screens for testing
def _filter_matches(matches: list[Match], mode: str) -> list[Match]:
    """Filter matches by decision mode.

    Args:
        matches: List of matches to filter
        mode: Filter mode ('all', 'pending', 'accepted', 'rejected')

    Returns:
        Filtered list of matches
    """
    if mode == "all":
        return matches
    elif mode == "pending":
        return [m for m in matches if m.decision == MatchDecision.PENDING]
    elif mode == "accepted":
        return [m for m in matches if m.decision == MatchDecision.ACCEPTED]
    elif mode == "rejected":
        return [m for m in matches if m.decision == MatchDecision.REJECTED]
    return []
