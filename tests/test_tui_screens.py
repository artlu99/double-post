"""Tests for TUI screen logic and pure functions.

Tests the non-interactive parts of TUI screens:
- MatchState dataclass
- Display utility functions (format_date, format_amount, truncate_string, get_tier_display)
- MatchReviewScreen state management
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd

from src.models import ConfidenceTier, MatchDecision
from src.tui import display_utils
from src.tui.screens import MatchReviewScreen, MatchState, MissingItemsScreen, SummaryScreen
from tests.factories import TestDataFactory


class TestMatchState:
    """Test MatchState dataclass."""

    def test_initialization_with_defaults(self) -> None:
        """Test MatchState initialization with default values."""
        match_result = TestDataFactory.create_match_result()
        state = MatchState(match_result=match_result)

        assert state.match_result == match_result
        assert state.filter_mode == "all"
        assert state.selected_match_idx == -1

    def test_initialization_with_custom_values(self) -> None:
        """Test MatchState initialization with custom values."""
        match_result = TestDataFactory.create_match_result()
        state = MatchState(
            match_result=match_result,
            filter_mode="pending",
            selected_match_idx=5,
        )

        assert state.filter_mode == "pending"
        assert state.selected_match_idx == 5


class TestDisplayUtils:
    """Test display utility functions."""

    def test_format_date_valid(self) -> None:
        """Test date formatting with valid date."""
        date_val = datetime(2024, 1, 15, 14, 30)
        result = display_utils.format_date(date_val)
        assert result == "2024-01-15"

    def test_format_date_none(self) -> None:
        """Test date formatting with None."""
        result = display_utils.format_date(None)
        assert result == "N/A"

    def test_format_date_nan(self) -> None:
        """Test date formatting with pandas NaT."""
        result = display_utils.format_date(pd.NaT)
        assert result == "N/A"

    def test_format_amount_positive(self) -> None:
        """Test amount formatting with positive value."""
        result = display_utils.format_amount(Decimal("123.45"))
        assert result == "$123.45"

    def test_format_amount_negative(self) -> None:
        """Test amount formatting with negative value."""
        result = display_utils.format_amount(Decimal("-50.00"))
        assert result == "$-50.00"

    def test_format_amount_none(self) -> None:
        """Test amount formatting with None."""
        result = display_utils.format_amount(None)
        assert result == "N/A"

    def test_truncate_short_string(self) -> None:
        """Test truncation of string shorter than max length."""
        result = display_utils.truncate_string("hello", 30)
        assert result == "hello"

    def test_truncate_exact_length(self) -> None:
        """Test truncation of string at exact max length."""
        result = display_utils.truncate_string("hello", 5)
        assert result == "hello"

    def test_truncate_long_string(self) -> None:
        """Test truncation of string longer than max length."""
        result = display_utils.truncate_string("This is a very long description", 10)
        assert result == "This is a ..."
        assert len(result) == 13  # "This is a ..." = 10 chars + "..."

    def test_get_tier_display_high(self) -> None:
        """Test tier display for HIGH confidence."""
        text, icon, color = display_utils.get_tier_display(ConfidenceTier.HIGH)
        assert text == "HIGH"
        assert icon == "⭐"
        assert color == "bold green"

    def test_get_tier_display_medium(self) -> None:
        """Test tier display for MEDIUM confidence."""
        text, icon, color = display_utils.get_tier_display(ConfidenceTier.MEDIUM)
        assert text == "MED"
        assert icon == "○"
        assert color == "yellow"

    def test_get_tier_display_low(self) -> None:
        """Test tier display for LOW confidence."""
        text, icon, color = display_utils.get_tier_display(ConfidenceTier.LOW)
        assert text == "LOW"
        assert icon == "○"
        assert color == "dim cyan"

    def test_get_tier_display_none(self) -> None:
        """Test tier display for NONE confidence."""
        text, icon, color = display_utils.get_tier_display(ConfidenceTier.NONE)
        assert text == "NONE"
        assert icon == "—"
        assert color == "dim"


class TestMatchReviewScreen:
    """Test MatchReviewScreen functionality."""

    def test_get_decision_icon_accepted(self) -> None:
        """Test decision icon for accepted match."""
        screen = self._create_screen()
        result = screen._get_decision_icon(MatchDecision.ACCEPTED)
        assert result == "[green]✓ Accepted[/]"

    def test_get_decision_icon_rejected(self) -> None:
        """Test decision icon for rejected match."""
        screen = self._create_screen()
        result = screen._get_decision_icon(MatchDecision.REJECTED)
        assert result == "[red]✗ Rejected[/]"

    def test_get_decision_icon_pending(self) -> None:
        """Test decision icon for pending match."""
        screen = self._create_screen()
        result = screen._get_decision_icon(MatchDecision.PENDING)
        assert result == "[dim]...Pending[/]"

    def test_get_filtered_matches_all(self) -> None:
        """Test filtering with 'all' mode."""
        screen = self._create_screen()
        screen.match_state.filter_mode = "all"

        result = screen.match_state.get_filtered_and_sorted_matches()

        assert len(result) == 3  # All matches returned

    def test_get_filtered_matches_pending(self) -> None:
        """Test filtering with 'pending' mode."""
        screen = self._create_screen()
        screen.match_state.filter_mode = "pending"

        result = screen.match_state.get_filtered_and_sorted_matches()

        assert len(result) == 1
        assert all(m.decision == MatchDecision.PENDING for m in result)

    def test_get_filtered_matches_accepted(self) -> None:
        """Test filtering with 'accepted' mode."""
        screen = self._create_screen()
        screen.match_state.filter_mode = "accepted"

        result = screen.match_state.get_filtered_and_sorted_matches()

        assert len(result) == 1
        assert all(m.decision == MatchDecision.ACCEPTED for m in result)

    def test_get_filtered_matches_rejected(self) -> None:
        """Test filtering with 'rejected' mode."""
        screen = self._create_screen()
        screen.match_state.filter_mode = "rejected"

        result = screen.match_state.get_filtered_and_sorted_matches()

        assert len(result) == 1
        assert all(m.decision == MatchDecision.REJECTED for m in result)

    def test_get_filtered_matches_unknown_mode(self) -> None:
        """Test filtering with unknown mode returns empty list."""
        screen = self._create_screen()
        # Manually set an invalid filter mode
        screen.match_state.filter_mode = "invalid"  # type: ignore

        result = screen.match_state.get_filtered_and_sorted_matches()

        assert result == []

    def _create_screen(self) -> MatchReviewScreen:
        """Helper to create a MatchReviewScreen for testing."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        # Create matches with different decisions
        matches = [
            TestDataFactory.create_match(
                source_idx=0, target_idx=0, decision=MatchDecision.PENDING
            ),
            TestDataFactory.create_match(
                source_idx=1, target_idx=1, decision=MatchDecision.ACCEPTED
            ),
            TestDataFactory.create_match(
                source_idx=2, target_idx=2, decision=MatchDecision.REJECTED
            ),
        ]
        match_result = TestDataFactory.create_match_result(matches=matches, missing_in_target=[])

        return MatchReviewScreen(source_df, target_df, match_result)


class TestMissingItemsScreen:
    """Test MissingItemsScreen functionality."""

    def _create_screen(self) -> MissingItemsScreen:
        """Helper to create a MissingItemsScreen for testing."""
        source_df = TestDataFactory.create_source_dataframe()
        match_result = TestDataFactory.create_match_result(missing_in_target=[0, 1])
        return MissingItemsScreen(source_df, match_result)


class TestSummaryScreen:
    """Test SummaryScreen functionality."""

    def _create_screen(self) -> SummaryScreen:
        """Helper to create a SummaryScreen for testing."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()
        return SummaryScreen(source_df, target_df, match_result, "source.csv", "target.csv")


class TestScreenInitialization:
    """Test screen initialization and state setup."""

    def test_match_review_screen_initialization_creates_state(self) -> None:
        """Test that MatchReviewScreen creates MatchState if not provided."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()

        screen = MatchReviewScreen(source_df, target_df, match_result, match_state=None)

        assert screen.match_state is not None
        assert screen.match_state.match_result == match_result
        assert screen.match_state.filter_mode == "all"
        assert screen.match_state.selected_match_idx == -1

    def test_match_review_screen_uses_provided_state(self) -> None:
        """Test that MatchReviewScreen uses provided MatchState."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()
        custom_state = MatchState(
            match_result=match_result, filter_mode="pending", selected_match_idx=2
        )

        screen = MatchReviewScreen(source_df, target_df, match_result, match_state=custom_state)

        assert screen.match_state == custom_state
        assert screen.match_state.filter_mode == "pending"
        assert screen.match_state.selected_match_idx == 2

    def test_missing_items_screen_initialization(self) -> None:
        """Test MissingItemsScreen initialization."""
        source_df = TestDataFactory.create_source_dataframe()
        match_result = TestDataFactory.create_match_result(missing_in_target=[0, 1, 2])

        screen = MissingItemsScreen(source_df, match_result)

        assert screen.source_df is source_df
        assert screen.match_result == match_result
        assert len(screen.match_result.missing_in_target) == 3

    def test_summary_screen_initialization(self) -> None:
        """Test SummaryScreen initialization."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()

        screen = SummaryScreen(source_df, target_df, match_result, "source.csv", "target.csv")

        assert screen.source_df is source_df
        assert screen.target_df is target_df
        assert screen.match_result == match_result
        assert screen.source_path == "source.csv"
        assert screen.target_path == "target.csv"
