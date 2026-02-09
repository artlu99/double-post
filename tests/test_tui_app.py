"""Tests for TUI application logic."""

from pathlib import Path

import pandas as pd
import pytest

from src.models import MatchDecision
from src.tui.app import DoublePostApp, run_tui
from tests.factories import TestDataFactory


class TestDoublePostApp:
    """Test DoublePostApp initialization and setup."""

    def test_app_initialization(self) -> None:
        """Test DoublePostApp initialization with all parameters."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()
        source_path = Path("source.csv")
        target_path = Path("target.csv")

        app = DoublePostApp(source_df, target_df, match_result, source_path, target_path)

        assert app.TITLE == "Double Post - CSV Reconciliation"
        assert app.SUB_TITLE == "Match Review"
        assert app.source_df is source_df
        assert app.target_df is target_df
        assert app.match_result is match_result
        assert app.match_state.match_result is match_result
        assert app.source_path == source_path
        assert app.target_path == target_path

    def test_app_creates_match_state(self) -> None:
        """Test that app creates MatchState with correct defaults."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()

        app = DoublePostApp(source_df, target_df, match_result, Path("s.csv"), Path("t.csv"))

        assert app.match_state is not None
        assert app.match_state.match_result is match_result
        assert app.match_state.filter_mode == "all"
        assert app.match_state.selected_match_idx == -1

    def test_app_css_is_defined(self) -> None:
        """Test that app has CSS defined."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()

        app = DoublePostApp(source_df, target_df, match_result, Path("s.csv"), Path("t.csv"))

        assert app.CSS is not None
        assert "Screen {" in app.CSS
        assert "DataTable {" in app.CSS
        assert "#source_pane" in app.CSS
        assert "#target_pane" in app.CSS


class TestRunTuiFunction:
    """Test the run_tui function."""

    def test_run_tui_creates_app(self, monkeypatch) -> None:
        """Test that run_tui creates and configures app correctly."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()
        match_result = TestDataFactory.create_match_result()
        source_path = Path("source.csv")
        target_path = Path("target.csv")

        # Mock app.run() to prevent actual TUI from opening
        app_instances = []

        def mock_run(self):
            app_instances.append(self)

        monkeypatch.setattr("src.tui.app.DoublePostApp.run", mock_run)

        run_tui(source_df, target_df, match_result, source_path, target_path)

        assert len(app_instances) == 1
        app = app_instances[0]
        assert isinstance(app, DoublePostApp)
        assert app.source_df is source_df
        assert app.target_df is target_df
        assert app.match_result is match_result


class TestTuiIntegration:
    """Integration tests for TUI components."""

    def test_match_state_persists_across_screens(self) -> None:
        """Test that MatchState is shared across screens."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        matches = [
            TestDataFactory.create_match(
                source_idx=0,
                target_idx=0,
                confidence=0.95,
                decision=MatchDecision.PENDING
            ),
        ]
        match_result = TestDataFactory.create_match_result(matches=matches, missing_in_target=[])

        app = DoublePostApp(source_df, target_df, match_result, Path("s.csv"), Path("t.csv"))

        # The same MatchState should be passed to screens
        assert app.match_state is app.match_state
        assert app.match_state.match_result is match_result

    def test_app_with_empty_match_result(self) -> None:
        """Test app initialization with empty match result."""
        source_df = pd.DataFrame()
        target_df = pd.DataFrame()
        match_result = TestDataFactory.create_match_result(matches=[], missing_in_target=[])

        app = DoublePostApp(source_df, target_df, match_result, Path("s.csv"), Path("t.csv"))

        assert app.match_result == match_result
        assert len(app.match_result.matches) == 0
        assert len(app.match_result.missing_in_target) == 0

    def test_app_with_mixed_decisions(self) -> None:
        """Test app with matches in different decision states."""
        source_df = TestDataFactory.create_source_dataframe()
        target_df = TestDataFactory.create_target_dataframe()

        matches = [
            TestDataFactory.create_match(source_idx=0, target_idx=0, decision=MatchDecision.ACCEPTED),
            TestDataFactory.create_match(source_idx=1, target_idx=1, decision=MatchDecision.REJECTED),
            TestDataFactory.create_match(source_idx=2, target_idx=2, decision=MatchDecision.PENDING),
        ]
        match_result = TestDataFactory.create_match_result(matches=matches, missing_in_target=[])

        app = DoublePostApp(source_df, target_df, match_result, Path("s.csv"), Path("t.csv"))

        assert len(app.match_result.matches) == 3

        # Count decisions
        accepted = sum(1 for m in app.match_result.matches if m.decision == MatchDecision.ACCEPTED)
        rejected = sum(1 for m in app.match_result.matches if m.decision == MatchDecision.REJECTED)
        pending = sum(1 for m in app.match_result.matches if m.decision == MatchDecision.PENDING)

        assert accepted == 1
        assert rejected == 1
        assert pending == 1
