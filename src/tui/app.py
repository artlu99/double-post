"""Main Textual TUI application for Double Post."""

from pathlib import Path

import pandas as pd
from textual.app import App

from src.models import MatchResult
from src.tui.screens import MatchReviewScreen, MatchState


class DoublePostApp(App):
    """Double Post TUI application.

    Attributes:
        source_df: Normalized source DataFrame (bank data)
        target_df: Normalized target DataFrame (personal records)
        match_state: Shared state for tracking match decisions
        source_path: Original source file path
        target_path: Original target file path
    """

    TITLE = "Double Post - CSV Reconciliation"
    SUB_TITLE = "Match Review"

    CSS = """
    Screen {
        background: $background;
    }

    #review_title, #missing_title {
        text-align: center;
        padding: 1 0;
        background: $surface;
    }

    #help_text {
        text-align: center;
        padding: 0 0 1 0;
        background: $surface;
    }

    #source_pane, #target_pane {
        height: 1fr;
    }

    #source_title, #target_title {
        text-align: center;
        text-style: bold;
        padding: 0 1;
        background: $primary;
    }

    DataTable {
        height: 1fr;
    }

    #summary_text {
        margin: 2 4;
        text-align: left;
    }
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        match_result: MatchResult,
        source_path: Path,
        target_path: Path,
    ) -> None:
        """Initialize the Double Post TUI application.

        Args:
            source_df: Normalized source DataFrame
            target_df: Normalized target DataFrame
            match_result: Results from matching operation
            source_path: Original source file path
            target_path: Original target file path
        """
        super().__init__()
        self.source_df = source_df
        self.target_df = target_df
        self.match_result = match_result
        self.match_state = MatchState(match_result=match_result)
        self.source_path = source_path
        self.target_path = target_path

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        # Start with the match review screen
        self.push_screen(
            MatchReviewScreen(self.source_df, self.target_df, self.match_result, self.match_state)
        )


def run_tui(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    match_result: MatchResult,
    source_path: Path,
    target_path: Path,
) -> None:
    """Run the Textual TUI application.

    Args:
        source_df: Normalized source DataFrame
        target_df: Normalized target DataFrame
        match_result: Results from matching operation
        source_path: Original source file path
        target_path: Original target file path
    """
    app = DoublePostApp(source_df, target_df, match_result, source_path, target_path)
    app.run()


__all__ = ["DoublePostApp", "run_tui"]
