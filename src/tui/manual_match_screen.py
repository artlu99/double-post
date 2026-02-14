"""Manual match screen for Double Post TUI.

Allows users to manually match an unmatched source record to a target record.
"""

from typing import Any

import pandas as pd
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from src.models import MatchResult


class ManualMatchScreen(Screen):
    """Screen for manually matching a source record to a target record.

    This screen is displayed when the user selects an unmatched source record
    and presses a key (e.g., 'm') to manually match it to a target.
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Cancel"),
        ("q", "app.pop_screen", "Cancel"),
        ("enter", "confirm_match", "Confirm Match"),
    ]

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_idx: int,
        match_result: MatchResult | None = None,
    ) -> None:
        """Initialize the manual match screen.

        Args:
            source_df: Normalized source DataFrame
            target_df: Normalized target DataFrame
            source_idx: Index of the source record to match
            match_result: Current match result (used to filter already-matched targets)
        """
        super().__init__()
        self.source_df = source_df
        self.target_df = target_df
        self.source_idx = source_idx
        self.match_result = match_result
        self.selected_target_idx: int | None = None

    def compose(self) -> Any:
        """Compose the manual match screen."""
        yield Header()
        yield Static(
            f"[bold]Manual Match - Source Record #{self.source_idx}[/]",
            id="title",
        )
        yield Static(
            "[dim]Select a target record and press ENTER to confirm, ESC to cancel[/]",
            id="help_text",
        )

        # Source record display
        source_record = self.get_source_record()
        yield Static(
            f"[bold]Source:[/] {source_record['description_clean']} | "
            f"${source_record['amount_clean']} | "
            f"{source_record['date_clean'].strftime('%Y-%m-%d')}",
            id="source_display",
        )

        yield Static("[bold]Available Targets:[/]", id="targets_label")
        yield DataTable(id="targets_table")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the targets table when screen is mounted."""
        table = self.query_one("#targets_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Index", "Date", "Amount", "Description", "Similarity")

        # Get available targets
        available_targets = self.get_available_targets()

        # Add rows for each available target
        for target_idx in available_targets:
            target_row = self.target_df.iloc[target_idx]
            source_row = self.source_df.iloc[self.source_idx]

            # Calculate similarity (simple character match for now)
            similarity = self._calculate_similarity(
                str(source_row["description_clean"]),
                str(target_row["description_clean"]),
            )

            table.add_row(
                str(target_idx),
                target_row["date_clean"].strftime("%Y-%m-%d"),
                f"${target_row['amount_clean']}",
                str(target_row["description_clean"])[:40],
                f"{similarity:.0%}",
            )

    def get_available_targets(self) -> list[int]:
        """Get list of available target indices.

        Filters out targets that are already matched in the current match result.

        Returns:
            List of available target indices
        """
        # Get all target indices
        all_targets = list(range(len(self.target_df)))

        # If no match result, return all targets
        if self.match_result is None:
            return all_targets

        # Filter out already-matched targets
        matched_targets = {
            m.target_idx for m in self.match_result.matches if m.target_idx is not None
        }
        available_targets = [idx for idx in all_targets if idx not in matched_targets]

        return available_targets

    def get_source_record(self) -> pd.Series:
        """Get the source record being matched.

        Returns:
            Source record as pandas Series

        Raises:
            IndexError: If source_idx is out of range
        """
        if self.source_idx < 0 or self.source_idx >= len(self.source_df):
            raise IndexError(f"Source index {self.source_idx} out of range")
        return self.source_df.iloc[self.source_idx]

    def _calculate_similarity(self, source_desc: str, target_desc: str) -> float:
        """Calculate description similarity score.

        Args:
            source_desc: Source description
            target_desc: Target description

        Returns:
            Similarity score from 0.0 to 1.0
        """
        from rapidfuzz import fuzz

        return fuzz.ratio(source_desc, target_desc) / 100.0

    def action_confirm_match(self) -> None:
        """Confirm the selected match.

        This action is triggered when the user presses ENTER.
        In a full implementation, this would create the manual match
        and return to the review screen.
        """
        table = self.query_one("#targets_table", DataTable)
        if table.cursor_row is not None:
            # Get the selected target index from the table
            cell_key = table.get_cell_at(table.cursor_row, 0)
            self.selected_target_idx = int(cell_key.value)

            # In a full implementation, we would:
            # 1. Create the manual match using create_manual_match()
            # 2. Add it to the match result
            # 3. Remove source from missing list
            # 4. Return to review screen

            # For now, just pop the screen
            self.app.pop_screen()
        else:
            self.app.notify(
                "No target selected - use arrow keys to select a row", severity="warning"
            )


__all__ = ["ManualMatchScreen"]
