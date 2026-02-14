"""Type stubs for textual.widgets module.

This stub file provides type hints for Textual's DataTable widget
to improve type safety in the Double Post codebase.
"""

from typing import Any

class DataTable:
    """A data table widget for displaying tabular data.

    This stub provides type hints for the most commonly used DataTable methods.
    """

    cursor_type: str
    cursor_row: int  # Read-only property
    zebra_striping: bool

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        """Add a column to the table."""
        ...

    def add_columns(self, *columns: str) -> None:
        """Add multiple columns to the table."""
        ...

    def add_row(self, *cells: Any) -> None:
        """Add a row to the table."""
        ...

    def clear(self) -> None:
        """Clear all rows from the table."""
        ...

    def move_cursor(
        self,
        *,
        row: int | None = None,
        column: int | None = None,
        animate: bool = False,
        scroll: bool = True,
    ) -> None:
        """Move the cursor to the given position.

        Args:
            row: The new row to move the cursor to
            column: The new column to move the cursor to
            animate: Whether to animate the change
            scroll: Scroll the cursor into view after moving
        """
        ...
