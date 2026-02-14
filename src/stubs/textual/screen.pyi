"""Type stubs for textual.screen module.

This stub file provides type hints for Textual's Screen class
to improve type safety in the Double Post codebase.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App
    from textual.widgets import Widget

class Screen:
    """A screen in the Textual TUI framework.

    This stub provides type hints for the most commonly used Screen methods.
    """

    # The app that owns this screen
    app: "App[Any]"

    def push_screen(self, screen: "Screen") -> None:
        """Push a new screen onto the screen stack.

        Args:
            screen: The screen to push
        """
        ...

    def pop_screen(self) -> None:
        """Pop the current screen from the screen stack."""
        ...

    def dismiss(self) -> None:
        """Dismiss the current screen."""
        ...
