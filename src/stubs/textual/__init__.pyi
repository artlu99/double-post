"""Type stubs for Textual framework.

This package provides enhanced type hints for Textual to improve
type safety in the Double Post codebase.

Note: These stubs are not comprehensive - they only cover the
Textual APIs actually used by Double Post. Extend as needed.
"""

from .screen import Screen
from .widgets import DataTable

__all__ = ["Screen", "DataTable"]
