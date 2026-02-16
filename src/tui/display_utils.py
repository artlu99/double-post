"""Shared display formatting utilities for TUI screens.

This module provides common formatting functions used across multiple
TUI screen classes to eliminate code duplication.
"""

from decimal import Decimal

import pandas as pd

from src.models import ConfidenceTier


def format_date(date_val) -> str:
    """Format a date value for display.

    Args:
        date_val: Date value to format (datetime, Timestamp, etc.)

    Returns:
        Formatted date string or "N/A" if value is NaN/None
    """
    if pd.isna(date_val):
        return "N/A"
    return date_val.strftime("%Y-%m-%d")


def format_amount(amount_val: Decimal) -> str:
    """Format an amount value for display.

    Args:
        amount_val: Amount value to format (Decimal, float, etc.)

    Returns:
        Formatted amount string with $ prefix or "N/A" if value is NaN/None
    """
    if pd.isna(amount_val):
        return "N/A"
    amount = Decimal(amount_val)
    return f"${amount:.2f}"


def truncate_string(s: str, max_len: int) -> str:
    """Truncate string to max length with ellipsis.

    Args:
        s: String to truncate
        max_len: Maximum length before truncation

    Returns:
        Truncated string with "..." appended if truncated, otherwise original string
    """
    return s[:max_len] + "..." if len(s) > max_len else s


def get_tier_display(tier: ConfidenceTier) -> tuple[str, str, str]:
    """Get tier text, icon, and color markup for display.

    Args:
        tier: ConfidenceTier enum value

    Returns:
        Tuple of (text, icon, color_markup) for the tier
        - text: Short tier name ("HIGH", "MED", "LOW", "NONE")
        - icon: Symbol representing tier ("⭐", "○", "—")
        - color_markup: Textual markup color ("bold green", "yellow", etc.)
    """
    tier_map = {
        ConfidenceTier.HIGH: ("HIGH", "⭐", "bold green"),
        ConfidenceTier.MEDIUM: ("MED", "○", "yellow"),
        ConfidenceTier.LOW: ("LOW", "○", "dim cyan"),
        ConfidenceTier.NONE: ("NONE", "—", "dim"),
    }
    return tier_map.get(tier, ("?", "?", "white"))


__all__ = ["format_date", "format_amount", "truncate_string", "get_tier_display"]
