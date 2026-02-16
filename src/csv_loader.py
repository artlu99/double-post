"""CSV loading and normalization with format detection.

Mitigation #3: Detect and normalize different amount formats (Debit/Credit columns, signed amounts)
Mitigation #4: Infer and parse various date formats (US, EU, ISO)
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dateutil import parser as date_parser
from rapidfuzz import fuzz, process

from src.models import ColumnMapping


def detect_column_mapping(df: pd.DataFrame, _source_type: str | None) -> ColumnMapping:
    """Detect column mappings and bank format type.

    Uses fuzzy matching to identify columns for date, amount, and description.
    Detects Chase, Amex, or generic format based on column patterns.

    Args:
        df: DataFrame to analyze
        source_type: Optional hint about source type

    Returns:
        ColumnMapping with detected format and column names
    """
    columns = df.columns.tolist()
    column_lower = [col.lower() for col in columns]

    # Fuzzy match for key columns
    def find_column(keywords: list[str]) -> str | None:
        # First try exact match
        for keyword in keywords:
            if keyword in column_lower:
                idx = column_lower.index(keyword)
                return columns[idx]
        # Then try fuzzy match with lower cutoff
        matches = process.extractOne(
            keywords,
            column_lower,
            scorer=fuzz.WRatio,
            score_cutoff=60,  # Lowered from 80 for better matching
        )
        if matches:
            return columns[matches[2]]
        return None

    # Detect date column (prefer Post Date over Transaction Date)
    date_col = find_column(["post date", "transaction date", "date", "dt", "trans date"])

    # Detect amount-related columns
    amount_col = find_column(["amount", "amt", "usd"])
    debit_col = find_column(["debit"])
    credit_col = find_column(["credit"])

    # Detect description column
    desc_col = find_column(["description", "desc", "descript", "memo", "merchant"])

    # Determine format type
    if debit_col and credit_col:
        format_type: Literal = "chase"
    else:
        format_type = "generic"

    return ColumnMapping(
        date=date_col,
        amount=amount_col,
        description=desc_col,
        debit=debit_col,
        credit=credit_col,
        type=None,
        format_type=format_type,
    )


def infer_date_format(dates: pd.Series) -> dict:
    """Infer date format from a sample of dates.

    Mitigation #4: Detect US (MDY) vs EU (DMY) vs ISO (YMD) formats.

    Args:
        dates: Series of date strings to analyze

    Returns:
        Dict with dayfirst and yearfirst hints for dateutil.parser
    """
    hints = {"dayfirst": False, "yearfirst": False}

    sample = dates.dropna().head(10)
    if len(sample) == 0:
        return hints

    for date_str in sample:
        date_str = str(date_str).strip()

        # Check for ISO format (YYYY-MM-DD)
        if date_str.startswith(("20", "19")) and "-" in date_str:
            parts = date_str.split("-")
            if len(parts) >= 3 and len(parts[0]) == 4:
                hints["yearfirst"] = True
                return hints

        # Check for slash format
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) >= 2:
                try:
                    first, second = int(parts[0]), int(parts[1])
                    if first > 12 and second <= 12:
                        # Day > 12 indicates DMY format
                        hints["dayfirst"] = True
                        return hints
                except ValueError:
                    continue

    return hints


def standardize_date(date_val: Any, format_hints: dict) -> datetime | None:
    """Parse and standardize a date value.

    Mitigation #4: Use detected format hints to parse ambiguous dates.

    Args:
        date_val: Raw date value from CSV
        format_hints: Dict with dayfirst/yearfirst preferences

    Returns:
        Parsed datetime or None if parsing fails
    """
    if pd.isna(date_val):
        return None

    try:
        return date_parser.parse(str(date_val), **format_hints)
    except (ValueError, TypeError):
        return None


def standardize_amount(row: pd.Series, mapping: ColumnMapping) -> Decimal | None:
    """Normalize amount to signed Decimal.

    Mitigation #3: Handle different bank formats (Chase Debit/Credit, Generic signed amounts)

    Args:
        row: Pandas Series representing a single row
        mapping: Column mapping for the CSV format

    Returns:
        Signed Decimal (negative = expense, positive = income) or None if parsing fails
    """
    try:
        if mapping.format_type == "chase":
            # Chase: Credit - Debit (expense in Debit → negative, payment in Credit → positive)
            debit_val = row.get(mapping.debit)
            credit_val = row.get(mapping.credit)

            debit = Decimal(str(debit_val)) if pd.notna(debit_val) and debit_val else Decimal(0)
            credit = Decimal(str(credit_val)) if pd.notna(credit_val) and credit_val else Decimal(0)

            return credit - debit  # Credit positive → income, Debit positive → expense (negative)

        else:  # generic
            # Generic: Signed amount column (may have negative for expenses)
            amount_val = row.get(mapping.amount)
            if pd.isna(amount_val):
                return None

            # Strip currency symbols and convert to Decimal
            amount_str = str(amount_val).replace("$", "").replace(",", "").strip()
            return Decimal(amount_str)

    except (ValueError, InvalidOperation):
        return None


def detect_sign_convention(df: pd.DataFrame, mapping: ColumnMapping) -> dict[str, str | int]:
    """Detect the sign convention for debits/credits in the CSV.

    Analyzes the DataFrame to determine whether expenses (debits) are stored as
    positive or negative values. The more frequent sign is assumed to be expenses.

    Args:
        df: Raw DataFrame to analyze
        mapping: Column mapping for the CSV format

    Returns:
        Dict with keys:
        - debit_sign: "positive", "negative", or "debit_col" (for Chase format)
        - credit_sign: "positive", "negative", or "credit_col" (for Chase format)
        - positive_count: Count of positive/credit values
        - negative_count: Count of negative/debit values (for Chase: debit_count, credit_count)
    """
    result: dict[str, str | int] = {
        "debit_sign": "negative",
        "credit_sign": "positive",
        "positive_count": 0,
        "negative_count": 0,
    }

    if mapping.format_type == "chase":
        # Chase format: Count non-null Debit and Credit entries
        debit_col = mapping.debit
        credit_col = mapping.credit

        debit_count = 0
        credit_count = 0

        if debit_col and debit_col in df.columns:
            debit_count = df[debit_col].notna().sum()

        if credit_col and credit_col in df.columns:
            credit_count = df[credit_col].notna().sum()

        result["debit_sign"] = "debit_col"
        result["credit_sign"] = "credit_col"
        result["debit_count"] = int(debit_count)
        result["credit_count"] = int(credit_count)

        # For Chase format, also track positive/negative for consistency
        result["positive_count"] = int(credit_count)
        result["negative_count"] = int(debit_count)

    else:
        # Generic format: Count positive and negative amounts
        amount_col = mapping.amount or "amount"

        if amount_col not in df.columns:
            return result

        # Convert amounts to numeric, ignoring errors
        amounts = pd.to_numeric(df[amount_col], errors="coerce")

        positive_count = int((amounts > 0).sum())
        negative_count = int((amounts < 0).sum())

        result["positive_count"] = positive_count
        result["negative_count"] = negative_count

        # Determine which sign represents expenses (more frequent = expenses)
        # NOTE: This assumption applies to credit card accounts where purchases (debits)
        # typically outnumber payments (credits). For other financial accounts like
        # brokerage statements, this logic will need to be extended to handle different
        # transaction patterns (e.g., buys vs sells, dividends, deposits, withdrawals).
        if positive_count > negative_count:
            result["debit_sign"] = "positive"
            result["credit_sign"] = "negative"
        elif negative_count > positive_count:
            result["debit_sign"] = "negative"
            result["credit_sign"] = "positive"
        # If equal, keep default (negative = debit)

    return result


def normalize_dataframe(df: pd.DataFrame, mapping: ColumnMapping, date_hints: dict) -> pd.DataFrame:
    """Apply normalization to DataFrame.

    Args:
        df: Raw DataFrame
        mapping: Detected column mapping
        date_hints: Date format hints

    Returns:
        Normalized DataFrame with date_clean, amount_clean, description_clean columns
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Get the raw column names
    date_col = mapping.date or "date"
    desc_col = mapping.description or "description"

    # Normalize dates
    if date_col in df.columns:
        df["date_clean"] = df[date_col].apply(lambda x: standardize_date(x, date_hints))
    else:
        df["date_clean"] = None

    # Normalize amounts
    df["amount_clean"] = df.apply(lambda row: standardize_amount(row, mapping), axis=1)

    # Normalize descriptions
    if desc_col in df.columns:
        df["description_clean"] = df[desc_col].astype(str).str.strip().str.lower()
    else:
        df["description_clean"] = ""

    # Filter out rows with failed normalization
    df = df.dropna(subset=["date_clean", "amount_clean"])

    return df


def load_csv(
    path: Path, source_type: str | None = None
) -> tuple[pd.DataFrame, ColumnMapping, dict]:
    """Load and normalize a CSV file.

    Args:
        path: Path to CSV file
        source_type: Optional hint about source type

    Returns:
        Tuple of (normalized DataFrame, column mapping, sign convention dict)
    """
    # Load CSV with encoding handling
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        # Try with different encoding
        df = pd.read_csv(path, encoding="latin-1")

    # Detect column mapping
    mapping = detect_column_mapping(df, source_type)

    # Detect sign convention (before normalization, on raw data)
    sign_convention = detect_sign_convention(df, mapping)

    # Infer date format from the date column
    date_col = mapping.date or "date"
    date_hints = infer_date_format(df[date_col]) if date_col in df.columns else {}

    # Normalize DataFrame
    normalized = normalize_dataframe(df, mapping, date_hints)

    return normalized, mapping, sign_convention
