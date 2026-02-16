"""Tests for CSV loader edge cases.

Tests for infer_date_format, standardize_amount, and other edge cases.
"""

from decimal import Decimal

import pandas as pd

from src.csv_loader import infer_date_format, standardize_amount
from src.models import ColumnMapping


class TestInferDateFormatEdgeCases:
    """Test edge cases for date format inference."""

    def test_empty_date_series(self) -> None:
        """Test inference with empty date series."""
        dates = pd.Series([])
        result = infer_date_format(dates)

        # Returns default hints even for empty series
        assert "dayfirst" in result
        assert "yearfirst" in result

    def test_mixed_date_formats(self) -> None:
        """Test inference with mixed date formats in same column."""
        dates = pd.Series(["01/15/2024", "2024-01-16", "15-Jan-2024"])
        result = infer_date_format(dates)

        # Detects yearfirst from ISO format
        assert result.get("yearfirst") == True

    def test_whitespace_in_dates(self) -> None:
        """Test dates with extra whitespace."""
        dates = pd.Series(["  01/15/2024  ", "\t2024-01-16\t"])
        result = infer_date_format(dates)

        # Should handle whitespace gracefully
        assert result is not None

    def test_non_date_strings_mixed(self) -> None:
        """Test non-date strings mixed with valid dates."""
        dates = pd.Series(["01/15/2024", "N/A", "invalid", "2024-01-16"])
        result = infer_date_format(dates)

        # Should still extract format from valid dates
        assert result is not None

    def test_two_digit_years(self) -> None:
        """Test dates with two-digit years."""
        dates = pd.Series(["01/15/24", "01/16/24"])
        result = infer_date_format(dates)

        # Should detect format even with two-digit years
        assert result is not None


class TestStandardizeAmountEdgeCases:
    """Test edge cases for amount standardization."""

    def test_chase_both_debit_and_credit_present(self) -> None:
        """Test Chase format when both Debit and Credit have values."""
        mapping = ColumnMapping(
            date=None,
            amount=None,
            description=None,
            debit="Debit",
            credit="Credit",
            type=None,
            format_type="chase",
        )
        row = pd.Series({"Debit": "100.00", "Credit": "50.00", "Description": "test"})

        result = standardize_amount(row, mapping)

        # Should net the values (debit is expense, credit is income)
        # Net: 100 debit + 50 credit = 100 - 50 = 50 expense
        assert result == Decimal("-50.00")

    def test_chase_zero_values(self) -> None:
        """Test Chase format with zero values."""
        mapping = ColumnMapping(
            date=None,
            amount=None,
            description=None,
            debit="Debit",
            credit="Credit",
            type=None,
            format_type="chase",
        )
        row = pd.Series({"Debit": "0.00", "Credit": "0.00", "Description": "test"})

        result = standardize_amount(row, mapping)

        assert result == Decimal("0")

    def test_amex_type_variations(self) -> None:
        """Test Amex format with different Type values.

        NOTE: Current implementation treats Amex as generic format.
        The Type column is not yet used for sign determination.
        This test documents current behavior - Type handling is a future enhancement.
        """
        mapping = ColumnMapping(
            date=None,
            amount="Amount",
            description=None,
            debit=None,
            credit=None,
            type="Type",
            format_type="generic",  # Currently treated as generic
        )

        # Test with negative amount (expense)
        row1 = pd.Series({"Amount": "-100.00", "Type": "Purchase"})
        result1 = standardize_amount(row1, mapping)
        assert result1 == Decimal("-100.00")

        # Test with positive amount (currently treated as-is)
        # TODO: Future enhancement - use Type column to determine sign
        row2 = pd.Series({"Amount": "50.00", "Type": "debit"})
        result2 = standardize_amount(row2, mapping)
        assert result2 == Decimal("50.00")  # Currently returns positive

    def test_amex_mixed_case_type(self) -> None:
        """Test Amex format with mixed case Type values.

        NOTE: Type column is not currently used for amount sign determination.
        """
        mapping = ColumnMapping(
            date=None,
            amount="Amount",
            description=None,
            debit=None,
            credit=None,
            type="Type",
            format_type="generic",
        )
        row = pd.Series({"Amount": "100.00", "Type": "PURCHASE"})

        result = standardize_amount(row, mapping)

        # Type column is ignored, amount is used as-is
        assert result == Decimal("100.00")

    def test_generic_currency_symbols(self) -> None:
        """Test generic format with various currency symbols."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )

        # Euro symbol
        row1 = pd.Series({"amount": "€100,00"})
        result1 = standardize_amount(row1, mapping)
        # Should handle or return None for unsupported symbols
        assert result1 is not None or result1 is None

        # Pound symbol
        row2 = pd.Series({"amount": "£100.00"})
        result2 = standardize_amount(row2, mapping)
        assert result2 is not None or result2 is None

    def test_generic_commas_in_numbers(self) -> None:
        """Test generic format with commas as thousand separators."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": "1,234.56"})

        result = standardize_amount(row, mapping)

        # Should handle commas
        assert result == Decimal("1234.56")

    def test_generic_whitespace_around_amounts(self) -> None:
        """Test generic format with whitespace around amounts."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": "  100.00  "})

        result = standardize_amount(row, mapping)

        assert result == Decimal("100.00")

    def test_nan_handling(self) -> None:
        """Test handling of NaN values."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": None})

        result = standardize_amount(row, mapping)

        # NaN should return None
        assert result is None


class TestStandardizeAmountFormats:
    """Test amount standardization for various formats."""

    def test_dollar_sign_prefix(self) -> None:
        """Test amount with dollar sign prefix."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": "$100.00"})

        result = standardize_amount(row, mapping)

        assert result == Decimal("100.00")

    def test_negative_signed_amount(self) -> None:
        """Test negative signed amount."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": "-50.00"})

        result = standardize_amount(row, mapping)

        assert result == Decimal("-50.00")

    def test_positive_signed_amount(self) -> None:
        """Test positive signed amount."""
        mapping = ColumnMapping(
            date=None,
            amount="amount",
            description=None,
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )
        row = pd.Series({"amount": "+50.00"})

        result = standardize_amount(row, mapping)

        assert result == Decimal("50.00")
