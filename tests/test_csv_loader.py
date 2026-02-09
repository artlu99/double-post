"""Tests for CSV loader with format detection and normalization."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.csv_loader import (
    detect_column_mapping,
    detect_sign_convention,
    infer_date_format,
    load_csv,
    normalize_dataframe,
    standardize_amount,
    standardize_date,
)
from src.models import ColumnMapping


class TestColumnMappingDetection:
    """Tests for automatic column mapping detection."""

    def test_detect_chase_format(self, fixtures_dir: Path):
        """Test Chase format detection with Debit/Credit columns."""
        import pandas as pd

        df = pd.read_csv(fixtures_dir / "chase.csv")
        mapping = detect_column_mapping(df, None)

        assert mapping.format_type == "chase"
        assert mapping.debit is not None
        assert mapping.credit is not None
        assert mapping.description is not None

    def test_detect_personal_format(self, fixtures_dir: Path):
        """Test personal/generic format detection."""
        import pandas as pd

        df = pd.read_csv(fixtures_dir / "personal.csv")
        mapping = detect_column_mapping(df, None)

        assert mapping.format_type == "generic"
        assert mapping.amount is not None
        assert mapping.description is not None


class TestDateFormatInference:
    """Tests for date format auto-detection (Mitigation #4)."""

    def test_infer_us_date_format(self):
        """Test US date format (MDY) detection."""
        import pandas as pd

        dates = pd.Series(["01/15/2024", "02/20/2024", "03/25/2024"])
        hints = infer_date_format(dates)

        assert hints["dayfirst"] is False
        assert hints["yearfirst"] is False

    def test_infer_eu_date_format(self):
        """Test EU date format (DMY) detection."""
        import pandas as pd

        # Day > 12 indicates DMY format
        dates = pd.Series(["15/01/2024", "20/02/2024", "25/03/2024"])
        hints = infer_date_format(dates)

        assert hints["dayfirst"] is True

    def test_infer_iso_date_format(self):
        """Test ISO date format (YYYY-MM-DD) detection."""
        import pandas as pd

        dates = pd.Series(["2024-01-15", "2024-02-20", "2024-03-25"])
        hints = infer_date_format(dates)

        assert hints["yearfirst"] is True


class TestDateStandardization:
    """Tests for date parsing with various formats."""

    def test_standardize_iso_date(self):
        """Test ISO 8601 date format."""
        result = standardize_date("2024-01-15", {})

        assert result == datetime(2024, 1, 15)

    def test_standardize_us_date_format(self):
        """Test US date format (MM/DD/YYYY)."""
        result = standardize_date("01/15/2024", {"dayfirst": False})

        assert result == datetime(2024, 1, 15)

    def test_standardize_eu_date_format(self):
        """Test EU date format (DD/MM/YYYY)."""
        result = standardize_date("15/01/2024", {"dayfirst": True})

        assert result == datetime(2024, 1, 15)

    def test_standardize_datetime_with_time(self):
        """Test datetime string with time component."""
        result = standardize_date("2024-01-15 14:30:00", {})

        assert result == datetime(2024, 1, 15, 14, 30, 0)

    def test_standardize_invalid_date(self):
        """Test that invalid dates return None."""
        result = standardize_date("not-a-date", {})

        assert result is None


class TestAmountStandardization:
    """Tests for amount normalization (Mitigation #3)."""

    def test_standardize_chase_debit(self):
        """Test Chase format: Debit column (expense)."""
        import pandas as pd

        mapping = ColumnMapping(
            date="Post Date",
            amount=None,
            description="Description",
            debit="Debit",
            credit="Credit",
            type=None,
            format_type="chase",
        )
        row = pd.Series({"Debit": Decimal("15.99"), "Credit": None})

        result = standardize_amount(row, mapping)

        assert result == Decimal("-15.99")

    def test_standardize_chase_credit(self):
        """Test Chase format: Credit column (income/payment)."""
        import pandas as pd

        mapping = ColumnMapping(
            date="Post Date",
            amount=None,
            description="Description",
            debit="Debit",
            credit="Credit",
            type=None,
            format_type="chase",
        )
        row = pd.Series({"Debit": None, "Credit": Decimal("200.00")})

        result = standardize_amount(row, mapping)

        assert result == Decimal("200.00")

    # NOTE: test_standardize_with_dollar_sign and test_standardize_generic_signed_amount
    # removed - these scenarios are covered more comprehensively in test_csv_loader_edge_cases.py


class TestCSVLoudAndNormalization:
    """End-to-end tests for CSV loading and normalization."""

    def test_load_chase_csv(self, fixtures_dir: Path):
        """Test loading and normalizing Chase CSV."""
        df, mapping, convention = load_csv(fixtures_dir / "chase.csv")

        assert mapping.format_type == "chase"
        assert "date_clean" in df.columns
        assert "amount_clean" in df.columns
        assert "description_clean" in df.columns

        # Check that amounts are normalized to Decimal
        assert df["amount_clean"].notna().sum() > 0

        # Check that dates are parsed
        assert df["date_clean"].notna().sum() > 0

    def test_load_amex_csv(self, fixtures_dir: Path):
        """Test loading Amex format with signed amounts (no Type column)."""
        df, mapping, convention = load_csv(fixtures_dir / "amex.csv")

        # Should be detected as generic (no Type column)
        assert mapping.format_type == "generic"
        assert "date_clean" in df.columns
        assert "amount_clean" in df.columns

        # Verify positive amounts are preserved
        positive_rows = df[df["amount_clean"] > 0]
        assert len(positive_rows) > 0

        # Verify negative amounts are preserved (credits/refunds)
        negative_rows = df[df["amount_clean"] < 0]
        assert len(negative_rows) > 0

        # Check specific values from fixture
        assert Decimal("62.50") in df["amount_clean"].values
        assert Decimal("-10.89") in df["amount_clean"].values
        assert Decimal("-25.00") in df["amount_clean"].values

    def test_load_personal_csv(self, fixtures_dir: Path):
        """Test loading and normalizing personal CSV."""
        df, mapping, convention = load_csv(fixtures_dir / "personal.csv")

        assert mapping.format_type == "generic"
        assert "date_clean" in df.columns
        assert "amount_clean" in df.columns

    def test_normalize_dataframe_removes_invalid_rows(self):
        """Test that rows with unparseable dates/amounts are filtered."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "date": ["2024-01-15", "invalid-date", "2024-01-17"],
                "amount": [10.0, 20.0, "invalid"],
                "description": ["A", "B", "C"],
            }
        )
        mapping = ColumnMapping(
            date="date",
            amount="amount",
            description="description",
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )

        normalized = normalize_dataframe(df, mapping, {})

        # Should filter out rows with invalid data
        assert len(normalized) < len(df)


class TestSignConventionDetection:
    """Tests for automatic detection of expense sign convention."""

    def test_detect_negative_expenses(self, fixtures_dir: Path):
        """Test detection when expenses are negative (like personal.csv)."""
        import pandas as pd

        df = pd.read_csv(fixtures_dir / "personal.csv")
        mapping = detect_column_mapping(df, None)

        # personal.csv has mostly negative amounts (expenses)
        # So negative should be detected as DEBIT (expenses)
        convention = detect_sign_convention(df, mapping)

        assert convention["debit_sign"] == "negative"
        assert convention["credit_sign"] == "positive"
        assert convention["negative_count"] > convention["positive_count"]

    def test_detect_positive_expenses(self, fixtures_dir: Path):
        """Test detection when expenses are positive (like amex_signed.csv)."""
        import pandas as pd

        df = pd.read_csv(fixtures_dir / "amex_signed.csv")
        mapping = detect_column_mapping(df, None)

        # amex_signed.csv has mostly positive amounts (expenses/charges)
        # So positive should be detected as DEBIT (expenses)
        convention = detect_sign_convention(df, mapping)

        assert convention["debit_sign"] == "positive"
        assert convention["credit_sign"] == "negative"
        assert convention["positive_count"] > convention["negative_count"]

    def test_detect_chase_format(self, fixtures_dir: Path):
        """Test detection for Chase format with Debit/Credit columns."""
        import pandas as pd

        df = pd.read_csv(fixtures_dir / "chase.csv")
        mapping = detect_column_mapping(df, None)

        # Chase: Debit column = expenses, Credit column = income/payments
        convention = detect_sign_convention(df, mapping)

        assert convention["debit_sign"] == "debit_col"
        assert convention["credit_sign"] == "credit_col"
        assert convention["debit_count"] > convention["credit_count"]

    def test_sign_convention_with_mixed_data(self):
        """Test detection with mixed positive/negative amounts."""
        import pandas as pd

        # Create test data: 6 negative (expenses), 2 positive (payments)
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "amount": [-10.50, -25.00, -15.99, 200.00],
                "description": ["A", "B", "C", "Payment"],
            }
        )
        mapping = ColumnMapping(
            date="date",
            amount="amount",
            description="description",
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )

        convention = detect_sign_convention(df, mapping)

        # More negatives → negative is debit (expenses)
        assert convention["debit_sign"] == "negative"
        assert convention["negative_count"] == 3
        assert convention["positive_count"] == 1

    def test_sign_convention_with_equal_counts(self):
        """Test detection when positive and negative counts are equal."""
        import pandas as pd

        # Create test data: 2 negative, 2 positive
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "amount": [-10.50, -25.00, 200.00, 100.00],
                "description": ["A", "B", "Payment", "Deposit"],
            }
        )
        mapping = ColumnMapping(
            date="date",
            amount="amount",
            description="description",
            debit=None,
            credit=None,
            type=None,
            format_type="generic",
        )

        convention = detect_sign_convention(df, mapping)

        # Equal counts → default to negative as debit (common convention)
        assert convention["debit_sign"] == "negative"
        assert convention["positive_count"] == convention["negative_count"]
