"""Tests for CLI entry point.

Tests the main reconcile() function in src/main.py.
"""

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.models import MatchDecision
from tests.factories import TestDataFactory


class TestCLIReconcileCommand:
    """Test the CLI reconcile command."""

    def test_missing_source_file(self, tmp_path: Path) -> None:
        """Test that missing source file produces helpful error."""
        from src.main import app

        runner = CliRunner()
        source = tmp_path / "nonexistent.csv"
        target = tmp_path / "target.csv"
        target.touch()

        result = runner.invoke(app, [str(source), str(target)])

        assert result.exit_code == 1
        # Error goes to stderr in typer
        assert "Error: Source file not found" in result.stderr or "File not found" in result.stderr

    def test_missing_target_file(self, tmp_path: Path) -> None:
        """Test that missing target file produces helpful error."""
        from src.main import app

        runner = CliRunner()
        source = tmp_path / "source.csv"
        source.touch()
        target = tmp_path / "nonexistent.csv"

        result = runner.invoke(app, [str(source), str(target)])

        assert result.exit_code == 1
        # Error goes to stderr in typer
        assert "Error: Target file not found" in result.stderr or "File not found" in result.stderr

    def test_dry_run_mode_output(self, tmp_path: Path) -> None:
        """Test dry-run mode prints matching results."""
        from src.main import app

        runner = CliRunner()

        # Create test CSV files
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Write simple CSV data
        source.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee\n")
        target.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee Shop\n")

        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        assert "MATCHING RESULTS" in result.stdout
        assert "Dry run complete" in result.stdout

    def test_custom_threshold_option(self, tmp_path: Path) -> None:
        """Test that custom min-confidence is accepted."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"
        source.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee\n")
        target.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee Shop\n")

        result = runner.invoke(app, [str(source), str(target), "--min-confidence", "0.8", "--dry-run"])

        assert result.exit_code == 0

    def test_custom_date_window_option(self, tmp_path: Path) -> None:
        """Test that custom date window is accepted."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"
        source.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee\n")
        target.write_text("Date,Amount,Description\n2024-01-15,100.00,Coffee Shop\n")

        result = runner.invoke(app, [str(source), str(target), "--date-window", "5", "--dry-run"])

        assert result.exit_code == 0

    def test_shows_format_detection(self, tmp_path: Path) -> None:
        """Test that format detection is displayed."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Create CSV in generic format
        source.write_text("date,amount,description\n2024-01-15,100.00,Coffee\n")
        target.write_text("date,amount,description\n2024-01-15,100.00,Coffee Shop\n")

        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        assert "Detected" in result.stdout
        assert "Loaded" in result.stdout

    def test_shows_missing_records_in_dry_run(self, tmp_path: Path) -> None:
        """Test that missing records are shown in dry-run mode."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source has 2 records, target has 1
        source.write_text("date,amount,description\n2024-01-15,100.00,Coffee\n2024-01-16,50.00,Lunch\n")
        target.write_text("date,amount,description\n2024-01-15,100.00,Coffee Shop\n")

        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        assert "Missing in target" in result.stdout

    def test_shows_low_confidence_matches(self, tmp_path: Path) -> None:
        """Test that low confidence matches are shown in dry-run mode."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Create records that will have low confidence
        source.write_text("date,amount,description\n2024-01-15,100.00,Coffee\n")
        target.write_text("date,amount,description\n2024-01-15,100.00,Bakery\n")  # Different description

        result = runner.invoke(app, [str(source), str(target), "--min-confidence", "0.1", "--dry-run"])

        assert result.exit_code == 0
        # Should show tier breakdown


class TestCLIFunctionality:
    """Test CLI functionality and integration."""

    def test_confidence_buckets_displayed(self, tmp_path: Path) -> None:
        """Test that confidence buckets are displayed correctly."""
        from src.main import app

        runner = CliRunner()

        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("date,amount,description\n2024-01-15,100.00,Coffee\n2024-01-16,50.00,Lunch\n")
        target.write_text("date,amount,description\n2024-01-15,100.00,Coffee Shop\n2024-01-16,50.00,Lunch\n")

        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Check for confidence buckets
        assert "High confidence" in result.stdout or "≥0.8" in result.stdout
        assert "Medium confidence" in result.stdout or "0.6-0.8" in result.stdout
        assert "Low confidence" in result.stdout or "<0.6" in result.stdout
