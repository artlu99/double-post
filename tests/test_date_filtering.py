"""Tests for target date filtering based on source date range."""

from pathlib import Path

from typer.testing import CliRunner

from src.main import app


class TestTargetDateFiltering:
    """Tests for filtering target records based on latest source date."""

    def test_no_filtering_when_dates_in_range(self, tmp_path: Path) -> None:
        """Test that no records are filtered when all target dates are within range."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source: dates up to 2024-01-15
        source.write_text("""date,amount,description
2024-01-10,100.00,Coffee
2024-01-15,50.00,Lunch
""")

        # Target: dates within source range
        target.write_text("""date,amount,description
2024-01-10,100.00,Coffee Shop
2024-01-15,50.00,Lunch Special
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed
        assert result.exit_code == 0
        # Should mention loaded 2 target records (none filtered)
        assert "Loaded 2 source records, 2 target records" in result.stdout

    def test_filters_target_records_after_cutoff(self, tmp_path: Path) -> None:
        """Test that target records after latest source date + 1 day are filtered."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source: latest date is 2024-01-15
        source.write_text("""date,amount,description
2024-01-10,100.00,Coffee
2024-01-15,50.00,Lunch
""")

        # Target: includes records after cutoff
        target.write_text("""date,amount,description
2024-01-10,100.00,Coffee Shop
2024-01-15,50.00,Lunch Special
2024-01-16,75.00,Dinner
2024-01-20,200.00,Groceries
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed
        assert result.exit_code == 0
        # Should filter only 2024-01-20 (after 2024-01-16 cutoff)
        # 2024-01-16 is kept because it's on the cutoff date (latest + 1 day)
        assert "Filtered 1 target records dated after 2024-01-16" in result.stdout

    def test_cutoff_includes_one_day_cushion(self, tmp_path: Path) -> None:
        """Test that cutoff is latest source date + 1 day."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source: latest date is 2024-01-15
        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        # Target: includes record on cutoff date (2024-01-16 = latest + 1)
        target.write_text("""date,amount,description
2024-01-15,50.00,Lunch Special
2024-01-16,75.00,Dinner
2024-01-17,100.00,Coffee
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed
        assert result.exit_code == 0
        # Should filter only records after 2024-01-16 (keeps 2024-01-16)
        assert "Filtered 1 target records dated after 2024-01-16" in result.stdout

    def test_handles_missing_date_column_gracefully(self, tmp_path: Path) -> None:
        """Test that files without date column are handled gracefully by CSV loader."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Create files without date column
        source.write_text("""amount,description
100.00,Coffee
""")

        target.write_text("""amount,description
100.00,Coffee Shop
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # CSV loader handles missing date column (creates date_clean as None)
        # Filtering is skipped when date_clean column doesn't exist
        assert result.exit_code == 0
        # Should not mention filtering
        assert "Filtered" not in result.stdout

    def test_handles_nan_dates_in_source(self, tmp_path: Path) -> None:
        """Test that NaN dates in source are handled gracefully."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source: has a NaN date
        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
,75.00,Unknown Date
""")

        target.write_text("""date,amount,description
2024-01-15,50.00,Lunch Special
2024-01-20,200.00,Groceries
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed - filtering should work with max() ignoring NaN
        assert result.exit_code == 0
        # Latest date is 2024-01-15, so cutoff is 2024-01-16
        # 2024-01-20 record should be filtered
        assert "Filtered 1 target records dated after 2024-01-16" in result.stdout

    def test_empty_source_dataframe(self, tmp_path: Path) -> None:
        """Test behavior when source has no valid dates."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        # Source: all dates are NaN/missing
        source.write_text("""date,amount,description
,50.00,Lunch
,75.00,Dinner
""")

        target.write_text("""date,amount,description
2024-01-15,50.00,Lunch Special
2024-01-20,200.00,Groceries
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed - filtering is skipped when all dates are NaN
        assert result.exit_code == 0
        # Should not mention filtering (cutoff can't be determined)
        assert "Filtered" not in result.stdout
