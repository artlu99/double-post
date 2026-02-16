"""Tests for filtering already-reconciled target records."""

from pathlib import Path

from typer.testing import CliRunner

from src.main import app


class TestReconciledFiltering:
    """Tests for filtering target records where reconciled=true."""

    def test_no_filtering_when_all_unreconciled(self, tmp_path: Path) -> None:
        """Test that no records are filtered when all have reconciled=false."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        target.write_text("""date,amount,description,reconciled
2024-01-15,50.00,Lunch Special,false
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should mention loaded 1 target record (none filtered)
        assert "Loaded 1 source records, 1 target records" in result.stdout
        # Should not mention filtering reconciled records
        assert "reconciled" not in result.stdout.lower()

    def test_filters_reconciled_records(self, tmp_path: Path) -> None:
        """Test that records with reconciled=true are filtered out."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-12,200.00,Payment
2024-01-15,50.00,Lunch
""")

        target.write_text("""date,amount,description,reconciled
2024-01-12,200.00,Credit Card Payment,true
2024-01-15,50.00,Lunch Special,false
2024-01-16,75.00,Dinner,false
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should mention filtering 1 reconciled record
        assert "Filtered 1 already-reconciled target record(s)" in result.stdout
        # Loaded message shows original count before filtering (3 records)
        assert "Loaded 2 source records, 3 target records" in result.stdout

    def test_multiple_reconciled_records_filtered(self, tmp_path: Path) -> None:
        """Test that multiple reconciled records are all filtered."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-08,35.50,Groceries
2024-01-10,25.00,Taxi
2024-01-15,50.00,Lunch
""")

        target.write_text("""date,amount,description,reconciled
2024-01-08,35.50,Trader Joe's,true
2024-01-10,25.00,Taxi ride,true
2024-01-15,50.00,Lunch Special,false
2024-01-16,75.00,Dinner,false
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should filter 2 reconciled records
        assert "Filtered 2 already-reconciled target record(s)" in result.stdout
        # Loaded message shows original count before filtering (4 records)
        assert "Loaded 3 source records, 4 target records" in result.stdout

    def test_handles_missing_reconciled_column(self, tmp_path: Path) -> None:
        """Test that missing reconciled column is handled gracefully."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        # Target CSV without reconciled column
        target.write_text("""date,amount,description
2024-01-15,50.00,Lunch Special
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        # Should succeed - no reconciled filtering when column is missing
        assert result.exit_code == 0
        # Should not mention reconciled filtering
        assert "reconciled" not in result.stdout.lower()

    def test_reconciled_values_are_case_insensitive(self, tmp_path: Path) -> None:
        """Test that 'True' and 'TRUE' and 'true' all trigger filtering."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        # Test various boolean representations
        # All dates on or before cutoff (2024-01-16 = source date + 1 day)
        target.write_text("""date,amount,description,reconciled
2024-01-15,50.00,Lunch 1,True
2024-01-16,75.00,Dinner,TRUE
2024-01-16,100.00,Coffee,true
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should filter all 3 as reconciled (case-insensitive)
        assert "Filtered 3 already-reconciled target record(s)" in result.stdout
        # Loaded message shows original count before filtering (3 records)
        assert "Loaded 1 source records, 3 target records" in result.stdout

    def test_false_values_not_filtered(self, tmp_path: Path) -> None:
        """Test that 'false' and '0' values are not filtered."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        target.write_text("""date,amount,description,reconciled
2024-01-15,50.00,Lunch 1,false
2024-01-16,75.00,Dinning,0
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should not filter false/0 values
        assert "reconciled" not in result.stdout.lower()
        assert "Loaded 1 source records, 2 target records" in result.stdout

    def test_filters_then_processes_remaining_records(self, tmp_path: Path) -> None:
        """Test that after filtering reconciled records, remaining are processed."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"

        source.write_text("""date,amount,description
2024-01-15,50.00,Lunch
""")

        target.write_text("""date,amount,description,reconciled
2024-01-12,200.00,Payment,true
2024-01-15,50.00,Lunch Special,false
""")

        runner = CliRunner()
        result = runner.invoke(app, [str(source), str(target), "--dry-run"])

        assert result.exit_code == 0
        # Should filter 1 reconciled record
        assert "Filtered 1 already-reconciled target record(s)" in result.stdout
        # Remaining record should be matched
        assert "Total matches: 1" in result.stdout
        # Should show the match in dry-run output
        assert "$50.00" in result.stdout
