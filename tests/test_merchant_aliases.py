"""Tests for merchant alias system.

Tests for SQLite-based merchant name alias storage and lookup.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.models import MatchConfig
from tests.factories import TestDataFactory


class TestAliasDatabaseInitialization:
    """Test AliasDatabase initialization and setup."""

    def test_database_creates_new_file(self, tmp_path: Path) -> None:
        """Test that a new database file is created."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        assert db_path.exists()

    def test_database_opens_existing_file(self, tmp_path: Path) -> None:
        """Test that existing database can be opened."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db1 = AliasDatabase(db_path)
        # Close and reopen
        db2 = AliasDatabase(db_path)

        assert db2 is not None

    def test_database_creates_aliases_table(self, tmp_path: Path) -> None:
        """Test that aliases table is created on initialization."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        # Check that table exists
        tables = db._execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [t[0] for t in tables]

        assert "aliases" in table_names

    def test_database_handles_invalid_path(self, tmp_path: Path) -> None:
        """Test that invalid database path is handled."""
        from src.aliases import AliasDatabase

        # Invalid path (directory that doesn't exist)
        invalid_path = tmp_path / "nonexistent" / "aliases.db"

        with pytest.raises(Exception):
            AliasDatabase(invalid_path)


class TestAliasAdd:
    """Test adding aliases to database."""

    def test_add_alias_success(self, tmp_path: Path) -> None:
        """Test successfully adding a new alias."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Verify alias was added
        aliases = db.list_aliases()
        assert len(aliases) == 1
        assert aliases[0].primary_name == "Netflix"
        assert aliases[0].alias == "netflix.com"

    def test_add_duplicate_alias_updates_existing(self, tmp_path: Path) -> None:
        """Test that adding duplicate alias updates existing."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.add_alias("Netflix Streaming", "netflix.com")  # Same alias, different primary

        # Should update the existing alias
        aliases = db.list_aliases()
        assert len(aliases) == 1
        assert aliases[0].primary_name == "Netflix Streaming"

    def test_add_alias_with_whitespace_normalized(self, tmp_path: Path) -> None:
        """Test that whitespace is normalized in aliases."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("  Netflix  ", "  netflix.com  ")

        aliases = db.list_aliases()
        assert aliases[0].primary_name == "Netflix"
        assert aliases[0].alias == "netflix.com"

    def test_add_alias_case_insensitive(self, tmp_path: Path) -> None:
        """Test that aliases are case-insensitive."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.add_alias("netflix", "NETFLIX.COM")  # Different case

        # Should treat as duplicate and update
        aliases = db.list_aliases()
        assert len(aliases) == 1

    def test_add_empty_primary_name_raises_error(self, tmp_path: Path) -> None:
        """Test that empty primary name raises error."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        with pytest.raises(ValueError, match="Primary name cannot be empty"):
            db.add_alias("", "netflix.com")

    def test_add_empty_alias_raises_error(self, tmp_path: Path) -> None:
        """Test that empty alias raises error."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        with pytest.raises(ValueError, match="Alias cannot be empty"):
            db.add_alias("Netflix", "")

    def test_add_alias_initializes_usage_count(self, tmp_path: Path) -> None:
        """Test that new alias has usage count of 0."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        aliases = db.list_aliases()
        assert aliases[0].usage_count == 0


class TestAliasLookup:
    """Test looking up aliases in database."""

    def test_get_primary_name_existing_alias(self, tmp_path: Path) -> None:
        """Test looking up existing alias returns primary name."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        result = db.get_primary_name("netflix.com")

        assert result == "Netflix"

    def test_get_primary_name_nonexistent_alias(self, tmp_path: Path) -> None:
        """Test looking up nonexistent alias returns None."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        result = db.get_primary_name("nonexistent")

        assert result is None

    def test_get_primary_name_case_insensitive(self, tmp_path: Path) -> None:
        """Test that lookup is case-insensitive."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Should find with different cases
        assert db.get_primary_name("NETFLIX.COM") == "Netflix"
        assert db.get_primary_name("Netflix.Com") == "Netflix"

    def test_get_primary_name_with_whitespace_variation(self, tmp_path: Path) -> None:
        """Test that lookup handles whitespace variations."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Should find with whitespace
        assert db.get_primary_name("  netflix.com  ") == "Netflix"

    def test_get_primary_name_increments_usage(self, tmp_path: Path) -> None:
        """Test that lookup increments usage count."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # First lookup
        db.get_primary_name("netflix.com")
        aliases = db.list_aliases()
        assert aliases[0].usage_count == 1

        # Second lookup
        db.get_primary_name("netflix.com")
        aliases = db.list_aliases()
        assert aliases[0].usage_count == 2


class TestAliasList:
    """Test listing aliases from database."""

    def test_list_aliases_empty_database(self, tmp_path: Path) -> None:
        """Test listing from empty database returns empty list."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        aliases = db.list_aliases()
        assert aliases == []

    def test_list_aliases_single_alias(self, tmp_path: Path) -> None:
        """Test listing single alias."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        aliases = db.list_aliases()
        assert len(aliases) == 1

    def test_list_aliases_multiple_aliases(self, tmp_path: Path) -> None:
        """Test listing multiple aliases."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.add_alias("Target", "target store")
        db.add_alias("Uber", "uber eats")

        aliases = db.list_aliases()
        assert len(aliases) == 3

    def test_list_aliases_sorted_by_usage(self, tmp_path: Path) -> None:
        """Test that aliases are sorted by usage count descending."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.add_alias("Target", "target")

        # Increment usage for Netflix
        db.get_primary_name("netflix.com")
        db.get_primary_name("netflix.com")

        aliases = db.list_aliases()
        # Netflix should be first (higher usage)
        assert aliases[0].primary_name == "Netflix"

    def test_list_aliases_includes_metadata(self, tmp_path: Path) -> None:
        """Test that listing includes created_at and usage_count."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        aliases = db.list_aliases()
        assert aliases[0].created_at is not None
        assert aliases[0].usage_count == 0


class TestAliasDelete:
    """Test deleting aliases from database."""

    def test_delete_existing_alias_returns_true(self, tmp_path: Path) -> None:
        """Test deleting existing alias returns True."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        result = db.delete_alias("netflix.com")

        assert result is True

    def test_delete_nonexistent_alias_returns_false(self, tmp_path: Path) -> None:
        """Test deleting nonexistent alias returns False."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        result = db.delete_alias("nonexistent")

        assert result is False

    def test_delete_case_insensitive(self, tmp_path: Path) -> None:
        """Test that delete is case-insensitive."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Delete with different case
        result = db.delete_alias("NETFLIX.COM")

        assert result is True
        assert len(db.list_aliases()) == 0

    def test_delete_removes_from_database(self, tmp_path: Path) -> None:
        """Test that delete actually removes alias."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.delete_alias("netflix.com")

        # Verify it's gone
        result = db.get_primary_name("netflix.com")
        assert result is None


class TestAliasSimilaritySearch:
    """Test finding similar aliases."""

    def test_find_similar_aliases_exact_match(self, tmp_path: Path) -> None:
        """Test finding exact matches."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        results = db.find_similar_aliases("netflix.com", threshold=0.8)
        assert len(results) == 1
        assert "netflix.com" in results

    def test_find_similar_aliases_fuzzy_match(self, tmp_path: Path) -> None:
        """Test finding fuzzy matches above threshold."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Similar but not exact
        results = db.find_similar_aliases("netflix", threshold=0.7)
        assert len(results) >= 1

    def test_find_similar_aliases_below_threshold(self, tmp_path: Path) -> None:
        """Test that matches below threshold are excluded."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")

        # Very different description
        results = db.find_similar_aliases("target store", threshold=0.9)
        assert len(results) == 0

    def test_find_similar_aliases_empty_database(self, tmp_path: Path) -> None:
        """Test finding similar aliases in empty database."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        results = db.find_similar_aliases("netflix", threshold=0.8)
        assert results == []

    def test_find_similar_aliases_sorts_by_similarity(self, tmp_path: Path) -> None:
        """Test that results are sorted by similarity score."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        db.add_alias("Netflix", "netflix.com")
        db.add_alias("Netflix", "netflix")  # More similar to "netflix"

        results = db.find_similar_aliases("netflix", threshold=0.5)
        # Most similar should be first
        assert results[0] == "netflix"


class TestAliasIntegrationWithMatcher:
    """Test alias integration with matching confidence calculation."""

    def test_alias_boosts_confidence(self, tmp_path: Path) -> None:
        """Test that alias match increases confidence."""
        from src.aliases import AliasDatabase
        from src.matcher import calculate_confidence

        db_path = tmp_path / "aliases.db"
        alias_db = AliasDatabase(db_path)

        # Add alias: "netflix.com" -> "Netflix"
        alias_db.add_alias("Netflix", "netflix.com")

        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "Netflix",  # Primary name
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix.com",  # Alias
        }])

        config = MatchConfig()

        # Confidence should be boosted due to alias
        confidence = calculate_confidence(
            source_df.iloc[0],
            target_df.iloc[0],
            config,
            alias_db=alias_db,
        )

        # With alias, should be high confidence despite different descriptions
        assert confidence >= 0.9

    def test_no_alias_normal_confidence(self, tmp_path: Path) -> None:
        """Test that without alias uses normal confidence."""
        from src.aliases import AliasDatabase
        from src.matcher import calculate_confidence

        db_path = tmp_path / "aliases.db"
        alias_db = AliasDatabase(db_path)

        # No aliases added

        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "Netflix",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix.com",
        }])

        config = MatchConfig()

        # Without alias, should use normal fuzzy matching
        confidence = calculate_confidence(
            source_df.iloc[0],
            target_df.iloc[0],
            config,
            alias_db=alias_db,
        )

        # Should be moderate (fuzzy match but not exact)
        assert 0.5 <= confidence <= 0.95

    def test_alias_with_amount_mismatch_reduced_confidence(self, tmp_path: Path) -> None:
        """Test that alias doesn't override amount mismatch."""
        from src.aliases import AliasDatabase
        from src.matcher import calculate_confidence

        db_path = tmp_path / "aliases.db"
        alias_db = AliasDatabase(db_path)

        alias_db.add_alias("Netflix", "netflix.com")

        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "Netflix",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("99.99"),  # Different amount
            "description_clean": "netflix.com",
        }])

        config = MatchConfig()

        confidence = calculate_confidence(
            source_df.iloc[0],
            target_df.iloc[0],
            config,
            alias_db=alias_db,
        )

        # Should still be reduced due to amount mismatch
        assert confidence < 0.9

    def test_alias_with_date_mismatch_reduced_confidence(self, tmp_path: Path) -> None:
        """Test that alias doesn't override date mismatch."""
        from src.aliases import AliasDatabase
        from src.matcher import calculate_confidence

        db_path = tmp_path / "aliases.db"
        alias_db = AliasDatabase(db_path)

        alias_db.add_alias("Netflix", "netflix.com")

        source_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "Netflix",
        }])
        target_df = pd.DataFrame([{
            "date_clean": datetime(2024, 1, 25),  # Different date
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix.com",
        }])

        config = MatchConfig(date_window_days=3)

        confidence = calculate_confidence(
            source_df.iloc[0],
            target_df.iloc[0],
            config,
            alias_db=alias_db,
        )

        # Should be reduced due to date mismatch
        assert confidence < 0.9


class TestAliasEdgeCases:
    """Test edge cases for alias system."""

    def test_alias_with_special_characters(self, tmp_path: Path) -> None:
        """Test handling special characters in names."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        # Special characters
        db.add_alias("Target Store #1234", "target")

        aliases = db.list_aliases()
        assert len(aliases) == 1

    def test_alias_with_unicode_characters(self, tmp_path: Path) -> None:
        """Test handling unicode characters."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        # Unicode/emoji
        db.add_alias("Café", "café")

        result = db.get_primary_name("café")
        assert result == "Café"

    def test_alias_very_long_names(self, tmp_path: Path) -> None:
        """Test handling very long merchant names."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"
        db = AliasDatabase(db_path)

        long_name = "A" * 500
        db.add_alias(long_name, "short")

        result = db.get_primary_name("short")
        assert result == long_name

    def test_concurrent_alias_access(self, tmp_path: Path) -> None:
        """Test that multiple database instances work correctly."""
        from src.aliases import AliasDatabase

        db_path = tmp_path / "aliases.db"

        db1 = AliasDatabase(db_path)
        db2 = AliasDatabase(db_path)

        db1.add_alias("Netflix", "netflix.com")

        # Both should be able to read
        assert db2.get_primary_name("netflix.com") == "Netflix"
