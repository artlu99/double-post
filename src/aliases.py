"""Merchant alias database for Double Post.

Stores and manages merchant name aliases to improve matching confidence.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sqlite3
from rapidfuzz import fuzz


@dataclass
class MerchantAlias:
    """Merchant name alias mapping.

    Attributes:
        primary_name: The canonical merchant name
        alias: An alternative name that maps to the primary
        created_at: When this alias was created
        usage_count: How many times this alias has been looked up
    """

    primary_name: str
    alias: str
    created_at: datetime
    usage_count: int


class AliasDatabase:
    """SQLite database for storing and managing merchant aliases.

    Provides CRUD operations for merchant name aliases and similarity search.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the alias database.

        Args:
            db_path: Path to SQLite database file (will be created if doesn't exist)

        Raises:
            Exception: If database cannot be created/opened
        """
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Create table if not exists
        self._create_table()

    def _create_table(self) -> None:
        """Create the aliases table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_name TEXT NOT NULL,
                alias TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def _execute_query(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result rows
        """
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()

    def add_alias(self, primary_name: str, alias: str) -> None:
        """Add a new alias mapping.

        If alias already exists, updates the primary_name.

        Args:
            primary_name: The canonical merchant name
            alias: An alternative name

        Raises:
            ValueError: If primary_name or alias is empty
        """
        # Normalize inputs
        primary_name = primary_name.strip()
        alias = alias.strip().lower()

        if not primary_name:
            raise ValueError("Primary name cannot be empty")
        if not alias:
            raise ValueError("Alias cannot be empty")

        # Check if alias exists
        existing = self._execute_query(
            "SELECT primary_name FROM aliases WHERE alias = ?",
            (alias,)
        )

        if existing:
            # Update existing alias
            self.conn.execute(
                "UPDATE aliases SET primary_name = ? WHERE alias = ?",
                (primary_name, alias)
            )
        else:
            # Insert new alias
            self.conn.execute(
                """INSERT INTO aliases (primary_name, alias, created_at, usage_count)
                   VALUES (?, ?, ?, 0)""",
                (primary_name, alias, datetime.now().isoformat())
            )

        self.conn.commit()

    def get_primary_name(self, alias: str) -> str | None:
        """Look up primary name for an alias.

        Increments usage count when found.

        Args:
            alias: The alias to look up

        Returns:
            Primary name if found, None otherwise
        """
        alias = alias.strip().lower()

        result = self._execute_query(
            "SELECT primary_name, usage_count FROM aliases WHERE alias = ?",
            (alias,)
        )

        if result:
            primary_name = result[0]["primary_name"]
            # Increment usage count
            new_count = result[0]["usage_count"] + 1
            self.conn.execute(
                "UPDATE aliases SET usage_count = ? WHERE alias = ?",
                (new_count, alias)
            )
            self.conn.commit()
            return primary_name

        return None

    def list_aliases(self) -> list[MerchantAlias]:
        """List all aliases in the database.

        Returns:
            List of MerchantAlias objects, sorted by usage_count descending
        """
        rows = self._execute_query(
            """SELECT primary_name, alias, created_at, usage_count
               FROM aliases
               ORDER BY usage_count DESC"""
        )

        return [
            MerchantAlias(
                primary_name=row["primary_name"],
                alias=row["alias"],
                created_at=datetime.fromisoformat(row["created_at"]),
                usage_count=row["usage_count"],
            )
            for row in rows
        ]

    def delete_alias(self, alias: str) -> bool:
        """Delete an alias from the database.

        Args:
            alias: The alias to delete

        Returns:
            True if deleted, False if not found
        """
        alias = alias.strip().lower()

        cursor = self.conn.execute(
            "DELETE FROM aliases WHERE alias = ?",
            (alias,)
        )
        self.conn.commit()

        return cursor.rowcount > 0

    def find_similar_aliases(self, description: str, threshold: float = 0.8) -> list[str]:
        """Find aliases similar to the given description.

        Args:
            description: Description to search for
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of matching aliases, sorted by similarity descending
        """
        description = description.strip().lower()

        # Get all aliases
        aliases = self.list_aliases()

        # Calculate similarity scores
        matches = []
        for alias_obj in aliases:
            similarity = fuzz.ratio(description, alias_obj.alias) / 100.0
            if similarity >= threshold:
                matches.append((similarity, alias_obj.alias))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[0], reverse=True)

        return [alias for _, alias in matches]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = ["MerchantAlias", "AliasDatabase"]
