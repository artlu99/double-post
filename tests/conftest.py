"""Pytest configuration and fixtures for Double Post tests."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.models import MatchResult
from tests.factories import TestDataFactory


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_source_df() -> pd.DataFrame:
    """Provide a sample source DataFrame for testing."""
    return TestDataFactory.create_source_dataframe()


@pytest.fixture
def sample_target_df() -> pd.DataFrame:
    """Provide a sample target DataFrame for testing."""
    return TestDataFactory.create_target_dataframe()


@pytest.fixture
def sample_match_result() -> MatchResult:
    """Provide a sample MatchResult for testing."""
    return TestDataFactory.create_match_result()


@pytest.fixture
def sample_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide matching source and target DataFrames."""
    source = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("-15.99"),
                "description_clean": "netflix",
            },
            {
                "date_clean": datetime(2024, 1, 16),
                "amount_clean": Decimal("50.00"),
                "description_clean": "deposit",
            },
        ]
    )
    target = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("-15.99"),
                "description_clean": "netflix.com",
            },
        ]
    )
    return source, target
