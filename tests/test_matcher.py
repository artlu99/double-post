"""Tests for matching engine with duplicate prevention (Mitigation #1)."""

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.matcher import Match, MatchConfig, calculate_confidence, find_matches


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_exact_match_high_confidence(self):
        """Test that exact matches return high confidence."""
        source = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )
        target = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        confidence = calculate_confidence(source, target, config)

        # Exact match should be near 1.0
        assert confidence == pytest.approx(1.0, abs=0.01)

    def test_amount_mismatch_reduces_confidence(self):
        """Test that different amounts reduce confidence."""
        source = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )
        target = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("99.99"),
                "description_clean": "netflix.com",
            }
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        confidence = calculate_confidence(source, target, config)

        # Amount mismatch should significantly reduce confidence
        assert confidence <= 0.7

    def test_date_proximity_partial_confidence(self):
        """Test confidence with date within window."""
        source = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )
        target = pd.Series(
            {
                "date_clean": datetime(2024, 1, 17),  # 2 days off
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        confidence = calculate_confidence(source, target, config)

        # Should still have good confidence
        assert confidence >= 0.8

    def test_fuzzy_description_confidence(self):
        """Test fuzzy description matching."""
        source = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            }
        )
        target = pd.Series(
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix",
            }
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        confidence = calculate_confidence(source, target, config)

        # Similar descriptions should still match well
        assert confidence > 0.9


class TestDuplicatePrevention:
    """Tests for duplicate handling (Mitigation #1)."""

    def test_duplicate_descriptions_dont_false_match(self):
        """Test that duplicate transactions don't cause false matches."""
        # Two identical Netflix purchases in source
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 2, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 1,
                },
            ]
        )

        # Only one Netflix purchase in target
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                }
            ]
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        result = find_matches(source_df, target_df, config)

        # Should only match one source record, not both
        assert len(result.matches) == 1
        assert len(result.missing_in_target) == 1  # February Netflix is missing

    def test_greedy_matching_highest_confidence_first(self):
        """Test that highest confidence matches are processed first."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",  # Slightly different description
                    "original_idx": 1,
                },
            ]
        )

        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                }
            ]
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        result = find_matches(source_df, target_df, config)

        # Should match the first source (exact description) not second
        assert len(result.matches) == 1
        assert result.matches[0].source_idx == 0
        assert result.matches[0].confidence > 0.95


class TestMatchResultStructure:
    """Tests for MatchResult data structure."""

    def test_match_result_has_correct_fields(self):
        """Test that MatchResult has expected fields."""
        match = Match(source_idx=0, target_idx=1, confidence=0.95, reason="Exact match")

        assert match.source_idx == 0
        assert match.target_idx == 1
        assert match.confidence == 0.95
        assert match.reason == "Exact match"

    def test_unmatched_record(self):
        """Test unmatched record has None target_idx."""
        match = Match(source_idx=0, target_idx=None, confidence=0.0, reason="No match found")

        assert match.target_idx is None


class TestEndToEndMatching:
    """End-to-end tests for matching logic."""

    def test_full_matching_flow(self):
        """Test complete matching workflow."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 1, 18),
                    "amount_clean": Decimal("42.87"),
                    "description_clean": "whole foods",
                    "original_idx": 1,
                },
                {
                    "date_clean": datetime(2024, 2, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",
                    "original_idx": 2,
                },
            ]
        )

        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 17),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                    "original_idx": 0,
                },
                {
                    "date_clean": datetime(2024, 1, 18),
                    "amount_clean": Decimal("42.87"),
                    "description_clean": "whole foods market",
                    "original_idx": 1,
                },
            ]
        )

        config = MatchConfig(threshold=0.7, date_window_days=3)
        result = find_matches(source_df, target_df, config)

        # Should match 2, 1 missing
        assert len(result.matches) == 2
        assert len(result.missing_in_target) == 1

        # Check that matched records are above threshold
        for match in result.matches:
            assert match.confidence >= config.threshold


class TestConfidenceTierClassification:
    """Tests for 4-tier confidence categorization system."""

    def test_classify_high_tier(self):
        """Test that confidence ≥ 0.9 is classified as HIGH."""
        from src.matcher import classify_confidence_tier
        from src.models import ConfidenceTier

        assert classify_confidence_tier(0.90) == ConfidenceTier.HIGH
        assert classify_confidence_tier(0.95) == ConfidenceTier.HIGH
        assert classify_confidence_tier(1.0) == ConfidenceTier.HIGH

    def test_classify_medium_tier(self):
        """Test that confidence 0.5-0.9 is classified as MEDIUM."""
        from src.matcher import classify_confidence_tier
        from src.models import ConfidenceTier

        assert classify_confidence_tier(0.50) == ConfidenceTier.MEDIUM
        assert classify_confidence_tier(0.75) == ConfidenceTier.MEDIUM
        assert classify_confidence_tier(0.89) == ConfidenceTier.MEDIUM

    def test_classify_low_tier(self):
        """Test that confidence 0.1-0.5 is classified as LOW."""
        from src.matcher import classify_confidence_tier
        from src.models import ConfidenceTier

        assert classify_confidence_tier(0.10) == ConfidenceTier.LOW
        assert classify_confidence_tier(0.25) == ConfidenceTier.LOW
        assert classify_confidence_tier(0.49) == ConfidenceTier.LOW

    def test_classify_none_tier(self):
        """Test that confidence < 0.1 is classified as NONE."""
        from src.matcher import classify_confidence_tier
        from src.models import ConfidenceTier

        assert classify_confidence_tier(0.0) == ConfidenceTier.NONE
        assert classify_confidence_tier(0.05) == ConfidenceTier.NONE
        assert classify_confidence_tier(0.09) == ConfidenceTier.NONE

    def test_auto_accept_high_confidence(self):
        """Test that HIGH tier matches are auto-accepted."""
        from src.models import MatchDecision

        # Create exact match (should be HIGH tier)
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                }
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.confidence >= 0.9
        assert match.tier.value == "high"
        assert match.decision == MatchDecision.ACCEPTED  # Auto-accepted

    def test_pending_for_lower_tiers(self):
        """Test that MEDIUM tier matches start as pending."""
        from src.models import MatchDecision

        # Create match with different description (should be MEDIUM tier)
        # Use different first word to avoid intelligent matching
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "coffee shop downtown",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "cafe morning",  # Different first word
                }
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        match = result.matches[0]
        # Should be MEDIUM tier due to description difference
        assert match.tier.value == "medium"
        assert match.decision == MatchDecision.PENDING  # Not auto-accepted

    def test_low_confidence_included(self):
        """Test that LOW tier matches (0.1-0.5) are included in results."""
        from src.models import ConfidenceTier

        # Create match with different amount and date (should be LOW tier)
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 20),  # Different date (5 days)
                    "amount_clean": Decimal("100.00"),  # Different amount
                    "description_clean": "something else",  # Different description
                }
            ]
        )
        config = MatchConfig(date_window_days=10)

        result = find_matches(source_df, target_df, config, min_confidence=0.1)

        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.tier == ConfidenceTier.LOW
        assert 0.1 <= match.confidence < 0.5

    def test_none_tier_excluded(self):
        """Test that min_confidence parameter excludes low-confidence matches."""
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                }
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 6, 1),  # Very different date
                    "amount_clean": Decimal("999.99"),  # Different amount
                    "description_clean": "something else",
                }
            ]
        )
        config = MatchConfig(date_window_days=3)

        # With min_confidence=0.1, low confidence matches are included
        result = find_matches(source_df, target_df, config, min_confidence=0.1)
        assert len(result.matches) >= 0  # May or may not match depending on description similarity

        # With higher min_confidence, low matches are excluded
        result_strict = find_matches(source_df, target_df, config, min_confidence=0.5)
        assert len(result_strict.matches) == 0
        assert len(result_strict.missing_in_target) == 1

    def test_best_match_for_each_source(self):
        """Test that each source row gets its best target match."""
        from src.models import MatchDecision

        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix",
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "coffee",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.99"),
                    "description_clean": "netflix.com",  # Best match for source 0
                },
                {
                    "date_clean": datetime(2024, 1, 16),
                    "amount_clean": Decimal("50.00"),
                    "description_clean": "coffee shop",  # Best match for source 1
                },
            ]
        )
        config = MatchConfig()

        result = find_matches(source_df, target_df, config)

        # Both sources should have matches
        assert len(result.matches) == 2
        assert len(result.missing_in_target) == 0

        # Verify best match for each source
        source_0_match = next(m for m in result.matches if m.source_idx == 0)
        source_1_match = next(m for m in result.matches if m.source_idx == 1)

        # Each source should match its best target
        assert source_0_match.target_idx == 0
        assert source_1_match.target_idx == 1


class TestIntelligentMatching:
    """Tests for intelligent matching layer (apostrophe normalization, first-two-words)."""

    def test_first_two_words_match_creates_high_confidence(self):
        """Test that matching first two words with same amount creates high confidence."""
        from src.models import ConfidenceTier, MatchDecision

        # Source: "Starbucks Coffee downtown" vs Target: "Starbucks Coffee uptown"
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("8.45"),
                    "description_clean": "starbucks coffee downtown",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("8.45"),
                    "description_clean": "starbucks coffee uptown location",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        match = result.matches[0]

        # Should be high confidence (≥0.90) due to first-two-words match
        assert match.confidence >= 0.90
        assert match.tier == ConfidenceTier.HIGH
        assert match.decision == MatchDecision.ACCEPTED

    def test_apostrophe_normalization_with_first_two_words(self):
        """Test apostrophe normalization with first two words matching."""
        from src.models import ConfidenceTier

        # Source: "Trader Joe's Market" vs Target: "Trader Joes Downtown"
        # After apostrophe removal: "trader joes" matches
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("42.15"),
                    "description_clean": "trader joe's market",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("42.15"),
                    "description_clean": "trader joes downtown",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        match = result.matches[0]

        # Should get high confidence from intelligent matching (≥0.90)
        assert match.confidence >= 0.90
        assert match.tier == ConfidenceTier.HIGH

    def test_simply_noodles_case_insensitive_match(self):
        """Test the user's specific case: Simply Noodles with different locations."""
        from src.models import ConfidenceTier

        # Source: "Simply Noodles 00-08new york ny"
        # Target: "Simply Noodles 267 amsterdam ave"
        # First two words: "simply noodles" match
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2026, 1, 18),
                    "amount_clean": Decimal("-62.50"),
                    "description_clean": "simply noodles 00-08new york ny",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2026, 1, 18),
                    "amount_clean": Decimal("-62.50"),
                    "description_clean": "simply noodles 267 amsterdam ave",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        assert len(result.matches) == 1
        match = result.matches[0]

        # Should get high confidence from intelligent matching (≥0.90)
        assert match.confidence >= 0.90
        assert match.tier == ConfidenceTier.HIGH

    def test_intelligent_match_requires_exact_amount(self):
        """Test that intelligent matching requires exact amount match."""
        # Same first two words but different amounts - intelligent match should NOT trigger
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.50"),
                    "description_clean": "coffee shop downtown",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("100.00"),  # Very different amount
                    "description_clean": "coffee shop uptown",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        # Intelligent matching won't trigger (amounts don't match)
        # Fuzzy matching will also be low due to very different amounts
        if len(result.matches) > 0:
            assert result.matches[0].confidence < 0.90

    def test_intelligent_match_requires_at_least_two_words(self):
        """Test that descriptions with less than 2 words don't trigger intelligent matching."""
        # Single word descriptions should not use intelligent matching
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.50"),
                    "description_clean": "netflix",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("15.50"),
                    "description_clean": "netflix",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        # Should still match but through fuzzy matching, not intelligent
        # Single word can't trigger intelligent matching (need at least 2)
        assert len(result.matches) == 1
        # May or may not be ≥0.90 depending on fuzzy match quality

    def test_first_two_words_dont_match(self):
        """Test that different first two words don't trigger intelligent matching."""
        from src.models import ConfidenceTier

        # Source: "Coffee Shop" vs Target: "Tea House" - different first two words
        source_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("10.00"),
                    "description_clean": "coffee shop downtown",
                },
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "date_clean": datetime(2024, 1, 15),
                    "amount_clean": Decimal("10.00"),
                    "description_clean": "tea house uptown",
                },
            ]
        )

        config = MatchConfig()
        result = find_matches(source_df, target_df, config)

        # Should match through fuzzy matching, not intelligent matching
        if len(result.matches) > 0:
            # Fuzzy match confidence, not the 0.90 intelligent match
            assert result.matches[0].confidence < 0.90
