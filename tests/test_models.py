"""Tests for data models in Double Post."""

from src.models import Match, MatchDecision, MatchResult


class TestMatchDecision:
    """Test the MatchDecision enum."""

    def test_match_decision_enum_values(self) -> None:
        """Test that MatchDecision has correct enum values."""
        assert MatchDecision.PENDING.value == "pending"
        assert MatchDecision.ACCEPTED.value == "accepted"
        assert MatchDecision.REJECTED.value == "rejected"

    def test_match_decision_enum_members(self) -> None:
        """Test that MatchDecision has correct enum members."""
        assert hasattr(MatchDecision, "PENDING")
        assert hasattr(MatchDecision, "ACCEPTED")
        assert hasattr(MatchDecision, "REJECTED")


class TestMatch:
    """Test the Match dataclass."""

    def test_match_creation_with_defaults(self) -> None:
        """Test creating a Match with default decision."""
        match = Match(
            source_idx=0,
            target_idx=1,
            confidence=0.85,
            reason="exact amount, same date",
        )
        assert match.source_idx == 0
        assert match.target_idx == 1
        assert match.confidence == 0.85
        assert match.reason == "exact amount, same date"
        assert match.decision == MatchDecision.PENDING

    def test_match_creation_with_explicit_decision(self) -> None:
        """Test creating a Match with explicit decision."""
        match = Match(
            source_idx=0,
            target_idx=1,
            confidence=0.85,
            reason="exact amount, same date",
            decision=MatchDecision.ACCEPTED,
        )
        assert match.decision == MatchDecision.ACCEPTED

    def test_match_unmatched_target(self) -> None:
        """Test creating a Match with no target (unmatched)."""
        match = Match(
            source_idx=0,
            target_idx=None,
            confidence=0.0,
            reason="no match found",
        )
        assert match.target_idx is None
        assert match.decision == MatchDecision.PENDING


class TestMatchResult:
    """Test the MatchResult dataclass."""

    def test_match_result_creation(self) -> None:
        """Test creating a MatchResult."""
        matches = [
            Match(source_idx=0, target_idx=0, confidence=0.95, reason="exact match"),
            Match(source_idx=1, target_idx=1, confidence=0.75, reason="similar description"),
        ]
        missing_in_target = [2, 3]

        result = MatchResult(
            matches=matches,
            missing_in_target=missing_in_target,
            duplicate_matches=[],
        )

        assert len(result.matches) == 2
        assert len(result.missing_in_target) == 2
        assert len(result.duplicate_matches) == 0

    def test_match_result_with_accepted_matches(self) -> None:
        """Test MatchResult contains matches with decisions."""
        matches = [
            Match(
                source_idx=0,
                target_idx=0,
                confidence=0.95,
                reason="exact match",
                decision=MatchDecision.ACCEPTED,
            ),
            Match(
                source_idx=1,
                target_idx=1,
                confidence=0.75,
                reason="similar description",
                decision=MatchDecision.REJECTED,
            ),
        ]

        result = MatchResult(
            matches=matches,
            missing_in_target=[],
            duplicate_matches=[],
        )

        assert result.matches[0].decision == MatchDecision.ACCEPTED
        assert result.matches[1].decision == MatchDecision.REJECTED
