#!/usr/bin/env python
"""Demo script to test the three new features.

Tests manual matching, edit records, and merchant aliases.
"""

from datetime import datetime
from decimal import Decimal

import pandas as pd

from src.aliases import AliasDatabase
from src.matcher import calculate_confidence, create_manual_match, find_matches
from src.models import MatchConfig, MatchResult


def demo_merchant_aliases():
    """Demonstrate merchant alias system."""
    print("\n" + "=" * 60)
    print("DEMO 1: Merchant Alias System")
    print("=" * 60)

    # Create database
    db = AliasDatabase("demo_aliases.db")

    # Add aliases
    print("\n📝 Adding aliases...")
    db.add_alias("Netflix", "netflix.com")
    db.add_alias("Netflix", "netflix")
    db.add_alias("AT&T", "att payment")
    print("  ✓ Netflix → netflix.com")
    print("  ✓ Netflix → netflix")
    print("  ✓ AT&T → att payment")

    # List aliases
    print("\n📋 All aliases (sorted by usage):")
    aliases = db.list_aliases()
    for alias in aliases:
        print(f"  - '{alias.alias}' → '{alias.primary_name}' (used {alias.usage_count}x)")

    # Lookup
    print("\n🔍 Looking up 'netflix.com'...")
    primary = db.get_primary_name("netflix.com")
    print(f"  → Primary name: {primary}")

    # Similarity search
    print("\n🎯 Finding aliases similar to 'netlix'...")
    similar = db.find_similar_aliases("netlix", threshold=0.7)
    print(f"  → Found: {similar}")

    # Integration test
    print("\n🔗 Testing integration with matcher...")

    source = pd.Series(
        {
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "Netflix",
        }
    )
    target = pd.Series(
        {
            "date_clean": datetime(2024, 1, 15),
            "amount_clean": Decimal("15.99"),
            "description_clean": "netflix.com",
        }
    )

    config = MatchConfig()

    # Without alias
    confidence_without = calculate_confidence(source, target, config, alias_db=None)
    print(f"  Confidence WITHOUT alias: {confidence_without:.2f}")

    # With alias
    confidence_with = calculate_confidence(source, target, config, alias_db=db)
    print(f"  Confidence WITH alias:    {confidence_with:.2f}")

    boost = confidence_with - confidence_without
    print(f"  ✨ Confidence boost:       +{boost:.2%}")

    db.close()
    print("\n✅ Merchant Alias System demo complete!")


def demo_manual_matching():
    """Demonstrate manual matching."""
    print("\n" + "=" * 60)
    print("DEMO 2: Manual Matching")
    print("=" * 60)

    # Create test data
    source_df = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "Netflix",
            },
            {
                "date_clean": datetime(2024, 1, 16),
                "amount_clean": Decimal("50.00"),
                "description_clean": "AT&T",
            },
        ]
    )

    target_df = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            },
        ]
    )

    print("\n📊 Test Data:")
    print(f"  Source records: {len(source_df)}")
    print(f"  Target records: {len(target_df)}")

    # Auto-match
    print("\n🤖 Auto-matching...")
    config = MatchConfig(threshold=0.7)
    result = find_matches(source_df, target_df, config)

    print(f"  Found {len(result.matches)} matches")
    print(f"  Missing: {result.missing_in_target}")

    if result.matches:
        match = result.matches[0]
        print(f"\n  Match:")
        print(f"    Source: {source_df.iloc[match.source_idx]['description_clean']}")
        print(f"    Target: {target_df.iloc[match.target_idx]['description_clean']}")
        print(f"    Confidence: {match.confidence:.2f}")
        print(f"    Reason: {match.reason}")
        print(f"    Manual: {match.manual}")

    # Manual match for missing record
    if result.missing_in_target:
        print(f"\n✋ Manual matching for source #{result.missing_in_target[0]}...")
        manual_match = create_manual_match(
            result.missing_in_target[0],
            0,
            source_df,
            target_df,
        )

        print(f"  Created manual match:")
        print(f"    Source: {source_df.iloc[manual_match.source_idx]['description_clean']}")
        print(f"    Target: {target_df.iloc[manual_match.target_idx]['description_clean']}")
        print(f"    Confidence: {manual_match.confidence:.2f}")
        print(f"    Reason: {manual_match.reason}")
        print(f"    Manual: {manual_match.manual}")

    print("\n✅ Manual Matching demo complete!")


def demo_edit_records():
    """Demonstrate record editing."""
    print("\n" + "=" * 60)
    print("DEMO 3: Edit Records")
    print("=" * 60)

    # Create test data
    source_df = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "netflix.com",
            },
        ]
    )

    print("\n📝 Original record:")
    print(f"  Description: '{source_df.iloc[0]['description_clean']}'")

    # Simulate editing
    print("\n✏️  Editing description...")
    new_description = "Netflix Subscription"
    source_df.at[0, "description_clean"] = new_description

    print(f"  New description: '{source_df.iloc[0]['description_clean']}'")

    # Show confidence impact
    target_df = pd.DataFrame(
        [
            {
                "date_clean": datetime(2024, 1, 15),
                "amount_clean": Decimal("15.99"),
                "description_clean": "Netflix",
            }
        ]
    )

    config = MatchConfig()

    print("\n🔗 Confidence impact:")

    # Before edit
    source_df.at[0, "description_clean"] = "netflix.com"
    confidence_before = calculate_confidence(
        source_df.iloc[0],
        target_df.iloc[0],
        config,
    )
    print(f"  Before edit: {confidence_before:.2f}")

    # After edit
    source_df.at[0, "description_clean"] = "Netflix Subscription"
    confidence_after = calculate_confidence(
        source_df.iloc[0],
        target_df.iloc[0],
        config,
    )
    print(f"  After edit:  {confidence_after:.2f}")

    print("\n✅ Edit Records demo complete!")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("DOUBLE POST - FEATURE DEMONSTRATION")
    print("Testing Manual Matching, Edit Records, and Merchant Aliases")
    print("=" * 60)

    try:
        demo_merchant_aliases()
        demo_manual_matching()
        demo_edit_records()

        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETE!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("  1. Run: uv run python -m src.main <source> <target>")
        print("  2. Test the features interactively in the TUI")
        print("  3. Try the keyboard shortcuts:")
        print("     - m: Manual match")
        print("     - e/E: Edit source/target")
        print("     - a: Accept match")
        print("     - r: Reject match")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
