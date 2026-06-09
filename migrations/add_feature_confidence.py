#!/usr/bin/env python3
"""
Database migration: Add feature_confidence column to geometric_features table.

Adds confidence tracking for each of 52 geometric feature dimensions.
Run this migration on all 3 database profiles (irl, 2d, 3d).

Usage:
    # IRL database
    DB_PROFILE=irl python3 migrations/add_feature_confidence.py

    # 2D database
    DB_PROFILE=2d python3 migrations/add_feature_confidence.py

    # 3D database
    DB_PROFILE=3d python3 migrations/add_feature_confidence.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.storage_manager import StorageManager
from src.config.settings import settings
from sqlalchemy import text


def migrate_up():
    """Add feature_confidence column to geometric_features table."""
    print(f"\n🔧 Adding feature_confidence column to geometric_features")
    print(f"   Profile: {settings.DB_PROFILE}")
    print(f"   Database: {settings.get_database_name()}\n")

    storage = StorageManager()

    with storage.engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='geometric_features'
            AND column_name='feature_confidence'
        """))

        if result.fetchone():
            print("   ⚠️  Column 'feature_confidence' already exists. Skipping.")
            return

        # Add column
        print("   Adding column...")
        conn.execute(text("""
            ALTER TABLE geometric_features
            ADD COLUMN feature_confidence REAL[]
        """))
        conn.commit()

        print("   ✅ Column added successfully!")

        # Get statistics
        result = conn.execute(text("""
            SELECT COUNT(*) FROM geometric_features
        """))
        total_features = result.fetchone()[0]

        print(f"\n   Statistics:")
        print(f"   - Total geometric features: {total_features}")
        print(f"   - feature_confidence column: nullable (backward compatible)")
        print(f"   - NULL values treated as all-valid (confidence=1.0)\n")


def migrate_down():
    """Remove feature_confidence column (rollback)."""
    print(f"\n🔄 Removing feature_confidence column from geometric_features")
    print(f"   Profile: {settings.DB_PROFILE}")
    print(f"   Database: {settings.get_database_name()}\n")

    storage = StorageManager()

    with storage.engine.connect() as conn:
        # Check if column exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='geometric_features'
            AND column_name='feature_confidence'
        """))

        if not result.fetchone():
            print("   ⚠️  Column 'feature_confidence' does not exist. Skipping.")
            return

        # Remove column
        print("   Removing column...")
        conn.execute(text("""
            ALTER TABLE geometric_features
            DROP COLUMN feature_confidence
        """))
        conn.commit()

        print("   ✅ Column removed successfully!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add feature_confidence column migration')
    parser.add_argument('--down', action='store_true', help='Rollback migration')
    args = parser.parse_args()

    try:
        if args.down:
            migrate_down()
        else:
            migrate_up()
        return 0
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
