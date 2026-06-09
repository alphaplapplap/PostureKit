#!/usr/bin/env python3
"""
Run the excluded_folders table migration on all database profiles.

This script adds the excluded_folders table to the irl, 2d, and 3d databases.
Run this after updating to the version with excluded folders support.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.storage_manager import StorageManager
from src.storage.models import Base
from src.config.settings import settings


def run_migration_for_profile(profile: str):
    """Run migration for a specific database profile."""
    print(f"\n{'='*60}")
    print(f"Running migration for profile: {profile}")
    print(f"{'='*60}")

    try:
        # Set the profile
        os.environ['DB_PROFILE'] = profile

        # Create storage manager for this profile
        storage = StorageManager(database_profile=profile)

        # Create all tables (idempotent - won't break existing tables)
        print(f"Creating excluded_folders table if it doesn't exist...")
        Base.metadata.create_all(storage.engine, checkfirst=True)

        # Test that the table works
        print(f"Testing excluded_folders table...")
        excluded = storage.get_excluded_folders()
        print(f"✓ Success! Table is working. Currently {len(excluded)} excluded folders.")

        storage.close()
        return True

    except Exception as e:
        print(f"✗ Error migrating profile '{profile}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run migration for all profiles."""
    print("="*60)
    print("PostureKit: Excluded Folders Migration")
    print("="*60)
    print("\nThis script will add the excluded_folders table to all databases.")
    print("It is safe to run multiple times (idempotent).\n")

    profiles = ['irl', '2d', '3d']
    results = {}

    for profile in profiles:
        results[profile] = run_migration_for_profile(profile)

    # Summary
    print(f"\n{'='*60}")
    print("Migration Summary")
    print(f"{'='*60}")

    for profile, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{profile:10s}: {status}")

    all_success = all(results.values())

    if all_success:
        print("\n✓ All migrations completed successfully!")
        print("\nYou can now use the excluded folders feature in Settings.")
        return 0
    else:
        print("\n✗ Some migrations failed. Check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
