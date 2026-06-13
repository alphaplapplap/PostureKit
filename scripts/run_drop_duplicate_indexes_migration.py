#!/usr/bin/env python3
"""
Drop the duplicate pose_id indexes from all database profiles.

geometric_features, visual_features, fused_features, and correction_statistics
each had pose_id declared unique=True (backed by a UNIQUE index *_pose_id_key)
AND a redundant plain btree index idx_*_pose_id. The plain indexes are strict
duplicates of the unique ones, so they are dropped here. The unique constraint
indexes are left intact.

This runs migrations/002_drop_duplicate_pose_id_indexes.sql against the irl,
2d, and 3d databases. It is idempotent (DROP INDEX IF EXISTS) and a no-op on
the empty 2d/3d profiles.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

from src.storage.storage_manager import StorageManager

MIGRATION_SQL = project_root / 'migrations' / '002_drop_duplicate_pose_id_indexes.sql'

# The redundant plain indexes this migration drops (for before/after reporting).
DUPLICATE_INDEXES = [
    'idx_geometric_features_pose_id',
    'idx_visual_features_pose_id',
    'idx_fused_features_pose_id',
    'idx_correction_statistics_pose_id',
]


def _present_indexes(session):
    """Return the subset of DUPLICATE_INDEXES that currently exist."""
    rows = session.execute(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE indexname = ANY(:names)"
        ),
        {'names': DUPLICATE_INDEXES},
    ).scalars().all()
    return set(rows)


def run_migration_for_profile(profile: str) -> bool:
    """Run the drop-duplicate-indexes migration for a specific profile."""
    print(f"\n{'='*60}")
    print(f"Running migration for profile: {profile}")
    print(f"{'='*60}")

    try:
        os.environ['DB_PROFILE'] = profile
        storage = StorageManager(database_profile=profile)

        sql = MIGRATION_SQL.read_text()

        with storage.session_scope() as session:
            before = _present_indexes(session)
            if before:
                print(f"Found {len(before)} duplicate index(es): {sorted(before)}")
            else:
                print("No duplicate pose_id indexes present (already clean).")

            # Execute each statement individually for clear errors. Strip
            # leading comment lines first so a statement that follows the
            # header comment block isn't skipped along with it.
            for chunk in sql.split(';'):
                statement = '\n'.join(
                    line for line in chunk.splitlines()
                    if not line.strip().startswith('--')
                ).strip()
                if not statement:
                    continue
                session.execute(text(statement))

            after = _present_indexes(session)
            dropped = sorted(before - after)
            if dropped:
                print(f"Dropped: {dropped}")
            if after:
                print(f"WARNING: still present after migration: {sorted(after)}")

        storage.close()
        print(f"Success for profile '{profile}'.")
        return True

    except Exception as e:
        print(f"Error migrating profile '{profile}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("PostureKit: Drop Duplicate pose_id Indexes Migration")
    print("=" * 60)
    print("\nThis removes the redundant idx_*_pose_id indexes from all databases.")
    print("It is safe to run multiple times (idempotent).\n")

    profiles = ['irl', '2d', '3d']
    results = {profile: run_migration_for_profile(profile) for profile in profiles}

    print(f"\n{'='*60}")
    print("Migration Summary")
    print(f"{'='*60}")
    for profile, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{profile:10s}: {status}")

    if all(results.values()):
        print("\nAll migrations completed successfully.")
        return 0
    print("\nSome migrations failed. Check the errors above.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
