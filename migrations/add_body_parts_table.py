"""
Migration: Add body_parts table for NudeNet semantic region detection

This migration adds a new table to store body part detections from NudeNet,
providing region-level bounding boxes that complement COCO-WholeBody keypoints.

Usage:
    python migrations/add_body_parts_table.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.storage_manager import StorageManager
from src.config.settings import settings
from sqlalchemy import text


def add_body_parts_table():
    """Create body_parts table with indexes."""
    storage = StorageManager(settings.DATABASE_URL)

    try:
        with storage.session_scope() as session:
            print("Checking if body_parts table exists...")

            # Check if table already exists
            result = session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public' AND table_name='body_parts'
            """))

            if result.fetchone():
                print("✓ body_parts table already exists, skipping migration")
                return

            print("Creating body_parts table...")
            session.execute(text("""
                CREATE TABLE body_parts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                    person_index INTEGER NOT NULL DEFAULT 0,
                    part_name TEXT NOT NULL,
                    canonical_region TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                    bbox REAL[] NOT NULL,
                    is_exposed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))

            print("Creating indexes on body_parts table...")

            # Index on image_id for fast lookups by image
            session.execute(text("""
                CREATE INDEX idx_body_parts_image_id ON body_parts(image_id)
            """))

            # Index on canonical_region for region-based queries
            session.execute(text("""
                CREATE INDEX idx_body_parts_canonical_region ON body_parts(canonical_region)
            """))

            # Index on part_name for specific part searches
            session.execute(text("""
                CREATE INDEX idx_body_parts_part_name ON body_parts(part_name)
            """))

            print("✓ Successfully created body_parts table with indexes")
            print("\nNext steps:")
            print("  1. Install NudeNet: pip install nudenet>=3.4.2")
            print("  2. Re-index your images to detect body parts")
            print("  3. Or run backfill: python backfill_body_parts.py")
            print("\nBody Part Detection provides:")
            print("  - 18 semantic regions (face, feet, torso, armpits, etc.)")
            print("  - Bounding boxes for each detected region")
            print("  - Exposure status (covered vs exposed)")
            print("  - Region-based similarity search filtering")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        storage.close()


if __name__ == "__main__":
    print("=" * 70)
    print("PostureKit Database Migration: Add Body Parts Table (NudeNet)")
    print("=" * 70)
    add_body_parts_table()
    print("=" * 70)
