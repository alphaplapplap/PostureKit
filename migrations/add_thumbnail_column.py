"""
Migration: Add thumbnail column to images table

This migration adds a BYTEA column to store pre-generated 200x200 JPEG thumbnails.
Run this before using the new thumbnail generation features.

Usage:
    python migrations/add_thumbnail_column.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.storage_manager import StorageManager
from src.config.settings import settings
from sqlalchemy import text

def add_thumbnail_column():
    """Add thumbnail column to images table."""
    storage = StorageManager(settings.DATABASE_URL)

    try:
        with storage.session_scope() as session:
            print("Checking if thumbnail column exists...")

            # Check if column already exists
            result = session.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='images' AND column_name='thumbnail'
            """))

            if result.fetchone():
                print("✓ Thumbnail column already exists, skipping migration")
                return

            print("Adding thumbnail column to images table...")
            session.execute(text("""
                ALTER TABLE images
                ADD COLUMN thumbnail BYTEA
            """))

            print("✓ Successfully added thumbnail column")
            print("\nNext steps:")
            print("  1. Re-index your images to generate thumbnails")
            print("  2. Use: python -c 'from src.swift_bridge import *; index_directory(\"/path/to/images\")'")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        storage.close()

if __name__ == "__main__":
    print("=" * 60)
    print("PostureKit Database Migration: Add Thumbnail Column")
    print("=" * 60)
    add_thumbnail_column()
    print("=" * 60)
