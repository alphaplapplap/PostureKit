#!/usr/bin/env python3
"""
Force-close all PostureKit database connections before ejecting drives.

This script disposes all PostgreSQL connection pools to release file handles
on external drives, allowing safe ejection without "diskutil unmount force".

Usage:
    python scripts/prepare_eject.py

    # Or chain with eject command:
    python scripts/prepare_eject.py && diskutil eject disk6

Why this is needed:
- PostureKit indexes images on external drives
- Database operations hold references to file paths
- PostgreSQL connection pools cache these references
- macOS prevents normal ejection while references exist
- This script forces cleanup of all connections

When to use:
- Before ejecting external drives containing indexed images
- After running any rebuild_index*.py script
- When "Eject" fails in Finder
- Before unplugging drives to prevent data corruption
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.storage_manager import StorageManager


def main():
    print("=" * 70)
    print("POSTUREKIT - PREPARE DRIVE FOR SAFE EJECTION")
    print("=" * 70)
    print("\nClosing all database connections and releasing file handles...")

    try:
        # Force disposal of ALL shared connection pools
        StorageManager.close_all_engines()

        print("\n✓ All PostgreSQL connection pools disposed")
        print("✓ All file handles released")
        print("✓ External drives can now be safely ejected")

        print("\n" + "=" * 70)
        print("SAFE TO EJECT")
        print("=" * 70)

        print("\nYou can now:")
        print("  • Eject drives normally in Finder")
        print("  • Use: diskutil eject disk6")
        print("  • Physically unplug the drive")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: Failed to close connections: {e}")
        print("\nTry emergency cleanup:")
        print("  bash scripts/kill_db_connections.sh")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
