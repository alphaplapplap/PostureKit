#!/usr/bin/env python3
"""
Test script to diagnose database synchronization issues.
Run this manually to see what errors occur.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("Testing Database Synchronization")
print("="*60)

try:
    print("\n1. Importing PostureKitBridge...")
    from src.swift_bridge import PostureKitBridge
    print("   ✓ Import successful")

    print("\n2. Initializing bridge (skip_models=True)...")
    bridge = PostureKitBridge(skip_models=True)
    print("   ✓ Bridge initialized")

    print("\n3. Calling synchronize_database()...")
    result = bridge.synchronize_database()
    print("   ✓ Sync completed")

    print("\n4. Results:")
    print(f"   Deleted images: {result.get('deleted_images', 'N/A')}")
    print(f"   Deleted poses: {result.get('deleted_poses', 'N/A')}")
    print(f"   Deleted from index: {result.get('deleted_from_index', 'N/A')}")
    print(f"   Duration: {result.get('scan_duration', 'N/A'):.2f}s")

    print("\n5. JSON output (what Swift expects):")
    import json
    print(json.dumps(result, indent=2))

    print("\n✓ SUCCESS - Sync completed without errors")

except Exception as e:
    print(f"\n✗ ERROR occurred:\n")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("This is the error that's causing the sync to fail.")
    print("="*60)
    sys.exit(1)
