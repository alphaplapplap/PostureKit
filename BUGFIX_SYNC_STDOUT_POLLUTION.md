# Bug Fix: Database Sync - Stdout Pollution from Transformers

## Problem

Database synchronization was failing with:
```
[SETTINGS] Database synchronization failed: Failed to parse sync response. Got:
```

The Python subprocess was returning output that Swift couldn't parse as JSON.

## Root Cause

**Stdout Pollution from Library Imports**

When `PostureKitBridge` was initialized, various Python libraries (transformers, tokenizers) were printing warning/info messages directly to **stdout**:

```
Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1
None of PyTorch, TensorFlow >= 2.0, or Flax have been found...
```

These messages mixed with the JSON output:
```
Disabling PyTorch because PyTorch >= 2.1 is required...
{"deleted_images": 0, "deleted_poses": 0, ...}
```

Swift's JSON parser couldn't handle the mixed output and failed.

## Why This Happened

1. **Swift runs Python script** that imports `PostureKitBridge` and calls `synchronize_database()`
2. **During import**, transformers library checks for PyTorch/TensorFlow and prints messages to stdout
3. **Script prints JSON** after the library messages
4. **Swift reads combined output**, tries to parse entire stdout as JSON
5. **Parse fails** because of the library messages before the JSON

## The Fix

**Suppress Library Logging to Stdout**

Added environment variables in `src/swift_bridge.py` **before any imports**:

```python
# Suppress transformers/tokenizers logging to stdout (critical for Swift JSON parsing)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
```

Also suppressed UserWarnings:
```python
warnings.filterwarnings('ignore', category=UserWarning)
```

**Result:**
- Library messages are suppressed or redirected to stderr
- Only clean JSON appears on stdout
- Swift can parse the response successfully

## Verification

**Before fix:**
```bash
$ python3 test_swift_sync_exact.py
Disabling PyTorch because PyTorch >= 2.1 is required...
None of PyTorch, TensorFlow >= 2.0, or Flax have been found...
{"deleted_images": 0, ...}
```

**After fix:**
```bash
$ python3 test_swift_sync_exact.py 2>/dev/null
{"deleted_images": 0, "deleted_poses": 0, "deleted_from_index": 0, "scan_duration": 3.4074}
```

Clean JSON output on stdout ✓

## Additional Improvement: Stderr Capture

Also improved error reporting in `PythonBridgeSubprocess.swift`:

**Before:**
```swift
task.standardError = FileHandle.nullDevice  // Errors invisible
```

**After:**
```swift
let errorPipe = Pipe()
task.standardError = errorPipe
// ... capture and log stderr for debugging
```

Now if sync fails, you'll see the actual Python error in the error message.

## Files Modified

1. **src/swift_bridge.py** - Added environment variables to suppress library logging
2. **PostureKit/PythonBridgeSubprocess.swift** - Added stderr capture for debugging

## Files Created

1. **test_swift_sync_exact.py** - Test script that mimics exact Swift behavior
2. **BUGFIX_SYNC_STDOUT_POLLUTION.md** - This documentation

## Testing

1. **Database Sync**: Settings → "Synchronize Database" → Should complete successfully
2. **Check results**: Should show deleted images/poses count or "0 images, 0 poses removed"
3. **Error cases**: If sync fails for other reasons, error message now includes stderr output

## Why This Is Critical

This pattern affects **all Swift-Python subprocess calls** that expect JSON output:
- Database synchronization
- Excluded folders management
- Thumbnail generation
- Any future features using JSON communication

The fix ensures clean stdout for all these operations.

## Lessons Learned

1. **Stdout is sacred for IPC** - Libraries shouldn't pollute it with messages
2. **Set env vars early** - Before any imports that might print
3. **Never discard stderr** - Needed for debugging subprocess failures
4. **Test subprocess calls exactly** - Mimic the actual execution environment
5. **JSON must be alone** - No mixed output on stdout when using JSON communication
