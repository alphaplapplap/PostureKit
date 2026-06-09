# Bug Fix: DetachedInstanceError Causing Zero Search Results

## Problem

After implementing the excluded folders feature, all searches returned **zero results** with this error:

```python
sqlalchemy.orm.exc.DetachedInstanceError: Instance <Image at 0x148a9d4f0> is not bound to a Session;
attribute refresh operation cannot proceed
```

## Root Cause

The exclusion check called `storage.is_path_excluded()` for each search result:

```python
# Line 773 in similarity_engine.py (BEFORE FIX)
if self.storage.is_path_excluded(image.file_path):
    continue
```

**The Problem:**
1. `is_path_excluded()` opens a **new database session** using `session_scope()`
2. After that session closes, the `image` object becomes **detached** from its original session
3. Later code tries to access `image.id` (line 779), causing SQLAlchemy to fail with DetachedInstanceError

## The Fix

**Pre-load excluded folders once at the start of search instead of querying per-result.**

### Changes Made

**File: `src/intelligence/similarity_engine.py`**

**1. Pre-load excluded folders (line 504-511):**
```python
# Pre-load excluded folders to avoid DetachedInstanceError during result iteration
excluded_folder_paths = []
try:
    excluded_folder_paths = [f['folder_path'] for f in self.storage.get_excluded_folders()]
    if excluded_folder_paths:
        logger.debug(f"Loaded {len(excluded_folder_paths)} excluded folders for filtering")
except Exception as e:
    logger.warning(f"Failed to load excluded folders, continuing without exclusions: {e}")
```

**2. Replace per-result query with inline check (line 781-784):**
```python
# Skip if image is in an excluded folder (using pre-loaded list)
if any(image.file_path.startswith(folder) for folder in excluded_folder_paths):
    logger.debug(f"Skipping excluded file: {image.file_path}")
    continue
```

## Benefits

✅ **Fixes the bug**: No more DetachedInstanceError
✅ **Improves performance**: 1 query instead of N queries (one per result)
✅ **Fail-safe**: If excluded folders can't be loaded, search continues without exclusions
✅ **Thread-safe**: Pre-loaded before entering the index lock

## How It Works

1. **Before search starts**: Load all excluded folder paths once into a list
2. **During search**: Check each result's file path against the pre-loaded list using string comparison
3. **If exclusion fails**: Log warning and continue with search (no excluded folders applied)

## Performance Improvement

**Before:**
- N database queries (one per search result)
- Each query opens/closes a session
- Session churn causes detached instance errors

**After:**
- 1 database query (at search start)
- Simple in-memory string comparison per result
- No session issues during iteration

## Testing

Search results should now work correctly:
- ✅ Returns results when no folders are excluded
- ✅ Filters results when folders are excluded
- ✅ Continues working even if excluded_folders table doesn't exist (with warning)
- ✅ No DetachedInstanceError
