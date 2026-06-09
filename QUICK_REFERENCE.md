# PostureKit Cross-Reference Quick Reference Card

## Status at a Glance

| Category | Result | Details |
|----------|--------|---------|
| **Breaking References** | ✓ PASS | Zero broken (0/7 functions) |
| **Configuration Wiring** | ✓ PASS | 13/15 properties working |
| **Dead Code** | ⚠ REVIEW | 3 items found (142 lines) |
| **Missing Wrappers** | ⚠ INCOMPLETE | 9 Python functions not exposed |

---

## 1-Minute Summary

### What Works
All pose detection, search, and indexing features are properly wired from UI to Python backend. Every slider and toggle that should affect results actually does.

### What's Broken
- Profile switching disabled (Python ready, Swift missing 3 wrappers)
- Folder watching disabled (Python ready, Swift missing 6 wrappers)

### What's Dead
- `runCancellableSearch()` - 142 unused lines (delete it)
- `kMultiplier` property - UI slider has no effect (delete it)
- `quickLookURL` property - Never used (delete or complete it)

---

## Function Cross-Reference Matrix

### Python → Called From Swift

| Function | Swift Caller | Status |
|----------|--------------|--------|
| `detect_multi_person_poses_from_file` | `detectAllPoses()` | ✓ USED |
| `detect_pose_from_file` | `detectPose()` | ✓ USED |
| `extract_features` | `extractFeatures()` | ✓ USED |
| `index_directory` | `startIndexing()` | ✓ USED |
| `backfill_thumbnails` | `backfillThumbnails()` | ✓ USED |
| `search_similar` | `executeSearch()` script | ✓ USED |
| `browse_by_body_parts` | `browseByBodyParts()` | ✓ USED |

### Python → NOT Called From Swift

| Function | Reason | Impact |
|----------|--------|--------|
| `get_current_profile` | No Swift wrapper | Can't get profile |
| `get_profile_stats` | No Swift wrapper | Can't show stats |
| `switch_profile` | No Swift wrapper | Can't switch profiles |
| `start_folder_watch` | No Swift wrapper | Can't start watching |
| `pause_folder_watch` | No Swift wrapper | Can't pause watching |
| `resume_folder_watch` | No Swift wrapper | Can't resume watching |
| `stop_folder_watch` | No Swift wrapper | Can't stop watching |
| `get_folder_watch_status` | No Swift wrapper | Can't check status |
| `index_single_file` | No Swift wrapper | Can't index 1 file |

---

## Configuration Slider Status

### Working (9 sliders)
```
Detection parameters (3):
- minPersonConf       [0.05-0.95] → Python min_person_conf ✓
- detectionThreshold  [0.05-0.95] → Python detection_threshold ✓
- cropPadding         [0.0-0.3]   → Python crop_padding ✓

Search parameters (5):
- numberOfResults     [5-500]     → Python k ✓
- minConfidence       [0-1]       → Python min_confidence ✓
- minFeatureConfidence[0-1]       → Python min_feature_confidence ✓
- minValidOverlap     [0-52]      → Python min_valid_overlap ✓
- minRegionConfidence [0-1]       → Python min_region_confidence ✓

Browse parameters (2):
- requiredBodyParts   (set)       → Python required_regions ✓
- categoryThresholds  (dict)      → Python category_thresholds ✓
```

### Not Working (1 slider)
```
- kMultiplier         [0-?]       → Python NEVER ✗ DEAD CODE
```

### Inverted Logic (Intentional)
```
- showMultiplePeoplePerImage      → Python deduplicate_images (inverted) ✓
  (true = show all, deduplicate = false = don't deduplicate)
```

---

## Dead Code Locations

### File: PythonBridgeSubprocess.swift

**`runCancellableSearch()` (lines 1083-1224)**
- Status: Orphaned, never called
- Size: 142 lines
- Reason: Replaced by persistent search server
- Action: **DELETE**

### File: PostureKitViewModel.swift

**`@Published var kMultiplier: Double = 1.0` (line 69)**
- Status: Dead property
- Comment: "DEPRECATED"
- Usage: NEVER passed to Python
- UI: Slider exists but meaningless
- Action: **DELETE**

**`@Published var quickLookURL: URL?` (line 96)**
- Status: Dead property
- Comment: "For Space bar Quick Look preview"
- Usage: NEVER used
- UI: Never bound
- Action: **DELETE** or **COMPLETE**

---

## Missing Swift Wrappers to Add

### Profile Management (3 methods needed)

```swift
// File: PythonBridgeSubprocess.swift

func getCurrentProfile() -> String {
    // Wrapper for bridge.get_current_profile()
    // Returns: 'irl', '2d', or '3d'
}

func getProfileStats(profile: String) -> [String: Any]? {
    // Wrapper for bridge.get_profile_stats(profile)
    // Returns: stats dict or nil on error
}

func switchProfile(to newProfile: String) -> Bool {
    // Wrapper for bridge.switch_profile(newProfile)
    // Returns: true on success
}
```

### Folder Watching (6 methods needed)

```swift
// File: PythonBridgeSubprocess.swift

func startFolderWatch(paths: [String], debounceSeconds: Double = 2.0) -> Bool {
    // Wrapper for bridge.start_folder_watch(watch_paths, debounce_seconds)
}

func pauseFolderWatch() -> Bool {
    // Wrapper for bridge.pause_folder_watch()
}

func resumeFolderWatch() -> Bool {
    // Wrapper for bridge.resume_folder_watch()
}

func stopFolderWatch() -> Bool {
    // Wrapper for bridge.stop_folder_watch()
}

func getFolderWatchStatus() -> [String: Any]? {
    // Wrapper for bridge.get_folder_watch_status()
}

func indexSingleFile(_ filePath: String) -> [String: Any]? {
    // Wrapper for bridge.index_single_file(file_path)
}
```

---

## Testing Checklist

### Pre-Cleanup (Delete Code)
- [ ] Search codebase for `runCancellableSearch` - 0 results expected
- [ ] Search codebase for `kMultiplier` - check only UI binding
- [ ] Search codebase for `quickLookURL` - 0 results expected
- [ ] Run full test suite - no failures expected

### Post-Addition (New Wrappers)
- [ ] Test `switchProfile('irl')` → succeeds, backend switches
- [ ] Test `startFolderWatch(['/path'])` → folder monitoring starts
- [ ] Test all 9 new wrappers for null/error handling
- [ ] Add unit tests for wrapper methods

---

## File Locations Reference

| Concern | File | Lines |
|---------|------|-------|
| Dead code | PythonBridgeSubprocess.swift | 1083-1224 |
| Dead properties | PostureKitViewModel.swift | 69, 96 |
| Profile API | swift_bridge.py | 1678-1884 |
| Folder watch API | swift_bridge.py | 1890-2034 |
| Search server | PythonBridgeSubprocess.swift | 52-260 |
| Statistics | PostureKitViewModel.swift | 129-137 |

---

## Impact Assessment

### User Impact: LOW
- No currently working features will break
- Profile/folder features already disabled
- Performance overhead is negligible

### Developer Impact: LOW
- Changes are localized
- No API changes to public interfaces
- Backward compatible

### Maintenance Impact: HIGH
- Enables 2 complete features
- Reduces codebase by 142 lines
- Improves code clarity

---

## Next Steps

1. **Immediate** (5 min): Delete 3 dead code items
2. **This Sprint** (3-5 hrs): Add 9 wrapper methods
3. **Next Sprint** (10-14 hrs): Add UI for profiles & folder watching

**Total effort to enable features: ~15-20 hours**

---

**Last Updated**: 2025-01-15  
**Report Location**: `/Users/linuxbabe/Hardware-Aware/PostureKit/CROSS_REFERENCE_VALIDATION_REPORT.md`
