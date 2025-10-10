

# GUI Model Selection Integration

**User-friendly model selection in the PostureKit GUI preferences.**

## Overview

Model selection is now available in the GUI through the Preferences dialog, allowing users to choose between RTMW-L, RTMW-X, or ensemble mode without touching code.

## User Interface

### Location
**Edit → Preferences → Detection Tab**

### Controls

1. **Pose Model Dropdown**
   - RTMW-L (Default - 220MB, Fast)
   - RTMW-X (353MB, Best Quality)
   - Ensemble: RTMW-L + RTMW-X (2.5x slower, 10-15% better)

2. **Fusion Method Dropdown** (visible only for Ensemble)
   - Confidence-Weighted (Recommended)
   - Weighted Average

3. **Two-Stage Detection Checkbox**
   - Enable two-stage detection (YOLO + pose)
   - Works with any model choice

4. **Info Label**
   - Shows real-time information about selected model
   - Displays expected performance

## Screenshots

```
┌─────────────────────────────────────────────────────────┐
│  Pose Estimation Model                                  │
│                                                          │
│  Pose model:  [RTMW-L (Default - 220MB, Fast) ▼]       │
│                                                          │
│  ☐ Enable two-stage detection (YOLO + pose)            │
│                                                          │
│  ℹ️ Fast, good quality. Recommended for most use cases. │
│     Detection time: ~0.6s                               │
└─────────────────────────────────────────────────────────┘
```

When Ensemble is selected:

```
┌─────────────────────────────────────────────────────────┐
│  Pose Estimation Model                                  │
│                                                          │
│  Pose model:  [Ensemble: RTMW-L + RTMW-X ▼]            │
│                                                          │
│  Fusion method: [Confidence-Weighted (Recommended) ▼]   │
│                                                          │
│  ☐ Enable two-stage detection (YOLO + pose)            │
│                                                          │
│  ℹ️ Uses both models for maximum accuracy.              │
│     Detection time: ~1.4s (2.5x slower)                 │
│     10-15% accuracy improvement                         │
└─────────────────────────────────────────────────────────┘
```

## Implementation

### Settings Storage

Settings are persisted using Qt's QSettings (platform-specific):
- **macOS**: `~/Library/Preferences/com.PostureKit.PostureKit.plist`
- **Windows**: Registry at `HKEY_CURRENT_USER\Software\PostureKit\PostureKit`
- **Linux**: `~/.config/PostureKit/PostureKit.conf`

### Settings Keys

```python
# Stored settings:
"detection/pose_model_index"      # int: 0=RTMW-L, 1=RTMW-X, 2=Ensemble
"detection/fusion_method_index"   # int: 0=confidence_weighted, 1=weighted_average
"detection/use_two_stage"          # bool: Enable two-stage detection
```

### Reading Preferences

Use the static helper method to get configuration:

```python
from PySide6.QtCore import QSettings
from gui_legacy.dialogs.preferences_dialog import PreferencesDialog

# Get configuration dictionary
config = PreferencesDialog.get_pose_detector_config()

# Returns:
# {
#     'pose_models': 'rtmw-l' | 'rtmw-x' | ['rtmw-l', 'rtmw-x'],
#     'fusion_method': 'confidence_weighted' | 'weighted_average',
#     'use_two_stage': True | False
# }
```

### Initializing PostureKitBridge

Apply user preferences when creating the bridge:

```python
from src.swift_bridge import PostureKitBridge
from gui_legacy.dialogs.preferences_dialog import PreferencesDialog

# Get user's preferred configuration
config = PreferencesDialog.get_pose_detector_config()

# Initialize bridge with user preferences
bridge = PostureKitBridge(**config)

# Or explicitly:
bridge = PostureKitBridge(
    pose_models=config['pose_models'],
    fusion_method=config['fusion_method'],
    use_two_stage=config['use_two_stage']
)
```

## Main Window Integration

Update your main window initialization:

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ... other initialization ...

        # Initialize PostureKitBridge with user preferences
        config = PreferencesDialog.get_pose_detector_config()
        self.bridge = PostureKitBridge(**config)

    def on_preferences_changed(self, changed_settings):
        """Handle preferences changes."""
        # Check if detection settings changed
        detection_keys = [
            'detection/pose_model_index',
            'detection/fusion_method_index',
            'detection/use_two_stage'
        ]

        if any(key in changed_settings for key in detection_keys):
            # Reinitialize bridge with new settings
            config = PreferencesDialog.get_pose_detector_config()
            self.bridge = PostureKitBridge(**config)

            # Notify user
            from gui.widgets.toast_notification import show_info
            show_info("Pose detector updated with new settings")
```

## Testing

### Interactive Test

Run the test GUI to try preferences:

```bash
python3 test_gui_preferences.py
```

Features:
- ✅ Open preferences dialog
- ✅ View current configuration
- ✅ Test PostureKitBridge initialization
- ✅ Settings persistence

### Manual Testing Checklist

1. **Model Selection**
   - [ ] Select RTMW-L → Info shows "~0.6s"
   - [ ] Select RTMW-X → Info shows "~0.9s (1.5x slower)"
   - [ ] Select Ensemble → Info shows "~1.4s (2.5x slower)"

2. **Fusion Method** (Ensemble only)
   - [ ] Fusion controls hidden for RTMW-L
   - [ ] Fusion controls hidden for RTMW-X
   - [ ] Fusion controls visible for Ensemble
   - [ ] Can select Confidence-Weighted
   - [ ] Can select Weighted Average

3. **Two-Stage Detection**
   - [ ] Checkbox works with RTMW-L
   - [ ] Checkbox works with RTMW-X
   - [ ] Checkbox works with Ensemble

4. **Persistence**
   - [ ] Settings saved when clicking Apply
   - [ ] Settings saved when clicking OK
   - [ ] Settings restored on next app launch
   - [ ] Restore Defaults resets to RTMW-L

5. **PostureKitBridge Integration**
   - [ ] Config read correctly
   - [ ] Bridge initializes with RTMW-L
   - [ ] Bridge initializes with RTMW-X
   - [ ] Bridge initializes with Ensemble
   - [ ] Fusion method applied correctly

## User Documentation

### For End Users

**Choosing a Pose Model:**

1. Open **Edit → Preferences** (or **PostureKit → Preferences** on macOS)
2. Go to the **Detection** tab
3. Select your preferred model from the dropdown:
   - **RTMW-L**: Best for general use (fast, good quality)
   - **RTMW-X**: Best for difficult poses (slower, higher quality)
   - **Ensemble**: Best for maximum accuracy (slowest, 10-15% better)
4. If using Ensemble, choose fusion method:
   - **Confidence-Weighted**: Better quality (recommended)
   - **Weighted Average**: Faster fusion
5. Optionally enable **Two-Stage Detection** for better multi-person handling
6. Click **Apply** or **OK**

**Performance Guide:**

| Configuration | Speed | Quality | Best For |
|--------------|-------|---------|----------|
| RTMW-L | Fast (0.6s) | Good | Most users |
| RTMW-X | Medium (0.9s) | Best | Difficult poses |
| Ensemble | Slow (1.4s) | Maximum | Quality-critical work |
| + Two-Stage | +0.5s | +5-10% | Multi-person images |

## Code Changes

### Files Modified

1. **src/gui_legacy/dialogs/preferences_dialog.py**
   - Added pose model selection controls
   - Added fusion method selection
   - Added two-stage detection checkbox
   - Added `get_pose_detector_config()` static method
   - Added `_on_model_changed()` callback
   - Updated `_load_settings()` and `_save_settings()`

### Files Created

1. **test_gui_preferences.py**
   - Interactive test application
   - Demonstrates preferences dialog
   - Tests bridge initialization

2. **GUI_MODEL_SELECTION.md**
   - This documentation file

## Migration Notes

### For Existing Code

If your code currently hardcodes model selection:

**Before:**
```python
bridge = PostureKitBridge(use_ensemble=True)
```

**After:**
```python
# Use user preferences
config = PreferencesDialog.get_pose_detector_config()
bridge = PostureKitBridge(**config)
```

### Backward Compatibility

The `use_ensemble` parameter still works for backward compatibility:

```python
# Old API still works
bridge = PostureKitBridge(use_ensemble=True)

# Equivalent to:
bridge = PostureKitBridge(pose_models=['rtmw-l', 'rtmw-x'])
```

## Troubleshooting

### Settings Not Persisting

**Problem**: Preferences reset after restarting

**Solution**: Ensure QApplication has correct organization/app name:
```python
app.setOrganizationName("PostureKit")
app.setApplicationName("PostureKit")
```

### Fusion Controls Not Showing

**Problem**: Fusion method dropdown doesn't appear for Ensemble

**Solution**: Check `_on_model_changed()` is connected:
```python
self.pose_model_combo.currentIndexChanged.connect(self._on_model_changed)
```

### Bridge Initialization Fails

**Problem**: PostureKitBridge raises error with preferences

**Solution**: Verify model files exist:
```bash
ls -lh data/models/*.pth
# Should show rtmw-dw-x-l_*.pth (220MB) and rtmw-x_*.pth (353MB)
```

## Future Enhancements

Potential improvements:

1. **Model Download**
   - Automatic download of missing models
   - Progress indicator during download

2. **Performance Indicators**
   - Real-time speed measurement
   - Show actual detection times

3. **Model Comparison**
   - Side-by-side comparison tool
   - Visual quality differences

4. **Quick Switch**
   - Toolbar dropdown for quick model switching
   - Keyboard shortcuts

5. **Batch Processing Profiles**
   - Save named configurations
   - Quick profile switching for different tasks

## See Also

- [MODEL_CONFIGURATION_GUIDE.md](MODEL_CONFIGURATION_GUIDE.md) - Programmatic API
- [test_flexible_config.py](test_flexible_config.py) - Configuration tests
- [test_config_proof.py](test_config_proof.py) - Rigorous validation
