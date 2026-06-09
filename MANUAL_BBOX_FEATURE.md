# Manual Bounding Box Feature for Improved Pose Detection

## Overview

This feature allows users to manually draw or adjust bounding boxes around people to improve pose detection accuracy, especially for:
- Partially occluded people
- People in crowded scenes
- Small or distant people
- Unusual poses that automatic detection misses

## Architecture

### Current Implementation (Automatic)
```
Image → YOLO Person Detection → Crop with padding → Pose Detection → Keypoints
                ↓
         Bounding Box [x, y, w, h]
```

### New Implementation (Manual Override)
```
Image → User Draws/Adjusts Bbox → Crop with padding → Pose Detection → Keypoints
                ↓
         Custom Bounding Box [x, y, w, h]
```

## Implementation Plan

### 1. Python Backend (src/swift_bridge.py)

Add new method `detect_pose_in_custom_bbox()`:
- Accepts image_array + custom bbox coordinates
- Crops to bbox region with configurable padding
- Runs pose detection on cropped region
- Maps keypoints back to original image coordinates

### 2. Swift Bridge (PythonBridgeSubprocess.swift)

Add new method `detectPoseInCustomBbox()`:
- Accepts NSImage + CGRect bbox
- Calls Python bridge with bbox coordinates
- Returns PoseDetectionResult with adjusted coordinates

### 3. SwiftUI Components

#### BoundingBoxEditor.swift
- Draggable rectangle overlay on image
- Resize handles at corners and edges
- Snap-to-grid option for precision
- Displays bbox dimensions in real-time

#### ManualDetectionView.swift
- Image viewer with bbox drawing capability
- "Draw Bbox" button to enable drawing mode
- "Detect in Region" button to run detection
- Option to refine existing YOLO-detected bboxes

### 4. Integration Points

#### ContentView.swift
- Add "Manual Adjust" button next to detected people
- Opens ManualDetectionView with current image
- Pre-populates with YOLO bbox if available

## API Additions

### Python (swift_bridge.py)

```python
def detect_pose_in_custom_bbox(
    self,
    image_array: np.ndarray,
    bbox: Tuple[int, int, int, int],  # (x, y, width, height)
    crop_padding: float = 0.1
) -> Optional[Dict[str, Any]]:
    """
    Detect pose within a custom bounding box region.

    Args:
        image_array: Full image as numpy array (H, W, 3) RGB
        bbox: Bounding box (x, y, width, height) in pixel coordinates
        crop_padding: Extra padding around bbox (default 0.1 = 10%)

    Returns:
        Dictionary with pose data in ORIGINAL image coordinates
    """
```

### Swift (PythonBridgeSubprocess.swift)

```swift
func detectPoseInCustomBbox(
    image: NSImage,
    bbox: CGRect,  // User-drawn bbox in image coordinates
    cropPadding: Double = 0.1
) -> PoseDetectionResult? {
    // Convert CGRect to [x, y, w, h]
    // Call Python with custom bbox
    // Return result with keypoints in original coordinates
}
```

### SwiftUI (BoundingBoxEditor.swift)

```swift
struct BoundingBoxEditor: View {
    @Binding var bbox: CGRect
    let imageSize: CGSize
    let onDetect: (CGRect) -> Void

    // Interactive bbox drawing/editing
    // Shows dimensions, confidence preview
    // "Detect Pose" button
}
```

## Benefits

1. **Improved Accuracy**: User can manually select hard-to-detect people
2. **Handles Edge Cases**: Works for occluded, small, or unusual poses
3. **Iterative Refinement**: User can adjust bbox and re-run detection
4. **Learning Tool**: Helps users understand what the detector sees
5. **Quality Control**: Manually verify/correct automatic detections

## UX Flow

### Scenario 1: Miss-Detection (YOLO missed a person)
1. User sees image with no detection or partial detection
2. Clicks "Draw Bounding Box" button
3. Drags rectangle around the missed person
4. Clicks "Detect Pose in Region"
5. System runs pose detection on that region
6. Keypoints appear, user can proceed with search

### Scenario 2: False Positive Refinement (YOLO bbox too large/small)
1. User sees detected person with incorrect bbox
2. Clicks "Adjust Bounding Box" on that person
3. Resizes/moves the bbox to better fit
4. Clicks "Re-detect Pose"
5. System re-runs detection with refined bbox
6. More accurate keypoints displayed

### Scenario 3: Crowded Scene (Multiple overlapping people)
1. User sees multiple people, some not detected
2. For each person, draws custom bbox
3. System treats each bbox as separate detection
4. User selects which person to use for search

## Technical Considerations

### Coordinate Space Mapping
- User draws in UI coordinates (may be scaled/transformed)
- Must map to actual pixel coordinates for Python
- Keypoints returned in pixel coordinates
- Must map back to UI coordinates for display

### Performance
- Custom bbox detection is FASTER than full-image detection
  (smaller region to process)
- Can run on-demand, no need to batch process

### Edge Cases
- Bbox too small: Increase padding automatically
- Bbox at image boundary: Clamp to valid region
- Bbox outside image: Show error, reject detection
- Multiple custom bboxes: Support multi-person manual mode

## Example Usage

```swift
// User draws bbox at (100, 150) with size (200, 300)
let customBbox = CGRect(x: 100, y: 150, width: 200, height: 300)

// Run detection in that region
let result = pythonBridge.detectPoseInCustomBbox(
    image: queryImage,
    bbox: customBbox,
    cropPadding: 0.15  // 15% padding
)

if let pose = result {
    // Display keypoints at positions relative to full image
    // User can now use this for search
    print("Detected pose with \(pose.keypoints.count) keypoints")
}
```

## Future Enhancements

1. **Save Custom Bboxes**: Store user-drawn bboxes in database
2. **Bbox Suggestions**: ML model learns from user corrections
3. **Batch Manual Mode**: Draw multiple bboxes, detect all at once
4. **Bbox Templates**: Pre-defined bbox sizes for common scenarios
5. **Confidence Heatmap**: Show where detector is "looking"
6. **Keypoint Refinement**: Allow manual adjustment of individual keypoints

## Implementation Priority

**Phase 1** (Core Functionality):
- [ ] Python: `detect_pose_in_custom_bbox()` method
- [ ] Swift Bridge: `detectPoseInCustomBbox()` method
- [ ] SwiftUI: Basic `BoundingBoxEditor` component

**Phase 2** (UX Polish):
- [ ] Integration with ContentView
- [ ] Resize handles and snap-to-grid
- [ ] Bbox dimension display
- [ ] Validation and error handling

**Phase 3** (Advanced Features):
- [ ] Multi-bbox mode
- [ ] Bbox history/undo
- [ ] Save custom bboxes to database
- [ ] Bbox adjustment for search results

## Related Files

- `src/swift_bridge.py` - Add custom bbox method
- `PostureKit/PythonBridgeSubprocess.swift` - Add Swift bridge
- `PostureKit/ContentView.swift` - Add UI integration
- `src/core/two_stage_detector.py` - Already has crop logic (reuse)

---

**Status**: Design Complete - Ready for Implementation
**Estimated Effort**: 6-8 hours (Phase 1), 4-6 hours (Phase 2)
**Dependencies**: None (uses existing detection pipeline)
