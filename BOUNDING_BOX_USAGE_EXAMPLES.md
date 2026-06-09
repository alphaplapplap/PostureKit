# Bounding Box Feature - Usage Examples

## Overview

The manual bounding box feature allows you to improve pose detection accuracy by manually specifying regions of interest. This is especially useful for:
- **Missed detections**: People that automatic YOLO detection didn't find
- **Partial occlusion**: People partially hidden by objects or other people
- **Crowded scenes**: Multiple overlapping people
- **Small/distant people**: People who are too small for reliable automatic detection
- **Unusual poses**: Poses that confuse the automatic detector

## Quick Start

### 1. Basic Usage in SwiftUI

```swift
import SwiftUI

struct MyView: View {
    @State private var showBboxEditor = false
    @State private var customBbox = CGRect.zero
    @State private var queryImage: NSImage?
    @State private var detectedPose: PoseDetectionResult?

    var body: some View {
        VStack {
            if let image = queryImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                Button("Draw Custom Bounding Box") {
                    showBboxEditor = true
                }
            }
        }
        .sheet(isPresented: $showBboxEditor) {
            if let image = queryImage {
                BoundingBoxEditor(
                    image: image,
                    bbox: $customBbox,
                    onDetect: { bbox in
                        detectPoseInCustomBbox(image: image, bbox: bbox)
                    },
                    onDismiss: {
                        showBboxEditor = false
                    }
                )
            }
        }
    }

    func detectPoseInCustomBbox(image: NSImage, bbox: CGRect) {
        guard let result = PythonBridgeSubprocess.shared.detectPoseInCustomBbox(
            in: image,
            bbox: bbox,
            cropPadding: 0.1
        ) else {
            print("Detection failed")
            return
        }

        detectedPose = result.result
        showBboxEditor = false

        print("Detected pose with \(result.result.keypoints.count) keypoints")
        print("Confidence: \(result.result.confidence)")
    }
}
```

### 2. Python-Only Usage

```python
import cv2
from src.swift_bridge import PostureKitBridge

# Initialize bridge
bridge = PostureKitBridge(
    pose_models='rtmw-l',
    use_two_stage=False,
    device='mps'
)

# Load image
image = cv2.imread('photo.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define custom bounding box [x, y, width, height]
# Example: Person at position (200, 300) with size (150, 400)
custom_bbox = [200, 300, 150, 400]

# Run detection in custom bbox
result = bridge.detect_pose_in_custom_bbox(
    image_rgb,
    bbox=custom_bbox,
    crop_padding=0.15  # 15% padding around bbox
)

if result:
    print(f"Detected pose with confidence: {result['confidence']:.2f}")
    print(f"Number of keypoints: {len(result['keypoints'])}")
    print(f"Detection method: {result['detection_method']}")

    # Keypoints are in ORIGINAL image coordinates
    # No need to remap - ready to use directly
    keypoints = result['keypoints']  # (133, 3) array
    bbox = result['bbox']  # [x, y, w, h] with padding applied
```

## Advanced Scenarios

### Scenario 1: Refining Automatic YOLO Detection

Sometimes YOLO detects a person but with an incorrect bounding box (too large, too small, or misaligned). You can refine it:

```swift
func refineExistingDetection(person: DetectedPerson, image: NSImage) {
    // Start with YOLO's bbox
    let yoloBbox = CGRect(
        x: person.bbox[0],
        y: person.bbox[1],
        width: person.bbox[2],
        height: person.bbox[3]
    )

    // User adjusts bbox in editor...
    // Then re-run detection with refined bbox
    let result = PythonBridgeSubprocess.shared.detectPoseInCustomBbox(
        in: image,
        bbox: adjustedBbox,  // User's refined bbox
        cropPadding: 0.1
    )

    // Compare confidence scores
    print("YOLO confidence: \(person.confidence)")
    print("Custom bbox confidence: \(result?.result.confidence ?? 0)")

    // Use whichever has higher confidence
    if let customResult = result?.result, customResult.confidence > person.confidence {
        print("Custom bbox improved detection by \((customResult.confidence - person.confidence) * 100)%")
        // Use customResult for search
    }
}
```

### Scenario 2: Multi-Region Detection (Crowded Scene)

Detect multiple people by drawing multiple bounding boxes:

```swift
struct MultiPersonDetector {
    func detectMultipleCustomRegions(image: NSImage, regions: [CGRect]) async -> [PoseDetectionResult] {
        var results: [PoseDetectionResult] = []

        for (index, bbox) in regions.enumerated() {
            print("Detecting person \(index + 1) of \(regions.count)...")

            if let result = PythonBridgeSubprocess.shared.detectPoseInCustomBbox(
                in: image,
                bbox: bbox,
                cropPadding: 0.1
            ) {
                results.append(result.result)
            }
        }

        print("Detected \(results.count) people from \(regions.count) custom regions")
        return results
    }
}

// Usage
let customRegions = [
    CGRect(x: 100, y: 200, width: 150, height: 300),  // Person 1
    CGRect(x: 500, y: 180, width: 160, height: 320),  // Person 2
    CGRect(x: 800, y: 220, width: 140, height: 290),  // Person 3
]

let people = await detector.detectMultipleCustomRegions(image: photo, regions: customRegions)
```

### Scenario 3: Interactive Bbox Adjustment with Real-Time Preview

```swift
struct InteractiveBboxEditor: View {
    @State private var bbox = CGRect.zero
    @State private var previewKeypoints: [[Double]]?
    @State private var previewConfidence: Double = 0

    var body: some View {
        VStack {
            BoundingBoxEditor(
                image: image,
                bbox: $bbox,
                onDetect: { bbox in
                    runDetectionPreview(bbox: bbox)
                },
                onDismiss: { }
            )

            if let confidence = previewConfidence, confidence > 0 {
                Text("Preview Confidence: \(Int(confidence * 100))%")
                    .foregroundColor(confidence > 0.7 ? .green : .orange)
            }
        }
    }

    func runDetectionPreview(bbox: CGRect) {
        // Run detection as preview
        if let result = PythonBridgeSubprocess.shared.detectPoseInCustomBbox(
            in: image,
            bbox: bbox,
            cropPadding: 0.1
        ) {
            previewKeypoints = result.result.keypoints
            previewConfidence = result.result.confidence

            // If confidence is low, suggest adjusting bbox
            if result.result.confidence < 0.5 {
                print("⚠️ Low confidence. Try adjusting the bounding box.")
            }
        }
    }
}
```

### Scenario 4: Batch Processing with Custom Bboxes

```python
import json
import cv2
from pathlib import Path
from src.swift_bridge import PostureKitBridge

# Load bbox annotations from JSON file
# Format: {"image1.jpg": [[x, y, w, h], ...], "image2.jpg": [[x, y, w, h], ...]}
with open('bbox_annotations.json') as f:
    annotations = json.load(f)

bridge = PostureKitBridge(pose_models='rtmw-l', device='mps')

results = {}
for image_path, bboxes in annotations.items():
    print(f"Processing {image_path} with {len(bboxes)} custom bboxes...")

    image = cv2.imread(str(Path('photos') / image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_results = []
    for bbox in bboxes:
        result = bridge.detect_pose_in_custom_bbox(
            image_rgb,
            bbox=bbox,
            crop_padding=0.1
        )

        if result:
            image_results.append({
                'bbox': bbox,
                'confidence': result['confidence'],
                'keypoints': result['keypoints']
            })

    results[image_path] = image_results
    print(f"  Detected {len(image_results)} people")

# Save results
with open('custom_bbox_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Coordinate System Notes

### Important: Pixel Coordinates vs Display Coordinates

The bounding box editor works in **display coordinates** (SwiftUI points), but the Python backend requires **pixel coordinates**. The conversion is handled automatically:

```swift
// In SwiftUI view (display coordinates)
let displayBbox = CGRect(x: 100, y: 150, width: 200, height: 300)

// Automatically converted to pixel coordinates based on image DPI/scale
// If image is 2x retina: pixelBbox = CGRect(x: 200, y: 300, width: 400, height: 600)
```

**The `detectPoseInCustomBbox()` method expects pixel coordinates**, which it receives from `saveImageToTemp()` that returns the actual pixel dimensions.

### Coordinate Mapping Flow

```
User draws bbox in UI (SwiftUI points)
    ↓
BoundingBoxEditor converts to pixel coords
    ↓
Swift bridge sends to Python (pixel coords)
    ↓
Python crops image at pixel coords
    ↓
Python detects pose in cropped region
    ↓
Python maps keypoints back to original image pixel coords
    ↓
Swift receives keypoints (pixel coords)
    ↓
UI displays keypoints (converts to SwiftUI points)
```

## Performance Characteristics

### Speed Comparison

| Detection Method | Time (M2 Pro) | Notes |
|-----------------|---------------|-------|
| Full image (auto) | ~0.6s | YOLO + RTMW-L on full image |
| Custom bbox | ~0.2-0.4s | Only processes cropped region (faster) |
| Custom bbox (ensemble) | ~0.8-1.2s | RTMW-L + RTMW-X ensemble on crop |

**Custom bbox detection is FASTER than full-image detection** because it only processes a small region.

### Memory Usage

- Full image: ~500MB (1080p image with models loaded)
- Custom bbox: ~200MB (processes only cropped region)
- Multiple custom bboxes: ~200MB per bbox (processed sequentially)

## Best Practices

### 1. Bbox Size Guidelines

```
Minimum size: 50×50 pixels (enforced by Python)
Recommended minimum: 100×100 pixels
Optimal range: 150-400 pixels width/height
Too large: >800 pixels (loses focus on specific person)
```

### 2. Padding Recommendations

```python
# Tight poses (yoga, stretching): More padding
result = bridge.detect_pose_in_custom_bbox(image, bbox, crop_padding=0.20)

# Normal standing poses: Default padding
result = bridge.detect_pose_in_custom_bbox(image, bbox, crop_padding=0.10)

# Crowded scenes (avoid overlap): Less padding
result = bridge.detect_pose_in_custom_bbox(image, bbox, crop_padding=0.05)
```

### 3. Error Handling

```swift
func safeCustomBboxDetection(image: NSImage, bbox: CGRect) -> PoseDetectionResult? {
    // Validate bbox size
    guard bbox.width >= 50 && bbox.height >= 50 else {
        print("⚠️ Bbox too small: \(bbox.width)×\(bbox.height). Minimum: 50×50")
        return nil
    }

    // Validate bbox is within image bounds
    let pixelSize = image.size  // Adjust for actual pixel size if needed
    guard bbox.maxX <= pixelSize.width && bbox.maxY <= pixelSize.height else {
        print("⚠️ Bbox extends outside image bounds")
        return nil
    }

    // Run detection
    guard let result = PythonBridgeSubprocess.shared.detectPoseInCustomBbox(
        in: image,
        bbox: bbox,
        cropPadding: 0.1
    ) else {
        print("⚠️ Detection failed in custom bbox region")
        return nil
    }

    // Validate confidence
    if result.result.confidence < 0.3 {
        print("⚠️ Low confidence: \(result.result.confidence). Consider adjusting bbox.")
    }

    return result.result
}
```

### 4. Logging Custom Detections

```python
# Python side - automatic logging
logger.info(
    f"Custom bbox detection: region=[{x}, {y}, {w}, {h}], "
    f"confidence={pose.overall_confidence:.2f}"
)

# Swift side - log for analytics
print("[CUSTOM_BBOX] Detection at [\(Int(bbox.origin.x)), \(Int(bbox.origin.y)), \(Int(bbox.width)), \(Int(bbox.height))], confidence=\(result.confidence)")
```

## Integration with Existing Workflow

### Add "Manual Adjust" to ContentView

```swift
// In ContentView.swift
ForEach(detectedPeople) { person in
    HStack {
        PersonCard(person: person)

        // Add manual adjustment button
        Button(action: {
            openBboxEditor(for: person)
        }) {
            Image(systemName: "crop")
            Text("Adjust Bbox")
        }
    }
}
```

### Fallback Pattern

```swift
func detectWithFallback(image: NSImage) -> [DetectedPerson] {
    // Try automatic detection first
    let (autoResults, pixelSize) = pythonBridge.detectAllPoses(
        in: image,
        minPersonConf: 0.15,
        detectionThreshold: 0.3,
        cropPadding: 0.1
    )

    if autoResults.isEmpty {
        // No automatic detections - prompt user for manual bbox
        print("No people detected automatically. Please draw a custom bounding box.")
        showBboxEditor = true
        return []
    }

    // Check confidence scores
    let lowConfidencePeople = autoResults.filter { $0.confidence < 0.5 }
    if !lowConfidencePeople.isEmpty {
        print("Found \(lowConfidencePeople.count) low-confidence detections. Consider manual adjustment.")
    }

    return autoResults.map { ... }
}
```

## Troubleshooting

### Issue: "No pose detected in custom bbox region"

**Causes:**
1. Bbox doesn't actually contain a person
2. Bbox is too small (< 50px)
3. Person is heavily occluded
4. Unusual pose the model can't recognize

**Solutions:**
- Increase `crop_padding` (try 0.15 or 0.20)
- Make bbox larger to include more context
- Use ensemble mode for better accuracy
- Check that person is actually visible in the region

### Issue: Low confidence score

**Causes:**
1. Poor lighting in cropped region
2. Heavy occlusion
3. Bbox includes multiple people (confuses detector)
4. Extreme pose (lying down, upside down, etc.)

**Solutions:**
- Adjust bbox to tighten around the person
- Increase padding if person is truncated
- Use RTMW-X or ensemble for better quality
- Try two-stage detection (`useTwoStage=true`)

### Issue: Keypoints in wrong location

**Cause:** Coordinate mapping bug (display coords vs pixel coords)

**Solution:** Ensure you're using pixel coordinates when calling the Python bridge:

```swift
// ✓ Correct - pixel coordinates from saveImageToTemp()
let (_, pixelSize) = saveImageToTemp(image)
let pixelBbox = CGRect(x: 100, y: 200, width: 150, height: 300)

// ✗ Incorrect - SwiftUI display coordinates
let displayBbox = someView.frame
```

## API Reference

### Python

```python
def detect_pose_in_custom_bbox(
    self,
    image_array: np.ndarray,  # (H, W, 3) RGB
    bbox: List[int],           # [x, y, width, height] in pixels
    crop_padding: float = 0.1  # Padding ratio (0.0-0.3)
) -> Optional[Dict[str, Any]]
```

**Returns:**
```python
{
    'keypoints': [[x, y, conf], ...],  # 133 keypoints in original image coords
    'visibility': [0, 1, 2, ...],      # COCO visibility (133)
    'bbox': [x, y, w, h],              # Actual bbox used (with padding)
    'confidence': 0.87,                # Overall detection confidence
    'person_id': 0,                    # Always 0 for custom bbox
    'detection_method': 'custom_bbox'  # Method identifier
}
```

### Swift

```swift
func detectPoseInCustomBbox(
    in image: NSImage,
    bbox: CGRect,              // Pixel coordinates
    cropPadding: Double = 0.1  // Padding ratio
) -> (result: PoseDetectionResult, pixelSize: CGSize)?
```

**Returns:** Tuple of `(PoseDetectionResult, pixelSize)` or `nil` if detection fails.

### SwiftUI

```swift
struct BoundingBoxEditor: View {
    let image: NSImage
    @Binding var bbox: CGRect
    let onDetect: (CGRect) -> Void
    let onDismiss: () -> Void
}
```

---

**For more details, see:**
- `MANUAL_BBOX_FEATURE.md` - Architecture and design
- `src/swift_bridge.py:698` - Python implementation
- `PostureKit/PythonBridgeSubprocess.swift:322` - Swift bridge
- `PostureKit/Views/BoundingBoxEditor.swift` - UI component
