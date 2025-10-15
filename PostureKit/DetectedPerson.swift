//
//  DetectedPerson.swift
//  PostureKit
//
//  Multi-person detection model for query images.
//  Stores bounding box, keypoints, and confidence for each detected person.
//

import Foundation
import CoreGraphics

/// Represents a single detected person in an image with their pose data
struct DetectedPerson: Identifiable, Equatable {
    /// Unique identifier for this detected person
    let id: UUID

    /// Person index in the detection results (0-based)
    let personId: Int

    /// Bounding box [x, y, width, height] in image coordinates
    let bbox: [Double]

    /// Keypoints array (133 x 3): [x, y, confidence] for each keypoint
    let keypoints: [[Double]]

    /// Overall detection confidence (0.0 - 1.0)
    let confidence: Double

    /// Original pose detection result (for feature extraction)
    let poseResult: PoseDetectionResult

    init(personId: Int, bbox: [Double], keypoints: [[Double]], confidence: Double, poseResult: PoseDetectionResult) {
        self.id = UUID()
        self.personId = personId
        self.bbox = bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.poseResult = poseResult
    }

    /// Convenience computed property to get bbox components
    var bboxComponents: (x: Double, y: Double, width: Double, height: Double) {
        guard bbox.count == 4 else {
            return (0, 0, 0, 0)
        }
        return (bbox[0], bbox[1], bbox[2], bbox[3])
    }

    /// Check if a point (in image coordinates) is inside this person's bbox
    func contains(point: CGPoint, imageSize: CGSize) -> Bool {
        let (x, y, width, height) = bboxComponents
        let bboxRect = CGRect(x: x, y: y, width: width, height: height)
        return bboxRect.contains(point)
    }

    static func == (lhs: DetectedPerson, rhs: DetectedPerson) -> Bool {
        return lhs.id == rhs.id
    }
}
