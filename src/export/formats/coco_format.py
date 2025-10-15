"""
COCO format exporter for PostureKit.
Transforms PostureKit annotations to COCO keypoint format.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class COCOFormatter:
    """
    Formats pose data for COCO keypoint dataset.

    COCO format is designed around the standard 17-keypoint COCO pose model,
    but we have 133 keypoints from RTMW. We handle this by:

    1. Including all 133 keypoints in the full format
    2. Optionally providing a 17-keypoint subset for compatibility
    3. Storing the skeleton structure for visualization

    The output matches the COCO format specification:
    http://cocodataset.org/#format-data
    """

    # RTMW keypoint names (133 total)
    # These are organized as: body (17) + feet (6) + face (68) + left_hand (21) + right_hand (21)
    KEYPOINT_NAMES = [
        # Body keypoints (0-16) - COCO compatible
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',

        # Feet keypoints (17-22)
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel',

        # Face keypoints (23-90) - 68 points
        *[f'face_{i}' for i in range(68)],

        # Left hand keypoints (91-111) - 21 points
        *[f'left_hand_{i}' for i in range(21)],

        # Right hand keypoints (112-132) - 21 points
        *[f'right_hand_{i}' for i in range(21)]
    ]

    # Skeleton connections for visualization
    # These define which keypoints connect to form the pose skeleton
    SKELETON = [
        # Body connections
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15], [12, 14], [14, 16],  # Legs

        # Feet connections
        [15, 17], [15, 18], [15, 19],  # Left foot
        [16, 20], [16, 21], [16, 22],  # Right foot
    ]

    def format(self, poses: List[Dict]) -> Dict:
        """
        Transform PostureKit poses to COCO format.

        Args:
            poses: List of pose dictionaries from database query

        Returns:
            Dictionary in COCO JSON format
        """
        # COCO format has three main sections: info, images, and annotations

        # Info section - metadata about the dataset
        coco_data = {
            'info': {
                'description': 'PostureKit Pose Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'PostureKit',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],  # Add license info if needed
            'categories': self._get_categories(poses),
            'images': [],
            'annotations': []
        }

        # Track unique images (multiple poses can come from one image)
        image_map = {}
        image_id_counter = 1
        annotation_id_counter = 1

        for pose in poses:
            # Add image if not already present
            if pose['image_id'] not in image_map:
                image_data = {
                    'id': image_id_counter,
                    'file_name': Path(pose['image_path']).name,
                    'width': pose['image_width'],
                    'height': pose['image_height']
                }
                coco_data['images'].append(image_data)
                image_map[pose['image_id']] = image_id_counter
                image_id_counter += 1

            # Create annotation for this pose
            annotation = {
                'id': annotation_id_counter,
                'image_id': image_map[pose['image_id']],
                'category_id': self._get_category_id(pose['category']),
                'keypoints': self._format_keypoints(pose['keypoints'], pose.get('visibility')),
                'num_keypoints': pose.get('visible_keypoint_count', self._count_visible_keypoints(pose['keypoints'])),
                'bbox': self._format_bbox(pose['bbox']),
                'area': self._calculate_area(pose['bbox']),
                'iscrowd': 0  # We don't have crowd annotations
            }

            # Add multi-person detection metadata if available
            if pose.get('person_bbox'):
                annotation['person_bbox'] = pose['person_bbox']
            if pose.get('person_confidence'):
                annotation['person_confidence'] = pose['person_confidence']

            coco_data['annotations'].append(annotation)
            annotation_id_counter += 1

        return coco_data

    def _get_categories(self, poses: List[Dict]) -> List[Dict]:
        """
        Extract unique categories and create COCO category definitions.

        COCO requires each category to have a unique ID, name, and skeleton.
        For pose estimation, we typically have a single "person" category,
        but PostureKit supports multiple pose categories (standing, sitting, etc.)
        """
        # Get unique categories from poses
        unique_categories = set()
        for pose in poses:
            if pose['category']:
                unique_categories.add(pose['category'])

        # If no categories, use default "person"
        if not unique_categories:
            unique_categories = {'person'}

        # Create COCO category definitions
        categories = []
        for idx, category_name in enumerate(sorted(unique_categories), 1):
            categories.append({
                'id': idx,
                'name': category_name,
                'supercategory': 'person',
                'keypoints': self.KEYPOINT_NAMES,
                'skeleton': self.SKELETON
            })

        return categories

    def _get_category_id(self, category_name: str) -> int:
        """Map category name to ID (simplified - in production, use lookup table)."""
        # In a full implementation, maintain a category name -> ID mapping
        # For now, use a simple hash-based approach
        if not category_name:
            return 1
        return hash(category_name) % 1000 + 1

    def _format_keypoints(self, keypoints: List[List[float]], visibility: List[int] = None) -> List[float]:
        """
        Format keypoints array for COCO.

        COCO expects keypoints as a flat list: [x1, y1, v1, x2, y2, v2, ...]
        where v is visibility flag (0=not labeled, 1=labeled but not visible, 2=labeled and visible)

        Args:
            keypoints: List of [x, y, confidence] for each keypoint (133, 3)
            visibility: Optional list of visibility flags (133,). If provided, uses these
                       instead of computing from confidence.

        Returns:
            Flat list in COCO format: [x1, y1, v1, x2, y2, v2, ...]
        """
        keypoints_array = np.array(keypoints)
        formatted = []

        for i, kp in enumerate(keypoints_array):
            x, y, conf = kp

            # Use stored visibility if available, otherwise compute from confidence
            if visibility is not None and i < len(visibility):
                vis_flag = int(visibility[i])
            else:
                # Fallback: Convert confidence to COCO visibility flag
                if conf > 0.5:
                    vis_flag = 2  # Labeled and visible
                elif conf > 0:
                    vis_flag = 1  # Labeled but occluded
                else:
                    vis_flag = 0  # Not labeled

            formatted.extend([float(x), float(y), vis_flag])

        return formatted

    def _count_visible_keypoints(self, keypoints: List[List[float]]) -> int:
        """Count keypoints with visibility > 0."""
        count = 0
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0:  # Has confidence > 0
                count += 1
        return count

    def _format_bbox(self, bbox: List[float]) -> List[float]:
        """
        Format bounding box for COCO.

        Our bbox is [x, y, w, h], which matches COCO format exactly.
        Just ensure it's a list of floats.
        """
        return [float(v) for v in bbox]

    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate bounding box area."""
        return float(bbox[2] * bbox[3])  # width * height
