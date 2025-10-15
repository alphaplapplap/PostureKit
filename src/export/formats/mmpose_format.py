"""
MMPose format exporter for PostureKit.
Transforms PostureKit annotations to MMPose training format.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class MMPoseFormatter:
    """
    Formats pose data for MMPose training.

    MMPose uses a specific JSON format that differs from COCO in several ways:

    1. Image paths are stored as relative paths from dataset root
    2. Each annotation has a 'joints_3d' field (we use zeros for z-coordinate)
    3. 'joints_3d_visible' flags indicate keypoint visibility
    4. Annotations include 'center' and 'scale' for cropping during training

    This format is designed to work seamlessly with MMPose's data loaders,
    making it easy to train custom models on your annotated data.
    """

    def format(self, poses: List[Dict]) -> Dict:
        """
        Transform PostureKit poses to MMPose format.

        Args:
            poses: List of pose dictionaries from database query

        Returns:
            Dictionary in MMPose JSON format
        """
        # MMPose format is simpler than COCO - just images and annotations
        mmpose_data = {
            'dataset_info': {
                'dataset_name': 'PostureKit Dataset',
                'paper_info': {
                    'author': 'PostureKit',
                    'title': 'PostureKit Annotated Pose Dataset',
                    'year': str(datetime.now().year)
                },
                'keypoint_info': self._get_keypoint_info()
            },
            'data_list': []
        }

        # Process each pose into MMPose format
        for pose in poses:
            # MMPose expects each annotation to be self-contained with image info
            annotation = {
                'img_path': str(Path(pose['image_path']).name),
                'img_shape': [pose['image_height'], pose['image_width']],

                # Keypoint annotations
                'joints_3d': self._format_joints_3d(pose['keypoints']),
                'joints_3d_visible': self._format_visibility(pose['keypoints']),

                # Bounding box and scale for data augmentation
                'bbox': self._format_bbox(pose['bbox']),
                'bbox_score': float(pose['confidence']),
                'center': self._calculate_center(pose['bbox']),
                'scale': self._calculate_scale(pose['bbox']),

                # Category and metadata
                'category_id': 1,  # MMPose typically uses single category
                'id': pose['pose_id']
            }

            # Add optional PostureKit-specific metadata
            # MMPose ignores unknown fields, so we can include extra info
            annotation['posturekit_metadata'] = {
                'category': pose['category'],
                'difficulty': pose['difficulty'],
                'viewpoint': pose['viewpoint']
            }

            mmpose_data['data_list'].append(annotation)

        return mmpose_data

    def _get_keypoint_info(self) -> Dict:
        """
        Define keypoint information for MMPose.

        This tells MMPose about the keypoint structure: how many keypoints,
        what they're called, how they connect, and which ones form symmetric pairs.
        This information is critical for MMPose's data augmentation (like flipping).
        """
        # Keypoint names matching RTMW-L structure
        keypoint_names = [
            # Body keypoints (0-16)
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',

            # Feet (17-22)
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel',

            # Face (23-90) - 68 facial landmarks
            *[f'face_{i}' for i in range(68)],

            # Hands (91-132) - 21 points each
            *[f'left_hand_{i}' for i in range(21)],
            *[f'right_hand_{i}' for i in range(21)]
        ]

        # Skeleton connections for visualization and loss computation
        skeleton_links = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
            [15, 17], [15, 18], [15, 19],  # Left foot
            [16, 20], [16, 21], [16, 22],  # Right foot
        ]

        # Symmetric keypoint pairs for horizontal flip augmentation
        # When an image is flipped, left and right keypoints swap
        flip_pairs = [
            [1, 2], [3, 4],  # Eyes and ears
            [5, 6], [7, 8], [9, 10],  # Arms
            [11, 12], [13, 14], [15, 16],  # Hips, knees, ankles
            [17, 20], [18, 21], [19, 22],  # Feet
            # Hand pairs would go here (91-111 with 112-132)
            *[[91 + i, 112 + i] for i in range(21)]
        ]

        return {
            'dataset_name': 'posturekit_133',
            'num_keypoints': 133,
            'keypoint_names': keypoint_names,
            'skeleton_links': skeleton_links,
            'flip_pairs': flip_pairs,
            'upper_body_ids': list(range(23)),  # Body + feet
            'lower_body_ids': list(range(11, 23)),  # Hips down to feet
        }

    def _format_joints_3d(self, keypoints: List[List[float]]) -> List[List[float]]:
        """
        Format keypoints as 3D joints for MMPose.

        MMPose expects joints in 3D even for 2D pose estimation. For 2D poses,
        we set the z-coordinate to zero. The format is (N, 3) where N is the
        number of keypoints and each row is [x, y, z].
        """
        keypoints_array = np.array(keypoints)

        # Extract x, y coordinates (first two columns)
        joints_2d = keypoints_array[:, :2]

        # Add z-coordinate of zero for 2D poses
        z_coords = np.zeros((len(joints_2d), 1))
        joints_3d = np.concatenate([joints_2d, z_coords], axis=1)

        return joints_3d.tolist()

    def _format_visibility(self, keypoints: List[List[float]]) -> List[List[int]]:
        """
        Format keypoint visibility flags for MMPose.

        MMPose uses a 3-element visibility flag for each keypoint:
        [visible, labeled, in_image]

        We derive this from our confidence scores:
        - visible: 1 if confidence > 0.5, else 0
        - labeled: 1 if confidence > 0, else 0
        - in_image: 1 (we assume all keypoints are within image bounds)
        """
        visibility = []

        for kp in keypoints:
            conf = kp[2] if len(kp) >= 3 else 0

            visible = 1 if conf > 0.5 else 0
            labeled = 1 if conf > 0 else 0
            in_image = 1  # Assume all keypoints are in image

            visibility.append([visible, labeled, in_image])

        return visibility

    def _format_bbox(self, bbox: List[float]) -> List[float]:
        """
        Format bounding box for MMPose.

        MMPose expects [x, y, w, h] format, which matches our storage format.
        We just ensure the values are floats.
        """
        return [float(v) for v in bbox]

    def _calculate_center(self, bbox: List[float]) -> List[float]:
        """
        Calculate bounding box center point.

        MMPose uses the center point for data augmentation transformations.
        The center is simply the middle of the bounding box.
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        return [float(center_x), float(center_y)]

    def _calculate_scale(self, bbox: List[float]) -> List[float]:
        """
        Calculate scale factor for MMPose data loading.

        MMPose uses a scale factor to normalize poses during training. The scale
        is typically the bounding box size divided by a reference size (200 pixels).
        This allows the model to handle poses of different sizes consistently.
        """
        x, y, w, h = bbox

        # Scale based on max of width and height
        # MMPose uses 200 as reference, but we can adjust based on dataset
        reference_size = 200.0
        scale = max(w, h) / reference_size

        # Return as [scale_x, scale_y] for potential non-uniform scaling
        return [float(scale), float(scale)]
