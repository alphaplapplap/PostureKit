"""
Custom format exporter for PostureKit.
Exports complete PostureKit data including all metadata and analysis.
"""
import numpy as np
from typing import List, Dict
from datetime import datetime
from pathlib import Path


class CustomFormatter:
    """
    Formats pose data in PostureKit's custom format.

    This format preserves ALL information from PostureKit's analysis pipeline:
    - Complete 133-keypoint annotations with confidence scores
    - Viewpoint estimates (camera elevation, azimuth, subject facing)
    - 52-dimensional geometric feature vectors
    - Category labels, difficulty ratings, tags, and user notes
    - Image metadata and processing history

    This rich format is ideal for:
    - Research requiring detailed pose analysis
    - Custom training pipelines that use geometric features
    - Dataset analysis and quality assessment
    - Building pose similarity search systems
    - Archiving complete annotation sessions
    """

    def format(self, poses: List[Dict]) -> Dict:
        """
        Transform PostureKit poses to custom format.

        Args:
            poses: List of pose dictionaries from database query

        Returns:
            Dictionary in PostureKit custom format
        """
        custom_data = {
            'format_version': '1.0',
            'export_info': {
                'export_timestamp': datetime.now().isoformat(),
                'posturekit_version': '0.1.0',
                'exporter': 'PostureKit Custom Format Exporter',
                'description': 'Complete PostureKit pose dataset with full metadata'
            },

            # Dataset-level metadata
            'dataset_info': {
                'total_poses': len(poses),
                'keypoint_count': 133,
                'feature_dimension': 52,
                'categories': self._extract_categories(poses),
                'difficulty_levels': self._extract_difficulties(poses)
            },

            # Individual pose annotations
            'poses': []
        }

        # Process each pose with complete information
        for pose in poses:
            pose_data = {
                # Identification
                'pose_id': pose['pose_id'],
                'image_id': pose['image_id'],

                # Image information
                'image': {
                    'path': pose['image_path'],
                    'filename': Path(pose['image_path']).name,
                    'width': pose['image_width'],
                    'height': pose['image_height']
                },

                # Keypoint annotations (133 points with confidence)
                'keypoints': {
                    'coordinates': self._format_keypoint_coordinates(pose['keypoints']),
                    'confidence_scores': self._extract_confidence_scores(pose['keypoints']),
                    'structure': 'RTMW-L 133-keypoint',
                    'format': 'Each keypoint: [x, y, confidence]'
                },

                # Bounding box (pose keypoints)
                'bounding_box': {
                    'x': float(pose['bbox'][0]),
                    'y': float(pose['bbox'][1]),
                    'width': float(pose['bbox'][2]),
                    'height': float(pose['bbox'][3]),
                    'format': '[x, y, width, height]',
                    'description': 'Bounding box computed from keypoints'
                },

                # Person detection bounding box (from YOLO person detector)
                'person_detection': {
                    'bbox': pose.get('person_bbox'),
                    'confidence': pose.get('person_confidence'),
                    'description': 'Person detector bounding box and confidence (YOLO)'
                } if pose.get('person_bbox') else None,

                # Detection confidence
                'detection_confidence': float(pose['confidence']),

                # Viewpoint analysis - this is unique to PostureKit
                'viewpoint': {
                    'camera_elevation': float(pose['viewpoint']['elevation']),
                    'camera_azimuth': float(pose['viewpoint']['azimuth']),
                    'subject_facing': float(pose['viewpoint']['facing']),
                    'estimation_confidence': float(pose['viewpoint']['confidence']),
                    'units': 'degrees',
                    'description': {
                        'elevation': 'Camera angle above/below subject (-90 to 90)',
                        'azimuth': 'Camera horizontal angle (0=front, 90=side, 180=back)',
                        'facing': 'Direction subject is facing (0=toward camera)'
                    }
                },

                # Geometric features - PostureKit's pose similarity vectors
                'geometric_features': {
                    'feature_vector': [float(f) for f in pose['features']],
                    'dimension': len(pose['features']),
                    'description': '52-dim geometric feature vector for similarity matching',
                    'components': {
                        'joint_angles': 'Dimensions 0-11 (12 angles)',
                        'limb_ratios': 'Dimensions 12-21 (10 ratios)',
                        'body_angles': 'Dimensions 22-36 (15 angles)',
                        'symmetry_scores': 'Dimensions 37-44 (8 scores)',
                        'occlusion_pattern': 'Dimensions 45-51 (7 binary flags)'
                    }
                },

                # Training labels and annotations
                'labels': {
                    'category': pose['category'],
                    'difficulty': pose['difficulty'],
                    'tags': pose['tags'] if pose['tags'] else [],
                    'notes': pose['notes']
                },

                # Quality metrics
                'quality': {
                    'visible_keypoints': self._count_visible_keypoints(pose['keypoints']),
                    'total_keypoints': len(pose['keypoints']),
                    'visibility_ratio': self._calculate_visibility_ratio(pose['keypoints']),
                    'overall_confidence': float(pose['confidence'])
                }
            }

            custom_data['poses'].append(pose_data)

        return custom_data

    def _extract_categories(self, poses: List[Dict]) -> List[str]:
        """Extract unique categories from poses."""
        categories = set()
        for pose in poses:
            if pose['category']:
                categories.add(pose['category'])
        return sorted(list(categories))

    def _extract_difficulties(self, poses: List[Dict]) -> List[str]:
        """Extract unique difficulty levels from poses."""
        difficulties = set()
        for pose in poses:
            if pose['difficulty']:
                difficulties.add(pose['difficulty'])
        return sorted(list(difficulties))

    def _format_keypoint_coordinates(self, keypoints: List[List[float]]) -> List[Dict]:
        """
        Format keypoints with semantic names and structure.

        Unlike COCO and MMPose which use flat arrays, our custom format
        provides named keypoints for better readability and self-documentation.
        """
        keypoint_names = [
            # Body (0-16)
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',

            # Feet (17-22)
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel',

            # Face (23-90)
            *[f'face_{i}' for i in range(68)],

            # Hands (91-132)
            *[f'left_hand_{i}' for i in range(21)],
            *[f'right_hand_{i}' for i in range(21)]
        ]

        formatted_keypoints = []
        for idx, kp in enumerate(keypoints):
            name = keypoint_names[idx] if idx < len(keypoint_names) else f'unknown_{idx}'

            formatted_keypoints.append({
                'id': idx,
                'name': name,
                'x': float(kp[0]),
                'y': float(kp[1]),
                'confidence': float(kp[2]) if len(kp) >= 3 else 0.0
            })

        return formatted_keypoints

    def _extract_confidence_scores(self, keypoints: List[List[float]]) -> List[float]:
        """Extract just the confidence scores for quick analysis."""
        return [float(kp[2]) if len(kp) >= 3 else 0.0 for kp in keypoints]

    def _count_visible_keypoints(self, keypoints: List[List[float]]) -> int:
        """Count keypoints with confidence above threshold."""
        threshold = 0.3
        return sum(1 for kp in keypoints if len(kp) >= 3 and kp[2] > threshold)

    def _calculate_visibility_ratio(self, keypoints: List[List[float]]) -> float:
        """Calculate ratio of visible to total keypoints."""
        if not keypoints:
            return 0.0
        visible = self._count_visible_keypoints(keypoints)
        return float(visible) / len(keypoints)
