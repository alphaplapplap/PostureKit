"""
Data Exporter for PostureKit.
Exports annotated pose data to various training formats.
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from sqlalchemy import select, and_

from src.storage.storage_manager import StorageManager
from src.storage.models import (
    Image, PoseDetection,
    GeometricFeatures, TrainingLabel
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    COCO = "coco"  # COCO keypoint format for general use
    MMPOSE = "mmpose"  # MMPose training format
    CUSTOM = "custom"  # PostureKit custom JSON with full metadata


@dataclass
class ExportConfig:
    """
    Configuration for dataset export.

    This encapsulates all the parameters needed for an export operation,
    making it easy to save, load, and modify export settings.
    """
    format: ExportFormat
    output_dir: Path
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test
    include_unlabeled: bool = False
    copy_images: bool = True  # Copy images to output dir or just link paths
    categories: Optional[List[str]] = None  # Export only these categories (None = all)
    min_keypoints: int = 5  # Minimum visible keypoints required

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure split ratios sum to 1.0
        if abs(sum(self.split_ratio) - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {sum(self.split_ratio)}"
            )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DataExportError(Exception):
    """Base exception for export errors."""
    pass


class DataExporter:
    """
    Exports annotated pose data to training formats.

    This class handles the complete export pipeline:
    1. Query database for annotated poses
    2. Filter based on export criteria
    3. Split into train/val/test sets
    4. Transform to target format
    5. Write files and copy images
    6. Generate metadata and statistics

    The exporter is designed to be robust, logging all operations and
    validating outputs to ensure exported datasets are correct.
    """

    def __init__(self, storage_manager: StorageManager):
        """
        Initialize Data Exporter.

        Args:
            storage_manager: StorageManager instance for database access
        """
        self.storage_manager = storage_manager
        logger.info("DataExporter initialized")

    def export_dataset(self, config: ExportConfig) -> Dict[str, any]:
        """
        Export dataset according to configuration.

        This is the main entry point for exports. It orchestrates the entire
        process and returns detailed statistics about what was exported.

        Args:
            config: ExportConfig with export parameters

        Returns:
            Dictionary with export statistics and file paths

        Raises:
            DataExportError: If export fails
        """
        logger.info(
            f"Starting dataset export to {config.format.value} format",
            extra={'extra_data': {
                'output_dir': str(config.output_dir),
                'split_ratio': config.split_ratio
            }}
        )

        try:
            # Step 1: Query and filter poses
            poses = self._query_poses(config)
            logger.info(f"Retrieved {len(poses)} poses from database")

            if len(poses) == 0:
                raise DataExportError("No poses match export criteria")

            # Step 2: Split into train/val/test sets
            # We use stratified splitting to ensure each split has a balanced
            # distribution of categories
            splits = self._split_dataset(poses, config.split_ratio)
            logger.info(
                f"Split dataset: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}"
            )

            # Step 3: Transform to target format
            # Each format transformer handles the specific schema requirements
            if config.format == ExportFormat.COCO:
                from src.export.formats.coco_format import COCOFormatter
                formatter = COCOFormatter()
            elif config.format == ExportFormat.MMPOSE:
                from src.export.formats.mmpose_format import MMPoseFormatter
                formatter = MMPoseFormatter()
            else:  # CUSTOM
                from src.export.formats.custom_format import CustomFormatter
                formatter = CustomFormatter()

            # Step 4: Export each split
            exported_files = {}
            for split_name, split_poses in splits.items():
                if len(split_poses) == 0:
                    logger.warning(f"Skipping empty {split_name} split")
                    continue

                logger.info(f"Exporting {split_name} split ({len(split_poses)} poses)")

                # Transform data to target format
                formatted_data = formatter.format(split_poses)

                # Write annotation file
                annotation_file = config.output_dir / f"{split_name}_annotations.json"
                with open(annotation_file, 'w') as f:
                    json.dump(formatted_data, f, indent=2)

                exported_files[split_name] = str(annotation_file)
                logger.info(f"Wrote {split_name} annotations to {annotation_file}")

                # Copy images if requested
                if config.copy_images:
                    self._copy_split_images(
                        split_poses,
                        config.output_dir / split_name
                    )

            # Step 5: Generate export metadata
            # This helps users understand what was exported and how to use it
            metadata = self._generate_metadata(config, splits, exported_files)
            metadata_file = config.output_dir / "export_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Export completed successfully",
                extra={'extra_data': metadata}
            )

            return metadata

        except Exception as e:
            if isinstance(e, DataExportError):
                raise
            logger.error(f"Export failed: {e}", exc_info=True)
            raise DataExportError(f"Export failed: {e}") from e

    def _query_poses(self, config: ExportConfig) -> List[Dict]:
        """
        Query poses from database based on export criteria.

        This method builds a complex query that joins all the relevant tables
        and filters based on the export configuration. The result is a list
        of dictionaries containing all the information needed for export.
        """
        with self.storage_manager.session_scope() as session:
            # Build base query joining all relevant tables
            # Note: ViewpointAnalysis table not implemented, using default values
            query = (
                select(
                    PoseDetection,
                    Image,
                    GeometricFeatures,
                    TrainingLabel
                )
                .join(Image, PoseDetection.image_id == Image.id)
                .join(GeometricFeatures, PoseDetection.id == GeometricFeatures.pose_id)
                .outerjoin(TrainingLabel, PoseDetection.id == TrainingLabel.pose_id)
            )

            # Apply filters based on configuration
            filters = []

            # Exclude unlabeled if requested
            if not config.include_unlabeled:
                filters.append(TrainingLabel.id.isnot(None))

            # Filter by categories if specified
            if config.categories:
                filters.append(TrainingLabel.category.in_(config.categories))

            # Apply minimum keypoint requirement
            # This ensures we only export poses with sufficient annotation quality
            if config.min_keypoints > 0:
                # Filter by actual visible keypoint count
                filters.append(PoseDetection.visible_keypoint_count >= config.min_keypoints)

            if filters:
                query = query.where(and_(*filters))

            # Execute query
            results = session.execute(query).all()

            # Transform to dictionaries for easier manipulation
            poses = []
            for pose_det, image, features, label in results:
                # Reshape keypoints from flat array (399,) to (133, 3)
                import numpy as np
                keypoints_flat = np.array(pose_det.keypoints, dtype=np.float32)
                keypoints_reshaped = keypoints_flat.reshape(133, 3)

                # Get visibility array
                visibility = np.array(pose_det.visibility, dtype=np.int8) if pose_det.visibility else np.zeros(133, dtype=np.int8)

                pose_data = {
                    'pose_id': str(pose_det.id),
                    'image_id': str(image.id),
                    'image_path': image.file_path,
                    'image_width': image.width,
                    'image_height': image.height,
                    'keypoints': keypoints_reshaped.tolist(),  # Convert to list for JSON serialization
                    'visibility': visibility.tolist(),  # Include visibility flags
                    'bbox': pose_det.bbox,
                    'confidence': pose_det.overall_confidence,
                    'visible_keypoint_count': pose_det.visible_keypoint_count,
                    'occluded_keypoint_count': pose_det.occluded_keypoint_count,
                    'missing_keypoint_count': pose_det.missing_keypoint_count,
                    'occlusion_percentage': pose_det.occlusion_percentage,
                    # Multi-person detection fields
                    'person_bbox': pose_det.person_bbox if hasattr(pose_det, 'person_bbox') else None,
                    'person_confidence': pose_det.person_confidence if hasattr(pose_det, 'person_confidence') else None,
                    # Viewpoint not stored in DB - use defaults for export compatibility
                    'viewpoint': {
                        'elevation': 0.0,
                        'azimuth': 0.0,
                        'facing': 0.0,
                        'confidence': 0.0
                    },
                    'features': features.feature_vector,
                    'category': label.category if label else None,
                    'difficulty': label.difficulty if label else None,
                    'tags': label.tags if label else [],
                    'notes': label.notes if label else None
                }
                poses.append(pose_data)

            return poses

    def _split_dataset(
        self,
        poses: List[Dict],
        split_ratio: Tuple[float, float, float]
    ) -> Dict[str, List[Dict]]:
        """
        Split dataset into train/val/test sets with stratification.

        Stratification ensures each split has a balanced distribution of
        categories. This is important because if all examples of a category
        end up in the training set, the model won't be properly evaluated.
        """
        import numpy as np
        from collections import defaultdict

        # Group poses by category for stratified splitting
        category_poses = defaultdict(list)
        for pose in poses:
            category = pose.get('category', 'unknown')
            category_poses[category].append(pose)

        # Initialize splits
        splits = {
            'train': [],
            'val': [],
            'test': []
        }

        train_ratio, val_ratio, test_ratio = split_ratio

        # Split each category separately to maintain distribution
        for category, cat_poses in category_poses.items():
            # Shuffle poses within category for randomization
            np.random.shuffle(cat_poses)

            n_total = len(cat_poses)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # Allocate to splits
            splits['train'].extend(cat_poses[:n_train])
            splits['val'].extend(cat_poses[n_train:n_train + n_val])
            splits['test'].extend(cat_poses[n_train + n_val:])

        return splits

    def _copy_split_images(self, poses: List[Dict], output_dir: Path):
        """
        Copy images for a split to output directory.

        This creates a clean dataset structure where images are organized
        alongside their annotations, making it easy to move the dataset
        or share it with others.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Track unique images (multiple poses might come from same image)
        copied_images = set()

        for pose in poses:
            image_path = Path(pose['image_path'])

            if image_path in copied_images:
                continue

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            # Copy with original filename to output directory
            dest_path = output_dir / image_path.name

            if not dest_path.exists():
                shutil.copy2(image_path, dest_path)
                logger.debug(f"Copied {image_path.name} to {output_dir}")

            copied_images.add(image_path)

        logger.info(f"Copied {len(copied_images)} images to {output_dir}")

    def _generate_metadata(
        self,
        config: ExportConfig,
        splits: Dict[str, List[Dict]],
        exported_files: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Generate export metadata for documentation and reproducibility.

        This metadata file serves as a README for the exported dataset,
        explaining what's included, how it was generated, and how to use it.
        """
        # Calculate category distribution across splits
        def count_categories(poses):
            from collections import Counter
            return dict(Counter(p.get('category', 'unknown') for p in poses))

        return {
            'export_timestamp': datetime.now().isoformat(),
            'format': config.format.value,
            'configuration': {
                'split_ratio': config.split_ratio,
                'include_unlabeled': config.include_unlabeled,
                'copy_images': config.copy_images,
                'categories_filter': config.categories,
                'min_keypoints': config.min_keypoints
            },
            'dataset_statistics': {
                'total_poses': sum(len(poses) for poses in splits.values()),
                'splits': {
                    'train': {
                        'count': len(splits['train']),
                        'categories': count_categories(splits['train'])
                    },
                    'val': {
                        'count': len(splits['val']),
                        'categories': count_categories(splits['val'])
                    },
                    'test': {
                        'count': len(splits['test']),
                        'categories': count_categories(splits['test'])
                    }
                }
            },
            'exported_files': exported_files,
            'usage_instructions': self._get_format_instructions(config.format)
        }

    def _get_format_instructions(self, format: ExportFormat) -> str:
        """
        Provide format-specific usage instructions.

        These instructions help users integrate the exported data into
        their training pipelines without consulting external documentation.
        """
        instructions = {
            ExportFormat.COCO: """
COCO Format Usage:

This dataset uses the COCO keypoint annotation format. To load in Python:
```python
import json
from pycocotools.coco import COCO

# Load annotations
coco = COCO('train_annotations.json')

# Get all person annotations
person_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=person_ids)

# Iterate through images
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    # Process annotations...
```
""",
            ExportFormat.MMPOSE: """
MMPose Format Usage:

This dataset uses MMPose's training format. Use in MMPose config:
```python
# In your MMPose config file
dataset_type = 'CocoDataset'
data_root = 'path/to/export/'
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_annotations.json',
        data_prefix=dict(img=''),
    )
)
```
""",
            ExportFormat.CUSTOM: """
PostureKit Custom Format Usage:

This format includes all PostureKit metadata. To load:
```python
import json

with open('train_annotations.json') as f:
    data = json.load(f)

# Access metadata
metadata = data['metadata']
keypoint_names = data['keypoint_names']

# Process poses
for pose in data['poses']:
    keypoints = pose['detection']['keypoints']
    viewpoint = pose['viewpoint']
    features = pose['features']
    # Your processing...
```
"""
        }
        return instructions[format]
