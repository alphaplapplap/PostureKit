"""
Export Dataset Dialog
Comprehensive dialog for exporting training datasets in various formats
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QLineEdit, QSlider, QSpinBox,
    QCheckBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QWidget, QScrollArea, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QPalette, QColor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import numpy as np

from src.storage.storage_manager import StorageManager
from src.storage.models import PoseDetection, Image, GeometricFeatures, TrainingLabel
from sqlalchemy import select, and_


class ExportFormat(Enum):
    """Available export formats"""
    COCO = "coco"
    MMPOSE = "mmpose"
    CUSTOM = "custom"


class SplitStrategy(Enum):
    """Dataset split strategies"""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SEQUENTIAL = "sequential"


@dataclass
class ExportConfig:
    """Export configuration"""
    format: ExportFormat
    output_dir: str

    # Split ratios (must sum to 100)
    train_ratio: int = 70
    val_ratio: int = 15
    test_ratio: int = 15
    split_strategy: SplitStrategy = SplitStrategy.RANDOM

    # Filters
    category_filter: Optional[str] = None
    min_confidence: float = 0.3
    max_occlusion: int = 20
    corrected_only: bool = False
    labeled_only: bool = False
    exclude_multi_person: bool = False

    # Options
    copy_images: bool = True
    include_features: bool = False
    generate_metadata: bool = True
    generate_readme: bool = True


class ExportWorker(QThread):
    """Background worker for export process"""

    progress = Signal(int, str)  # percentage, status_message
    finished = Signal(bool, str)  # success, message

    def __init__(self, config: ExportConfig, pose_data: List[Dict]):
        super().__init__()
        self.config = config
        self.pose_data = pose_data
        self.is_cancelled = False

    def run(self):
        """Execute export"""
        try:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Filter poses
            self.progress.emit(5, "Filtering poses...")
            filtered_poses = self._filter_poses()

            if not filtered_poses:
                self.finished.emit(False, "No poses match the filter criteria")
                return

            # Split dataset
            self.progress.emit(15, f"Splitting dataset ({self.config.split_strategy.value})...")
            splits = self._split_dataset(filtered_poses)

            # Export based on format
            if self.config.format == ExportFormat.COCO:
                self.progress.emit(30, "Exporting COCO format...")
                self._export_coco(splits, output_path)
            elif self.config.format == ExportFormat.MMPOSE:
                self.progress.emit(30, "Exporting MMPose format...")
                self._export_mmpose(splits, output_path)
            elif self.config.format == ExportFormat.CUSTOM:
                self.progress.emit(30, "Exporting PostureKit format...")
                self._export_custom(splits, output_path)

            # Copy images if requested
            if self.config.copy_images:
                self.progress.emit(70, "Copying images...")
                self._copy_images(splits, output_path)

            # Generate metadata
            if self.config.generate_metadata:
                self.progress.emit(90, "Generating metadata...")
                self._generate_metadata(splits, output_path)

            # Generate README
            if self.config.generate_readme:
                self.progress.emit(95, "Generating README...")
                self._generate_readme(splits, output_path)

            self.progress.emit(100, "Export complete!")
            self.finished.emit(True, f"Successfully exported {len(filtered_poses)} poses")

        except Exception as e:
            self.finished.emit(False, f"Export failed: {str(e)}")

    def cancel(self):
        """Cancel export"""
        self.is_cancelled = True

    def _filter_poses(self) -> List[Dict]:
        """Apply filters to poses"""
        filtered = []

        for pose in self.pose_data:
            # Category filter
            if self.config.category_filter and self.config.category_filter != "All Categories":
                if pose.get('category') != self.config.category_filter:
                    continue

            # Confidence filter
            if pose.get('confidence', 1.0) < self.config.min_confidence:
                continue

            # Occlusion filter
            occlusion_pct = pose.get('occlusion_percentage', 0)
            if occlusion_pct > self.config.max_occlusion:
                continue

            # Corrected only
            if self.config.corrected_only and not pose.get('is_corrected', False):
                continue

            # Labeled only
            if self.config.labeled_only and not pose.get('category'):
                continue

            # Multi-person filter
            if self.config.exclude_multi_person and pose.get('num_people', 1) > 1:
                continue

            filtered.append(pose)

        return filtered

    def _split_dataset(self, poses: List[Dict]) -> Dict[str, List[Dict]]:
        """Split dataset into train/val/test"""
        import random
        import numpy as np

        total = len(poses)
        train_count = int(total * self.config.train_ratio / 100)
        val_count = int(total * self.config.val_ratio / 100)
        test_count = total - train_count - val_count

        if self.config.split_strategy == SplitStrategy.RANDOM:
            # Random shuffle
            shuffled = poses.copy()
            random.shuffle(shuffled)

        elif self.config.split_strategy == SplitStrategy.STRATIFIED:
            # Stratified by category
            categories = {}
            for pose in poses:
                cat = pose.get('category', 'uncategorized')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(pose)

            # Split each category proportionally
            shuffled = []
            for cat_poses in categories.values():
                random.shuffle(cat_poses)
                shuffled.extend(cat_poses)

        else:  # SEQUENTIAL
            # Keep original order
            shuffled = poses

        return {
            'train': shuffled[:train_count],
            'val': shuffled[train_count:train_count + val_count],
            'test': shuffled[train_count + val_count:]
        }

    def _export_coco(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Export in COCO format"""
        for split_name, poses in splits.items():
            if not poses:
                continue

            coco_data = {
                'info': {
                    'description': 'PostureKit COCO-WholeBody Dataset',
                    'version': '1.0',
                    'year': 2025,
                    'contributor': 'PostureKit',
                },
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': [{
                    'id': 1,
                    'name': 'person',
                    'keypoints': self._get_coco_keypoint_names(),
                    'skeleton': self._get_coco_skeleton()
                }]
            }

            for idx, pose in enumerate(poses):
                # Image entry
                coco_data['images'].append({
                    'id': idx,
                    'file_name': pose['image_path'],
                    'width': pose['image_width'],
                    'height': pose['image_height']
                })

                # Annotation entry
                keypoints = []
                for kpt in pose['keypoints']:
                    keypoints.extend([kpt['x'], kpt['y'], kpt['visibility']])

                coco_data['annotations'].append({
                    'id': idx,
                    'image_id': idx,
                    'category_id': 1,
                    'bbox': pose['bbox'],
                    'area': pose['bbox'][2] * pose['bbox'][3],
                    'keypoints': keypoints,
                    'num_keypoints': pose['visible_keypoints'],
                    'iscrowd': 0
                })

            # Write JSON
            output_file = output_path / f'{split_name}_coco.json'
            with open(output_file, 'w') as f:
                json.dump(coco_data, f, indent=2)

    def _export_mmpose(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Export in MMPose format"""
        for split_name, poses in splits.items():
            if not poses:
                continue

            mmpose_data = []
            for pose in poses:
                mmpose_data.append({
                    'img_path': pose['image_path'],
                    'bbox': pose['bbox'],
                    'keypoints': [[kpt['x'], kpt['y'], kpt['visibility']]
                                 for kpt in pose['keypoints']],
                    'keypoints_visible': [kpt['visibility'] for kpt in pose['keypoints']],
                    'category': pose.get('category', 'person')
                })

            output_file = output_path / f'{split_name}_mmpose.json'
            with open(output_file, 'w') as f:
                json.dump(mmpose_data, f, indent=2)

    def _export_custom(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Export in PostureKit custom format"""
        for split_name, poses in splits.items():
            if not poses:
                continue

            custom_data = {
                'format': 'PostureKit Custom v1.0',
                'keypoint_schema': 'COCO-WholeBody-133',
                'poses': poses
            }

            output_file = output_path / f'{split_name}_posturekit.json'
            with open(output_file, 'w') as f:
                json.dump(custom_data, f, indent=2)

    def _copy_images(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Copy images to output directory"""
        import shutil

        for split_name, poses in splits.items():
            split_dir = output_path / 'images' / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for pose in poses:
                src = Path(pose['image_path'])
                dst = split_dir / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

    def _generate_metadata(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Generate dataset metadata"""
        metadata = {
            'export_config': {
                'format': self.config.format.value,
                'split_strategy': self.config.split_strategy.value,
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'filters': {
                    'category': self.config.category_filter,
                    'min_confidence': self.config.min_confidence,
                    'max_occlusion': self.config.max_occlusion,
                    'corrected_only': self.config.corrected_only,
                    'labeled_only': self.config.labeled_only
                }
            },
            'statistics': {
                split: {
                    'num_poses': len(poses),
                    'avg_confidence': sum(p['confidence'] for p in poses) / len(poses) if poses else 0,
                    'categories': self._get_category_counts(poses)
                }
                for split, poses in splits.items()
            }
        }

        output_file = output_path / 'metadata.json'
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_readme(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Generate README file"""
        total_poses = sum(len(poses) for poses in splits.values())

        readme = f"""# PostureKit Dataset Export

## Overview
- **Format**: {self.config.format.value.upper()}
- **Total Poses**: {total_poses}
- **Export Date**: {self._get_current_date()}

## Dataset Split
- **Train**: {len(splits.get('train', []))} poses ({self.config.train_ratio}%)
- **Validation**: {len(splits.get('val', []))} poses ({self.config.val_ratio}%)
- **Test**: {len(splits.get('test', []))} poses ({self.config.test_ratio}%)
- **Strategy**: {self.config.split_strategy.value.capitalize()}

## Filters Applied
- **Category**: {self.config.category_filter or 'All'}
- **Min Confidence**: {self.config.min_confidence}
- **Max Occlusion**: {self.config.max_occlusion}%
- **Corrected Only**: {self.config.corrected_only}
- **Labeled Only**: {self.config.labeled_only}

## File Structure
```
{output_path.name}/
├── train_{self.config.format.value}.json
├── val_{self.config.format.value}.json
├── test_{self.config.format.value}.json
├── metadata.json
├── README.md
└── images/
    ├── train/
    ├── val/
    └── test/
```

## Keypoint Schema
COCO-WholeBody format with 133 keypoints:
- Body: 17 keypoints
- Feet: 6 keypoints
- Face: 68 keypoints
- Left Hand: 21 keypoints
- Right Hand: 21 keypoints

## Usage
Load annotations using your preferred framework:

### PyTorch (COCO format)
```python
from pycocotools.coco import COCO

coco = COCO('train_coco.json')
image_ids = coco.getImgIds()
```

### MMPose
```python
import json

with open('train_mmpose.json', 'r') as f:
    dataset = json.load(f)
```

## Citation
If you use this dataset, please cite PostureKit.
"""

        output_file = output_path / 'README.md'
        with open(output_file, 'w') as f:
            f.write(readme)

    @staticmethod
    def _get_coco_keypoint_names() -> List[str]:
        """Get COCO-WholeBody keypoint names"""
        return [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            # Feet
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel',
            # Face (68 points)
            *[f'face_{i}' for i in range(68)],
            # Left hand (21 points)
            *[f'left_hand_{i}' for i in range(21)],
            # Right hand (21 points)
            *[f'right_hand_{i}' for i in range(21)]
        ]

    @staticmethod
    def _get_coco_skeleton() -> List[List[int]]:
        """Get COCO skeleton connections"""
        return [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 11], [6, 12], [11, 12],  # Torso
            [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
        ]

    @staticmethod
    def _get_category_counts(poses: List[Dict]) -> Dict[str, int]:
        """Get category distribution"""
        counts = {}
        for pose in poses:
            cat = pose.get('category', 'uncategorized')
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    @staticmethod
    def _get_current_date() -> str:
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')


class ExportDialog(QDialog):
    """
    Export dataset dialog matching spec requirements
    """

    def __init__(self, storage_manager: StorageManager, parent=None):
        super().__init__(parent)

        self.storage_manager = storage_manager
        self.pose_data = self._load_poses()
        self.categories = self._load_categories()
        self.export_worker = None

        self.setWindowTitle("Export Training Dataset")
        self.setMinimumSize(700, 750)

        self._setup_ui()
        self._update_preview()

    def _load_poses(self) -> List[Dict]:
        """Load all poses from database"""
        with self.storage_manager.session_scope() as session:
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

            results = session.execute(query).all()

            poses = []
            for pose_det, image, features, label in results:
                keypoints_flat = np.array(pose_det.keypoints, dtype=np.float32)
                keypoints_reshaped = keypoints_flat.reshape(133, 3)

                poses.append({
                    'pose_id': str(pose_det.id),
                    'image_id': str(image.id),
                    'image_path': image.file_path,
                    'keypoints': keypoints_reshaped.tolist(),
                    'confidence': pose_det.overall_confidence,
                    'occlusion_percentage': pose_det.occlusion_percentage or 0,
                    'is_corrected': pose_det.is_corrected or False,
                    'num_people': 1,  # Single person for now
                    'category': label.category if label else None,
                })

            return poses

    def _load_categories(self) -> List[str]:
        """Load unique categories from database"""
        with self.storage_manager.session_scope() as session:
            query = select(TrainingLabel.category).distinct().where(TrainingLabel.category.isnot(None))
            results = session.execute(query).scalars().all()
            return sorted([cat for cat in results if cat])

    def _setup_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)

        # Format selection
        format_group = self._create_format_group()
        layout.addWidget(format_group)

        # Output directory
        output_group = self._create_output_group()
        layout.addWidget(output_group)

        # Dataset split
        split_group = self._create_split_group()
        layout.addWidget(split_group)

        # Filters
        filters_group = self._create_filters_group()
        layout.addWidget(filters_group)

        # Options
        options_group = self._create_options_group()
        layout.addWidget(options_group)

        # Preview
        preview_label = QLabel()
        preview_label.setText("<b>Preview:</b> <span id='count'>0 poses match criteria</span>")
        preview_label.setTextFormat(Qt.RichText)
        layout.addWidget(preview_label)
        self.preview_label = preview_label

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._show_preview)
        button_layout.addWidget(preview_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._start_export)
        button_layout.addWidget(self.export_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _create_format_group(self) -> QGroupBox:
        """Create format selection group"""
        group = QGroupBox("Format")
        layout = QVBoxLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItem("COCO Format (Universal)", ExportFormat.COCO)
        self.format_combo.addItem("MMPose Format", ExportFormat.MMPOSE)
        self.format_combo.addItem("PostureKit Custom", ExportFormat.CUSTOM)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        layout.addWidget(self.format_combo)

        # Description
        self.format_description = QTextEdit()
        self.format_description.setReadOnly(True)
        self.format_description.setMaximumHeight(60)
        self.format_description.setPlainText(
            "Standard COCO-WholeBody format (133 keypoints). "
            "Compatible with all major frameworks."
        )
        layout.addWidget(self.format_description)

        group.setLayout(layout)
        return group

    def _create_output_group(self) -> QGroupBox:
        """Create output directory group"""
        group = QGroupBox("Output Directory")
        layout = QHBoxLayout()

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("/path/to/output")
        layout.addWidget(self.output_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_output)
        layout.addWidget(browse_btn)

        group.setLayout(layout)
        return group

    def _create_split_group(self) -> QGroupBox:
        """Create dataset split group"""
        group = QGroupBox("Dataset Split")
        layout = QVBoxLayout()

        # Sliders with percentages
        grid = QGridLayout()

        # Train
        grid.addWidget(QLabel("Train:"), 0, 0)
        self.train_slider = QSlider(Qt.Horizontal)
        self.train_slider.setRange(0, 100)
        self.train_slider.setValue(70)
        self.train_slider.valueChanged.connect(self._update_split_sliders)
        grid.addWidget(self.train_slider, 0, 1)
        self.train_label = QLabel("70%")
        grid.addWidget(self.train_label, 0, 2)

        # Progress bar visualization
        self.train_bar = QProgressBar()
        self.train_bar.setValue(70)
        self.train_bar.setTextVisible(False)
        grid.addWidget(self.train_bar, 0, 3)

        # Val
        grid.addWidget(QLabel("Val:"), 1, 0)
        self.val_slider = QSlider(Qt.Horizontal)
        self.val_slider.setRange(0, 100)
        self.val_slider.setValue(15)
        self.val_slider.valueChanged.connect(self._update_split_sliders)
        grid.addWidget(self.val_slider, 1, 1)
        self.val_label = QLabel("15%")
        grid.addWidget(self.val_label, 1, 2)

        self.val_bar = QProgressBar()
        self.val_bar.setValue(15)
        self.val_bar.setTextVisible(False)
        grid.addWidget(self.val_bar, 1, 3)

        # Test
        grid.addWidget(QLabel("Test:"), 2, 0)
        self.test_slider = QSlider(Qt.Horizontal)
        self.test_slider.setRange(0, 100)
        self.test_slider.setValue(15)
        self.test_slider.valueChanged.connect(self._update_split_sliders)
        grid.addWidget(self.test_slider, 2, 1)
        self.test_label = QLabel("15%")
        grid.addWidget(self.test_label, 2, 2)

        self.test_bar = QProgressBar()
        self.test_bar.setValue(15)
        self.test_bar.setTextVisible(False)
        grid.addWidget(self.test_bar, 2, 3)

        layout.addLayout(grid)

        # Strategy buttons
        strategy_layout = QHBoxLayout()

        self.strategy_random = QPushButton("Random")
        self.strategy_random.setCheckable(True)
        self.strategy_random.setChecked(True)
        strategy_layout.addWidget(self.strategy_random)

        self.strategy_stratified = QPushButton("Stratified")
        self.strategy_stratified.setCheckable(True)
        strategy_layout.addWidget(self.strategy_stratified)

        self.strategy_sequential = QPushButton("Sequential")
        self.strategy_sequential.setCheckable(True)
        strategy_layout.addWidget(self.strategy_sequential)

        # Button group for exclusive selection
        self.strategy_group = QButtonGroup()
        self.strategy_group.addButton(self.strategy_random)
        self.strategy_group.addButton(self.strategy_stratified)
        self.strategy_group.addButton(self.strategy_sequential)

        layout.addLayout(strategy_layout)

        group.setLayout(layout)
        return group

    def _create_filters_group(self) -> QGroupBox:
        """Create filters group"""
        group = QGroupBox("Filters")
        layout = QGridLayout()

        # Category filter
        layout.addWidget(QLabel("Category:"), 0, 0)
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories")
        self.category_combo.addItems(self.categories)
        self.category_combo.currentTextChanged.connect(self._update_preview)
        layout.addWidget(self.category_combo, 0, 1, 1, 2)

        # Min confidence
        layout.addWidget(QLabel("Min Confidence:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(30)
        self.conf_slider.valueChanged.connect(self._on_conf_slider_changed)
        layout.addWidget(self.conf_slider, 1, 1)
        self.conf_label = QLabel("0.30")
        layout.addWidget(self.conf_label, 1, 2)

        # Max occlusion
        layout.addWidget(QLabel("Max Occlusion:"), 2, 0)
        self.occlusion_slider = QSlider(Qt.Horizontal)
        self.occlusion_slider.setRange(0, 100)
        self.occlusion_slider.setValue(20)
        self.occlusion_slider.valueChanged.connect(self._on_occlusion_slider_changed)
        layout.addWidget(self.occlusion_slider, 2, 1)
        self.occlusion_label = QLabel("20%")
        layout.addWidget(self.occlusion_label, 2, 2)

        # Checkboxes
        self.corrected_check = QCheckBox("Corrected poses only")
        self.corrected_check.stateChanged.connect(self._update_preview)
        layout.addWidget(self.corrected_check, 3, 0, 1, 3)

        self.labeled_check = QCheckBox("Labeled poses only")
        self.labeled_check.stateChanged.connect(self._update_preview)
        layout.addWidget(self.labeled_check, 4, 0, 1, 3)

        self.exclude_multi_check = QCheckBox("Exclude multi-person images")
        self.exclude_multi_check.stateChanged.connect(self._update_preview)
        layout.addWidget(self.exclude_multi_check, 5, 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_options_group(self) -> QGroupBox:
        """Create options group"""
        group = QGroupBox("Options")
        layout = QVBoxLayout()

        self.copy_images_check = QCheckBox("Copy images to export directory")
        self.copy_images_check.setChecked(True)
        layout.addWidget(self.copy_images_check)

        self.include_features_check = QCheckBox("Include visual features (512-dim)")
        layout.addWidget(self.include_features_check)

        self.metadata_check = QCheckBox("Generate metadata.json")
        self.metadata_check.setChecked(True)
        layout.addWidget(self.metadata_check)

        self.readme_check = QCheckBox("Create README.md")
        self.readme_check.setChecked(True)
        layout.addWidget(self.readme_check)

        group.setLayout(layout)
        return group

    def _on_format_changed(self, index):
        """Update description when format changes"""
        format_type = self.format_combo.currentData()

        descriptions = {
            ExportFormat.COCO: "Standard COCO-WholeBody format (133 keypoints). Compatible with all major frameworks.",
            ExportFormat.MMPOSE: "MMPose training format. Optimized for MMPose framework with direct loading support.",
            ExportFormat.CUSTOM: "PostureKit custom format with full metadata. Includes all annotations, corrections, and training labels."
        }

        self.format_description.setPlainText(descriptions.get(format_type, ""))

    def _browse_output(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.output_edit.setText(directory)

    def _update_split_sliders(self):
        """Keep split ratios summing to 100%"""
        sender = self.sender()

        # Auto-adjust other sliders to maintain 100% total
        train = self.train_slider.value()
        val = self.val_slider.value()
        test = self.test_slider.value()

        total = train + val + test

        if total != 100:
            # Adjust the slider that wasn't changed
            diff = 100 - total
            if sender == self.train_slider:
                # Distribute diff between val and test
                self.val_slider.setValue(max(0, val + diff // 2))
                self.test_slider.setValue(100 - train - self.val_slider.value())
            elif sender == self.val_slider:
                # Distribute diff between train and test
                self.test_slider.setValue(max(0, test + diff))
            else:
                # Distribute diff between train and val
                self.val_slider.setValue(max(0, val + diff))

        # Update labels and bars
        self.train_label.setText(f"{self.train_slider.value()}%")
        self.val_label.setText(f"{self.val_slider.value()}%")
        self.test_label.setText(f"{self.test_slider.value()}%")

        self.train_bar.setValue(self.train_slider.value())
        self.val_bar.setValue(self.val_slider.value())
        self.test_bar.setValue(self.test_slider.value())

    def _on_conf_slider_changed(self):
        """Update confidence label"""
        value = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{value:.2f}")
        self._update_preview()

    def _on_occlusion_slider_changed(self):
        """Update occlusion label"""
        value = self.occlusion_slider.value()
        self.occlusion_label.setText(f"{value}%")
        self._update_preview()

    def _update_preview(self):
        """Update preview count"""
        config = self._get_config()

        # Count matching poses
        count = 0
        for pose in self.pose_data:
            if self._matches_filters(pose, config):
                count += 1

        self.preview_label.setText(f"<b>Preview:</b> {count} poses match criteria")

    def _matches_filters(self, pose: Dict, config: ExportConfig) -> bool:
        """Check if pose matches filters"""
        if config.category_filter and config.category_filter != "All Categories":
            if pose.get('category') != config.category_filter:
                return False

        if pose.get('confidence', 1.0) < config.min_confidence:
            return False

        if pose.get('occlusion_percentage', 0) > config.max_occlusion:
            return False

        if config.corrected_only and not pose.get('is_corrected', False):
            return False

        if config.labeled_only and not pose.get('category'):
            return False

        if config.exclude_multi_person and pose.get('num_people', 1) > 1:
            return False

        return True

    def _show_preview(self):
        """Show preview of matching poses"""
        config = self._get_config()
        matching_poses = [p for p in self.pose_data if self._matches_filters(p, config)]

        # Show preview dialog
        from PySide6.QtWidgets import QTextBrowser
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Export Preview")
        preview_dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(preview_dialog)

        browser = QTextBrowser()

        text = f"<h3>Export Preview</h3>"
        text += f"<p><b>Total matching poses:</b> {len(matching_poses)}</p>"

        # Split counts
        total = len(matching_poses)
        train_count = int(total * config.train_ratio / 100)
        val_count = int(total * config.val_ratio / 100)
        test_count = total - train_count - val_count

        text += f"<p><b>Split:</b></p>"
        text += f"<ul>"
        text += f"<li>Train: {train_count} poses ({config.train_ratio}%)</li>"
        text += f"<li>Val: {val_count} poses ({config.val_ratio}%)</li>"
        text += f"<li>Test: {test_count} poses ({config.test_ratio}%)</li>"
        text += f"</ul>"

        # Category distribution
        categories = {}
        for pose in matching_poses:
            cat = pose.get('category', 'uncategorized')
            categories[cat] = categories.get(cat, 0) + 1

        text += f"<p><b>Categories:</b></p><ul>"
        for cat, count in sorted(categories.items()):
            text += f"<li>{cat}: {count} poses</li>"
        text += "</ul>"

        browser.setHtml(text)
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_btn)

        preview_dialog.exec()

    def _get_config(self) -> ExportConfig:
        """Get current export configuration"""
        strategy = SplitStrategy.RANDOM
        if self.strategy_stratified.isChecked():
            strategy = SplitStrategy.STRATIFIED
        elif self.strategy_sequential.isChecked():
            strategy = SplitStrategy.SEQUENTIAL

        return ExportConfig(
            format=self.format_combo.currentData(),
            output_dir=self.output_edit.text(),
            train_ratio=self.train_slider.value(),
            val_ratio=self.val_slider.value(),
            test_ratio=self.test_slider.value(),
            split_strategy=strategy,
            category_filter=self.category_combo.currentText() if self.category_combo.currentText() != "All Categories" else None,
            min_confidence=self.conf_slider.value() / 100.0,
            max_occlusion=self.occlusion_slider.value(),
            corrected_only=self.corrected_check.isChecked(),
            labeled_only=self.labeled_check.isChecked(),
            exclude_multi_person=self.exclude_multi_check.isChecked(),
            copy_images=self.copy_images_check.isChecked(),
            include_features=self.include_features_check.isChecked(),
            generate_metadata=self.metadata_check.isChecked(),
            generate_readme=self.readme_check.isChecked()
        )

    def _start_export(self):
        """Start export process"""
        config = self._get_config()

        # Validate
        if not config.output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory")
            return

        if config.train_ratio + config.val_ratio + config.test_ratio != 100:
            QMessageBox.warning(self, "Error", "Split ratios must sum to 100%")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.export_btn.setEnabled(False)

        # Start worker
        self.export_worker = ExportWorker(config, self.pose_data)
        self.export_worker.progress.connect(self._on_progress)
        self.export_worker.finished.connect(self._on_finished)
        self.export_worker.start()

    def _on_progress(self, percentage: int, message: str):
        """Update progress"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def _on_finished(self, success: bool, message: str):
        """Handle export completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.export_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """Clean up threads before closing."""
        # Terminate any running worker threads
        if hasattr(self, "export_worker") and self.export_worker and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait(2000)

            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.export_worker.progress.disconnect()
                self.export_worker.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.export_worker.deleteLater()

        super().closeEvent(event)


# Example usage
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Sample data
    pose_data = [
        {
            'image_path': 'image1.jpg',
            'image_width': 1920,
            'image_height': 1080,
            'bbox': [100, 100, 200, 400],
            'keypoints': [],
            'confidence': 0.95,
            'visible_keypoints': 128,
            'occlusion_percentage': 5,
            'category': 'yoga',
            'is_corrected': True,
            'num_people': 1
        }
    ]

    dialog = ExportDialog(pose_data, ['yoga', 'standing', 'action'])
    dialog.exec()
