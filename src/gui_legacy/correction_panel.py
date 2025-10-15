"""
Correction Panel for PostureKit.
Allows manual correction of pose keypoints and labeling.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QScrollArea, QFrame, QGridLayout, QCheckBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import numpy as np
from typing import Optional
from pathlib import Path

from src.core.geometric_feature_extractor import GeometricFeatures, GeometricFeatureExtractor
from src.core.image_ingestor import ImageIngestor
from src.core.pose_detector import PoseResult
from src.gui_legacy.widgets.keypoint_editor import KeypointEditor
from src.gui_legacy.widgets.label_input import LabelInput
from src.gui_legacy.widgets.occlusion_editor import OcclusionEditor
from src.gui_legacy.undo_stack import UndoStack, UndoCommand
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureDisplayWidget(QWidget):
    """
    Widget for displaying geometric features in an organized, readable format.
    This is a self-contained component that can be added to the existing panel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._current_features: Optional[GeometricFeatures] = None

    def _init_ui(self):
        """Initialize the feature display UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)  # Reduced vertical margins
        layout.setSpacing(5)  # Reduced from 8 to 5

        # Header
        header = QLabel("Geometric Analysis")
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        explanation = QLabel(
            "These features describe the pose geometry and are used for similarity search."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(explanation)

        # Joint Angles Section
        self.joint_angles_group = self._create_joint_angles_section()
        layout.addWidget(self.joint_angles_group)

        # Limb Proportions Section
        self.limb_proportions_group = self._create_limb_proportions_section()
        layout.addWidget(self.limb_proportions_group)

        # Body Orientation Section
        self.body_angles_group = self._create_body_angles_section()
        layout.addWidget(self.body_angles_group)

        # Symmetry Scores Section
        self.symmetry_group = self._create_symmetry_section()
        layout.addWidget(self.symmetry_group)

        # Occlusion Pattern Section
        self.occlusion_group = self._create_occlusion_section()
        layout.addWidget(self.occlusion_group)

        layout.addStretch()

    def _create_joint_angles_section(self) -> QGroupBox:
        """Create the joint angles display section."""
        group = QGroupBox("Joint Angles (degrees)")
        layout = QGridLayout()
        layout.setSpacing(4)

        self.joint_angle_labels = {}

        joint_pairs = [
            ('left_elbow', 'right_elbow', 'Elbows'),
            ('left_shoulder', 'right_shoulder', 'Shoulders'),
            ('left_hip', 'right_hip', 'Hips'),
            ('left_knee', 'right_knee', 'Knees'),
        ]

        for row, (left_key, right_key, name) in enumerate(joint_pairs):
            name_label = QLabel(f"{name}:")
            layout.addWidget(name_label, row, 0)

            left_label = QLabel("--°")
            left_label.setMinimumWidth(50)
            left_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.joint_angle_labels[left_key] = left_label
            layout.addWidget(left_label, row, 1)

            right_label = QLabel("--°")
            right_label.setMinimumWidth(50)
            right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.joint_angle_labels[right_key] = right_label
            layout.addWidget(right_label, row, 2)

        group.setLayout(layout)
        return group

    def _create_limb_proportions_section(self) -> QGroupBox:
        """Create the limb proportions display section."""
        group = QGroupBox("Limb Proportions (normalized)")
        layout = QGridLayout()
        layout.setSpacing(4)

        self.limb_proportion_labels = {}

        limb_pairs = [
            ('left_upper_arm', 'right_upper_arm', 'Upper Arms'),
            ('left_forearm', 'right_forearm', 'Forearms'),
            ('left_thigh', 'right_thigh', 'Thighs'),
            ('left_shin', 'right_shin', 'Shins'),
        ]

        for row, (left_key, right_key, name) in enumerate(limb_pairs):
            name_label = QLabel(f"{name}:")
            layout.addWidget(name_label, row, 0)

            left_label = QLabel("--")
            left_label.setMinimumWidth(50)
            left_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.limb_proportion_labels[left_key] = left_label
            layout.addWidget(left_label, row, 1)

            right_label = QLabel("--")
            right_label.setMinimumWidth(50)
            right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.limb_proportion_labels[right_key] = right_label
            layout.addWidget(right_label, row, 2)

        group.setLayout(layout)
        return group

    def _create_body_angles_section(self) -> QGroupBox:
        """Create the body orientation angles section."""
        group = QGroupBox("Body Orientation (degrees)")
        layout = QGridLayout()
        layout.setSpacing(4)

        self.body_angle_labels = {}

        key_angles = [
            ('torso_lean', 'Torso Lean'),
            ('head_tilt', 'Head Tilt'),
        ]

        for row, (key, name) in enumerate(key_angles):
            name_label = QLabel(f"{name}:")
            layout.addWidget(name_label, row, 0)

            value_label = QLabel("--°")
            value_label.setMinimumWidth(60)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.body_angle_labels[key] = value_label
            layout.addWidget(value_label, row, 1)

        group.setLayout(layout)
        return group

    def _create_symmetry_section(self) -> QGroupBox:
        """Create the symmetry scores section."""
        group = QGroupBox("Symmetry Analysis")
        layout = QGridLayout()
        layout.setSpacing(4)

        self.symmetry_labels = {}

        symmetry_measures = [
            ('elbow_symmetry', 'Elbows'),
            ('shoulder_symmetry', 'Shoulders'),
            ('hip_symmetry', 'Hips'),
            ('knee_symmetry', 'Knees')
        ]

        for row, (key, name) in enumerate(symmetry_measures):
            name_label = QLabel(f"{name}:")
            layout.addWidget(name_label, row, 0)

            indicator = QLabel("●")
            indicator.setMinimumWidth(30)
            indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.symmetry_labels[key] = indicator
            layout.addWidget(indicator, row, 1)

            score_label = QLabel("--")
            score_label.setMinimumWidth(50)
            score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.symmetry_labels[f"{key}_score"] = score_label
            layout.addWidget(score_label, row, 2)

        legend = QLabel("● Green: Symmetric | Yellow: Moderate | Red: Asymmetric")
        legend.setStyleSheet("color: #666; font-size: 8pt;")
        layout.addWidget(legend, len(symmetry_measures), 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_occlusion_section(self) -> QGroupBox:
        """Create the occlusion pattern section."""
        group = QGroupBox("Visible Body Regions")
        layout = QHBoxLayout()

        self.occlusion_labels = {}

        regions = [
            ('head', 'Head'),
            ('left_arm', 'L Arm'),
            ('right_arm', 'R Arm'),
            ('torso', 'Torso'),
            ('left_leg', 'L Leg'),
            ('right_leg', 'R Leg'),
            ('feet', 'Feet')
        ]

        for key, name in regions:
            region_widget = QWidget()
            region_layout = QVBoxLayout(region_widget)
            region_layout.setContentsMargins(2, 2, 2, 2)
            region_layout.setSpacing(2)

            icon_label = QLabel("✓")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
            self.occlusion_labels[key] = icon_label

            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("font-size: 8pt;")

            region_layout.addWidget(icon_label)
            region_layout.addWidget(name_label)

            layout.addWidget(region_widget)

        group.setLayout(layout)
        return group

    def update_features(self, features: Optional[GeometricFeatures]):
        """Update the display with new geometric features."""
        if features is None:
            self._clear_display()
            return

        self._current_features = features

        # Update joint angles with color coding
        for key, label in self.joint_angle_labels.items():
            if key in features.joint_angles:
                angle = features.joint_angles[key]
                label.setText(f"{angle:.1f}°")

                if angle < 30:
                    label.setStyleSheet("color: #d32f2f;")
                elif angle > 150:
                    label.setStyleSheet("color: #1976d2;")
                else:
                    label.setStyleSheet("color: #388e3c;")

        # Update limb proportions
        for key, label in self.limb_proportion_labels.items():
            if key in features.limb_ratios:
                ratio = features.limb_ratios[key]
                label.setText(f"{ratio:.3f}")

        # Update body angles
        for key, label in self.body_angle_labels.items():
            if key in features.body_angles:
                angle = features.body_angles[key]
                label.setText(f"{angle:.1f}°")

        # Update symmetry with color coding
        for key in ['elbow_symmetry', 'shoulder_symmetry', 'hip_symmetry', 'knee_symmetry']:
            if key in features.symmetry_scores:
                score = features.symmetry_scores[key]

                indicator = self.symmetry_labels[key]
                if score < 0.2:
                    indicator.setStyleSheet("color: #4caf50; font-size: 16pt;")
                elif score < 0.5:
                    indicator.setStyleSheet("color: #ff9800; font-size: 16pt;")
                else:
                    indicator.setStyleSheet("color: #f44336; font-size: 16pt;")

                score_label = self.symmetry_labels[f"{key}_score"]
                score_label.setText(f"{score:.3f}")

        # Update occlusion pattern
        occlusion_map = {
            'head': 0, 'left_arm': 1, 'right_arm': 2, 'torso': 3,
            'left_leg': 4, 'right_leg': 5, 'feet': 6
        }

        for key, idx in occlusion_map.items():
            if idx < len(features.occlusion_pattern):
                is_visible = features.occlusion_pattern[idx] > 0.5
                label = self.occlusion_labels[key]

                if is_visible:
                    label.setText("✓")
                    label.setStyleSheet("color: #4caf50; font-size: 16pt; font-weight: bold;")
                else:
                    label.setText("✗")
                    label.setStyleSheet("color: #9e9e9e; font-size: 16pt; font-weight: bold;")

    def _clear_display(self):
        """Clear all feature displays."""
        for label in self.joint_angle_labels.values():
            label.setText("--°")
            label.setStyleSheet("")

        for label in self.limb_proportion_labels.values():
            label.setText("--")

        for label in self.body_angle_labels.values():
            label.setText("--°")

        for key, label in self.symmetry_labels.items():
            if key.endswith('_score'):
                label.setText("--")
            else:
                label.setText("●")
                label.setStyleSheet("color: #9e9e9e; font-size: 16pt;")

        for label in self.occlusion_labels.values():
            label.setText("✗")
            label.setStyleSheet("color: #9e9e9e; font-size: 16pt; font-weight: bold;")


class CorrectionPanel(QWidget):
    """
    Panel for correcting pose detections.
    Includes geometric feature display and pose labeling.
    """

    # Signals that MainWindow may be connected to
    correction_saved = Signal(dict)
    keypoint_edited = Signal(int, float, float, float)  # index, x, y, confidence
    correction_reset = Signal()  # Emitted when corrections are discarded
    save_new_pose_requested = Signal(dict)  # Emitted when saving new pose
    occlusionDisplayChanged = Signal(bool, bool, bool)  # show_visible, show_occluded, show_missing
    person_delete_requested = Signal(int)  # person_index
    person_next_requested = Signal()  # Navigate to next person
    labelsChanged = Signal(object)  # Training labels changed
    saveRequested = Signal()  # Save button clicked
    deleteRequested = Signal()  # Delete button clicked
    resetRequested = Signal()  # Reset button clicked
    keypointSelectedInList = Signal(int)  # keypoint_index - emitted when keypoint selected in occlusion editor
    keypointsSelectedInList = Signal(list)  # list of keypoint_indices - emitted when multiple keypoints selected
    visibilityChanged = Signal(int, int)  # keypoint_index, visibility - emitted when visibility changed

    def __init__(self, parent=None, storage_manager=None):
        super().__init__(parent)

        # Storage manager for correction tracking
        self.storage_manager = storage_manager

        # Undo/Redo system
        self.undo_stack = UndoStack(max_size=100)
        self.undo_stack.can_undo_changed.connect(self._on_undo_state_changed)
        self.undo_stack.can_redo_changed.connect(self._on_redo_state_changed)

        # Track keypoint state for undo
        self._current_keypoints: Optional[np.ndarray] = None
        self._original_keypoints: Optional[np.ndarray] = None  # Original keypoints for correction tracking
        self._last_keypoint_state: dict = {}  # {index: (pos, conf)}

        self._init_ui()
        self._current_features: Optional[GeometricFeatures] = None
        self._has_changes = False

        # MainWindow compatibility attributes
        self.current_pose_id = None
        self.current_pose_data = None
        self._current_pose_result = None  # Store full PoseResult for visibility stats
        self.current_image_path = None  # Store image path for database save

        # Multi-person state
        self._all_poses = []  # List of all PoseResult objects
        self._current_person_index = 0  # Currently selected person

    def _init_ui(self):
        """Initialize the correction panel UI with scroll area."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Content widget
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 5, 8, 5)  # Reduced margins: top, left, right, bottom
        content_layout.setSpacing(6)  # Reduced from 10 to 6

        # Title
        title = QLabel("Pose Correction")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        content_layout.addWidget(title)

        # Multi-person info header
        person_info_group = QGroupBox("Person Info")
        person_info_layout = QVBoxLayout()

        # Person label and confidence
        info_row = QHBoxLayout()
        self.person_label = QLabel("Person 1 of 1")
        self.person_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        info_row.addWidget(self.person_label)
        info_row.addStretch()

        self.confidence_label = QLabel("Confidence: --")
        info_row.addWidget(self.confidence_label)
        person_info_layout.addLayout(info_row)

        # Visible keypoints count
        self.visible_keypoints_label = QLabel("Visible keypoints: --")
        person_info_layout.addWidget(self.visible_keypoints_label)

        person_info_group.setLayout(person_info_layout)
        content_layout.addWidget(person_info_group)

        # Person navigation buttons - REMOVED duplicate save button
        # nav_layout = QHBoxLayout()
        # self.save_button = QPushButton("Save Labels")
        # self.save_button.clicked.connect(self._on_save_clicked)
        # nav_layout.addWidget(self.save_button)
        # self.delete_button = QPushButton("Delete Person")
        # self.delete_button.clicked.connect(self._on_delete_clicked)
        # nav_layout.addWidget(self.delete_button)
        # Next Person button removed per user request
        # self.next_button = QPushButton("Next Person")
        # self.next_button.clicked.connect(self._on_next_clicked)
        # nav_layout.addWidget(self.next_button)
        # content_layout.addLayout(nav_layout)

        # Keypoint editor (for MainWindow compatibility)
        self.keypoint_editor = KeypointEditor()
        self.keypoint_editor.keypoint_moved.connect(self._on_keypoint_moved)
        self.keypoint_editor.confidence_changed.connect(self._on_keypoint_confidence_changed)
        content_layout.addWidget(self.keypoint_editor)

        # Occlusion Editor (NEW)
        self.occlusion_editor = OcclusionEditor()
        self.occlusion_editor.visibility_changed.connect(self._on_visibility_changed)
        self.occlusion_editor.keypoint_selected.connect(self._on_keypoint_selected_in_list)
        self.occlusion_editor.keypoints_selected.connect(self._on_keypoints_selected_in_list)
        content_layout.addWidget(self.occlusion_editor)

        # Occlusion Display Controls (NEW)
        occlusion_group = QGroupBox("Occlusion Display")
        occlusion_layout = QVBoxLayout()

        # Checkboxes for visibility filtering
        checkbox_layout = QHBoxLayout()

        # Checkbox styling for better visibility
        checkbox_style = """
            QCheckBox {
                color: #e0e0e0;
                font-size: 10pt;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #888;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #4caf50;
                border-color: #4caf50;
            }
            QCheckBox::indicator:hover {
                border-color: #aaa;
            }
        """

        self.show_visible_checkbox = QCheckBox("Show Visible")
        self.show_visible_checkbox.setChecked(True)
        self.show_visible_checkbox.setStyleSheet(checkbox_style)
        self.show_visible_checkbox.stateChanged.connect(self._on_occlusion_display_changed)
        checkbox_layout.addWidget(self.show_visible_checkbox)

        self.show_occluded_checkbox = QCheckBox("Show Occluded")
        self.show_occluded_checkbox.setChecked(True)
        self.show_occluded_checkbox.setStyleSheet(checkbox_style)
        self.show_occluded_checkbox.stateChanged.connect(self._on_occlusion_display_changed)
        checkbox_layout.addWidget(self.show_occluded_checkbox)

        self.show_missing_checkbox = QCheckBox("Show Missing")
        self.show_missing_checkbox.setChecked(False)
        self.show_missing_checkbox.setStyleSheet(checkbox_style)
        self.show_missing_checkbox.stateChanged.connect(self._on_occlusion_display_changed)
        checkbox_layout.addWidget(self.show_missing_checkbox)

        occlusion_layout.addLayout(checkbox_layout)

        # Occlusion statistics display
        self.occlusion_stats_label = QLabel()
        self.occlusion_stats_label.setWordWrap(True)
        self._update_occlusion_stats(None)
        occlusion_layout.addWidget(self.occlusion_stats_label)

        occlusion_group.setLayout(occlusion_layout)
        content_layout.addWidget(occlusion_group)

        # Geometric Features Display (NEW)
        self.feature_display = FeatureDisplayWidget()
        content_layout.addWidget(self.feature_display)

        # Label inputs (for MainWindow compatibility) - REMOVED per user request
        # self.label_input = LabelInput()
        # self.label_input.label_changed.connect(self._on_change_detected)
        # content_layout.addWidget(self.label_input)

        # Undo/Redo buttons
        undo_layout = QHBoxLayout()

        self.undo_button = QPushButton("⟲ Undo")
        self.undo_button.setShortcut("Ctrl+Z")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        undo_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("⟳ Redo")
        self.redo_button.setShortcut("Ctrl+Y")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self._on_redo_clicked)
        undo_layout.addWidget(self.redo_button)

        content_layout.addLayout(undo_layout)

        # Save/Reset buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        # Emit signal to main window instead of saving directly
        # This allows main window to show selection dialog for multi-person saves
        self.save_button.clicked.connect(self.saveRequested.emit)
        button_layout.addWidget(self.save_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_button)

        content_layout.addLayout(button_layout)
        content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def updatePerson(self, person):
        """Update panel with data from a Person object."""
        logger.info(f"updatePerson called for person ID: {person.id}")

        # Store pose ID for correction tracking
        # For newly detected poses, person.id is just an index (0, 1, 2...)
        # For poses from database, person.id is a UUID string
        # We only set current_pose_id if it's a valid UUID, otherwise None (new pose)
        try:
            from uuid import UUID
            # Try to parse as UUID - if it works, it's from database
            UUID(str(person.id))
            self.current_pose_id = person.id
            logger.debug(f"Set current_pose_id to database UUID: {person.id}")
        except (ValueError, AttributeError):
            # Not a valid UUID - this is a newly detected pose
            self.current_pose_id = None
            logger.debug(f"Person ID {person.id} is not a UUID - treating as new pose")

        pose_result = None
        try:
            from src.core.pose_detector import PoseResult
            from src.core.geometric_feature_extractor import GeometricFeatureExtractor

            # Convert Person to PoseResult format
            keypoints_array = np.array([[kp.x, kp.y, kp.confidence] for kp in person.keypoints])
            visibility_array = np.array([kp.visibility for kp in person.keypoints])

            # Convert bbox tuple to numpy array
            bbox_array = np.array(person.bbox, dtype=np.float32)

            logger.debug(f"Creating PoseResult with {len(keypoints_array)} keypoints")

            pose_result = PoseResult(
                keypoints=keypoints_array,
                bbox=bbox_array,
                overall_confidence=person.confidence,
                visibility=visibility_array,
                person_id=person.id
            )

            # Update internal state
            self._current_pose_result = pose_result
            self._current_keypoints = keypoints_array.copy()
            self._original_keypoints = keypoints_array.copy()  # Store original for correction tracking

            # Load pose into occlusion editor
            self.occlusion_editor.load_pose(keypoints_array, visibility_array)
            logger.debug("Occlusion editor updated")

            # Extract geometric features and update display
            logger.debug("Extracting geometric features...")
            extractor = GeometricFeatureExtractor()
            features = extractor.extract(pose_result)
            logger.info(f"Features extracted: {len(features.feature_vector)} dimensions")

            self.feature_display.update_features(features)
            logger.info("Geometric features extracted and displayed successfully")

        except Exception as e:
            logger.error(f"Failed in updatePerson: {e}", exc_info=True)
            self.feature_display.update_features(None)

        # Update _all_poses so confidence display works
        if pose_result is not None:
            self._all_poses = [pose_result]
            self._current_person_index = 0

        # Update UI components
        self._update_person_info()
        if pose_result is not None:
            self._update_occlusion_stats(pose_result)

    def set_features(self, features: GeometricFeatures):
        """
        Set the geometric features to display.
        """
        self._current_features = features
        self.feature_display.update_features(features)
        logger.debug("Set geometric features")

    def setPoseResult(self, pose_result, image_path):
        """
        Store PoseResult and image path for database save.

        Args:
            pose_result: PoseResult object from pose detector
            image_path: Path to the source image
        """
        self._current_pose_result = pose_result
        self.current_image_path = image_path
        logger.debug(f"Stored PoseResult and image path: {image_path}")

    def set_pose(self, pose_result, person_index: int = 0, all_poses: list = None):
        """
        Update panel with new pose result.

        Args:
            pose_result: PoseResult object with keypoints and visibility data
            person_index: Index of currently selected person
            all_poses: List of all PoseResult objects in the image
        """
        self._current_pose_result = pose_result
        self._current_person_index = person_index
        self._all_poses = all_poses if all_poses else [pose_result]

        # Initialize keypoints for undo tracking
        if pose_result is not None and hasattr(pose_result, 'keypoints'):
            self._current_keypoints = pose_result.keypoints.copy()
            self._original_keypoints = pose_result.keypoints.copy()  # Store original for correction tracking
        else:
            self._current_keypoints = None
            self._original_keypoints = None

        # Load pose into occlusion editor
        if pose_result is not None and hasattr(pose_result, 'keypoints'):
            visibility = pose_result.visibility if hasattr(pose_result, 'visibility') else None
            self.occlusion_editor.load_pose(pose_result.keypoints, visibility)

        # Clear undo stack when loading new pose
        self.undo_stack.clear()

        self._update_occlusion_stats(pose_result)
        self._update_person_info()
        logger.debug(f"Set pose result for person {person_index + 1} of {len(self._all_poses)}")

    def set_selected_keypoint(self, index: int, x: float, y: float, confidence: float):
        """
        Set currently selected keypoint for editing (MainWindow compatibility).

        Args:
            index: Keypoint index
            x: X coordinate
            y: Y coordinate
            confidence: Confidence score
        """
        self.keypoint_editor.set_keypoint(index, x, y, confidence)

    def get_correction_data(self) -> dict:
        """
        Get the current correction data as a dictionary.
        MainWindow likely calls this to retrieve user edits.
        """
        # Label inputs removed per user request - return empty values
        # label_data = self.label_input.get_labels().to_dict()

        # Get corrected keypoints and visibility
        corrections = {
            'category': '',
            'difficulty': '',
            'tags': [],
            'notes': ''
        }

        # Add keypoints if modified
        if self._current_keypoints is not None:
            corrections['keypoints'] = self._current_keypoints.copy()

        # Add visibility if modified
        if hasattr(self.occlusion_editor, 'visibility') and self.occlusion_editor.visibility is not None:
            corrections['visibility'] = self.occlusion_editor.get_visibility()

        # Determine correction type
        corrections['correction_type'] = self._determine_correction_type()

        return corrections

    def has_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes.
        MainWindow uses this to warn users before navigating away.
        """
        return self._has_changes

    def _on_keypoint_moved(self, index: int, x: float, y: float):
        """Handle keypoint position change."""
        self._has_changes = True

        # Get current state
        confidence = self.keypoint_editor.conf_spinbox.value()
        new_position = np.array([x, y], dtype=np.float32)

        # Get old state for undo
        if self._current_keypoints is not None and index < len(self._current_keypoints):
            old_position = self._current_keypoints[index, :2].copy()
            old_confidence = self._current_keypoints[index, 2]

            # Create undo command
            command = UndoCommand(
                name=f"Move keypoint {index}",
                keypoint_index=index,
                old_position=old_position,
                new_position=new_position,
                old_confidence=old_confidence,
                new_confidence=confidence
            )
            self.undo_stack.push(command)

            # Update current keypoints
            self._current_keypoints[index, :2] = new_position
            self._current_keypoints[index, 2] = confidence

        self.keypoint_edited.emit(index, x, y, confidence)
        logger.debug(f"Keypoint {index} moved to ({x}, {y})")

    def _on_keypoint_confidence_changed(self, index: int, confidence: float):
        """Handle keypoint confidence change."""
        self._has_changes = True

        # Get current position
        x = self.keypoint_editor.x_spinbox.value()
        y = self.keypoint_editor.y_spinbox.value()
        position = np.array([x, y], dtype=np.float32)

        # Get old state for undo
        if self._current_keypoints is not None and index < len(self._current_keypoints):
            old_confidence = self._current_keypoints[index, 2]

            # Only create undo command if confidence actually changed
            if abs(old_confidence - confidence) > 0.001:
                command = UndoCommand(
                    name=f"Change keypoint {index} confidence",
                    keypoint_index=index,
                    old_position=position.copy(),
                    new_position=position.copy(),
                    old_confidence=old_confidence,
                    new_confidence=confidence
                )
                self.undo_stack.push(command)

                # Update current keypoints
                self._current_keypoints[index, 2] = confidence

        self.keypoint_edited.emit(index, x, y, confidence)
        logger.debug(f"Keypoint {index} confidence changed to {confidence}")

    def _on_change_detected(self):
        """Called when any input changes."""
        self._has_changes = True

    def _on_visibility_changed(self, keypoint_idx: int, new_visibility: int):
        """Handle visibility change from occlusion editor."""
        if self._current_pose_result is not None and hasattr(self._current_pose_result, 'visibility'):
            self._current_pose_result.visibility[keypoint_idx] = new_visibility
            self._has_changes = True

            # Update occlusion stats display
            self._update_occlusion_stats(self._current_pose_result)

            # Emit signal so MainWindow can update the Person object
            self.visibilityChanged.emit(keypoint_idx, new_visibility)

            logger.debug(f"Visibility changed for keypoint {keypoint_idx}: {new_visibility}")

    def _on_keypoint_selected_in_list(self, keypoint_idx: int):
        """Handle single keypoint selection in occlusion editor list."""
        logger.debug(f"Keypoint {keypoint_idx} selected in list, forwarding to viewer")
        self.keypointSelectedInList.emit(keypoint_idx)

    def _on_keypoints_selected_in_list(self, keypoint_indices: list):
        """Handle multiple keypoint selection in occlusion editor list."""
        logger.debug(f"Multiple keypoints {keypoint_indices} selected in list, forwarding to viewer")
        self.keypointsSelectedInList.emit(keypoint_indices)

    def _determine_correction_type(self) -> Optional[str]:
        """Determine what type of correction was made."""
        # Check if any edits were made - use the internal _has_changes flag
        has_position_changes = self._has_changes
        has_visibility_changes = self.occlusion_editor.has_changes() if hasattr(self, 'occlusion_editor') else False

        if has_position_changes and has_visibility_changes:
            return 'both'
        elif has_position_changes:
            return 'position'
        elif has_visibility_changes:
            return 'occlusion'
        else:
            return None

    def _update_occlusion_stats(self, pose_result):
        """Update occlusion statistics display.

        Args:
            pose_result: PoseResult object or None
        """
        if pose_result is None or not hasattr(pose_result, 'visibility') or pose_result.visibility is None:
            self.occlusion_stats_label.setText("No pose loaded")
            return

        visibility = pose_result.visibility
        visible = np.sum(visibility == 2)
        occluded = np.sum(visibility == 1)
        missing = np.sum(visibility == 0)

        # Calculate occlusion percentage
        occluded_or_missing = occluded + missing
        occlusion_pct = (occluded_or_missing / 133.0) * 100.0

        stats_text = f"""
<b>Keypoint Visibility:</b><br>
<span style="color: #4caf50;">● Visible:</span> {visible}/133 ({visible/133*100:.1f}%)<br>
<span style="color: #ff9800;">● Occluded:</span> {occluded}/133 ({occluded/133*100:.1f}%)<br>
<span style="color: #9e9e9e;">● Missing:</span> {missing}/133 ({missing/133*100:.1f}%)<br>
<b>Occlusion Level:</b> {occlusion_pct:.1f}%
"""
        self.occlusion_stats_label.setText(stats_text)

    def _update_person_info(self):
        """Update person info labels."""
        if not self._all_poses:
            self.person_label.setText("Person 1 of 1")
            self.confidence_label.setText("Confidence: --")
            self.visible_keypoints_label.setText("Visible keypoints: --")
            return

        total = len(self._all_poses)
        current = self._current_person_index + 1
        self.person_label.setText(f"Person {current} of {total}")

        if self._current_pose_result:
            # Average confidence of visible keypoints
            confidences = self._current_pose_result.keypoints[:, 2]
            visible_conf = confidences[confidences > 0]
            avg_conf = visible_conf.mean() if len(visible_conf) > 0 else 0.0
            self.confidence_label.setText(f"Confidence: {avg_conf:.2f}")

            # Count visible keypoints
            visible_count = (confidences > 0).sum()
            self.visible_keypoints_label.setText(f"Visible keypoints: {visible_count}/133")

    def _on_occlusion_display_changed(self):
        """Notify viewer to update rendering."""
        show_visible = self.show_visible_checkbox.isChecked()
        show_occluded = self.show_occluded_checkbox.isChecked()
        show_missing = self.show_missing_checkbox.isChecked()

        self.occlusionDisplayChanged.emit(show_visible, show_occluded, show_missing)
        logger.debug(f"Occlusion display changed: visible={show_visible}, occluded={show_occluded}, missing={show_missing}")

    def _on_undo_clicked(self):
        """Handle undo button click."""
        command = self.undo_stack.undo()
        if command:
            # Restore old state
            if self._current_keypoints is not None and command.keypoint_index < len(self._current_keypoints):
                self._current_keypoints[command.keypoint_index, :2] = command.old_position
                self._current_keypoints[command.keypoint_index, 2] = command.old_confidence

                # Update keypoint editor
                self.keypoint_editor.set_keypoint(
                    command.keypoint_index,
                    command.old_position[0],
                    command.old_position[1],
                    command.old_confidence
                )

                # Notify that keypoints changed
                self.keypoint_edited.emit(
                    command.keypoint_index,
                    command.old_position[0],
                    command.old_position[1],
                    command.old_confidence
                )

                logger.info(f"Undo: {command.name}")

    def _on_redo_clicked(self):
        """Handle redo button click."""
        command = self.undo_stack.redo()
        if command:
            # Restore new state
            if self._current_keypoints is not None and command.keypoint_index < len(self._current_keypoints):
                self._current_keypoints[command.keypoint_index, :2] = command.new_position
                self._current_keypoints[command.keypoint_index, 2] = command.new_confidence

                # Update keypoint editor
                self.keypoint_editor.set_keypoint(
                    command.keypoint_index,
                    command.new_position[0],
                    command.new_position[1],
                    command.new_confidence
                )

                # Notify that keypoints changed
                self.keypoint_edited.emit(
                    command.keypoint_index,
                    command.new_position[0],
                    command.new_position[1],
                    command.new_confidence
                )

                logger.info(f"Redo: {command.name}")

    def _on_undo_state_changed(self, can_undo: bool):
        """Update undo button state."""
        self.undo_button.setEnabled(can_undo)
        if can_undo:
            undo_text = self.undo_stack.get_undo_text()
            self.undo_button.setToolTip(f"Undo: {undo_text}")
        else:
            self.undo_button.setToolTip("Nothing to undo")

    def _on_redo_state_changed(self, can_redo: bool):
        """Update redo button state."""
        self.redo_button.setEnabled(can_redo)
        if can_redo:
            redo_text = self.undo_stack.get_redo_text()
            self.redo_button.setToolTip(f"Redo: {redo_text}")
        else:
            self.redo_button.setToolTip("Nothing to redo")

    def _on_save_clicked(self):
        """Handle save labels button click."""
        from PySide6.QtWidgets import QMessageBox

        logger.info("=" * 60)
        logger.info("SAVE BUTTON CLICKED")
        logger.info("=" * 60)

        # Debug: Log all relevant state
        logger.debug(f"storage_manager exists: {self.storage_manager is not None}")
        logger.debug(f"current_pose_id: {self.current_pose_id}")
        logger.debug(f"_current_pose_result exists: {self._current_pose_result is not None}")
        logger.debug(f"current_image_path: {self.current_image_path}")
        logger.debug(f"_current_keypoints exists: {hasattr(self, '_current_keypoints') and self._current_keypoints is not None}")

        correction_data = self.get_correction_data()
        correction_data['person_index'] = self._current_person_index
        logger.info("Saving labels for current person", extra={'extra_data': correction_data})

        # Save to database if we have StorageManager
        if not self.storage_manager:
            msg = "No StorageManager available - database saving disabled"
            logger.warning(msg)
            QMessageBox.warning(None, "Save Failed", msg)
            self._has_changes = False
            self.correction_saved.emit(correction_data)
            return

        try:
            # Case 1: New pose (never saved to database before)
            if self.current_pose_id is None:
                logger.info("Detected NEW POSE (current_pose_id is None)")

                if not self._current_pose_result:
                    msg = "No pose result available. Please run pose detection first."
                    logger.warning(msg)
                    QMessageBox.warning(None, "Save Failed", msg)
                    return

                if not self.current_image_path:
                    msg = "No image path available. Please load an image first."
                    logger.warning(msg)
                    QMessageBox.warning(None, "Save Failed", msg)
                    return

                logger.info(f"Saving new pose to database... (image: {self.current_image_path})")

                # Reconstruct PoseResult with corrected keypoints
                corrected_pose_result = PoseResult(
                    keypoints=self._current_keypoints,
                    visibility=self._current_pose_result.visibility,
                    bbox=self._current_pose_result.bbox,
                    overall_confidence=self._current_pose_result.overall_confidence,
                    person_id=self._current_pose_result.person_id
                )
                logger.debug("Created corrected PoseResult")

                # Extract geometric features from corrected keypoints
                feature_extractor = GeometricFeatureExtractor()
                geometric_features = feature_extractor.extract(corrected_pose_result)
                logger.debug(f"Extracted geometric features: {len(geometric_features.feature_vector)} dimensions")

                # Get image metadata - need to load image first
                import cv2
                image = cv2.imread(self.current_image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {self.current_image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                ingestor = ImageIngestor()
                image_metadata = ingestor.extract_metadata(Path(self.current_image_path), image)
                logger.debug(f"Got image metadata: {image_metadata.original_width}x{image_metadata.original_height}")

                # Store detection to database
                logger.info("Calling storage_manager.store_detection()...")
                pose_id = self.storage_manager.store_detection(
                    image_path=Path(self.current_image_path),
                    image_metadata=image_metadata,
                    pose_result=corrected_pose_result,
                    features=geometric_features,
                    visual_features=None,  # Optional
                    fused_features=None    # Optional
                )

                # Store pose_id for future saves
                self.current_pose_id = pose_id
                success_msg = f"✓ Pose saved to database successfully!\nPose ID: {pose_id}"
                logger.info(success_msg)
                QMessageBox.information(None, "Save Successful", success_msg)

            # Case 2: Existing pose (update detection with corrections)
            elif self.current_pose_id and self._current_pose_result:
                logger.info(f"Detected EXISTING POSE (pose_id: {self.current_pose_id})")

                # Check if keypoints were modified
                if self._original_keypoints is not None and self._current_keypoints is not None:
                    keypoints_changed = not np.array_equal(self._original_keypoints, self._current_keypoints)
                    logger.debug(f"Keypoints changed: {keypoints_changed}")

                    if keypoints_changed:
                        logger.info(f"Updating detection with corrections for pose {self.current_pose_id}")

                        # Reconstruct PoseResult with corrected keypoints
                        corrected_pose_result = PoseResult(
                            keypoints=self._current_keypoints,
                            visibility=self._current_pose_result.visibility,
                            bbox=self._current_pose_result.bbox,
                            overall_confidence=self._current_pose_result.overall_confidence,
                            person_id=self._current_pose_result.person_id
                        )

                        # Re-extract geometric features from corrected keypoints
                        feature_extractor = GeometricFeatureExtractor()
                        corrected_features = feature_extractor.extract(corrected_pose_result)

                        # Update detection in database (includes storing correction statistics)
                        self.storage_manager.update_detection(
                            pose_id=self.current_pose_id,
                            corrected_keypoints=self._current_keypoints,
                            corrected_visibility=self._current_pose_result.visibility,
                            corrected_features=corrected_features,
                            correction_type='position',  # Can be extended to detect type
                            store_statistics=True
                        )
                        success_msg = f"✓ Pose updated successfully!\nPose ID: {self.current_pose_id}"
                        logger.info(success_msg)
                        QMessageBox.information(None, "Update Successful", success_msg)
                    else:
                        msg = "No keypoint changes detected - nothing to update"
                        logger.info(msg)
                        QMessageBox.information(None, "No Changes", msg)
                else:
                    msg = "Original or current keypoints not available"
                    logger.warning(msg)
            else:
                msg = f"Cannot save: Missing data\n- pose_id: {self.current_pose_id}\n- pose_result: {self._current_pose_result is not None}\n- image_path: {self.current_image_path}"
                logger.warning(msg)
                QMessageBox.warning(None, "Save Failed", msg)

        except Exception as e:
            error_msg = f"Failed to save to database:\n{str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(None, "Save Error", error_msg)
            return

        self._has_changes = False
        # Emit correction_saved signal for listeners
        # Note: saveRequested is now only emitted from the save button, not here
        self.correction_saved.emit(correction_data)
        logger.info("=" * 60)

    def _on_delete_clicked(self):
        """Handle delete person button click."""
        if not self._all_poses or len(self._all_poses) <= 1:
            logger.warning("Cannot delete: only one person in image")
            return

        logger.info(f"Requesting deletion of person {self._current_person_index}")
        self.person_delete_requested.emit(self._current_person_index)

    def _on_next_clicked(self):
        """Handle next person button click."""
        if not self._all_poses:
            return

        logger.info(f"Requesting navigation to next person")
        self.person_next_requested.emit()

    def _on_reset_clicked(self):
        """Reset all keypoints to original detection values."""
        logger.info("Reset button clicked")
        self._has_changes = False
        # Emit signal to main window to reload the current person from original pose results
        self.resetRequested.emit()
        logger.debug("Emitted resetRequested signal to main window")

    def _set_enabled(self, enabled: bool):
        """
        Enable or disable the correction panel (MainWindow compatibility).

        Args:
            enabled: True to enable, False to disable
        """
        self.setEnabled(enabled)
        logger.debug(f"Correction panel {'enabled' if enabled else 'disabled'}")

    def clear(self):
        """Clear all displayed data."""
        self._current_features = None
        self._current_pose_result = None
        self._current_keypoints = None
        self._has_changes = False

        # Clear undo/redo history
        self.undo_stack.clear()

        # self.label_input.clear()  # Removed per user request
        self.keypoint_editor.clear()

        # Clear occlusion editor
        if hasattr(self, 'occlusion_editor'):
            self.occlusion_editor.visibility = None
            self.occlusion_editor.keypoints = None

        self.feature_display.update_features(None)
        self._update_occlusion_stats(None)

        logger.debug("Cleared correction panel")
