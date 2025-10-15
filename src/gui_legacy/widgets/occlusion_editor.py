"""
Occlusion Editor Widget for PostureKit.
Allows marking keypoints as visible, occluded, or not present.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QButtonGroup, QRadioButton, QGroupBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OcclusionEditor(QWidget):
    """
    Widget for editing keypoint visibility flags.

    Signals:
        visibility_changed: Emitted when visibility of any keypoint changes
                           Args: (keypoint_index, new_visibility)
    """

    visibility_changed = Signal(int, int)  # keypoint_index, visibility (0/1/2)
    keypoint_selected = Signal(int)  # keypoint_index - emitted when user selects a keypoint in the list
    keypoints_selected = Signal(list)  # list of keypoint_indices - emitted when multiple keypoints selected

    # Generate names for all 133 keypoints
    @staticmethod
    def _generate_keypoint_names():
        """Generate names for all 133 RTMW-L keypoints."""
        from src.core.models import KEYPOINT_NAMES
        names = []
        for i in range(133):
            if i in KEYPOINT_NAMES:
                names.append(KEYPOINT_NAMES[i])
            elif 23 <= i <= 90:
                names.append(f"Face {i-22}")  # Face keypoints 1-68
            elif 91 <= i <= 111:
                names.append(f"Left Hand {i-90}")  # Left hand 1-21
            elif 112 <= i <= 132:
                names.append(f"Right Hand {i-111}")  # Right hand 1-21
            else:
                names.append(f"Keypoint {i}")
        return names

    KEYPOINT_NAMES = _generate_keypoint_names.__func__()

    # Show all keypoints with confidence > threshold (dynamically filtered)
    MAJOR_KEYPOINTS = list(range(133))  # All keypoints available

    def __init__(self, parent=None):
        super().__init__(parent)

        self.visibility = None  # (133,) array
        self.keypoints = None   # (133, 3) array

        self._setup_ui()

        logger.debug("OcclusionEditor initialized")

    def _setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Keypoint Visibility")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Select keypoint(s), then mark visibility.\n"
            "Shift+Click: Range | Cmd/Ctrl+Click: Toggle\n\n"
            "• Visible: Clearly visible in image\n"
            "• Occluded: Hidden by another body part\n"
            "• Not Present: Outside image bounds"
        )
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Keypoint list with multi-selection enabled
        self.keypoint_list = QListWidget()
        self.keypoint_list.setMinimumHeight(300)
        self.keypoint_list.setMaximumHeight(500)
        self.keypoint_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # Enable Shift/Ctrl multi-select
        self.keypoint_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.keypoint_list)

        # Visibility controls
        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout()

        self.button_group = QButtonGroup(self)

        self.visible_radio = QRadioButton("✓ Visible (default)")
        self.visible_radio.setStyleSheet("color: green;")
        self.button_group.addButton(self.visible_radio, 2)
        visibility_layout.addWidget(self.visible_radio)

        self.occluded_radio = QRadioButton("◐ Occluded (hidden)")
        self.occluded_radio.setStyleSheet("color: orange;")
        self.button_group.addButton(self.occluded_radio, 1)
        visibility_layout.addWidget(self.occluded_radio)

        self.not_present_radio = QRadioButton("✕ Not Present (ignore)")
        self.not_present_radio.setStyleSheet("color: red;")
        self.button_group.addButton(self.not_present_radio, 0)
        visibility_layout.addWidget(self.not_present_radio)

        self.button_group.idClicked.connect(self._on_visibility_changed)

        visibility_group.setLayout(visibility_layout)
        layout.addWidget(visibility_group)

        # Batch operations
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QVBoxLayout()

        mark_all_visible_btn = QPushButton("Mark All Visible")
        mark_all_visible_btn.clicked.connect(lambda: self._mark_all(2))
        batch_layout.addWidget(mark_all_visible_btn)

        mark_low_conf_occluded_btn = QPushButton("Mark Low Confidence as Occluded")
        mark_low_conf_occluded_btn.clicked.connect(self._mark_low_confidence_occluded)
        batch_layout.addWidget(mark_low_conf_occluded_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def load_pose(self, keypoints: np.ndarray, visibility: Optional[np.ndarray] = None):
        """
        Load pose data into editor.

        Args:
            keypoints: (133, 3) array [x, y, confidence]
            visibility: (133,) array [0, 1, 2] (defaults to 2 for all if None)
        """
        assert keypoints.shape == (133, 3), f"Expected (133, 3), got {keypoints.shape}"

        self.keypoints = keypoints.copy()

        if visibility is None:
            # Default: visible (2) if confidence > 0.3, occluded (1) otherwise
            self.visibility = np.where(keypoints[:, 2] > 0.3, 2, 1).astype(np.int32)
        else:
            assert visibility.shape == (133,), f"Expected (133,), got {visibility.shape}"
            self.visibility = visibility.copy()

        self._populate_list()
        self._update_statistics()

        logger.debug("Loaded pose into occlusion editor")

    def _populate_list(self):
        """Populate keypoint list with current data."""
        self.keypoint_list.clear()

        # Group keypoints by category for better organization
        categories = {
            'Body': range(0, 17),
            'Feet': range(17, 23),
            'Face': range(23, 91),
            'Left Hand': range(91, 112),
            'Right Hand': range(112, 133),
        }

        for category_name, indices in categories.items():
            # Add category header if any keypoint in category has conf > 0.1
            category_keypoints = [idx for idx in indices if self.keypoints[idx, 2] > 0.1]

            if category_keypoints:
                # Add category header
                header_item = QListWidgetItem(f"─── {category_name} ───")
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Not selectable
                header_item.setForeground(QColor(100, 100, 100))
                self.keypoint_list.addItem(header_item)

                # Add keypoints in this category
                for idx in category_keypoints:
                    name = self.KEYPOINT_NAMES[idx] if idx < len(self.KEYPOINT_NAMES) else f"Keypoint {idx}"
                    confidence = self.keypoints[idx, 2]
                    visibility = self.visibility[idx]

                    # Visibility symbols
                    vis_symbol = "✓" if visibility == 2 else "◐" if visibility == 1 else "✕"

                    # Format item text with visibility symbol
                    item_text = f"  {vis_symbol} {idx}: {name} (conf: {confidence:.2f})"

                    # Create item
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, idx)  # Store keypoint index

                    # Color code by confidence (matching image viewer)
                    if confidence > 0.7:
                        item.setForeground(QColor(16, 185, 129))  # Green
                    elif confidence > 0.4:
                        item.setForeground(QColor(251, 191, 36))  # Yellow/Orange
                    else:
                        item.setForeground(QColor(239, 68, 68))  # Red

                    self.keypoint_list.addItem(item)

    def _on_selection_changed(self):
        """Handle selection change (single or multiple keypoints)."""
        selected_items = self.keypoint_list.selectedItems()

        # Filter out header items and extract keypoint indices
        selected_indices = []
        for item in selected_items:
            keypoint_idx = item.data(Qt.ItemDataRole.UserRole)
            if keypoint_idx is not None:  # Skip header items
                selected_indices.append(keypoint_idx)

        if not selected_indices:
            logger.debug("No keypoints selected")
            # Disable visibility controls
            self.button_group.blockSignals(True)
            self.visible_radio.setChecked(False)
            self.occluded_radio.setChecked(False)
            self.not_present_radio.setChecked(False)
            self.button_group.blockSignals(False)
            return

        logger.debug(f"Selected {len(selected_indices)} keypoint(s): {selected_indices}")

        # Emit signals for image viewer
        if len(selected_indices) == 1:
            self.keypoint_selected.emit(selected_indices[0])
        else:
            self.keypoints_selected.emit(selected_indices)

        # Update radio buttons based on selection
        self._update_visibility_controls(selected_indices)

    def _update_visibility_controls(self, selected_indices):
        """Update visibility radio buttons based on selected keypoints."""
        if not selected_indices:
            return

        # Block signals while updating
        self.button_group.blockSignals(True)

        # Get visibility states of all selected keypoints
        visibilities = [self.visibility[idx] for idx in selected_indices]

        if len(set(visibilities)) == 1:
            # All selected keypoints have same visibility - show it
            visibility = visibilities[0]
            if visibility == 2:
                self.visible_radio.setChecked(True)
            elif visibility == 1:
                self.occluded_radio.setChecked(True)
            else:  # 0
                self.not_present_radio.setChecked(True)

            logger.debug(f"All {len(selected_indices)} selected keypoints have visibility={visibility}")
        else:
            # Mixed visibility states - show indeterminate (none checked)
            self.visible_radio.setChecked(False)
            self.occluded_radio.setChecked(False)
            self.not_present_radio.setChecked(False)

            logger.debug(f"Selected keypoints have mixed visibility: {set(visibilities)}")

        self.button_group.blockSignals(False)

    def _on_visibility_changed(self, visibility_value):
        """Handle visibility radio button change - applies to all selected keypoints."""
        selected_items = self.keypoint_list.selectedItems()

        # Filter out header items and extract keypoint indices
        selected_indices = []
        for item in selected_items:
            keypoint_idx = item.data(Qt.ItemDataRole.UserRole)
            if keypoint_idx is not None:
                selected_indices.append(keypoint_idx)

        if not selected_indices:
            logger.warning("Visibility changed but no keypoints selected - ignoring")
            return

        logger.debug(f"Applying visibility={visibility_value} to {len(selected_indices)} keypoint(s)")

        # Apply to all selected keypoints
        for item in selected_items:
            keypoint_idx = item.data(Qt.ItemDataRole.UserRole)
            if keypoint_idx is None:
                continue  # Skip header items

            old_visibility = self.visibility[keypoint_idx]

            if old_visibility == visibility_value:
                continue  # Already has this visibility

            # Update visibility
            self.visibility[keypoint_idx] = visibility_value

            # Update item text with new visibility symbol
            name = self.KEYPOINT_NAMES[keypoint_idx] if keypoint_idx < len(self.KEYPOINT_NAMES) else f"Keypoint {keypoint_idx}"
            confidence = self.keypoints[keypoint_idx, 2]
            vis_symbol = "✓" if visibility_value == 2 else "◐" if visibility_value == 1 else "✕"
            item_text = f"  {vis_symbol} {keypoint_idx}: {name} (conf: {confidence:.2f})"
            item.setText(item_text)

            # Color stays based on confidence (not visibility)
            # No color change needed as confidence doesn't change

            # Emit signal for each changed keypoint
            self.visibility_changed.emit(keypoint_idx, visibility_value)

            logger.info(f"✓ Changed visibility of keypoint {keypoint_idx}: {old_visibility} → {visibility_value}")

        self._update_statistics()

    def _mark_all(self, visibility_value: int):
        """Mark all keypoints with specified visibility."""
        self.visibility[:] = visibility_value
        self._populate_list()
        self._update_statistics()

        logger.info(f"Marked all keypoints as visibility={visibility_value}")

    def _mark_low_confidence_occluded(self):
        """Mark keypoints with confidence < 0.3 as occluded."""
        low_conf_mask = self.keypoints[:, 2] < 0.3
        self.visibility[low_conf_mask] = 1

        count = low_conf_mask.sum()
        self._populate_list()
        self._update_statistics()

        logger.info(f"Marked {count} low-confidence keypoints as occluded")

    def _update_statistics(self):
        """Update statistics display."""
        visible_count = (self.visibility == 2).sum()
        occluded_count = (self.visibility == 1).sum()
        not_present_count = (self.visibility == 0).sum()

        stats_text = (
            f"Visible: {visible_count} | "
            f"Occluded: {occluded_count} | "
            f"Not Present: {not_present_count}"
        )

        self.stats_label.setText(stats_text)

    def get_visibility(self) -> np.ndarray:
        """Get current visibility array."""
        return self.visibility.copy()

    def has_changes(self) -> bool:
        """Check if visibility has been modified from default."""
        if self.keypoints is None or self.visibility is None:
            return False

        # Compare with default (all visible if conf > 0.3)
        default_visibility = np.where(self.keypoints[:, 2] > 0.3, 2, 1)
        return not np.array_equal(self.visibility, default_visibility)
