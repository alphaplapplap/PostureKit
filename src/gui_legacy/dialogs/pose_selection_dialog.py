"""
Pose Selection Dialog for Multi-Person Save Operations
Allows users to select which poses to save from a multi-person image
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
    QPushButton, QLabel, QScrollArea, QWidget, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


class PoseSelectionDialog(QDialog):
    """
    Dialog for selecting which poses to save in a multi-person scenario.
    Each pose will be saved as an individual database entry.
    """

    poses_selected = Signal(list)  # Emits list of selected pose indices

    def __init__(self, poses, current_person_idx=None, parent=None):
        """
        Args:
            poses: List of pose objects to display
            current_person_idx: Index of currently selected pose (will be pre-checked)
            parent: Parent widget
        """
        super().__init__(parent)
        self.poses = poses
        self.current_person_idx = current_person_idx
        self.checkboxes = []
        self.selected_indices = set()

        self.setWindowTitle("Select Poses to Save")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(300)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Header
        header_label = QLabel("Select which poses to save:")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Info label
        info_label = QLabel(
            "Each selected pose will be saved as a separate entry in the database.\n"
            "Unselected poses will not be saved."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.StyledPanel)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        # Create checkbox for each pose
        for idx, pose in enumerate(self.poses):
            checkbox = self._create_pose_checkbox(idx, pose)
            self.checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

            # Pre-check current person
            if idx == self.current_person_idx:
                checkbox.setChecked(True)
                self.selected_indices.add(idx)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        # Select all / Deselect all buttons
        select_buttons_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        select_buttons_layout.addWidget(deselect_all_btn)

        select_buttons_layout.addStretch()
        layout.addLayout(select_buttons_layout)

        # Selected count label
        self.count_label = QLabel()
        self._update_count_label()
        self.count_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.count_label)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.save_btn = QPushButton("Save Selected")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self._update_save_button()
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_pose_checkbox(self, idx, pose):
        """Create checkbox widget for a pose"""
        checkbox = QCheckBox()

        # Build label text with pose info
        person_label = f"Person {idx + 1}"

        # Add confidence if available
        confidence_text = ""
        if hasattr(pose, 'overall_confidence'):
            confidence_pct = pose.overall_confidence * 100
            confidence_text = f" (Confidence: {confidence_pct:.1f}%)"

        # Add bbox info if available
        bbox_text = ""
        if hasattr(pose, 'bbox'):
            x, y, w, h = pose.bbox
            bbox_text = f" [{int(x)}, {int(y)}, {int(w)}x{int(h)}]"

        checkbox.setText(f"{person_label}{confidence_text}{bbox_text}")

        # Style current person
        if idx == self.current_person_idx:
            checkbox.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #2196F3;
                    padding: 5px;
                    background-color: #E3F2FD;
                    border-radius: 4px;
                }
            """)
        else:
            checkbox.setStyleSheet("padding: 5px;")

        # Connect signal
        checkbox.stateChanged.connect(lambda state, i=idx: self._on_checkbox_changed(i, state))

        return checkbox

    def _on_checkbox_changed(self, idx, state):
        """Handle checkbox state change"""
        if state == Qt.Checked:
            self.selected_indices.add(idx)
        else:
            self.selected_indices.discard(idx)

        self._update_count_label()
        self._update_save_button()

    def _select_all(self):
        """Select all poses"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all poses"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def _update_count_label(self):
        """Update the selected count label"""
        count = len(self.selected_indices)
        total = len(self.poses)
        self.count_label.setText(f"Selected: {count} of {total} pose(s)")

    def _update_save_button(self):
        """Enable/disable save button based on selection"""
        self.save_btn.setEnabled(len(self.selected_indices) > 0)

    def _on_save_clicked(self):
        """Handle save button click"""
        if not self.selected_indices:
            logger.warning("Save clicked but no poses selected")
            return

        selected_list = sorted(list(self.selected_indices))
        logger.info(f"User selected poses for saving: {selected_list}")
        self.poses_selected.emit(selected_list)
        self.accept()

    def get_selected_indices(self) -> List[int]:
        """Get list of selected pose indices"""
        return sorted(list(self.selected_indices))
