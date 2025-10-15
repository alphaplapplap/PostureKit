"""
Occlusion visualization overlay for pose analysis.
Shows which keypoints are detected vs occluded/guessed.
"""
import numpy as np
from typing import Optional, List, Tuple
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from dataclasses import dataclass


@dataclass
class OcclusionAnalysis:
    """Analysis of pose occlusions."""
    total_keypoints: int
    visible_keypoints: List[int]
    occluded_keypoints: List[int]
    low_confidence_keypoints: List[int]
    occlusion_regions: List[str]  # Body regions with occlusions
    severity: str  # 'none', 'mild', 'moderate', 'severe'


class OcclusionVisualizer:
    """
    Visualizes occlusion patterns on pose.

    Strategies:
    1. Color-code keypoints by confidence (detected vs guessed)
    2. Highlight occluded body regions
    3. Show connection quality (solid vs dashed lines)
    4. Region-level occlusion indicators
    """

    # Occlusion severity thresholds
    SEVERITY_THRESHOLDS = {
        'none': (0, 0.05),      # <5% occluded
        'mild': (0.05, 0.20),   # 5-20% occluded
        'moderate': (0.20, 0.40),  # 20-40% occluded
        'severe': (0.40, 1.0)   # >40% occluded
    }

    def analyze_occlusion(
        self,
        keypoints: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> OcclusionAnalysis:
        """
        Analyze occlusion patterns in pose.

        Args:
            keypoints: Keypoints array (133, 3)
            confidence_threshold: Threshold for visible keypoint

        Returns:
            OcclusionAnalysis with detailed occlusion info
        """
        confidences = keypoints[:, 2]

        # Identify visible vs occluded
        visible = np.where(confidences >= confidence_threshold)[0].tolist()
        occluded = np.where(confidences < confidence_threshold)[0].tolist()
        low_conf = np.where((confidences >= confidence_threshold) &
                           (confidences < 0.6))[0].tolist()

        # Analyze regions
        occluded_regions = self._identify_occluded_regions(keypoints, confidence_threshold)

        # Calculate severity
        occlusion_rate = len(occluded) / len(keypoints)
        severity = self._get_severity(occlusion_rate)

        return OcclusionAnalysis(
            total_keypoints=len(keypoints),
            visible_keypoints=visible,
            occluded_keypoints=occluded,
            low_confidence_keypoints=low_conf,
            occlusion_regions=occluded_regions,
            severity=severity
        )

    def _identify_occluded_regions(
        self,
        keypoints: np.ndarray,
        threshold: float
    ) -> List[str]:
        """Identify which body regions are occluded."""
        regions = []

        # Check body regions
        region_definitions = {
            'Head': [0, 1, 2, 3, 4],
            'Left Arm': [5, 7, 9],
            'Right Arm': [6, 8, 10],
            'Torso': [5, 6, 11, 12],
            'Left Leg': [11, 13, 15],
            'Right Leg': [12, 14, 16],
            'Left Hand': list(range(91, 112)),
            'Right Hand': list(range(112, 133)),
            'Face': list(range(23, 91)),
            'Feet': [17, 18, 19, 20, 21, 22]
        }

        for region_name, kp_indices in region_definitions.items():
            # Check if majority of keypoints in region are occluded
            valid_indices = [i for i in kp_indices if i < len(keypoints)]
            if not valid_indices:
                continue

            region_confs = keypoints[valid_indices, 2]
            occluded_count = (region_confs < threshold).sum()

            if occluded_count / len(valid_indices) > 0.5:
                regions.append(region_name)

        return regions

    def _get_severity(self, occlusion_rate: float) -> str:
        """Determine occlusion severity."""
        for severity, (min_rate, max_rate) in self.SEVERITY_THRESHOLDS.items():
            if min_rate <= occlusion_rate < max_rate:
                return severity
        return 'severe'

    def render_occlusion_overlay(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        analysis: OcclusionAnalysis
    ) -> np.ndarray:
        """
        Render occlusion indicators on image.

        Strategy:
        - Occluded keypoints: Red X marks
        - Low confidence: Yellow circles
        - Visible: Normal rendering
        - Region labels for occluded areas
        """
        import cv2

        overlay = image.copy()

        # Draw occluded keypoints with X marks
        for kp_idx in analysis.occluded_keypoints:
            if kp_idx >= len(keypoints):
                continue

            kp = keypoints[kp_idx]
            center = (int(kp[0]), int(kp[1]))

            # Red X
            cv2.line(overlay,
                    (center[0] - 5, center[1] - 5),
                    (center[0] + 5, center[1] + 5),
                    (0, 0, 255), 2)
            cv2.line(overlay,
                    (center[0] + 5, center[1] - 5),
                    (center[0] - 5, center[1] + 5),
                    (0, 0, 255), 2)

        # Draw low confidence with yellow outline
        for kp_idx in analysis.low_confidence_keypoints:
            if kp_idx >= len(keypoints):
                continue

            kp = keypoints[kp_idx]
            center = (int(kp[0]), int(kp[1]))

            cv2.circle(overlay, center, 8, (0, 255, 255), 2)

        # Add region labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        for region in analysis.occlusion_regions:
            cv2.putText(overlay, f"Occluded: {region}",
                       (10, y_offset), font, 0.6, (0, 0, 255), 2)
            y_offset += 25

        return overlay


class OcclusionPanel(QWidget):
    """
    UI panel for occlusion analysis and visualization.

    Signals:
        overlay_toggled: Emitted when overlay visibility changes
        region_selected: Emitted when region clicked for focus
    """

    overlay_toggled = Signal(bool)
    region_selected = Signal(str)

    def __init__(self):
        super().__init__()

        self.analysis = None
        self.visualizer = OcclusionVisualizer()

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Occlusion Analysis")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Show overlay toggle
        self.show_overlay_cb = QCheckBox("Show Occlusion Overlay")
        self.show_overlay_cb.setChecked(True)
        self.show_overlay_cb.toggled.connect(self.overlay_toggled.emit)
        layout.addWidget(self.show_overlay_cb)

        # Severity indicator
        self.severity_label = QLabel("Severity: Unknown")
        layout.addWidget(self.severity_label)

        # Statistics
        self.stats_label = QLabel("No data")
        self.stats_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self.stats_label)

        # Occluded regions list
        regions_label = QLabel("Occluded Regions:")
        regions_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(regions_label)

        self.regions_label = QLabel("None")
        self.regions_label.setWordWrap(True)
        self.regions_label.setStyleSheet("color: #E53935;")
        layout.addWidget(self.regions_label)

        # Action: Focus on occluded regions
        self.focus_btn = QPushButton("Focus on Occluded Areas")
        self.focus_btn.clicked.connect(self._on_focus_clicked)
        self.focus_btn.setEnabled(False)
        layout.addWidget(self.focus_btn)

        layout.addStretch()

    def update_analysis(self, keypoints: np.ndarray, threshold: float = 0.3):
        """Update occlusion analysis."""
        self.analysis = self.visualizer.analyze_occlusion(keypoints, threshold)

        # Update severity
        severity_colors = {
            'none': '#4CAF50',
            'mild': '#FFC107',
            'moderate': '#FF9800',
            'severe': '#E53935'
        }

        color = severity_colors.get(self.analysis.severity, '#888')
        self.severity_label.setText(
            f"Severity: <span style='color: {color}; font-weight: bold;'>"
            f"{self.analysis.severity.upper()}</span>"
        )

        # Update stats
        visible_pct = len(self.analysis.visible_keypoints) / self.analysis.total_keypoints * 100
        occluded_pct = len(self.analysis.occluded_keypoints) / self.analysis.total_keypoints * 100

        self.stats_label.setText(
            f"Visible: {len(self.analysis.visible_keypoints)} ({visible_pct:.1f}%) | "
            f"Occluded: {len(self.analysis.occluded_keypoints)} ({occluded_pct:.1f}%)"
        )

        # Update regions
        if self.analysis.occluded_regions:
            self.regions_label.setText(", ".join(self.analysis.occluded_regions))
            self.focus_btn.setEnabled(True)
        else:
            self.regions_label.setText("None - Full visibility ✓")
            self.regions_label.setStyleSheet("color: #4CAF50;")
            self.focus_btn.setEnabled(False)

    def _on_focus_clicked(self):
        """Handle focus button click."""
        if self.analysis and self.analysis.occluded_regions:
            # Emit first occluded region for focus
            self.region_selected.emit(self.analysis.occluded_regions[0])
