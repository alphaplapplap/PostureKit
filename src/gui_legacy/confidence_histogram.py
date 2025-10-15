"""
Confidence histogram widget for pose quality assessment.
"""
import numpy as np
from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from dataclasses import dataclass


@dataclass
class ConfidenceStats:
    """Statistics about keypoint confidences."""
    total_keypoints: int
    valid_count: int  # >= threshold
    low_count: int    # < 0.3
    medium_count: int # 0.3-0.6
    high_count: int   # > 0.6
    mean: float
    median: float
    min: float
    max: float


class ConfidenceHistogram(QWidget):
    """
    Histogram showing distribution of keypoint confidences.

    Features:
    - Visual distribution (bars)
    - Color-coded quality zones (low/medium/high)
    - Statistics overlay
    - Threshold indicator
    - Click to filter by confidence range
    """

    def __init__(self):
        super().__init__()

        self.keypoints = None
        self.stats = None
        self.threshold = 0.3
        self.num_bins = 20

        self.setMinimumHeight(150)
        self.setMinimumWidth(250)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("Confidence Distribution")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Stats display
        self.stats_label = QLabel("No data")
        self.stats_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self.stats_label)

        # Canvas for histogram
        self.canvas = HistogramCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(self._create_legend_item("Low", QColor(244, 67, 54)))
        legend_layout.addWidget(self._create_legend_item("Medium", QColor(255, 193, 7)))
        legend_layout.addWidget(self._create_legend_item("High", QColor(76, 175, 80)))
        legend_layout.addStretch()

        layout.addLayout(legend_layout)

    def _create_legend_item(self, text: str, color: QColor) -> QWidget:
        """Create legend item."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(4)

        # Color box
        color_box = QLabel()
        color_box.setFixedSize(12, 12)
        color_box.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
            f"border: 1px solid #666;"
        )

        # Label
        label = QLabel(text)
        label.setStyleSheet("font-size: 10px; color: #888;")

        layout.addWidget(color_box)
        layout.addWidget(label)

        return widget

    def update_keypoints(self, keypoints: np.ndarray, threshold: float = 0.3):
        """Update histogram with new keypoint data."""
        self.keypoints = keypoints
        self.threshold = threshold

        # Calculate statistics
        confidences = keypoints[:, 2]

        self.stats = ConfidenceStats(
            total_keypoints=len(keypoints),
            valid_count=int((confidences >= threshold).sum()),
            low_count=int((confidences < 0.3).sum()),
            medium_count=int(((confidences >= 0.3) & (confidences < 0.6)).sum()),
            high_count=int((confidences >= 0.6).sum()),
            mean=float(confidences.mean()),
            median=float(np.median(confidences)),
            min=float(confidences.min()),
            max=float(confidences.max())
        )

        # Update stats display
        self.stats_label.setText(
            f"Mean: {self.stats.mean:.2f} | "
            f"Low: {self.stats.low_count} | "
            f"Med: {self.stats.medium_count} | "
            f"High: {self.stats.high_count}"
        )

        # Update canvas
        self.canvas.update_data(keypoints[:, 2], threshold)
        self.canvas.update()


class HistogramCanvas(QWidget):
    """Canvas for drawing histogram."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.confidences = None
        self.threshold = 0.3
        self.num_bins = 20
        self.hist_data = None

    def update_data(self, confidences: np.ndarray, threshold: float):
        """Update histogram data."""
        self.confidences = confidences
        self.threshold = threshold

        # Calculate histogram
        self.hist_data, bin_edges = np.histogram(
            confidences,
            bins=self.num_bins,
            range=(0.0, 1.0)
        )

        self.bin_edges = bin_edges

    def paintEvent(self, event):
        """Paint histogram."""
        if self.hist_data is None:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Get drawing area
            width = self.width()
            height = self.height()
            margin = 20

            draw_width = width - 2 * margin
            draw_height = height - 2 * margin

            # Calculate bar dimensions
            bar_width = draw_width / self.num_bins
            max_count = self.hist_data.max()

            if max_count == 0:
                return

            # Draw bars
            for i, count in enumerate(self.hist_data):
                # Calculate bar position
                x = margin + i * bar_width
                bar_height = (count / max_count) * draw_height
                y = margin + draw_height - bar_height

                # Determine color based on bin position
                bin_center = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2

                if bin_center < 0.3:
                    color = QColor(244, 67, 54, 180)  # Red
                elif bin_center < 0.6:
                    color = QColor(255, 193, 7, 180)  # Yellow
                else:
                    color = QColor(76, 175, 80, 180)  # Green

                # Draw bar
                painter.fillRect(
                    int(x), int(y),
                    int(bar_width - 1), int(bar_height),
                    QBrush(color)
                )

            # Draw threshold line
            threshold_x = margin + (self.threshold * draw_width)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            painter.drawLine(
                int(threshold_x), margin,
                int(threshold_x), margin + draw_height
            )

            # Draw axes
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawLine(margin, margin + draw_height, width - margin, margin + draw_height)  # X-axis
            painter.drawLine(margin, margin, margin, margin + draw_height)  # Y-axis

            # Draw labels
            painter.setPen(QColor(150, 150, 150))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            painter.drawText(margin, height - 5, "0.0")
            painter.drawText(width - margin - 20, height - 5, "1.0")
            painter.drawText(int(threshold_x) - 15, margin - 5, f"{self.threshold:.1f}")
        finally:
            painter.end()
