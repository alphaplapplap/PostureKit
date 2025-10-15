"""
Live Similarity Panel - Dockable widget showing similar poses in real-time.

This panel integrates into the MainWindow and automatically updates to show
similar poses whenever the current image changes, enabling continuous
pose-aware navigation through your photo collection.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame, QSpinBox, QSlider, QCheckBox, QComboBox,
    QToolButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSize
from PySide6.QtGui import QPixmap, QIcon

from src.intelligence.similarity_engine import SimilarityEngine
from src.storage.models import PoseDetection, GeometricFeatures

logger = logging.getLogger(__name__)


class QuickSearchWorker(QThread):
    """Background worker for non-blocking similarity search."""

    finished = Signal(list)  # results
    error = Signal(str)

    def __init__(self, engine: SimilarityEngine, pose_id: UUID,
                 k: int, min_confidence: float,
                 min_feature_confidence: float = 0.5,
                 min_valid_overlap: int = 20):
        super().__init__()
        self.engine = engine
        self.pose_id = pose_id
        self.k = k
        self.min_confidence = min_confidence
        self.min_feature_confidence = min_feature_confidence
        self.min_valid_overlap = min_valid_overlap

    def run(self):
        """Perform search in background."""
        try:
            results = self.engine.search_by_pose_id(
                self.pose_id,
                k=self.k,
                exclude_self=True,
                min_confidence=self.min_confidence,
                min_feature_confidence=self.min_feature_confidence,
                min_valid_overlap=self.min_valid_overlap
            )
            self.finished.emit(results)
        except Exception as e:
            logger.error(f"Quick search failed: {e}", exc_info=True)
            self.error.emit(str(e))


class CompactResultCard(QFrame):
    """Compact thumbnail card for similarity results."""

    clicked = Signal(str, str)  # pose_id, image_path

    def __init__(self, result: Dict, show_details: bool = True, parent=None):
        super().__init__(parent)
        self.result = result
        self.show_details = show_details
        self.setup_ui()

    def setup_ui(self):
        """Build compact card UI."""
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)

        # Adjust size based on detail level
        if self.show_details:
            self.setFixedSize(140, 170)
            img_size = 130
        else:
            self.setFixedSize(100, 120)
            img_size = 90

        # Hover effect
        self.setStyleSheet("""
            CompactResultCard {
                background-color: white;
            }
            CompactResultCard:hover {
                background-color: #e8f4ff;
                border: 2px solid #0066cc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Image
        image_label = QLabel()
        image_label.setFixedSize(img_size, img_size)
        image_label.setScaledContents(True)
        image_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(self.result['image_path'])
        if not pixmap.isNull():
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("Error")
            image_label.setStyleSheet("font-size: 8pt; color: #999;")

        layout.addWidget(image_label)

        if self.show_details:
            # Metadata
            similarity = self.result['similarity_score']

            info_text = f"{similarity:.0%}"
            if self.result['is_corrected']:
                info_text += " ✓"

            # Add valid dimensions count if confidence-aware search was used
            if 'valid_dimensions' in self.result:
                valid_dims = self.result['valid_dimensions']
                info_text += f"\n{valid_dims}/52"

            info_label = QLabel(info_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 9pt; font-weight: bold; color: #0066cc;")
            layout.addWidget(info_label)

    def mousePressEvent(self, event):
        """Emit click signal."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.result['pose_id'], self.result['image_path'])


class SimilarityPanel(QWidget):
    """Dockable panel showing similar poses to current image."""

    pose_selected = Signal(str, str)  # pose_id, image_path
    search_requested = Signal()  # Open full search dialog

    def __init__(self, similarity_engine: SimilarityEngine,
                 storage_manager, parent=None):
        super().__init__(parent)
        self.engine = similarity_engine
        self.storage = storage_manager

        self.current_pose_id: Optional[UUID] = None
        self.current_results: List[Dict] = []
        self.search_worker: Optional[QuickSearchWorker] = None

        # Debounce timer for auto-refresh
        self.refresh_timer = QTimer()
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self._execute_search)

        self.setup_ui()

    def setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel("Similar Poses")
        title_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Refresh button
        self.refresh_btn = QToolButton()
        self.refresh_btn.setText("🔄")
        self.refresh_btn.setToolTip("Refresh similar poses")
        self.refresh_btn.clicked.connect(self.refresh_results)
        header_layout.addWidget(self.refresh_btn)

        # Full search button
        full_search_btn = QToolButton()
        full_search_btn.setText("🔍")
        full_search_btn.setToolTip("Open full search dialog")
        full_search_btn.clicked.connect(self.search_requested.emit)
        header_layout.addWidget(full_search_btn)

        layout.addLayout(header_layout)

        # Settings
        settings_layout = self.create_settings_section()
        layout.addLayout(settings_layout)

        # Status label
        self.status_label = QLabel("No pose loaded")
        self.status_label.setStyleSheet("font-size: 9pt; color: #666;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Results scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(5)
        self.results_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, stretch=1)

        # Footer actions
        footer_layout = QHBoxLayout()

        self.auto_refresh_check = QCheckBox("Auto-refresh")
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.setToolTip("Automatically search for similar poses when image changes")
        footer_layout.addWidget(self.auto_refresh_check)

        footer_layout.addStretch()

        self.show_details_check = QCheckBox("Details")
        self.show_details_check.setChecked(True)
        self.show_details_check.setToolTip("Show similarity scores on thumbnails")
        self.show_details_check.toggled.connect(self._refresh_display)
        footer_layout.addWidget(self.show_details_check)

        layout.addLayout(footer_layout)

    def create_settings_section(self) -> QHBoxLayout:
        """Create compact settings controls."""
        layout = QVBoxLayout()
        layout.setSpacing(3)

        # Row 1: Results count and overall confidence
        row1 = QHBoxLayout()
        row1.setSpacing(5)

        row1.addWidget(QLabel("Show:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(3, 20)
        self.count_spin.setValue(8)
        self.count_spin.setMaximumWidth(60)
        self.count_spin.setToolTip("Number of similar poses to show")
        self.count_spin.valueChanged.connect(self._on_settings_changed)
        row1.addWidget(self.count_spin)

        row1.addWidget(QLabel("Min Pose:"))
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(30)
        self.confidence_spin.setSuffix("%")
        self.confidence_spin.setMaximumWidth(70)
        self.confidence_spin.setToolTip("Minimum overall pose confidence")
        self.confidence_spin.valueChanged.connect(self._on_settings_changed)
        row1.addWidget(self.confidence_spin)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Confidence-aware matching parameters
        row2 = QHBoxLayout()
        row2.setSpacing(5)

        row2.addWidget(QLabel("Feature Conf:"))
        self.feature_conf_spin = QSpinBox()
        self.feature_conf_spin.setRange(0, 100)
        self.feature_conf_spin.setValue(50)
        self.feature_conf_spin.setSuffix("%")
        self.feature_conf_spin.setMaximumWidth(70)
        self.feature_conf_spin.setToolTip("Minimum confidence for feature dimensions to be used in matching")
        self.feature_conf_spin.valueChanged.connect(self._on_settings_changed)
        row2.addWidget(self.feature_conf_spin)

        row2.addWidget(QLabel("Min Dims:"))
        self.min_dims_spin = QSpinBox()
        self.min_dims_spin.setRange(10, 52)
        self.min_dims_spin.setValue(20)
        self.min_dims_spin.setMaximumWidth(60)
        self.min_dims_spin.setToolTip("Minimum valid dimensions required for matching (out of 52)")
        self.min_dims_spin.valueChanged.connect(self._on_settings_changed)
        row2.addWidget(self.min_dims_spin)

        row2.addStretch()
        layout.addLayout(row2)

        return layout

    def set_current_pose(self, pose_id: UUID, debounce_ms: int = 500):
        """
        Set the current pose and trigger similarity search.

        Args:
            pose_id: UUID of current pose
            debounce_ms: Delay before executing search (for rapid navigation)
        """
        self.current_pose_id = pose_id

        if not self.auto_refresh_check.isChecked():
            self.status_label.setText("Auto-refresh disabled - click refresh to search")
            return

        # Check if index exists
        stats = self.engine.get_statistics()
        if not stats['index_exists'] or stats['total_poses'] == 0:
            self.status_label.setText("⚠ No search index - build index first")
            self.clear_results()
            return

        # Debounce for smooth navigation
        self.refresh_timer.stop()
        self.refresh_timer.start(debounce_ms)
        self.status_label.setText("Searching...")

    def _execute_search(self):
        """Execute the similarity search."""
        if not self.current_pose_id:
            return

        # Cancel any running search
        if self.search_worker and self.search_worker.isRunning():
            self.search_worker.terminate()
            self.search_worker.wait()
            self.search_worker.deleteLater()

        # Start new search with confidence-aware parameters
        k = self.count_spin.value()
        min_conf = self.confidence_spin.value() / 100.0
        min_feat_conf = self.feature_conf_spin.value() / 100.0
        min_dims = self.min_dims_spin.value()

        self.search_worker = QuickSearchWorker(
            self.engine, self.current_pose_id, k, min_conf,
            min_feature_confidence=min_feat_conf,
            min_valid_overlap=min_dims
        )
        self.search_worker.finished.connect(self._on_search_finished)
        self.search_worker.error.connect(self._on_search_error)
        self.search_worker.start()

    def _on_search_finished(self, results: List[Dict]):
        """Handle search completion."""
        self.current_results = results

        if not results:
            self.status_label.setText("No similar poses found")
            self.clear_results()
        else:
            self.status_label.setText(f"Found {len(results)} similar poses")
            self.display_results(results)

        if self.search_worker:
            self.search_worker.deleteLater()
        self.search_worker = None

    def _on_search_error(self, error: str):
        """Handle search error."""
        self.status_label.setText(f"Search error: {error}")
        self.clear_results()
        if self.search_worker:
            self.search_worker.deleteLater()
        self.search_worker = None

    def display_results(self, results: List[Dict]):
        """Display results as compact cards."""
        self.clear_results()

        show_details = self.show_details_check.isChecked()

        for result in results:
            card = CompactResultCard(result, show_details=show_details)
            card.clicked.connect(self._on_result_clicked)
            self.results_layout.addWidget(card)

        # Add stretch at bottom
        self.results_layout.addStretch()

    def clear_results(self):
        """Clear all result cards."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_result_clicked(self, pose_id: str, image_path: str):
        """Handle result card click."""
        self.pose_selected.emit(pose_id, image_path)

    def refresh_results(self):
        """Manually refresh results."""
        if self.current_pose_id:
            self._execute_search()
        else:
            self.status_label.setText("No pose loaded")

    def _on_settings_changed(self):
        """Handle settings change."""
        if self.current_pose_id and self.auto_refresh_check.isChecked():
            self.refresh_results()

    def _refresh_display(self):
        """Refresh display without re-searching."""
        if self.current_results:
            self.display_results(self.current_results)

    def get_state(self) -> Dict:
        """Get panel state for persistence."""
        return {
            'count': self.count_spin.value(),
            'confidence': self.confidence_spin.value(),
            'feature_confidence': self.feature_conf_spin.value(),
            'min_dims': self.min_dims_spin.value(),
            'auto_refresh': self.auto_refresh_check.isChecked(),
            'show_details': self.show_details_check.isChecked()
        }

    def set_state(self, state: Dict):
        """Restore panel state."""
        if 'count' in state:
            self.count_spin.setValue(state['count'])
        if 'confidence' in state:
            self.confidence_spin.setValue(state['confidence'])
        if 'feature_confidence' in state:
            self.feature_conf_spin.setValue(state['feature_confidence'])
        if 'min_dims' in state:
            self.min_dims_spin.setValue(state['min_dims'])
        if 'auto_refresh' in state:
            self.auto_refresh_check.setChecked(state['auto_refresh'])
        if 'show_details' in state:
            self.show_details_check.setChecked(state['show_details'])
