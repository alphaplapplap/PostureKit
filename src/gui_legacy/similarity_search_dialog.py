"""
Similarity Search Dialog - Visual pose similarity search interface.

Provides drag-and-drop reference image selection and gallery view of similar poses.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID
import cv2

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QSlider, QGroupBox, QScrollArea, QWidget, QGridLayout,
    QProgressBar, QFrame, QSizePolicy, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread, QSize, QMimeData
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QPainter, QColor

from src.intelligence.similarity_engine import SimilarityEngine
from src.core.pose_detector import PoseDetector
from src.core.geometric_feature_extractor import GeometricFeatureExtractor

logger = logging.getLogger(__name__)


class IndexBuildWorker(QThread):
    """Background worker for building FAISS index."""
    
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, similarity_engine: SimilarityEngine, force_rebuild: bool = False):
        super().__init__()
        self.engine = similarity_engine
        self.force_rebuild = force_rebuild
    
    def run(self):
        """Build index in background thread."""
        try:
            def progress_callback(current, total, message):
                self.progress.emit(current, total, message)
            
            self.engine.build_index(
                force_rebuild=self.force_rebuild,
                progress_callback=progress_callback
            )
            
            stats = self.engine.get_statistics()
            message = f"Index ready with {stats['total_poses']} poses"
            self.finished.emit(True, message)
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            self.finished.emit(False, f"Error: {str(e)}")


class SearchWorker(QThread):
    """Background worker for performing similarity search."""

    finished = Signal(list)  # results
    error = Signal(str)

    def __init__(self, engine: SimilarityEngine, pose_id: Optional[UUID],
                 feature_vector: Optional[List[float]], k: int, min_confidence: float,
                 min_feature_confidence: float = 0.5,
                 min_valid_overlap: int = 20):
        super().__init__()
        self.engine = engine
        self.pose_id = pose_id
        self.feature_vector = feature_vector
        self.k = k
        self.min_confidence = min_confidence
        self.min_feature_confidence = min_feature_confidence
        self.min_valid_overlap = min_valid_overlap

    def run(self):
        """Perform search in background thread."""
        try:
            if self.pose_id:
                results = self.engine.search_by_pose_id(
                    self.pose_id,
                    k=self.k,
                    min_confidence=self.min_confidence,
                    min_feature_confidence=self.min_feature_confidence,
                    min_valid_overlap=self.min_valid_overlap
                )
            elif self.feature_vector is not None:
                results = self.engine.search_by_feature(
                    self.feature_vector,
                    k=self.k,
                    min_confidence=self.min_confidence,
                    min_feature_confidence=self.min_feature_confidence,
                    min_valid_overlap=self.min_valid_overlap
                )
            else:
                self.error.emit("No search query provided")
                return

            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            self.error.emit(str(e))


class DropZoneWidget(QLabel):
    """Drag-and-drop zone for reference images."""
    
    image_dropped = Signal(str)  # file_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.setLineWidth(2)
        
        # Style
        self.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #999;
                border-radius: 8px;
                color: #666;
                font-size: 14pt;
                padding: 20px;
            }
            QLabel:hover {
                background-color: #e8e8e8;
                border-color: #666;
            }
        """)
        
        self.default_text = "Drag & Drop Reference Image\n\n(or click Browse)"
        self.setText(self.default_text)
        self.current_image = None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events with image files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle dropped image."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.set_image(file_path)
            self.image_dropped.emit(file_path)
    
    def set_image(self, file_path: str):
        """Display thumbnail of dropped image."""
        self.current_image = file_path
        pixmap = QPixmap(file_path)
        
        if not pixmap.isNull():
            # Scale to fit while maintaining aspect ratio
            scaled = pixmap.scaled(
                self.width() - 40, 
                self.height() - 40,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        else:
            self.setText("Invalid image file")
    
    def clear_image(self):
        """Reset to default state."""
        self.current_image = None
        self.clear()
        self.setText(self.default_text)


class ResultThumbnail(QFrame):
    """Single result thumbnail with metadata."""
    
    clicked = Signal(str, str)  # pose_id, image_path
    
    def __init__(self, result: Dict, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
    
    def setup_ui(self):
        """Build thumbnail UI."""
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(200, 240)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Image
        image_label = QLabel()
        image_label.setFixedSize(190, 190)
        image_label.setScaledContents(True)
        image_label.setAlignment(Qt.AlignCenter)
        
        pixmap = QPixmap(self.result['image_path'])
        if not pixmap.isNull():
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("Error loading")
        
        layout.addWidget(image_label)
        
        # Metadata
        rank = self.result['rank']
        similarity = self.result['similarity_score']
        confidence = self.result['detection_confidence']
        
        info_text = f"#{rank} | Similarity: {similarity:.2%}\nConfidence: {confidence:.2f}"
        
        if self.result['is_corrected']:
            info_text += " ✓"
        
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 10pt; color: #333;")
        info_label.setWordWrap(True)
        
        layout.addWidget(info_label)
    
    def mousePressEvent(self, event):
        """Emit click signal."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.result['pose_id'], self.result['image_path'])


class SimilaritySearchDialog(QDialog):
    """Main similarity search dialog."""
    
    pose_selected = Signal(str, str)  # pose_id, image_path
    
    def __init__(self, similarity_engine: SimilarityEngine, pose_detector: PoseDetector,
                 storage_manager, parent=None):
        super().__init__(parent)
        self.engine = similarity_engine
        self.detector = pose_detector
        self.storage = storage_manager
        self.feature_extractor = GeometricFeatureExtractor()
        
        self.current_results: List[Dict] = []
        self.search_worker: Optional[SearchWorker] = None
        self.index_worker: Optional[IndexBuildWorker] = None
        
        self.setup_ui()
        self.check_index_status()
    
    def setup_ui(self):
        """Build dialog UI."""
        self.setWindowTitle("Find Similar Poses")
        self.setMinimumSize(1000, 700)
        
        main_layout = QVBoxLayout(self)
        
        # Index status section
        index_group = self.create_index_section()
        main_layout.addWidget(index_group)
        
        # Search section
        search_group = self.create_search_section()
        main_layout.addWidget(search_group)
        
        # Results section
        results_group = self.create_results_section()
        main_layout.addWidget(results_group, stretch=1)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn, alignment=Qt.AlignRight)
    
    def create_index_section(self) -> QGroupBox:
        """Create index management section."""
        group = QGroupBox("Index Status")
        layout = QVBoxLayout(group)
        
        # Status label
        self.index_status_label = QLabel("Checking index...")
        layout.addWidget(self.index_status_label)
        
        # Progress bar
        self.index_progress = QProgressBar()
        self.index_progress.setVisible(False)
        layout.addWidget(self.index_progress)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.build_index_btn = QPushButton("Build Index")
        self.build_index_btn.clicked.connect(lambda: self.build_index(force_rebuild=False))
        btn_layout.addWidget(self.build_index_btn)
        
        self.rebuild_index_btn = QPushButton("Rebuild Index")
        self.rebuild_index_btn.clicked.connect(lambda: self.build_index(force_rebuild=True))
        btn_layout.addWidget(self.rebuild_index_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return group
    
    def create_search_section(self) -> QGroupBox:
        """Create search configuration section."""
        group = QGroupBox("Search Configuration")
        layout = QVBoxLayout(group)
        
        # Drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.image_dropped.connect(self.on_image_dropped)
        layout.addWidget(self.drop_zone)
        
        # Browse button
        browse_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse for Image...")
        browse_btn.clicked.connect(self.browse_for_image)
        browse_layout.addWidget(browse_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_search)
        browse_layout.addWidget(clear_btn)
        
        browse_layout.addStretch()
        layout.addLayout(browse_layout)
        
        # Search parameters
        params_layout = QHBoxLayout()
        
        # Number of results
        params_layout.addWidget(QLabel("Results:"))
        self.results_spin = QSpinBox()
        self.results_spin.setRange(1, 100)
        self.results_spin.setValue(20)
        params_layout.addWidget(self.results_spin)
        
        # Confidence threshold
        params_layout.addWidget(QLabel("Min Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(30)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        params_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.30")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )
        params_layout.addWidget(self.confidence_label)

        # Feature confidence threshold
        params_layout.addWidget(QLabel("Feature Conf:"))
        self.feature_conf_spin = QSpinBox()
        self.feature_conf_spin.setRange(0, 100)
        self.feature_conf_spin.setValue(50)
        self.feature_conf_spin.setSuffix("%")
        self.feature_conf_spin.setMaximumWidth(70)
        self.feature_conf_spin.setToolTip("Minimum confidence for feature dimensions")
        params_layout.addWidget(self.feature_conf_spin)

        # Minimum valid overlap
        params_layout.addWidget(QLabel("Min Dims:"))
        self.min_dims_spin = QSpinBox()
        self.min_dims_spin.setRange(10, 52)
        self.min_dims_spin.setValue(20)
        self.min_dims_spin.setMaximumWidth(60)
        self.min_dims_spin.setToolTip("Minimum valid dimensions (out of 52)")
        params_layout.addWidget(self.min_dims_spin)

        # Only corrected
        self.corrected_only_check = QCheckBox("Corrected Only")
        params_layout.addWidget(self.corrected_only_check)

        params_layout.addStretch()
        layout.addLayout(params_layout)
        
        # Search button
        self.search_btn = QPushButton("Search Similar Poses")
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self.perform_search)
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-size: 12pt;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        layout.addWidget(self.search_btn)
        
        return group
    
    def create_results_section(self) -> QGroupBox:
        """Create results display section."""
        group = QGroupBox("Similar Poses")
        layout = QVBoxLayout(group)
        
        # Results info label
        self.results_info_label = QLabel("No search performed yet")
        self.results_info_label.setStyleSheet("font-size: 11pt; color: #666;")
        layout.addWidget(self.results_info_label)
        
        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.results_container = QWidget()
        self.results_layout = QGridLayout(self.results_container)
        self.results_layout.setSpacing(10)
        
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll)
        
        return group
    
    def check_index_status(self):
        """Check and display index status."""
        stats = self.engine.get_statistics()
        
        if stats['index_exists'] and stats['total_poses'] > 0:
            self.index_status_label.setText(
                f"✓ Index ready with {stats['total_poses']} poses"
            )
            self.index_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.build_index_btn.setEnabled(False)
            self.rebuild_index_btn.setEnabled(True)
        else:
            self.index_status_label.setText(
                "⚠ No index found - build index to enable search"
            )
            self.index_status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.build_index_btn.setEnabled(True)
            self.rebuild_index_btn.setEnabled(False)
    
    def build_index(self, force_rebuild: bool = False):
        """Build FAISS index in background."""
        self.build_index_btn.setEnabled(False)
        self.rebuild_index_btn.setEnabled(False)
        self.index_progress.setVisible(True)
        self.index_progress.setRange(0, 3)
        
        self.index_worker = IndexBuildWorker(self.engine, force_rebuild)
        self.index_worker.progress.connect(self.on_index_progress)
        self.index_worker.finished.connect(self.on_index_finished)
        self.index_worker.start()
    
    def on_index_progress(self, current: int, total: int, message: str):
        """Update index building progress."""
        self.index_progress.setValue(current)
        self.index_status_label.setText(message)
    
    def on_index_finished(self, success: bool, message: str):
        """Handle index build completion."""
        self.index_progress.setVisible(False)
        self.index_status_label.setText(message)
        
        if success:
            self.index_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.rebuild_index_btn.setEnabled(True)
        else:
            self.index_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.build_index_btn.setEnabled(True)

        if self.index_worker:
            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.index_worker.progress.disconnect()
                self.index_worker.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.index_worker.deleteLater()
        self.index_worker = None
    
    def browse_for_image(self):
        """Open file dialog to select reference image."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.drop_zone.set_image(file_path)
            self.on_image_dropped(file_path)
    
    def on_image_dropped(self, file_path: str):
        """Handle reference image selection."""
        logger.info(f"Reference image: {file_path}")
        self.search_btn.setEnabled(True)
        self.results_info_label.setText(f"Ready to search with: {Path(file_path).name}")
    
    def clear_search(self):
        """Clear current search."""
        self.drop_zone.clear_image()
        self.search_btn.setEnabled(False)
        self.clear_results()
        self.results_info_label.setText("No search performed yet")
    
    def perform_search(self):
        """Execute similarity search."""
        if not self.drop_zone.current_image:
            return
        
        # Check index
        if not self.engine.index or self.engine.index.ntotal == 0:
            self.results_info_label.setText("⚠ Please build index first")
            return
        
        self.search_btn.setEnabled(False)
        self.results_info_label.setText("Detecting pose...")
        
        try:
            # Load reference image
            image_path = self.drop_zone.current_image
            image = cv2.imread(image_path)
            if image is None:
                self.results_info_label.setText("Failed to load reference image")
                self.search_btn.setEnabled(True)
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect pose in reference image
            detections = self.detector.detect(image)

            if not detections:
                self.results_info_label.setText("No pose detected in reference image")
                self.search_btn.setEnabled(True)
                return
            
            # Use first person detected
            pose = detections[0]
            
            # Extract geometric features
            self.results_info_label.setText("Extracting features...")
            features = self.feature_extractor.extract(pose)
            feature_vector = features.feature_vector
            
            if feature_vector is None:
                self.results_info_label.setText("Failed to extract pose features")
                self.search_btn.setEnabled(True)
                return
            
            # Perform search
            self.results_info_label.setText("Searching...")
            k = self.results_spin.value()
            min_conf = self.confidence_slider.value() / 100.0
            min_feat_conf = self.feature_conf_spin.value() / 100.0
            min_dims = self.min_dims_spin.value()

            self.search_worker = SearchWorker(
                self.engine, None, feature_vector, k, min_conf,
                min_feature_confidence=min_feat_conf,
                min_valid_overlap=min_dims
            )
            self.search_worker.finished.connect(self.on_search_finished)
            self.search_worker.error.connect(self.on_search_error)
            self.search_worker.start()
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            self.results_info_label.setText(f"Error: {str(e)}")
            self.search_btn.setEnabled(True)
    
    def on_search_finished(self, results: List[Dict]):
        """Display search results."""
        self.current_results = results
        self.search_btn.setEnabled(True)
        
        # Filter if needed
        if self.corrected_only_check.isChecked():
            results = [r for r in results if r['is_corrected']]
        
        self.results_info_label.setText(f"Found {len(results)} similar poses")

        # Display results
        self.display_results(results)

        if self.search_worker:
            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.search_worker.finished.disconnect()
                self.search_worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.search_worker.deleteLater()
        self.search_worker = None
    
    def on_search_error(self, error: str):
        """Handle search error."""
        self.results_info_label.setText(f"Search failed: {error}")
        self.search_btn.setEnabled(True)
        if self.search_worker:
            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.search_worker.finished.disconnect()
                self.search_worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.search_worker.deleteLater()
        self.search_worker = None
    
    def display_results(self, results: List[Dict]):
        """Populate results grid with thumbnails."""
        self.clear_results()
        
        if not results:
            self.results_info_label.setText("No similar poses found")
            return
        
        # Add thumbnails in grid (4 columns)
        cols = 4
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            
            thumbnail = ResultThumbnail(result)
            thumbnail.clicked.connect(self.on_result_clicked)
            
            self.results_layout.addWidget(thumbnail, row, col)
        
        # Add stretch to push thumbnails to top
        self.results_layout.setRowStretch(len(results) // cols + 1, 1)
    
    def clear_results(self):
        """Clear results display."""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def on_result_clicked(self, pose_id: str, image_path: str):
        """Handle result thumbnail click."""
        logger.info(f"Result clicked: {pose_id}")
        self.pose_selected.emit(pose_id, image_path)

    def closeEvent(self, event):
        """Clean up threads before closing."""
        # Terminate search worker if running
        if hasattr(self, "search_worker") and self.search_worker and self.search_worker.isRunning():
            self.search_worker.terminate()
            self.search_worker.wait(2000)

            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.search_worker.finished.disconnect()
                self.search_worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.search_worker.deleteLater()

        super().closeEvent(event)

