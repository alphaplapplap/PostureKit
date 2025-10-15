"""
Directory Indexing Dialog for PostureKit

Integrates into MainWindow to provide batch directory indexing functionality.
Detects poses in all images within a selected directory and builds FAISS search index.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4
import cv2

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox, QCheckBox,
    QSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread

from src.core.pose_detector import PoseDetector
from src.core.geometric_feature_extractor import GeometricFeatureExtractor
from src.core.image_ingestor import ImageMetadata
from src.storage.storage_manager import StorageManager
from src.intelligence.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)


class DirectoryIndexWorker(QThread):
    """Background worker for indexing directory of images."""

    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(bool, str, int)  # success, message, pose_count

    def __init__(self, directory: Path, recursive: bool,
                 pose_detector: PoseDetector,
                 feature_extractor: GeometricFeatureExtractor,
                 storage_manager: StorageManager,
                 min_confidence: float = 0.3):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.detector = pose_detector
        self.feature_extractor = feature_extractor
        self.storage = storage_manager
        self.min_confidence = min_confidence
        self.pose_count = 0
        self.image_count = 0
        self.failed_count = 0

    def run(self):
        """Index all images in directory using streaming for memory efficiency."""
        try:
            # Find all images using generator (memory efficient)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

            # Generator function for streaming file discovery
            def stream_image_files():
                if self.recursive:
                    for ext in image_extensions:
                        yield from self.directory.glob(f'**/*{ext}')
                        yield from self.directory.glob(f'**/*{ext.upper()}')
                else:
                    for ext in image_extensions:
                        yield from self.directory.glob(f'*{ext}')
                        yield from self.directory.glob(f'*{ext.upper()}')

            # Collect unique files (sorted for consistent processing)
            image_files = sorted(set(stream_image_files()))
            total = len(image_files)

            if total == 0:
                self.finished.emit(False, "No images found in directory", 0)
                return

            self.progress.emit(0, total, f"Found {total} images to process")

            # Process each image
            for i, image_path in enumerate(image_files, 1):
                self.progress.emit(i, total, f"Processing {image_path.name}...")

                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.error(f"Failed to load {image_path.name}")
                        self.failed_count += 1
                        continue

                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect poses
                    detections = self.detector.detect(image)

                    if not detections:
                        logger.warning(f"No poses detected in {image_path.name}")
                        self.failed_count += 1
                        continue

                    self.image_count += 1

                    # Save each detected pose
                    for person_idx, pose in enumerate(detections):
                        # Check confidence
                        if pose.overall_confidence < self.min_confidence:
                            continue

                        # Extract geometric features
                        features = self.feature_extractor.extract(pose)

                        if features.feature_vector is None:
                            continue

                        # Create ImageMetadata from loaded image
                        metadata = ImageMetadata(
                            file_path=image_path,
                            original_width=image.shape[1],
                            original_height=image.shape[0],
                            file_size_bytes=image_path.stat().st_size,
                            channels=image.shape[2] if len(image.shape) == 3 else 1,
                            dtype=str(image.dtype)
                        )

                        # Store detection using proper StorageManager API
                        self.storage.store_detection(
                            image_path=image_path,
                            image_metadata=metadata,
                            pose_result=pose,
                            features=features
                        )

                        self.pose_count += 1

                except Exception as e:
                    logger.error(f"Failed to process {image_path.name}: {e}")
                    self.failed_count += 1
                    continue

            # Build final message
            message = f"Successfully indexed {self.pose_count} poses from {self.image_count} images"
            if self.failed_count > 0:
                message += f" ({self.failed_count} images failed)"

            self.finished.emit(True, message, self.pose_count)

        except Exception as e:
            logger.error(f"Directory indexing failed: {e}", exc_info=True)
            self.finished.emit(False, f"Error: {str(e)}", 0)


class IndexBuildWorker(QThread):
    """Background worker for building FAISS index."""

    progress = Signal(int, int, str)
    finished = Signal(bool, str, int)  # success, message, pose_count

    def __init__(self, similarity_engine: SimilarityEngine):
        super().__init__()
        self.engine = similarity_engine

    def run(self):
        """Build FAISS index from database."""
        try:
            def progress_callback(current, total, message):
                self.progress.emit(current, total, message)

            self.engine.build_index(
                force_rebuild=True,
                progress_callback=progress_callback
            )

            stats = self.engine.get_statistics()
            self.finished.emit(
                True,
                f"Search index built successfully",
                stats['total_poses']
            )

        except Exception as e:
            logger.error(f"Index building failed: {e}", exc_info=True)
            self.finished.emit(False, f"Error: {str(e)}", 0)


class DirectoryIndexDialog(QDialog):
    """Dialog for indexing entire directories of images."""

    index_built = Signal(int)  # Emits number of poses indexed

    def __init__(self, pose_detector: PoseDetector,
                 feature_extractor: GeometricFeatureExtractor,
                 storage_manager: StorageManager,
                 similarity_engine: SimilarityEngine,
                 state_manager,
                 current_db_profile: str,
                 parent=None):
        super().__init__(parent)
        self.detector = pose_detector
        self.feature_extractor = feature_extractor
        self.storage = storage_manager
        self.engine = similarity_engine
        self.state_manager = state_manager
        self.current_db_profile = current_db_profile

        self.index_worker: Optional[DirectoryIndexWorker] = None
        self.build_worker: Optional[IndexBuildWorker] = None

        self.setup_ui()

    def setup_ui(self):
        """Build dialog UI."""
        self.setWindowTitle("Index Directory")
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)

        # Instructions
        intro = QLabel(
            "Select a directory of images to index. The system will detect poses "
            "in all images and build a searchable index for similarity search."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px;")
        layout.addWidget(intro)

        # Directory selection group
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout(dir_group)

        # Directory path display
        path_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("padding: 8px; background: white; border: 1px solid #ccc;")
        path_layout.addWidget(self.dir_label, stretch=1)

        select_btn = QPushButton("Browse...")
        select_btn.clicked.connect(self.select_directory)
        path_layout.addWidget(select_btn)

        dir_layout.addLayout(path_layout)

        # Options
        options_layout = QHBoxLayout()

        self.recursive_check = QCheckBox("Include subdirectories")
        self.recursive_check.setChecked(False)
        options_layout.addWidget(self.recursive_check)

        options_layout.addWidget(QLabel("Min Confidence:"))
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(30)
        self.confidence_spin.setSuffix("%")
        options_layout.addWidget(self.confidence_spin)

        options_layout.addStretch()

        dir_layout.addLayout(options_layout)

        layout.addWidget(dir_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Indexing")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                padding: 10px 20px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.start_btn.clicked.connect(self.start_indexing)
        button_layout.addWidget(self.start_btn)

        self.build_btn = QPushButton("Build Search Index")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self.build_index)
        button_layout.addWidget(self.build_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def select_directory(self):
        """Open directory selection dialog."""
        # Get last indexed directory for current profile
        setting_key = f"indexing/last_dir_{self.current_db_profile}"
        last_dir = self.state_manager.get_simple(setting_key, "")

        # Open dialog starting at last directory
        directory = QFileDialog.getExistingDirectory(
            self,
            f"Select Directory to Index ({self.current_db_profile.upper()} Database)",
            last_dir if last_dir else "",
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self.current_directory = Path(directory)
            self.dir_label.setText(str(self.current_directory))
            self.start_btn.setEnabled(True)
            self.status_text.append(f"Selected: {directory}")

            # Save for next time
            self.state_manager.set_simple(setting_key, directory)
            self.state_manager.save()

    def start_indexing(self):
        """Start indexing process."""
        if not hasattr(self, 'current_directory'):
            return

        # Disable controls
        self.start_btn.setEnabled(False)
        self.recursive_check.setEnabled(False)
        self.confidence_spin.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.status_text.append("\nStarting indexing process...")

        # Create worker
        min_conf = self.confidence_spin.value() / 100.0

        self.index_worker = DirectoryIndexWorker(
            self.current_directory,
            self.recursive_check.isChecked(),
            self.detector,
            self.feature_extractor,
            self.storage,
            min_conf
        )

        self.index_worker.progress.connect(self.on_index_progress)
        self.index_worker.finished.connect(self.on_index_finished)
        self.index_worker.start()

    def on_index_progress(self, current: int, total: int, message: str):
        """Update indexing progress."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_text.append(f"[{current}/{total}] {message}")

        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

    def on_index_finished(self, success: bool, message: str, pose_count: int):
        """Handle indexing completion."""
        self.progress_bar.setVisible(False)
        self.status_text.append(f"\n{message}")

        if success and pose_count > 0:
            self.status_text.append("\nIndexing complete! Now build the search index.")
            self.build_btn.setEnabled(True)

            # Show summary
            QMessageBox.information(
                self,
                "Indexing Complete",
                f"{message}\n\nClick 'Build Search Index' to enable similarity search."
            )
        else:
            self.status_text.append("\nIndexing failed or found no poses.")
            self.start_btn.setEnabled(True)
            self.recursive_check.setEnabled(True)
            self.confidence_spin.setEnabled(True)

    def build_index(self):
        """Build FAISS search index."""
        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_text.append("\nBuilding FAISS search index...")

        self.build_worker = IndexBuildWorker(self.engine)
        self.build_worker.progress.connect(self.on_build_progress)
        self.build_worker.finished.connect(self.on_build_finished)
        self.build_worker.start()

    def on_build_progress(self, current: int, total: int, message: str):
        """Update build progress."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_text.append(message)

    def on_build_finished(self, success: bool, message: str, pose_count: int):
        """Handle build completion."""
        self.progress_bar.setVisible(False)
        self.status_text.append(f"\n{message}")

        if success:
            self.status_text.append(
                f"\nSearch index ready with {pose_count} poses! "
                "You can now use similarity search."
            )

            # Emit signal
            self.index_built.emit(pose_count)

            QMessageBox.information(
                self,
                "Success",
                f"Search index built successfully with {pose_count} poses!\n\n"
                "You can now search for similar poses using the Search menu."
            )
        else:
            self.status_text.append("\nFailed to build search index.")
            self.build_btn.setEnabled(True)

    def closeEvent(self, event):
        """Clean up threads before closing."""
        # Terminate index worker if running
        if hasattr(self, "index_worker") and self.index_worker and self.index_worker.isRunning():
            self.index_worker.terminate()
            self.index_worker.wait(2000)  # Wait up to 2 seconds

            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.index_worker.progress.disconnect()
                self.index_worker.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.index_worker.deleteLater()

        # Terminate build worker if running
        if hasattr(self, "build_worker") and self.build_worker and self.build_worker.isRunning():
            self.build_worker.terminate()
            self.build_worker.wait(2000)

            # Disconnect signals before deleteLater() to prevent use-after-free
            try:
                self.build_worker.progress.disconnect()
                self.build_worker.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.build_worker.deleteLater()

        super().closeEvent(event)

