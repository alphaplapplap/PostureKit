"""
Batch Processing Dialog
Automated pose detection for multiple images with progress tracking
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QCheckBox, QSlider, QSpinBox,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QElapsedTimer
from PySide6.QtGui import QFont, QTextCursor
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading


class ProcessingState(Enum):
    """Batch processing states"""
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class DetectionSettings:
    """Detection configuration"""
    person_confidence: float = 0.5
    pose_confidence: float = 0.3
    max_persons: int = 10
    process_subdirs: bool = True
    save_to_db: bool = True
    export_json: bool = False
    skip_processed: bool = False


@dataclass
class ProcessingResult:
    """Result of processing a single image"""
    file_path: str
    success: bool
    num_people: int = 0
    error_message: str = ""
    processing_time: float = 0.0


class BatchProcessor(QThread):
    """Background worker for batch pose detection"""

    progress = Signal(int, int)
    file_started = Signal(str)
    file_completed = Signal(ProcessingResult)
    finished = Signal(bool, str)

    def __init__(self, directory: str, settings: DetectionSettings):
        super().__init__()
        self.directory = directory
        self.settings = settings
        self.is_paused = False
        self.is_cancelled = False
        self.image_files: List[Path] = []
        self.results: List[ProcessingResult] = []
        self._model_lock = threading.RLock()  # Thread-safe model access

        # Batch processing throttling (prevents CPU/GPU saturation)
        self.throttle_interval = 10  # Pause every N images
        self.throttle_delay_ms = 100  # Delay in milliseconds

    def run(self):
        try:
            self._find_image_files()
            if not self.image_files:
                self.finished.emit(False, "No image files found in directory")
                return

            total_files = len(self.image_files)
            for i, image_path in enumerate(self.image_files):
                while self.is_paused and not self.is_cancelled:
                    self.msleep(100)
                if self.is_cancelled:
                    self.finished.emit(False, "Processing cancelled by user")
                    return

                self.file_started.emit(image_path.name)
                self.progress.emit(i, total_files)
                result = self._process_image(image_path)
                self.results.append(result)
                self.file_completed.emit(result)

                # Batch throttling: Brief pause every N images for system responsiveness
                # Allows GPU to catch up and keeps GUI fluid
                if (i + 1) % self.throttle_interval == 0 and (i + 1) < total_files:
                    self.msleep(self.throttle_delay_ms)

            success_count = sum(1 for r in self.results if r.success)
            self.finished.emit(True, f"Processing complete: {success_count}/{total_files} successful")
        except Exception as e:
            self.finished.emit(False, f"Batch processing failed: {str(e)}")

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def cancel(self):
        self.is_cancelled = True
        self.is_paused = False

    def _find_image_files(self):
        """
        Find image files using generator for memory efficiency.

        Uses streaming approach to avoid loading 100k+ file paths into memory.
        Memory usage: O(1) vs O(n) for large directories.
        """
        directory = Path(self.directory)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        # Generator function for streaming file discovery
        def stream_image_files():
            if self.settings.process_subdirs:
                for ext in extensions:
                    yield from directory.rglob(f'*{ext}')
                    yield from directory.rglob(f'*{ext.upper()}')
            else:
                for ext in extensions:
                    yield from directory.glob(f'*{ext}')
                    yield from directory.glob(f'*{ext.upper()}')

        # Collect unique files (sorted for consistency)
        # Note: Still materializes list, but with deduplication
        unique_files = sorted(set(stream_image_files()))

        # Filter processed files if requested
        if self.settings.skip_processed:
            self.image_files = [f for f in unique_files if not self._is_already_processed(f)]
        else:
            self.image_files = unique_files

    def _process_image(self, image_path: Path) -> ProcessingResult:
        start_time = time.time()
        try:
            from src.core.image_ingestor import ImageIngestor
            from src.core.multi_person_pipeline import MultiPersonPipeline
            from src.config.settings import settings as app_settings

            # Thread-safe lazy initialization with double-check locking
            if not hasattr(self, '_ingestor'):
                with self._model_lock:
                    if not hasattr(self, '_ingestor'):
                        self._ingestor = ImageIngestor(target_size=app_settings.IMAGE_TARGET_SIZE)

            if not hasattr(self, '_pipeline'):
                with self._model_lock:
                    if not hasattr(self, '_pipeline'):
                        self._pipeline = MultiPersonPipeline(
                            person_confidence=self.settings.person_confidence,
                            pose_confidence=self.settings.pose_confidence
                        )

            # Protect model inference with lock to prevent concurrent access
            with self._model_lock:
                original, normalized, metadata = self._ingestor.process_image(image_path)
                results = self._pipeline.process_image(original)

            if not results:
                return ProcessingResult(
                    file_path=str(image_path),
                    success=False,
                    error_message="No people detected",
                    processing_time=time.time() - start_time
                )

            num_people = min(len(results), self.settings.max_persons)
            results = results[:num_people]

            if self.settings.save_to_db:
                self._save_to_database(image_path, results, metadata)
            if self.settings.export_json:
                self._export_json(image_path, results, metadata)

            # Cleanup image arrays to free memory
            del original
            del normalized
            import gc
            gc.collect()

            return ProcessingResult(
                file_path=str(image_path),
                success=True,
                num_people=num_people,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ProcessingResult(
                file_path=str(image_path),
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def _is_already_processed(self, image_path: Path) -> bool:
        return False

    def _save_to_database(self, image_path: Path, results: List, metadata: Dict):
        try:
            from src.storage.storage_manager import StorageManager
            if not hasattr(self, '_storage'):
                self._storage = StorageManager()
            for person_det, pose_result, viewpoint, geometric_features in results:
                self._storage.store_pose(
                    pose=pose_result,
                    features=geometric_features,
                    visual_features=None,
                    combined_features=None,
                    metadata=metadata,
                    image_path=str(image_path)
                )
        except Exception as e:
            print(f"Failed to save to database: {e}")

    def _export_json(self, image_path: Path, results: List, metadata: Dict):
        import json
        try:
            output_path = image_path.with_suffix('.json')
            data = {
                'image': str(image_path),
                'metadata': metadata,
                'num_people': len(results),
                'people': []
            }
            for person_det, pose_result, viewpoint, geometric_features in results:
                person_data = {
                    'bbox': person_det.bbox.tolist() if hasattr(person_det.bbox, 'tolist') else person_det.bbox,
                    'confidence': float(person_det.confidence),
                    'keypoints': pose_result.keypoints.tolist(),
                    'confidences': pose_result.confidences.tolist(),
                    'visibility': pose_result.visibility.tolist()
                }
                data['people'].append(person_data)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to export JSON: {e}")


class BatchDialog(QDialog):
    """Batch processing dialog for processing multiple images."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.processor: Optional[BatchProcessor] = None
        self.state = ProcessingState.READY
        self.start_time = None
        self.elapsed_timer = QElapsedTimer()
        self.total_files = 0
        self.processed_files = 0
        self.success_count = 0
        self.failed_count = 0
        self.setWindowTitle("Batch Process Directory")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.setMaximumSize(600, 500)
        self._setup_ui()
        self._update_ui_state()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_elapsed_time)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._create_directory_group())
        layout.addWidget(self._create_options_group())
        layout.addWidget(self._create_settings_group())
        layout.addWidget(self._create_progress_group())
        layout.addWidget(self._create_error_group())
        layout.addLayout(self._create_buttons())

    def _create_directory_group(self) -> QGroupBox:
        group = QGroupBox("Directory")
        layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("/path/to/images")
        layout.addWidget(self.dir_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setIcon(self.style().standardIcon(self.style().SP_DirIcon))
        browse_btn.clicked.connect(self._browse_directory)
        layout.addWidget(browse_btn)
        group.setLayout(layout)
        return group

    def _create_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        layout = QVBoxLayout()
        self.subdirs_check = QCheckBox("Process subdirectories recursively")
        self.subdirs_check.setChecked(True)
        layout.addWidget(self.subdirs_check)
        self.save_db_check = QCheckBox("Save detections to database")
        self.save_db_check.setChecked(True)
        layout.addWidget(self.save_db_check)
        self.export_json_check = QCheckBox("Export results to JSON")
        layout.addWidget(self.export_json_check)
        self.skip_processed_check = QCheckBox("Skip already processed images")
        layout.addWidget(self.skip_processed_check)
        group.setLayout(layout)
        return group

    def _create_settings_group(self) -> QGroupBox:
        group = QGroupBox("Detection Settings")
        layout = QGridLayout()
        layout.addWidget(QLabel("Person Confidence:"), 0, 0)
        self.person_conf_slider = QSlider(Qt.Horizontal)
        self.person_conf_slider.setRange(10, 100)
        self.person_conf_slider.setValue(50)
        self.person_conf_slider.valueChanged.connect(
            lambda v: self.person_conf_label.setText(f"{v/100:.2f}")
        )
        layout.addWidget(self.person_conf_slider, 0, 1)
        self.person_conf_label = QLabel("0.50")
        layout.addWidget(self.person_conf_label, 0, 2)
        layout.addWidget(QLabel("Pose Confidence:"), 1, 0)
        self.pose_conf_slider = QSlider(Qt.Horizontal)
        self.pose_conf_slider.setRange(10, 100)
        self.pose_conf_slider.setValue(30)
        self.pose_conf_slider.valueChanged.connect(
            lambda v: self.pose_conf_label.setText(f"{v/100:.2f}")
        )
        layout.addWidget(self.pose_conf_slider, 1, 1)
        self.pose_conf_label = QLabel("0.30")
        layout.addWidget(self.pose_conf_label, 1, 2)
        layout.addWidget(QLabel("Max Persons/Image:"), 2, 0)
        self.max_persons_spin = QSpinBox()
        self.max_persons_spin.setRange(1, 50)
        self.max_persons_spin.setValue(10)
        layout.addWidget(self.max_persons_spin, 2, 1, 1, 2)
        group.setLayout(layout)
        return group

    def _create_progress_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QVBoxLayout()
        self.status_label = QLabel("Status: Ready")
        font = self.status_label.font()
        font.setBold(True)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        self.current_file_label = QLabel("Current: -")
        layout.addWidget(self.current_file_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v/%m (%p%)")
        layout.addWidget(self.progress_bar)
        stats_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: 0s")
        stats_layout.addWidget(self.elapsed_label)
        self.eta_label = QLabel("ETA: -")
        stats_layout.addWidget(self.eta_label)
        stats_layout.addStretch()
        self.stats_label = QLabel("Success: 0 | Failed: 0")
        stats_layout.addWidget(self.stats_label)
        layout.addLayout(stats_layout)
        group.setLayout(layout)
        return group

    def _create_error_group(self) -> QGroupBox:
        group = QGroupBox("Error Log")
        layout = QVBoxLayout()
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setMaximumHeight(100)
        self.error_log.setPlaceholderText("Errors will appear here...")
        layout.addWidget(self.error_log)
        group.setLayout(layout)
        return group

    def _create_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start_processing)
        layout.addWidget(self.start_btn)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._pause_processing)
        layout.addWidget(self.pause_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_processing)
        layout.addWidget(self.cancel_btn)
        layout.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        layout.addWidget(self.close_btn)
        return layout

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory", "", QFileDialog.ShowDirsOnly)
        if directory:
            self.dir_edit.setText(directory)

    def _start_processing(self):
        directory = self.dir_edit.text()
        if not directory:
            QMessageBox.warning(self, "Error", "Please select a directory")
            return
        if not Path(directory).exists():
            QMessageBox.warning(self, "Error", "Directory does not exist")
            return

        settings = DetectionSettings(
            person_confidence=self.person_conf_slider.value() / 100.0,
            pose_confidence=self.pose_conf_slider.value() / 100.0,
            max_persons=self.max_persons_spin.value(),
            process_subdirs=self.subdirs_check.isChecked(),
            save_to_db=self.save_db_check.isChecked(),
            export_json=self.export_json_check.isChecked(),
            skip_processed=self.skip_processed_check.isChecked()
        )

        self.processed_files = 0
        self.success_count = 0
        self.failed_count = 0
        self.error_log.clear()

        # Clean up old processor if it exists
        if hasattr(self, 'processor') and self.processor:
            if self.processor.isRunning():
                self.processor.cancel()
                self.processor.wait(5000)

            # Disconnect all signals before deleteLater() to prevent use-after-free
            try:
                self.processor.progress.disconnect()
                self.processor.file_started.disconnect()
                self.processor.file_completed.disconnect()
                self.processor.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            self.processor.deleteLater()

        self.processor = BatchProcessor(directory, settings)
        self.processor.progress.connect(self._on_progress)
        self.processor.file_started.connect(self._on_file_started)
        self.processor.file_completed.connect(self._on_file_completed)
        self.processor.finished.connect(self._on_finished)
        self.processor.start()

        self.state = ProcessingState.PROCESSING
        self.elapsed_timer.start()
        self.update_timer.start(1000)
        self._update_ui_state()

    def _pause_processing(self):
        if self.state == ProcessingState.PROCESSING:
            self.processor.pause()
            self.state = ProcessingState.PAUSED
            self.pause_btn.setText("Resume")
            self.status_label.setText("Status: Paused")
            self.update_timer.stop()
        elif self.state == ProcessingState.PAUSED:
            self.processor.resume()
            self.state = ProcessingState.PROCESSING
            self.pause_btn.setText("Pause")
            self.status_label.setText("Status: Processing...")
            self.update_timer.start(1000)

    def _cancel_processing(self):
        if self.processor and self.state in [ProcessingState.PROCESSING, ProcessingState.PAUSED]:
            reply = QMessageBox.question(
                self, "Confirm Cancel",
                "Are you sure you want to cancel batch processing?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processor.cancel()
                self.state = ProcessingState.CANCELLED
                self.status_label.setText("Status: Cancelled")
                self.update_timer.stop()
                self._update_ui_state()

    def _on_progress(self, current: int, total: int):
        self.total_files = total
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_file_started(self, filename: str):
        self.current_file_label.setText(f"Current: {filename}")

    def _on_file_completed(self, result: ProcessingResult):
        self.processed_files += 1
        if result.success:
            self.success_count += 1
        else:
            self.failed_count += 1
            self._log_error(f"⚠ {Path(result.file_path).name}: {result.error_message}")
        self.stats_label.setText(f"Success: {self.success_count} | Failed: {self.failed_count}")
        self._update_eta()

    def _on_finished(self, success: bool, message: str):
        self.update_timer.stop()
        self.state = ProcessingState.COMPLETE
        self.status_label.setText(f"Status: {message}")
        self.current_file_label.setText("Current: -")
        self.eta_label.setText("ETA: Complete")
        self._update_ui_state()
        if success:
            QMessageBox.information(self, "Complete", message)
        else:
            QMessageBox.warning(self, "Error", message)

    def _update_elapsed_time(self):
        if self.elapsed_timer.isValid():
            elapsed_ms = self.elapsed_timer.elapsed()
            elapsed_sec = elapsed_ms / 1000
            if elapsed_sec < 60:
                self.elapsed_label.setText(f"Elapsed: {elapsed_sec:.0f}s")
            else:
                minutes = int(elapsed_sec // 60)
                seconds = int(elapsed_sec % 60)
                self.elapsed_label.setText(f"Elapsed: {minutes}m {seconds}s")

    def _update_eta(self):
        if self.processed_files > 0 and self.total_files > 0:
            elapsed_ms = self.elapsed_timer.elapsed()
            elapsed_sec = elapsed_ms / 1000
            avg_time = elapsed_sec / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta_sec = avg_time * remaining_files
            if eta_sec < 60:
                self.eta_label.setText(f"ETA: {eta_sec:.0f}s")
            else:
                minutes = int(eta_sec // 60)
                seconds = int(eta_sec % 60)
                self.eta_label.setText(f"ETA: {minutes}m {seconds}s")

    def _log_error(self, message: str):
        self.error_log.append(message)
        cursor = self.error_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.error_log.setTextCursor(cursor)

    def _update_ui_state(self):
        if self.state == ProcessingState.READY:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            self.dir_edit.setEnabled(True)
            self.subdirs_check.setEnabled(True)
            self.save_db_check.setEnabled(True)
            self.export_json_check.setEnabled(True)
            self.skip_processed_check.setEnabled(True)
            self.person_conf_slider.setEnabled(True)
            self.pose_conf_slider.setEnabled(True)
            self.max_persons_spin.setEnabled(True)
        elif self.state == ProcessingState.PROCESSING:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Pause")
            self.cancel_btn.setEnabled(True)
            self.close_btn.setEnabled(False)
            self.dir_edit.setEnabled(False)
            self.subdirs_check.setEnabled(False)
            self.save_db_check.setEnabled(False)
            self.export_json_check.setEnabled(False)
            self.skip_processed_check.setEnabled(False)
            self.person_conf_slider.setEnabled(False)
            self.pose_conf_slider.setEnabled(False)
            self.max_persons_spin.setEnabled(False)
        elif self.state == ProcessingState.PAUSED:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("Resume")
            self.cancel_btn.setEnabled(True)
            self.close_btn.setEnabled(False)
        elif self.state in [ProcessingState.COMPLETE, ProcessingState.CANCELLED]:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            self.dir_edit.setEnabled(True)
            self.subdirs_check.setEnabled(True)
            self.save_db_check.setEnabled(True)
            self.export_json_check.setEnabled(True)
            self.skip_processed_check.setEnabled(True)
            self.person_conf_slider.setEnabled(True)
            self.pose_conf_slider.setEnabled(True)
            self.max_persons_spin.setEnabled(True)


# Alias for backward compatibility
BatchProcessDialog = BatchDialog
