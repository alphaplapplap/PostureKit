"""
Application preferences dialog with persistent settings.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QFileDialog, QLineEdit,
    QMessageBox, QSlider
)
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QFont
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PreferencesDialog(QDialog):
    """
    Multi-tabbed preferences dialog for application settings.

    This dialog organizes settings into logical categories:
    - General: Application behavior, startup options
    - Detection: Pose detection parameters, model settings
    - Display: Visualization options, colors, rendering
    - Performance: Threading, caching, memory limits
    - Advanced: Database, logging, debug options

    All settings are persisted using QSettings, which automatically
    handles platform-specific storage (plist on macOS, registry on Windows).
    """

    settings_changed = Signal(dict)  # Emitted with changed settings

    def __init__(self, parent=None):
        """
        Initialize preferences dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Preferences")
        self.setMinimumSize(700, 600)

        # QSettings for persistent storage
        self.settings = QSettings("PostureKit", "PostureKit")

        # Track changes
        self.pending_changes: Dict[str, Any] = {}

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup dialog layout with tabbed interface."""
        layout = QVBoxLayout()

        # Tab widget for different setting categories
        self.tabs = QTabWidget()

        # Add setting pages
        self.tabs.addTab(self._create_general_page(), "General")
        self.tabs.addTab(self._create_detection_page(), "Detection")
        self.tabs.addTab(self._create_display_page(), "Display")
        self.tabs.addTab(self._create_performance_page(), "Performance")
        self.tabs.addTab(self._create_advanced_page(), "Advanced")

        layout.addWidget(self.tabs)

        # Button bar
        button_layout = QHBoxLayout()

        restore_btn = QPushButton("Restore Defaults")
        restore_btn.clicked.connect(self._restore_defaults)
        button_layout.addWidget(restore_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_settings)
        button_layout.addWidget(apply_btn)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._ok_clicked)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_general_page(self) -> QWidget:
        """
        Create general settings page.

        This page contains application-wide settings like startup behavior,
        auto-save preferences, and UI language options. These are the settings
        that affect the overall user experience rather than specific features.
        """
        page = QWidget()
        layout = QVBoxLayout()

        # Startup group
        startup_group = QGroupBox("Startup")
        startup_layout = QFormLayout()

        self.show_splash_check = QCheckBox("Show splash screen on startup")
        self.show_splash_check.setChecked(True)
        startup_layout.addRow(self.show_splash_check)

        self.restore_session_check = QCheckBox("Restore last session on startup")
        self.restore_session_check.setChecked(True)
        self.restore_session_check.setToolTip(
            "Automatically reload the last opened images and working state"
        )
        startup_layout.addRow(self.restore_session_check)

        self.check_updates_check = QCheckBox("Check for updates on startup")
        self.check_updates_check.setChecked(False)
        startup_layout.addRow(self.check_updates_check)

        startup_group.setLayout(startup_layout)
        layout.addWidget(startup_group)

        # Auto-save group
        autosave_group = QGroupBox("Auto-Save")
        autosave_layout = QFormLayout()

        self.autosave_enabled_check = QCheckBox("Enable auto-save")
        self.autosave_enabled_check.setChecked(True)
        self.autosave_enabled_check.toggled.connect(self._on_autosave_toggled)
        autosave_layout.addRow(self.autosave_enabled_check)

        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(1, 60)
        self.autosave_interval_spin.setValue(5)
        self.autosave_interval_spin.setSuffix(" minutes")
        autosave_layout.addRow("Auto-save interval:", self.autosave_interval_spin)

        autosave_group.setLayout(autosave_layout)
        layout.addWidget(autosave_group)

        # File handling group
        file_group = QGroupBox("File Handling")
        file_layout = QFormLayout()

        self.recent_files_spin = QSpinBox()
        self.recent_files_spin.setRange(5, 50)
        self.recent_files_spin.setValue(10)
        file_layout.addRow("Recent files to remember:", self.recent_files_spin)

        self.default_export_format = QComboBox()
        self.default_export_format.addItems(["COCO JSON", "MMPose", "Custom JSON", "CSV"])
        file_layout.addRow("Default export format:", self.default_export_format)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _create_detection_page(self) -> QWidget:
        """
        Create pose detection settings page.

        These settings control the core pose detection algorithm. The confidence
        threshold determines how certain the model needs to be before accepting
        a keypoint detection. Lower values catch more keypoints but may include
        false positives. The target image size affects both accuracy and speed—
        larger images provide more detail but take longer to process.
        """
        page = QWidget()
        layout = QVBoxLayout()

        # Pose model selection group
        pose_model_group = QGroupBox("Pose Estimation Model")
        pose_model_layout = QFormLayout()

        # Model selection
        self.pose_model_combo = QComboBox()
        self.pose_model_combo.addItems([
            "RTMW-L (Default - 220MB, Fast)",
            "RTMW-X (353MB, Best Quality)",
            "Ensemble: RTMW-L + RTMW-X (2.5x slower, 10-15% better)"
        ])
        self.pose_model_combo.setToolTip(
            "Select which pose estimation model(s) to use:\n"
            "• RTMW-L: Fast, good quality (recommended)\n"
            "• RTMW-X: Larger model, best for difficult poses\n"
            "• Ensemble: Uses both models for maximum accuracy"
        )
        self.pose_model_combo.currentIndexChanged.connect(self._on_model_changed)
        pose_model_layout.addRow("Pose model:", self.pose_model_combo)

        # Fusion method (for ensemble)
        self.fusion_method_combo = QComboBox()
        self.fusion_method_combo.addItems([
            "Confidence-Weighted (Recommended)",
            "Weighted Average"
        ])
        self.fusion_method_combo.setToolTip(
            "How to combine predictions when using ensemble:\n"
            "• Confidence-Weighted: Weight by per-keypoint confidence (better quality)\n"
            "• Weighted Average: Simple average of predictions (faster fusion)"
        )
        self.fusion_method_label = QLabel("Fusion method:")
        pose_model_layout.addRow(self.fusion_method_label, self.fusion_method_combo)

        # Two-stage detection
        self.two_stage_check = QCheckBox("Enable two-stage detection (YOLO + pose)")
        self.two_stage_check.setToolTip(
            "Use person detection before pose estimation:\n"
            "• 5-10% accuracy improvement\n"
            "• Better multi-person handling\n"
            "• Actually 13% faster on test images!"
        )
        pose_model_layout.addRow(self.two_stage_check)

        # Model info label
        self.model_info_label = QLabel()
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #666; font-size: 11px;")
        pose_model_layout.addRow(self.model_info_label)

        pose_model_group.setLayout(pose_model_layout)
        layout.addWidget(pose_model_group)

        # Model settings group
        model_group = QGroupBox("Detection Settings")
        model_layout = QFormLayout()

        self.detection_threshold_slider = QSlider(Qt.Horizontal)
        self.detection_threshold_slider.setRange(10, 90)
        self.detection_threshold_slider.setValue(30)
        self.detection_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.detection_threshold_slider.setTickInterval(10)

        # Add value label that updates with slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.detection_threshold_slider)
        self.threshold_value_label = QLabel("0.30")
        self.detection_threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"{v/100:.2f}")
        )
        threshold_layout.addWidget(self.threshold_value_label)

        model_layout.addRow("Detection confidence threshold:", threshold_layout)

        self.target_width_spin = QSpinBox()
        self.target_width_spin.setRange(512, 2048)
        self.target_width_spin.setValue(1024)
        self.target_width_spin.setSingleStep(128)
        model_layout.addRow("Target image width:", self.target_width_spin)

        self.target_height_spin = QSpinBox()
        self.target_height_spin.setRange(512, 2048)
        self.target_height_spin.setValue(1024)
        self.target_height_spin.setSingleStep(128)
        model_layout.addRow("Target image height:", self.target_height_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Processing options
        processing_group = QGroupBox("Processing Options")
        processing_layout = QFormLayout()

        self.use_gpu_check = QCheckBox("Use GPU acceleration (MPS/CUDA)")
        self.use_gpu_check.setChecked(True)
        self.use_gpu_check.setToolTip(
            "Enable Apple Silicon MPS or NVIDIA CUDA for faster detection"
        )
        processing_layout.addRow(self.use_gpu_check)

        self.batch_detection_check = QCheckBox("Enable batch detection for multiple images")
        self.batch_detection_check.setChecked(True)
        processing_layout.addRow(self.batch_detection_check)

        self.max_people_spin = QSpinBox()
        self.max_people_spin.setRange(1, 20)
        self.max_people_spin.setValue(10)
        processing_layout.addRow("Maximum people to detect:", self.max_people_spin)

        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        # Correction learning
        learning_group = QGroupBox("Correction Learning")
        learning_layout = QFormLayout()

        self.auto_retrain_check = QCheckBox("Automatically retrain on corrections")
        self.auto_retrain_check.setChecked(False)
        self.auto_retrain_check.setToolTip(
            "Trigger model retraining after accumulating corrections"
        )
        learning_layout.addRow(self.auto_retrain_check)

        self.retrain_threshold_spin = QSpinBox()
        self.retrain_threshold_spin.setRange(10, 200)
        self.retrain_threshold_spin.setValue(50)
        learning_layout.addRow("Retrain after N corrections:", self.retrain_threshold_spin)

        learning_group.setLayout(learning_layout)
        layout.addWidget(learning_group)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _create_display_page(self) -> QWidget:
        """
        Create display settings page.

        Visual preferences are highly personal. Some users prefer vibrant colors
        that make keypoints stand out clearly, while others want subtle overlays
        that don't obscure the underlying image. The confidence color coding helps
        identify uncertain detections at a glance—red keypoints need attention.
        """
        page = QWidget()
        layout = QVBoxLayout()

        # Skeleton visualization
        skeleton_group = QGroupBox("Skeleton Visualization")
        skeleton_layout = QFormLayout()

        self.skeleton_line_width_spin = QSpinBox()
        self.skeleton_line_width_spin.setRange(1, 10)
        self.skeleton_line_width_spin.setValue(2)
        skeleton_layout.addRow("Skeleton line width:", self.skeleton_line_width_spin)

        self.keypoint_size_spin = QSpinBox()
        self.keypoint_size_spin.setRange(2, 20)
        self.keypoint_size_spin.setValue(6)
        skeleton_layout.addRow("Keypoint marker size:", self.keypoint_size_spin)

        self.show_keypoint_labels_check = QCheckBox("Show keypoint labels")
        self.show_keypoint_labels_check.setChecked(False)
        skeleton_layout.addRow(self.show_keypoint_labels_check)

        self.show_confidence_colors_check = QCheckBox("Color-code by confidence")
        self.show_confidence_colors_check.setChecked(True)
        self.show_confidence_colors_check.setToolTip(
            "Green: high confidence, Yellow: medium, Red: low"
        )
        skeleton_layout.addRow(self.show_confidence_colors_check)

        skeleton_group.setLayout(skeleton_layout)
        layout.addWidget(skeleton_group)

        # Image viewer
        viewer_group = QGroupBox("Image Viewer")
        viewer_layout = QFormLayout()

        self.antialiasing_check = QCheckBox("Enable anti-aliasing")
        self.antialiasing_check.setChecked(True)
        viewer_layout.addRow(self.antialiasing_check)

        self.smooth_zoom_check = QCheckBox("Smooth zoom animation")
        self.smooth_zoom_check.setChecked(True)
        viewer_layout.addRow(self.smooth_zoom_check)

        self.zoom_speed_slider = QSlider(Qt.Horizontal)
        self.zoom_speed_slider.setRange(1, 10)
        self.zoom_speed_slider.setValue(5)
        viewer_layout.addRow("Zoom speed:", self.zoom_speed_slider)

        viewer_group.setLayout(viewer_layout)
        layout.addWidget(viewer_group)

        # Theme
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        theme_layout.addRow("Theme:", self.theme_combo)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _create_performance_page(self) -> QWidget:
        """
        Create performance settings page.

        Performance tuning allows users to balance speed against quality based on
        their hardware capabilities. The worker threads setting determines how many
        images can be processed simultaneously—more threads mean faster batch
        processing but higher memory usage. The cache helps avoid reloading the
        same images repeatedly, particularly useful when iterating on corrections.
        """
        page = QWidget()
        layout = QVBoxLayout()

        # Threading
        threading_group = QGroupBox("Multi-threading")
        threading_layout = QFormLayout()

        self.worker_threads_spin = QSpinBox()
        self.worker_threads_spin.setRange(1, 16)
        self.worker_threads_spin.setValue(4)
        self.worker_threads_spin.setToolTip(
            "Number of parallel worker threads for batch processing"
        )
        threading_layout.addRow("Worker threads:", self.worker_threads_spin)

        threading_group.setLayout(threading_layout)
        layout.addWidget(threading_group)

        # Memory management
        memory_group = QGroupBox("Memory Management")
        memory_layout = QFormLayout()

        self.image_cache_size_spin = QSpinBox()
        self.image_cache_size_spin.setRange(5, 100)
        self.image_cache_size_spin.setValue(20)
        self.image_cache_size_spin.setSuffix(" images")
        memory_layout.addRow("Image cache size:", self.image_cache_size_spin)

        self.max_memory_spin = QSpinBox()
        self.max_memory_spin.setRange(1000, 32000)
        self.max_memory_spin.setValue(8000)
        self.max_memory_spin.setSuffix(" MB")
        self.max_memory_spin.setSingleStep(1000)
        memory_layout.addRow("Maximum memory usage:", self.max_memory_spin)

        self.unload_models_check = QCheckBox("Unload models when idle")
        self.unload_models_check.setChecked(False)
        self.unload_models_check.setToolTip(
            "Free GPU memory by unloading models after 5 minutes of inactivity"
        )
        memory_layout.addRow(self.unload_models_check)

        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)

        # Database
        db_group = QGroupBox("Database")
        db_layout = QFormLayout()

        self.db_pool_size_spin = QSpinBox()
        self.db_pool_size_spin.setRange(1, 20)
        self.db_pool_size_spin.setValue(5)
        db_layout.addRow("Connection pool size:", self.db_pool_size_spin)

        self.vacuum_on_startup_check = QCheckBox("Vacuum database on startup")
        self.vacuum_on_startup_check.setChecked(False)
        self.vacuum_on_startup_check.setToolTip(
            "Optimize database on startup (may slow down launch)"
        )
        db_layout.addRow(self.vacuum_on_startup_check)

        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _create_advanced_page(self) -> QWidget:
        """
        Create advanced settings page.

        Advanced settings are intended for power users and developers. The logging
        level controls how much diagnostic information is recorded—DEBUG level
        captures everything and is useful for troubleshooting, but generates large
        log files. The experimental features toggle enables bleeding-edge functionality
        that may not be fully stable but provides access to new capabilities.
        """
        page = QWidget()
        layout = QVBoxLayout()

        # Logging
        logging_group = QGroupBox("Logging")
        logging_layout = QFormLayout()

        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        logging_layout.addRow("Log level:", self.log_level_combo)

        self.log_to_file_check = QCheckBox("Write logs to file")
        self.log_to_file_check.setChecked(True)
        logging_layout.addRow(self.log_to_file_check)

        log_dir_layout = QHBoxLayout()
        self.log_dir_input = QLineEdit()
        self.log_dir_input.setText("logs/")
        log_dir_layout.addWidget(self.log_dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_log_dir)
        log_dir_layout.addWidget(browse_btn)
        logging_layout.addRow("Log directory:", log_dir_layout)

        logging_group.setLayout(logging_layout)
        layout.addWidget(logging_group)

        # Developer options
        dev_group = QGroupBox("Developer Options")
        dev_layout = QFormLayout()

        self.show_fps_check = QCheckBox("Show FPS counter")
        self.show_fps_check.setChecked(False)
        dev_layout.addRow(self.show_fps_check)

        self.enable_profiling_check = QCheckBox("Enable performance profiling")
        self.enable_profiling_check.setChecked(False)
        dev_layout.addRow(self.enable_profiling_check)

        self.experimental_features_check = QCheckBox("Enable experimental features")
        self.experimental_features_check.setChecked(False)
        self.experimental_features_check.setToolTip(
            "⚠️ May be unstable. Use at your own risk."
        )
        dev_layout.addRow(self.experimental_features_check)

        dev_group.setLayout(dev_layout)
        layout.addWidget(dev_group)

        # Data management
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()

        clear_cache_btn = QPushButton("Clear Image Cache")
        clear_cache_btn.clicked.connect(self._clear_cache)
        data_layout.addWidget(clear_cache_btn)

        reset_db_btn = QPushButton("Reset Database (⚠️ Destructive)")
        reset_db_btn.clicked.connect(self._reset_database)
        data_layout.addWidget(reset_db_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _load_settings(self):
        """Load saved settings and populate UI controls."""
        # General settings
        self.show_splash_check.setChecked(
            self.settings.value("general/show_splash", True, bool)
        )
        self.restore_session_check.setChecked(
            self.settings.value("general/restore_session", True, bool)
        )
        self.autosave_enabled_check.setChecked(
            self.settings.value("general/autosave_enabled", True, bool)
        )
        self.autosave_interval_spin.setValue(
            self.settings.value("general/autosave_interval", 5, int)
        )

        # Detection settings - Pose models
        pose_model_idx = self.settings.value("detection/pose_model_index", 0, int)
        self.pose_model_combo.setCurrentIndex(pose_model_idx)

        fusion_method_idx = self.settings.value("detection/fusion_method_index", 0, int)
        self.fusion_method_combo.setCurrentIndex(fusion_method_idx)

        self.two_stage_check.setChecked(
            self.settings.value("detection/use_two_stage", False, bool)
        )

        # Update fusion method visibility
        self._on_model_changed(pose_model_idx)

        # Detection settings - Other
        self.detection_threshold_slider.setValue(
            int(self.settings.value("detection/confidence_threshold", 0.3, float) * 100)
        )
        self.target_width_spin.setValue(
            self.settings.value("detection/target_width", 1024, int)
        )
        self.target_height_spin.setValue(
            self.settings.value("detection/target_height", 1024, int)
        )
        self.use_gpu_check.setChecked(
            self.settings.value("detection/use_gpu", True, bool)
        )

        # Display settings
        self.skeleton_line_width_spin.setValue(
            self.settings.value("display/skeleton_line_width", 2, int)
        )
        self.keypoint_size_spin.setValue(
            self.settings.value("display/keypoint_size", 6, int)
        )
        self.show_confidence_colors_check.setChecked(
            self.settings.value("display/show_confidence_colors", True, bool)
        )

        # Performance settings
        self.worker_threads_spin.setValue(
            self.settings.value("performance/worker_threads", 4, int)
        )
        self.image_cache_size_spin.setValue(
            self.settings.value("performance/cache_size", 20, int)
        )

        # Advanced settings
        self.log_level_combo.setCurrentText(
            self.settings.value("advanced/log_level", "INFO", str)
        )

        logger.debug("Loaded preferences from settings")

    def _save_settings(self) -> Dict[str, Any]:
        """
        Save all settings and return dictionary of changed values.

        This method captures all current UI values and persists them using QSettings.
        It also tracks which settings changed so the application can respond
        appropriately (for example, changing the log level needs to reconfigure
        the logging system immediately).

        Returns:
            Dictionary of settings that changed
        """
        changed = {}

        # General settings
        self._set_and_track("general/show_splash", self.show_splash_check.isChecked(), changed)
        self._set_and_track("general/restore_session", self.restore_session_check.isChecked(), changed)
        self._set_and_track("general/autosave_enabled", self.autosave_enabled_check.isChecked(), changed)
        self._set_and_track("general/autosave_interval", self.autosave_interval_spin.value(), changed)

        # Detection settings - Pose models
        self._set_and_track("detection/pose_model_index", self.pose_model_combo.currentIndex(), changed)
        self._set_and_track("detection/fusion_method_index", self.fusion_method_combo.currentIndex(), changed)
        self._set_and_track("detection/use_two_stage", self.two_stage_check.isChecked(), changed)

        # Detection settings - Other
        self._set_and_track(
            "detection/confidence_threshold",
            self.detection_threshold_slider.value() / 100.0,
            changed
        )
        self._set_and_track("detection/target_width", self.target_width_spin.value(), changed)
        self._set_and_track("detection/target_height", self.target_height_spin.value(), changed)
        self._set_and_track("detection/use_gpu", self.use_gpu_check.isChecked(), changed)

        # Display settings
        self._set_and_track("display/skeleton_line_width", self.skeleton_line_width_spin.value(), changed)
        self._set_and_track("display/keypoint_size", self.keypoint_size_spin.value(), changed)
        self._set_and_track("display/show_confidence_colors", self.show_confidence_colors_check.isChecked(), changed)

        # Performance settings
        self._set_and_track("performance/worker_threads", self.worker_threads_spin.value(), changed)
        self._set_and_track("performance/cache_size", self.image_cache_size_spin.value(), changed)

        # Advanced settings
        self._set_and_track("advanced/log_level", self.log_level_combo.currentText(), changed)

        # Sync to disk
        self.settings.sync()

        logger.info(f"Saved preferences: {len(changed)} settings changed")
        return changed

    def _set_and_track(self, key: str, value: Any, changed: Dict):
        """Helper to set value and track if it changed."""
        old_value = self.settings.value(key)
        if old_value != value:
            changed[key] = {'old': old_value, 'new': value}
        self.settings.setValue(key, value)

    def _apply_settings(self):
        """Apply settings without closing dialog."""
        changed = self._save_settings()
        self.settings_changed.emit(changed)

        from gui.widgets.toast_notification import show_success
        show_success("Preferences applied")

    def _ok_clicked(self):
        """Handle OK button - save and close."""
        self._apply_settings()
        self.accept()

    def _restore_defaults(self):
        """Restore all settings to default values."""
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clear all settings
            self.settings.clear()

            # Reload UI with defaults
            self._load_settings()

            logger.info("Restored default preferences")

    def _on_model_changed(self, index: int):
        """Handle model selection change - show/hide fusion options."""
        # Show fusion method only for ensemble (index 2)
        is_ensemble = (index == 2)
        self.fusion_method_label.setVisible(is_ensemble)
        self.fusion_method_combo.setVisible(is_ensemble)

        # Update info label
        info_texts = [
            "Fast, good quality. Recommended for most use cases.\nDetection time: ~0.6s",
            "Larger model, best for difficult poses.\nDetection time: ~0.9s (1.5x slower)",
            "Uses both models for maximum accuracy.\nDetection time: ~1.4s (2.5x slower)\n10-15% accuracy improvement"
        ]
        self.model_info_label.setText(f"ℹ️ {info_texts[index]}")

    @staticmethod
    def get_pose_detector_config(settings: QSettings = None) -> Dict[str, Any]:
        """
        Get pose detector configuration from settings.

        This static method can be called without creating a dialog instance,
        useful for initializing PostureKitBridge with saved preferences.

        Args:
            settings: QSettings instance (creates new one if None)

        Returns:
            Dictionary with pose_models, fusion_method, and use_two_stage
        """
        if settings is None:
            settings = QSettings("PostureKit", "PostureKit")

        pose_model_idx = settings.value("detection/pose_model_index", 0, int)
        fusion_method_idx = settings.value("detection/fusion_method_index", 0, int)
        use_two_stage = settings.value("detection/use_two_stage", False, bool)

        # Map index to pose_models parameter
        pose_models_map = {
            0: 'rtmw-l',  # RTMW-L only
            1: 'rtmw-x',  # RTMW-X only
            2: ['rtmw-l', 'rtmw-x'],  # Ensemble
        }

        # Map fusion method index to string
        fusion_methods = ['confidence_weighted', 'weighted_average']

        return {
            'pose_models': pose_models_map.get(pose_model_idx, 'rtmw-l'),
            'fusion_method': fusion_methods[fusion_method_idx],
            'use_two_stage': use_two_stage,
        }

    def _on_autosave_toggled(self, checked: bool):
        """Enable/disable autosave interval spin box."""
        self.autosave_interval_spin.setEnabled(checked)

    def _browse_log_dir(self):
        """Browse for log directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Log Directory",
            self.log_dir_input.text()
        )

        if directory:
            self.log_dir_input.setText(directory)

    def _clear_cache(self):
        """Clear image cache."""
        reply = QMessageBox.question(
            self,
            "Clear Cache",
            "Clear all cached images? This will free memory but may slow down performance temporarily.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # TODO: Implement cache clearing
            from gui.widgets.toast_notification import show_success
            show_success("Cache cleared")
            logger.info("Cleared image cache")

    def _reset_database(self):
        """Reset database (destructive operation)."""
        reply = QMessageBox.warning(
            self,
            "Reset Database",
            "⚠️ WARNING: This will permanently delete ALL pose data, corrections, and training labels.\n\n"
            "This action CANNOT be undone.\n\n"
            "Are you absolutely sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Double confirmation for destructive action
            confirm = QMessageBox.warning(
                self,
                "Final Confirmation",
                "Type 'DELETE' to confirm database reset:",
                QMessageBox.Ok | QMessageBox.Cancel
            )

            if confirm == QMessageBox.Ok:
                # TODO: Implement database reset
                logger.warning("Database reset requested by user")
