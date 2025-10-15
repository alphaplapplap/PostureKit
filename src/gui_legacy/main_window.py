"""
Main application window for PostureKit
Complete implementation with all menu items, shortcuts, and features from specification
"""

import sys
import os
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime
from functools import partial
import json

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QStatusBar, QLabel, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QProgressBar, QDockWidget, QComboBox, QSizePolicy
)
from PySide6.QtCore import (
    Qt, Signal, QSettings, QSize, QTimer, QThread,
    QByteArray, QPoint, Slot, QDateTime, QUrl
)
from PySide6.QtGui import (
    QAction, QKeySequence, QCloseEvent, QIcon,
    QPixmap, QDesktopServices, QActionGroup
)

# Import all required components
from src.gui_legacy.image_viewer import ImageViewer
from src.gui_legacy.correction_panel import CorrectionPanel
from src.gui_legacy.training_intelligence_panel import TrainingIntelligencePanel
from src.gui_legacy.dialogs.preferences_dialog import PreferencesDialog
from src.gui_legacy.dialogs.about_dialog import AboutDialog
from src.gui_legacy.dialogs.statistics_dialog import StatisticsDialog
from src.gui_legacy.shortcuts_dialog import ShortcutsDialog
from src.gui_legacy.batch_dialog import BatchProcessDialog
from src.gui_legacy.export_dialog import ExportDialog
from src.gui_legacy.similarity_search_dialog import SimilaritySearchDialog
from src.gui_legacy.theme import ThemeManager
from src.gui_legacy.state_manager import StateManager
from src.gui_legacy.undo_stack import UndoStack
from src.gui_legacy.widgets.toast_notification import ToastNotification

# Core imports
from src.core.pose_detector import PoseDetector
from src.core.database_manager import DatabaseManager
from src.core.models import Pose, Person, Keypoint
from src.core.geometric_feature_extractor import GeometricFeatureExtractor
from src.intelligence.similarity_engine import SimilarityEngine
from src.utils.logging_config import get_logger
from src.utils.postgres_manager import ensure_postgres_running, stop_postgres

logger = get_logger(__name__)


class ViewMode(Enum):
    """View display modes"""
    NORMAL = "normal"
    SKELETON = "skeleton"
    KEYPOINTS = "keypoints"
    CONFIDENCE = "confidence"
    OCCLUSION = "occlusion"
    BOUNDING_BOX = "bbox"


class MainWindow(QMainWindow):
    """Main application window for PostureKit"""

    # Signals
    imageLoaded = Signal(str)
    poseDetected = Signal(list)  # List[Person]
    projectSaved = Signal()
    correctionsSaved = Signal()
    databaseStatusChanged = Signal(bool)  # connected

    def __init__(self):
        super().__init__()

        # Ensure PostgreSQL is running
        logger.info("Ensuring PostgreSQL is running...")
        if not ensure_postgres_running():
            logger.warning("PostgreSQL failed to start - database features may not work")

        # Core components
        self.state_manager = StateManager()
        self.theme_manager = ThemeManager()
        self.undo_stack = UndoStack(max_size=100)

        # Get active database profile from settings
        from src.gui_legacy.state_manager import SettingsGroup
        self.current_db_profile = self.state_manager.get_simple(
            SettingsGroup.DATABASE_PROFILE.key,
            SettingsGroup.DATABASE_PROFILE.default_value
        )
        logger.info(f"Loading database profile: {self.current_db_profile}")

        # Initialize database manager with profile (no global state mutation needed)
        self.db_manager = DatabaseManager(database_profile=self.current_db_profile)

        # Pass storage_manager to enable bias correction learning
        self.pose_detector = PoseDetector(storage_manager=self.db_manager.storage)

        # Initialize similarity engine with profile-aware index directory
        self.similarity_engine = SimilarityEngine(
            storage_manager=self.db_manager.storage,
            database_profile=self.current_db_profile
        )

        # State tracking
        self.current_image_path = None
        self.current_poses: List[Pose] = []
        self.current_person_idx = 0
        self.is_modified = False
        self.recent_files: List[str] = []
        self.max_recent_files = 10

        # Directory navigation
        self.image_files: List[str] = []
        self.current_image_index: int = -1
        self.current_directory: Optional[str] = None

        # Auto-save timer for crash recovery
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.autoSave)
        self.auto_save_interval = 2 * 60 * 1000  # 2 minutes in ms

        # Progress tracking
        self.current_operation = None
        self.operation_thread = None

        # Initialize directory indexing and similarity search integration
        # MUST be called BEFORE initUI() so create_similarity_menu method exists when menus are created
        from src.gui_legacy.similarity_integration_complete import add_similarity_search_to_mainwindow
        add_similarity_search_to_mainwindow(self)

        # Initialize UI
        self.initUI()
        self.loadSettings()

        # Set database selector to current profile
        profile_to_text = {"irl": "IRL", "2d": "2D Illustrations", "3d": "3D Illustrations"}
        self.database_selector.blockSignals(True)  # Prevent triggering change event
        self.database_selector.setCurrentText(profile_to_text.get(self.current_db_profile, "IRL"))
        self.database_selector.blockSignals(False)

        self.connectSignals()
        self.setupAutoSave()

        # Initialize enhanced similarity panel integration
        from src.gui_legacy.similarity_panel_integration import add_enhanced_similarity_to_mainwindow
        add_enhanced_similarity_to_mainwindow(self)

        # Check database connection
        self.checkDatabaseConnection()

        # Update window title and status bar with database info
        self._update_window_title_for_database()

    def initUI(self):
        """Initialize the complete user interface"""
        # Window title will be updated with database name
        profile_names = {"irl": "IRL", "2d": "2D Illustrations", "3d": "3D Illustrations"}
        db_name = profile_names.get(self.current_db_profile, "IRL")
        self.setWindowTitle(f"PostureKit - {db_name}")
        self.setGeometry(100, 100, 1400, 900)

        # Set application icon
        self.setWindowIcon(QIcon("resources/icons/app_icon.png"))

        # Apply theme
        self.applyTheme(self.state_manager.get_simple("theme", "dark"))

        # Create all UI components
        self.createActions()
        self.createMenus()
        self.createToolbars()
        self.setupCentralWidget()
        self.createDockWidgets()
        self.createStatusBar()

    def createActions(self):
        """Create all actions with shortcuts and icons"""
        # File actions
        self.action_open_image = QAction(QIcon("resources/icons/open.svg"), "Open Image...", self)
        self.action_open_image.setShortcut("Ctrl+O")
        self.action_open_image.triggered.connect(self.openImage)

        self.action_open_directory = QAction(QIcon("resources/icons/folder.svg"), "Open Directory...", self)
        self.action_open_directory.setShortcut("Ctrl+Shift+O")
        self.action_open_directory.triggered.connect(self.openDirectory)

        self.action_save_corrections = QAction(QIcon("resources/icons/save.svg"), "Save Corrections", self)
        self.action_save_corrections.setShortcut("Ctrl+S")
        self.action_save_corrections.triggered.connect(self.saveCorrections)

        self.action_export_corrections = QAction(QIcon("resources/icons/export.svg"), "Export Corrections...", self)
        self.action_export_corrections.setShortcut("Ctrl+E")
        self.action_export_corrections.triggered.connect(self.exportCorrections)

        self.action_export_dataset = QAction(QIcon("resources/icons/export_dataset.svg"), "Export Training Dataset...", self)
        self.action_export_dataset.setShortcut("Ctrl+Shift+E")
        self.action_export_dataset.triggered.connect(self.exportDataset)

        self.action_exit = QAction("Exit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)

        # Edit actions
        self.action_undo = QAction(QIcon("resources/icons/undo.svg"), "Undo", self)
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.triggered.connect(self.undo)
        self.action_undo.setEnabled(False)

        self.action_redo = QAction(QIcon("resources/icons/redo.svg"), "Redo", self)
        self.action_redo.setShortcut("Ctrl+Shift+Z")
        self.action_redo.triggered.connect(self.redo)
        self.action_redo.setEnabled(False)

        self.action_reset_corrections = QAction(QIcon("resources/icons/reset.svg"), "Reset Corrections", self)
        self.action_reset_corrections.setShortcut("Ctrl+R")
        self.action_reset_corrections.triggered.connect(self.resetCorrections)

        self.action_preferences = QAction(QIcon("resources/icons/settings.svg"), "Preferences...", self)
        self.action_preferences.setShortcut("Ctrl+,")
        self.action_preferences.triggered.connect(self.openPreferences)

        # View actions (checkable)
        self.action_zoom_in = QAction(QIcon("resources/icons/zoom_in.svg"), "Zoom In", self)
        self.action_zoom_in.setShortcut("Ctrl++")
        self.action_zoom_in.triggered.connect(self.zoomIn)

        self.action_zoom_out = QAction(QIcon("resources/icons/zoom_out.svg"), "Zoom Out", self)
        self.action_zoom_out.setShortcut("Ctrl+-")
        self.action_zoom_out.triggered.connect(self.zoomOut)

        self.action_fit_window = QAction("Fit to Window", self)
        self.action_fit_window.setShortcut("Ctrl+0")
        self.action_fit_window.triggered.connect(self.fitToWindow)

        self.action_reset_view = QAction("Reset View", self)
        self.action_reset_view.setShortcut("Ctrl+H")
        self.action_reset_view.setToolTip("Reset to 1:1 zoom and center image")
        self.action_reset_view.triggered.connect(self.resetView)

        self.action_actual_size = QAction("Actual Size (100%)", self)
        self.action_actual_size.setShortcut("Ctrl+1")
        self.action_actual_size.triggered.connect(self.actualSize)

        self.action_zoom_person = QAction("Zoom to Person", self)
        self.action_zoom_person.setShortcut("Ctrl+2")
        self.action_zoom_person.triggered.connect(self.zoomToPerson)

        self.action_show_skeleton = QAction("Show Skeleton", self)
        self.action_show_skeleton.setShortcut("Ctrl+K")
        self.action_show_skeleton.setCheckable(True)
        self.action_show_skeleton.setChecked(True)
        self.action_show_skeleton.toggled.connect(self.toggleSkeleton)

        self.action_show_keypoints = QAction("Show Keypoints", self)
        self.action_show_keypoints.setShortcut("Ctrl+J")
        self.action_show_keypoints.setCheckable(True)
        self.action_show_keypoints.setChecked(True)
        self.action_show_keypoints.toggled.connect(self.toggleKeypoints)

        self.action_show_confidence = QAction("Show Confidence Colors", self)
        self.action_show_confidence.setCheckable(True)
        self.action_show_confidence.setChecked(True)
        self.action_show_confidence.toggled.connect(self.toggleConfidenceColors)

        self.action_show_bbox = QAction("Show Bounding Boxes", self)
        self.action_show_bbox.setCheckable(True)
        self.action_show_bbox.setChecked(True)
        self.action_show_bbox.toggled.connect(self.toggleBoundingBoxes)

        self.action_color_coded = QAction("Color-Coded Body Parts", self)
        self.action_color_coded.setCheckable(True)
        self.action_color_coded.setChecked(False)
        self.action_color_coded.toggled.connect(self.toggleColorCoded)

        self.action_show_correction_panel = QAction("Show Correction Panel", self)
        self.action_show_correction_panel.setShortcut("Ctrl+P")
        self.action_show_correction_panel.setCheckable(True)
        self.action_show_correction_panel.setChecked(True)
        self.action_show_correction_panel.toggled.connect(self.toggleCorrectionPanel)

        self.action_show_training_intel = QAction("Show Training Intelligence", self)
        self.action_show_training_intel.setCheckable(True)
        self.action_show_training_intel.triggered.connect(self.toggleTrainingIntelligence)

        # Process actions
        self.action_detect_poses = QAction(QIcon("resources/icons/detect.svg"), "Detect Poses", self)
        self.action_detect_poses.setShortcut("Ctrl+D")
        self.action_detect_poses.triggered.connect(self.detectPoses)

        self.action_batch_process = QAction(QIcon("resources/icons/batch.svg"), "Batch Process...", self)
        self.action_batch_process.setShortcut("Ctrl+B")
        self.action_batch_process.triggered.connect(self.openBatchDialog)

        # Database actions
        self.action_delete_pose = QAction("Delete Pose...", self)
        self.action_delete_pose.setShortcut("Ctrl+Shift+D")
        self.action_delete_pose.triggered.connect(self.deletePose)

        self.action_clear_labels = QAction("Clear Labels...", self)
        self.action_clear_labels.triggered.connect(self.clearLabels)

        self.action_view_statistics = QAction("View Statistics", self)
        self.action_view_statistics.setShortcut("Ctrl+I")
        self.action_view_statistics.triggered.connect(self.viewStatistics)

        self.action_similarity_search = QAction("Find Similar Poses...", self)
        self.action_similarity_search.setShortcut("Ctrl+F")
        self.action_similarity_search.triggered.connect(self.similaritySearch)

        # Navigation actions
        self.action_prev_person = QAction(QIcon("resources/icons/prev.svg"), "Previous Person", self)
        self.action_prev_person.setShortcut("Shift+Tab")
        self.action_prev_person.triggered.connect(self.previousPerson)

        self.action_next_person = QAction(QIcon("resources/icons/next.svg"), "Next Person", self)
        self.action_next_person.setShortcut("Tab")
        self.action_next_person.triggered.connect(self.nextPerson)

        self.action_prev_image = QAction("Previous Image", self)
        self.action_prev_image.setShortcut("Ctrl+Left")
        self.action_prev_image.triggered.connect(self.previousImage)

        self.action_next_image = QAction("Next Image", self)
        self.action_next_image.setShortcut("Ctrl+Right")
        self.action_next_image.triggered.connect(self.nextImage)

        # Window actions
        self.action_fullscreen = QAction(QIcon("resources/icons/fullscreen.svg"), "Fullscreen", self)
        self.action_fullscreen.setShortcut("F11")
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.triggered.connect(self.toggleFullscreen)

        # Help actions
        self.action_shortcuts = QAction("Keyboard Shortcuts", self)
        self.action_shortcuts.setShortcut("F1")
        self.action_shortcuts.triggered.connect(self.showShortcuts)

        self.action_documentation = QAction("Documentation", self)
        self.action_documentation.triggered.connect(self.openDocumentation)

        self.action_about = QAction("About PostureKit", self)
        self.action_about.triggered.connect(self.showAbout)

    def createMenus(self):
        """Create complete menu system"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.action_open_image)
        file_menu.addAction(self.action_open_directory)

        # Recent Files submenu
        self.recent_menu = file_menu.addMenu(QIcon("resources/icons/recent.svg"), "Recent Files")
        self.updateRecentFilesMenu()

        file_menu.addSeparator()
        file_menu.addAction(self.action_save_corrections)
        file_menu.addAction(self.action_export_corrections)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_reset_corrections)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_preferences)

        # View Menu
        view_menu = menubar.addMenu("View")

        # Zoom submenu
        zoom_menu = view_menu.addMenu("Zoom")
        zoom_menu.addAction(self.action_zoom_in)
        zoom_menu.addAction(self.action_zoom_out)
        zoom_menu.addSeparator()
        zoom_menu.addAction(self.action_fit_window)
        zoom_menu.addAction(self.action_reset_view)
        zoom_menu.addAction(self.action_actual_size)
        zoom_menu.addAction(self.action_zoom_person)

        view_menu.addSeparator()

        # Display options
        view_menu.addAction(self.action_show_skeleton)
        view_menu.addAction(self.action_show_keypoints)
        view_menu.addAction(self.action_show_confidence)
        view_menu.addAction(self.action_color_coded)
        view_menu.addAction(self.action_show_bbox)

        view_menu.addSeparator()

        # Panel visibility
        view_menu.addAction(self.action_show_correction_panel)
        view_menu.addAction(self.action_show_training_intel)

        view_menu.addSeparator()

        # Theme submenu with radio buttons
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)

        self.action_theme_light = QAction("Light Theme", self)
        self.action_theme_light.setCheckable(True)
        self.action_theme_light.triggered.connect(partial(self.setTheme, "light"))
        theme_group.addAction(self.action_theme_light)
        theme_menu.addAction(self.action_theme_light)

        self.action_theme_dark = QAction("Dark Theme", self)
        self.action_theme_dark.setCheckable(True)
        self.action_theme_dark.setChecked(True)
        self.action_theme_dark.triggered.connect(partial(self.setTheme, "dark"))
        theme_group.addAction(self.action_theme_dark)
        theme_menu.addAction(self.action_theme_dark)

        # Process Menu
        process_menu = menubar.addMenu("Process")
        process_menu.addAction(self.action_detect_poses)
        process_menu.addAction(self.action_batch_process)

        # Search Menu (Directory Indexing & Similarity Search)
        search_menu = menubar.addMenu("Search")
        if hasattr(self, 'create_similarity_menu'):
            self.create_similarity_menu(search_menu)

        # Export Menu
        export_menu = menubar.addMenu("Export")
        export_menu.addAction(self.action_export_dataset)
        export_menu.addAction(self.action_export_corrections)

        # Database Menu
        database_menu = menubar.addMenu("Database")
        database_menu.addAction(self.action_delete_pose)
        database_menu.addAction(self.action_clear_labels)
        database_menu.addSeparator()
        database_menu.addAction(self.action_view_statistics)
        database_menu.addAction(self.action_similarity_search)

        # Help Menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(self.action_shortcuts)
        help_menu.addAction(self.action_documentation)
        help_menu.addSeparator()
        help_menu.addAction(self.action_about)

    def createToolbars(self):
        """Create main toolbar with all tools"""
        # Main toolbar
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)

        # File operations
        toolbar.addAction(self.action_open_image)
        toolbar.addAction(self.action_detect_poses)
        toolbar.addAction(self.action_batch_process)
        toolbar.addAction(self.action_save_corrections)
        toolbar.addSeparator()

        # Export
        toolbar.addAction(self.action_export_dataset)
        toolbar.addSeparator()

        # Navigation
        toolbar.addAction(self.action_prev_person)
        toolbar.addAction(self.action_next_person)
        toolbar.addSeparator()

        # View controls
        toolbar.addAction(self.action_zoom_in)
        toolbar.addAction(self.action_zoom_out)
        toolbar.addAction(self.action_reset_view)

        toolbar.addSeparator()

        # Other tools
        toolbar.addAction(self.action_fullscreen)
        toolbar.addAction(self.action_preferences)

        # Add spacer to push database selector to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Database selector (prominent, always visible)
        toolbar.addSeparator()
        db_label = QLabel("Database:")
        db_label.setStyleSheet("font-weight: bold; font-size: 11pt; padding-right: 5px;")
        toolbar.addWidget(db_label)

        self.database_selector = QComboBox()
        self.database_selector.addItems(["IRL", "2D Illustrations", "3D Illustrations"])
        self.database_selector.setStyleSheet("""
            QComboBox {
                font-size: 11pt;
                font-weight: bold;
                padding: 5px 10px;
                min-width: 150px;
                background-color: #E3F2FD;
                border: 2px solid #2196F3;
                border-radius: 4px;
            }
            QComboBox:hover {
                background-color: #BBDEFB;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.database_selector.setToolTip("Switch between isolated database profiles")
        self.database_selector.currentTextChanged.connect(self._on_database_changed)
        toolbar.addWidget(self.database_selector)

    def setupCentralWidget(self):
        """Setup the central widget with main layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Horizontal splitter for viewer and panel
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(1)

        # Image viewer (left, 65%)
        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumWidth(600)
        self.main_splitter.addWidget(self.image_viewer)

        # Correction panel (right, 35%)
        self.correction_panel = CorrectionPanel()
        self.correction_panel.setMaximumWidth(450)
        self.main_splitter.addWidget(self.correction_panel)

        # Set initial splitter sizes
        self.main_splitter.setSizes([900, 500])
        self.main_splitter.splitterMoved.connect(self.onSplitterMoved)

        main_layout.addWidget(self.main_splitter)

        central_widget.setLayout(main_layout)

    def createDockWidgets(self):
        """Create dockable widgets"""
        # Training Intelligence Panel (dockable)
        self.training_dock = TrainingIntelligencePanel(self)
        self.training_dock.setVisible(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.training_dock)

    def createStatusBar(self):
        """Create comprehensive status bar"""
        status = self.statusBar()

        # Database connection status with indicator
        self.db_status_indicator = QLabel("●")
        self.db_status_indicator.setStyleSheet("color: #10b981; font-size: 16px;")
        status.addPermanentWidget(self.db_status_indicator)

        self.db_status_label = QLabel("DB: Connected")
        status.addPermanentWidget(self.db_status_label)

        status.addPermanentWidget(self.createSeparator())

        # Pose statistics
        self.pose_count_label = QLabel("Poses: 0")
        status.addPermanentWidget(self.pose_count_label)

        status.addPermanentWidget(self.createSeparator())

        # Image counter for directory navigation
        self.image_counter_label = QLabel("Image: -")
        status.addPermanentWidget(self.image_counter_label)

        status.addPermanentWidget(self.createSeparator())

        # Current pose confidence
        self.confidence_label = QLabel("Confidence: -")
        status.addPermanentWidget(self.confidence_label)

        status.addPermanentWidget(self.createSeparator())

        # Occlusion stats
        self.occlusion_label = QLabel("Occlusion: -")
        status.addPermanentWidget(self.occlusion_label)

        status.addPermanentWidget(self.createSeparator())

        # Corrections count
        self.corrections_label = QLabel("Corrections: 0")
        status.addPermanentWidget(self.corrections_label)

        status.addPermanentWidget(self.createSeparator())

        # Zoom level
        self.zoom_label = QLabel("Zoom: 100%")
        status.addPermanentWidget(self.zoom_label)

        status.addPermanentWidget(self.createSeparator())

        # Device/acceleration status
        self.device_label = QLabel("Device: MPS")
        status.addPermanentWidget(self.device_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        status.addPermanentWidget(self.progress_bar)

    def createSeparator(self):
        """Create a separator for status bar"""
        sep = QLabel(" | ")
        sep.setStyleSheet("color: #6b7280;")
        return sep

    def connectSignals(self):
        """Connect all internal signals"""
        # Image viewer signals
        self.image_viewer.zoomChanged.connect(self.updateZoomLabel)
        self.image_viewer.personSelected.connect(self.onPersonSelected)
        self.image_viewer.keypointMoved.connect(self.onKeypointMoved)
        self.image_viewer.modificationMade.connect(self.setModified)
        self.image_viewer.imageDropped.connect(self.loadImage)

        # Correction panel signals
        self.correction_panel.labelsChanged.connect(self.onLabelsChanged)
        self.correction_panel.saveRequested.connect(self.saveCorrections)
        self.correction_panel.deleteRequested.connect(self.deletePose)
        self.correction_panel.resetRequested.connect(self.resetCorrections)
        self.correction_panel.occlusionDisplayChanged.connect(self.image_viewer.setOcclusionDisplay)
        self.correction_panel.keypointSelectedInList.connect(self.onKeypointSelectedInList)
        self.correction_panel.keypointsSelectedInList.connect(self.onKeypointsSelectedInList)
        self.correction_panel.visibilityChanged.connect(self.onVisibilityChanged)

        # Undo stack signals
        self.undo_stack.canUndoChanged.connect(self.action_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.action_redo.setEnabled)

        # Database signals
        self.db_manager.connectionChanged.connect(self.onDatabaseConnectionChanged)

        # Training intelligence signals
        if hasattr(self, 'training_dock'):
            self.training_dock.poseSelected.connect(self.loadPoseById)

    def setupAutoSave(self):
        """Setup auto-save for crash recovery"""
        if self.state_manager.get_simple("crash_recovery_enabled", True):
            interval = self.state_manager.get_simple("crash_recovery_interval", 2) * 60 * 1000
            self.auto_save_timer.setInterval(interval)
            self.auto_save_timer.start()
            logger.info(f"Auto-save enabled with {interval/60000:.1f} minute interval")

    def autoSave(self):
        """Perform auto-save for crash recovery"""
        if self.is_modified and self.current_poses:
            try:
                recovery_path = self.state_manager.getRecoveryPath()
                recovery_data = {
                    'timestamp': QDateTime.currentDateTime().toString(),
                    'image_path': self.current_image_path,
                    'poses': [pose.to_dict() for pose in self.current_poses],
                    'current_person': self.current_person_idx,
                    'corrections': self.undo_stack.getHistory()
                }

                with open(recovery_path, 'w') as f:
                    json.dump(recovery_data, f)

                logger.debug("Auto-save completed")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def checkCrashRecovery(self):
        """Check for crash recovery file on startup"""
        recovery_path = self.state_manager.getRecoveryPath()
        if os.path.exists(recovery_path):
            reply = QMessageBox.question(
                self, "Recover Previous Session",
                "A previous session was not properly closed. Would you like to recover your work?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.recoverSession(recovery_path)
            else:
                os.remove(recovery_path)

    def recoverSession(self, recovery_path):
        """Recover previous session from auto-save"""
        try:
            with open(recovery_path, 'r') as f:
                recovery_data = json.load(f)

            # Restore image
            if recovery_data['image_path'] and os.path.exists(recovery_data['image_path']):
                self.loadImage(recovery_data['image_path'])

            # Restore poses
            self.current_poses = [Pose.from_dict(p) for p in recovery_data['poses']]
            self.image_viewer.setPoses(self.current_poses)

            # Restore selection
            self.current_person_idx = recovery_data.get('current_person', 0)
            self.selectPerson(self.current_person_idx)

            # Restore undo history
            if 'corrections' in recovery_data:
                self.undo_stack.restoreHistory(recovery_data['corrections'])

            self.showToast("Session recovered successfully", ToastNotification.SUCCESS)
            os.remove(recovery_path)

        except Exception as e:
            logger.error(f"Failed to recover session: {e}")
            self.showToast("Failed to recover session", ToastNotification.ERROR)

    # File operations
    def openImage(self):
        """Open image file dialog"""
        last_dir = self.state_manager.get_simple("last_directory", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", last_dir,
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;All Files (*.*)"
        )

        if file_path:
            self.loadImage(file_path)
            self.addToRecentFiles(file_path)
            self.state_manager.set_simple("last_directory", os.path.dirname(file_path))

    def loadImage(self, file_path):
        """Load an image file"""
        try:
            self.image_viewer.loadImage(file_path)
            self.current_image_path = file_path
            self.setWindowTitle(f"PostureKit - {os.path.basename(file_path)}")
            self.imageLoaded.emit(file_path)
            self.action_detect_poses.setEnabled(True)

            # Check if poses exist in database
            existing_pose_data = self.db_manager.getPosesForImage(file_path)
            if existing_pose_data:
                # Extract all persons from pose objects
                persons = []
                for pose in existing_pose_data:
                    persons.extend(pose.persons)

                self.current_poses = persons
                self.image_viewer.setPoses(persons)
                self.updatePoseCount()
                self.showToast(f"Loaded {len(persons)} person(s)", ToastNotification.INFO)
                # Auto-select first person
                self.selectPerson(0)

            # Update similarity panel if visible and we have poses
            if existing_pose_data and len(existing_pose_data) > 0:
                if hasattr(self, 'similarity_panel') and hasattr(self, 'similarity_dock'):
                    if self.similarity_dock.isVisible():
                        # Get the first pose's ID to trigger panel update
                        first_pose = existing_pose_data[0]
                        if hasattr(first_pose, 'id'):
                            self.similarity_panel.set_current_pose(first_pose.id)

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def openDirectory(self):
        """Open directory browser"""
        last_dir = self.state_manager.get_simple("last_directory", "")
        dir_path = QFileDialog.getExistingDirectory(
            self, "Open Directory", last_dir
        )

        if dir_path:
            self.loadDirectory(dir_path)
            self.state_manager.set_simple("last_directory", dir_path)

    def loadDirectory(self, dir_path):
        """Load all images from a directory"""
        if not os.path.isdir(dir_path):
            QMessageBox.warning(self, "Invalid Directory", f"The path is not a directory: {dir_path}")
            return

        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

        # Scan directory for image files
        self.image_files = []
        try:
            for file in sorted(os.listdir(dir_path)):
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(dir_path, file)
                    self.image_files.append(full_path)

            if not self.image_files:
                QMessageBox.information(self, "No Images", f"No supported image files found in:\n{dir_path}")
                return

            # Set directory state
            self.current_directory = dir_path
            self.current_image_index = 0

            # Load first image
            self.loadImage(self.image_files[0])

            # Update status
            self.updateImageCounter()
            self.showToast(f"Loaded directory with {len(self.image_files)} images", ToastNotification.SUCCESS)

        except Exception as e:
            logger.error(f"Failed to load directory {dir_path}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load directory:\n{e}")

    def saveCorrections(self):
        """Save corrections to database"""
        if not self.current_poses:
            logger.debug("No poses to save")
            return

        try:
            # Count modified poses
            modified_count = sum(1 for pose in self.current_poses if pose.is_modified)

            if modified_count == 0:
                logger.debug("No modified poses to save")
                self.showToast("No changes to save", ToastNotification.INFO)
                return

            # Save all modified poses
            saved_count = 0
            for pose in self.current_poses:
                if pose.is_modified:
                    success = self.db_manager.updatePose(pose)
                    if success:
                        saved_count += 1

            self.is_modified = False
            self.correctionsSaved.emit()
            logger.info(f"Saved {saved_count}/{modified_count} corrections")
            self.showToast(f"Saved {saved_count} correction(s)", ToastNotification.SUCCESS)

            # Auto-training: Train every 10 corrections for incremental learning
            if self.pose_detector.bias_corrector is not None:
                correction_count = self.db_manager.storage.count_corrected_poses()

                # Train when we hit 10, 20, 30, etc. corrections
                if correction_count >= 10 and correction_count % 10 == 0:
                    logger.info(f"Auto-training triggered at {correction_count} corrections")
                    try:
                        stats = self.pose_detector.bias_corrector.train()
                        self.showToast(
                            f"Auto-trained from {correction_count} corrections "
                            f"({stats['biases_learned']} biases learned)",
                            ToastNotification.INFO
                        )

                        # Update training panel if visible
                        if hasattr(self, 'training_dock'):
                            self.training_dock._update_bias_status()

                    except Exception as e:
                        logger.warning(f"Auto-training failed: {e}", exc_info=True)
                        # Don't show error to user - training failure shouldn't block saves

        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save corrections:\n{e}")

    def exportCorrections(self):
        """Export current corrections to JSON"""
        if not self.current_poses:
            QMessageBox.warning(self, "No Poses", "No poses to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Corrections", "",
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                export_data = {
                    'image': self.current_image_path,
                    'timestamp': QDateTime.currentDateTime().toString(),
                    'poses': [pose.to_dict() for pose in self.current_poses]
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                self.showToast("Corrections exported successfully", ToastNotification.SUCCESS)

            except Exception as e:
                logger.error(f"Failed to export corrections: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export:\n{e}")

    def exportDataset(self):
        """Open export dataset dialog"""
        dialog = ExportDialog(self.db_manager.storage, self)
        if dialog.exec():
            self.showToast("Dataset export dialog completed", ToastNotification.INFO)

    def addToRecentFiles(self, file_path):
        """Add file to recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)

        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:self.max_recent_files]

        self.state_manager.set_simple("recent_files", self.recent_files)
        self.updateRecentFilesMenu()

    def updateRecentFilesMenu(self):
        """Update recent files menu"""
        self.recent_menu.clear()

        if not self.recent_files:
            action = self.recent_menu.addAction("(No recent files)")
            action.setEnabled(False)
            return

        for i, file_path in enumerate(self.recent_files, 1):
            if i <= 9:
                text = f"&{i}. {os.path.basename(file_path)}"
            else:
                text = os.path.basename(file_path)

            action = self.recent_menu.addAction(QIcon("resources/icons/image.svg"), text)
            action.setData(file_path)
            action.triggered.connect(partial(self.loadImage, file_path))

    # Edit operations
    def undo(self):
        """Undo last operation"""
        self.undo_stack.undo()
        self.image_viewer.update()
        self.correction_panel.refresh()

    def redo(self):
        """Redo last undone operation"""
        self.undo_stack.redo()
        self.image_viewer.update()
        self.correction_panel.refresh()

    def resetCorrections(self):
        """Reset all corrections with confirmation"""
        if not self.current_poses:
            return

        reply = QMessageBox.question(
            self, 'Reset Corrections',
            'Are you sure you want to reset all corrections to the original detection?\n\n'
            'This will undo all manual edits.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for pose in self.current_poses:
                pose.resetToOriginal()

            self.image_viewer.update()
            self.correction_panel.refresh()
            self.undo_stack.clear()
            self.showToast("Corrections reset to original", ToastNotification.INFO)

    # View operations
    def zoomIn(self):
        """Zoom in the image viewer"""
        self.image_viewer.zoomIn()

    def zoomOut(self):
        """Zoom out the image viewer"""
        self.image_viewer.zoomOut()

    def fitToWindow(self):
        """Fit image to window"""
        self.image_viewer.fitToWindow()

    def resetView(self):
        """Reset view to 1:1 zoom and center"""
        self.image_viewer.resetView()

    def actualSize(self):
        """Reset to actual size (100%)"""
        self.image_viewer.setZoom(1.0)

    def zoomToPerson(self):
        """Zoom to selected person's bounding box"""
        if self.current_poses and self.current_person_idx < len(self.current_poses):
            person = self.current_poses[self.current_person_idx]
            self.image_viewer.zoomToBounds(person.bbox)

    def resetView(self):
        """Reset viewer zoom and pan"""
        self.image_viewer.resetView()

    def toggleSkeleton(self, checked):
        """Toggle skeleton visibility"""
        self.image_viewer.show_skeleton = checked
        self.image_viewer.update()

    def toggleKeypoints(self, checked):
        """Toggle keypoint visibility"""
        self.image_viewer.show_keypoints = checked
        self.image_viewer.update()

    def toggleConfidenceColors(self, checked):
        """Toggle confidence-based coloring"""
        self.image_viewer.show_confidence_colors = checked
        self.image_viewer.update()

    def toggleBoundingBoxes(self, checked):
        """Toggle bounding box visibility"""
        self.image_viewer.show_bboxes = checked
        self.image_viewer.update()

    def toggleColorCoded(self, checked):
        """Toggle color-coded body parts"""
        self.image_viewer.use_colored_parts = checked
        self.image_viewer.update()

    def toggleCorrectionPanel(self, checked):
        """Toggle correction panel visibility"""
        self.correction_panel.setVisible(checked)
        if checked:
            # Restore previous splitter sizes
            sizes = self.state_manager.get_simple("splitter_sizes", [900, 500])
            self.main_splitter.setSizes(sizes)
        else:
            # Maximize viewer
            self.main_splitter.setSizes([self.main_splitter.width(), 0])

    def toggleTrainingIntelligence(self, checked):
        """Toggle training intelligence panel"""
        self.training_dock.setVisible(checked)

    def toggleFullscreen(self, checked):
        """Toggle fullscreen mode"""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    # Process operations
    def detectPoses(self):
        """Run pose detection on current image"""
        print(f"[DEBUG] detectPoses() called")  # Debug log
        print(f"[DEBUG] current_image_path = {self.current_image_path}")  # Debug log

        if not self.current_image_path:
            print(f"[ERROR] No image loaded, showing warning")  # Debug log
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return

        print(f"[DEBUG] Starting pose detection on: {self.current_image_path}")  # Debug log
        self.showProgress("Detecting poses...")

        try:
            # Load image as numpy array
            import cv2
            print(f"[DEBUG] Loading image with cv2.imread()...")  # Debug log
            image = cv2.imread(self.current_image_path)

            if image is None:
                error_msg = f"Failed to load image with cv2.imread(): {self.current_image_path}. Check file exists and is readable."
                print(f"[ERROR] {error_msg}")  # Debug log
                raise ValueError(error_msg)

            print(f"[DEBUG] Image loaded: {image.shape}, dtype={image.dtype}")  # Debug log

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"[DEBUG] Image converted to RGB")  # Debug log

            # Run detection
            print(f"[DEBUG] Calling pose_detector.detect()...")  # Debug log
            pose_results = self.pose_detector.detect(image)
            print(f"[DEBUG] Detection complete, found {len(pose_results)} poses")  # Debug log

            # Convert PoseResult objects to Person objects for GUI
            from core.models import Person, Keypoint, KEYPOINT_NAMES
            persons = []
            for person_id, result in enumerate(pose_results):
                # Convert numpy keypoints to Keypoint objects
                keypoints = []
                for idx in range(133):
                    x, y, conf = result.keypoints[idx]
                    visibility = result.visibility[idx] if result.visibility is not None else 2

                    # Apply confidence-based visibility correction
                    # Low confidence keypoints are likely hallucinated or occluded
                    if conf < 0.3:
                        visibility = 0  # Missing - very low confidence
                    elif conf < 0.7:
                        visibility = 1  # Occluded - medium confidence, uncertain

                    # Debug logging for foot keypoints
                    if idx in [15, 16, 17, 18, 19, 20, 21, 22]:  # Ankles and feet
                        print(f"[DEBUG] Person {person_id}, Keypoint {idx} ({KEYPOINT_NAMES.get(idx, f'point_{idx}')}): "
                              f"vis={visibility}, conf={conf:.2f}, pos=({x:.1f}, {y:.1f})")

                    kp = Keypoint(
                        index=idx,
                        name=KEYPOINT_NAMES.get(idx, f"point_{idx}"),
                        x=float(x),
                        y=float(y),
                        confidence=float(conf),
                        visibility=int(visibility),
                        is_corrected=False
                    )
                    keypoints.append(kp)

                # Create Person object
                person = Person(
                    id=person_id,
                    bbox=tuple(int(b) for b in result.bbox),
                    keypoints=keypoints,
                    confidence=float(result.overall_confidence),
                    total_count=len(pose_results)
                )
                persons.append(person)

            self.current_poses = persons
            self.image_viewer.setPoses(persons)
            self.updatePoseCount()
            self.hideProgress()
            self.showToast(f"Detected {len(persons)} pose(s)", ToastNotification.SUCCESS)

            # Auto-select first person if poses detected
            if persons:
                self.selectPerson(0)

        except Exception as e:
            self.hideProgress()
            logger.error(f"Pose detection failed: {e}")
            QMessageBox.critical(self, "Detection Failed", f"Failed to detect poses:\n{e}")

    def openBatchDialog(self):
        """Open batch processing dialog"""
        dialog = BatchProcessDialog(self)
        dialog.exec()

    # Database operations
    def deletePose(self):
        """Delete current pose from database"""
        if not self.current_poses or self.current_person_idx >= len(self.current_poses):
            return

        reply = QMessageBox.question(
            self, 'Delete Pose',
            'Are you sure you want to delete this pose?\n\n'
            'This action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            pose = self.current_poses[self.current_person_idx]
            try:
                self.db_manager.deletePose(pose.id)
                self.current_poses.pop(self.current_person_idx)
                self.image_viewer.setPoses(self.current_poses)
                self.updatePoseCount()

                # Select next person or previous if last
                if self.current_person_idx >= len(self.current_poses):
                    self.current_person_idx = max(0, len(self.current_poses) - 1)

                if self.current_poses:
                    self.selectPerson(self.current_person_idx)

                self.showToast("Pose deleted", ToastNotification.INFO)

            except Exception as e:
                logger.error(f"Failed to delete pose: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete pose:\n{e}")

    def clearLabels(self):
        """Clear training labels from poses"""
        reply = QMessageBox.question(
            self, 'Clear Labels',
            'Clear all training labels from selected poses?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Implementation
            pass

    def viewStatistics(self):
        """Open statistics dialog"""
        dialog = StatisticsDialog(self.db_manager, self)
        dialog.exec()

    def similaritySearch(self):
        """Open similarity search dialog"""
        try:
            dialog = SimilaritySearchDialog(
                similarity_engine=self.similarity_engine,
                pose_detector=self.pose_detector,
                storage_manager=self.db_manager.storage,
                parent=self
            )

            # Connect pose selection signal
            dialog.pose_selected.connect(self.onSimilarPoseSelected)

            dialog.exec()

        except Exception as e:
            logger.error(f"Failed to open similarity search dialog: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open similarity search:\n{e}")

    def onSimilarPoseSelected(self, pose_id: str, image_path: str):
        """Handle pose selection from similarity search"""
        try:
            # Load the image containing the selected pose
            self.loadImage(image_path)

            # Find and select the specific pose in the loaded image
            # The pose should already be loaded from database in loadImage()
            # We need to find which person index matches the pose_id
            from uuid import UUID
            target_uuid = UUID(pose_id)

            for idx, person in enumerate(self.current_poses):
                if hasattr(person, 'db_id') and person.db_id == target_uuid:
                    self.selectPerson(idx)
                    self.showToast(f"Loaded similar pose from {os.path.basename(image_path)}", ToastNotification.SUCCESS)
                    break
            else:
                # If pose not found in current poses, just show the image
                self.showToast(f"Loaded image: {os.path.basename(image_path)}", ToastNotification.INFO)

        except Exception as e:
            logger.error(f"Failed to load similar pose: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load pose:\n{e}")

    def checkDatabaseConnection(self):
        """Check and display database connection status"""
        is_connected = self.db_manager.isConnected()
        self.onDatabaseConnectionChanged(is_connected)

    def onDatabaseConnectionChanged(self, connected):
        """Handle database connection change"""
        if connected:
            self.db_status_indicator.setStyleSheet("color: #10b981;")  # Green
            self.db_status_label.setText("DB: Connected")

            # Update pose count
            count = self.db_manager.getPoseCount()
            self.pose_count_label.setText(f"Poses: {count}")
        else:
            self.db_status_indicator.setStyleSheet("color: #ef4444;")  # Red
            self.db_status_label.setText("DB: Disconnected")
            self.pose_count_label.setText("Poses: -")

    # Navigation
    def previousPerson(self):
        """Select previous person"""
        if self.current_poses and len(self.current_poses) > 1:
            self.current_person_idx = (self.current_person_idx - 1) % len(self.current_poses)
            self.selectPerson(self.current_person_idx)

    def nextPerson(self):
        """Select next person"""
        if self.current_poses and len(self.current_poses) > 1:
            self.current_person_idx = (self.current_person_idx + 1) % len(self.current_poses)
            self.selectPerson(self.current_person_idx)

    def selectPerson(self, index):
        """Select a person by index"""
        if 0 <= index < len(self.current_poses):
            self.current_person_idx = index
            person = self.current_poses[index]

            # Update viewer (will emit personSelected signal - handled by onPersonSelected)
            self.image_viewer.selectPerson(index)

            # Update correction panel directly
            self.correction_panel.updatePerson(person)

            # Update status bar
            self.updatePersonStats(person)

    def previousImage(self):
        """Navigate to previous image in directory"""
        if not self.image_files:
            self.showToast("No directory loaded. Use File → Open Directory", ToastNotification.WARNING)
            return

        # Check for unsaved changes
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before navigating to previous image?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )

            if reply == QMessageBox.Save:
                self.saveCorrections()
            elif reply == QMessageBox.Cancel:
                return

        # Navigate to previous image (wrap around)
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        self.loadImage(self.image_files[self.current_image_index])
        self.updateImageCounter()

    def nextImage(self):
        """Navigate to next image in directory"""
        if not self.image_files:
            self.showToast("No directory loaded. Use File → Open Directory", ToastNotification.WARNING)
            return

        # Check for unsaved changes
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before navigating to next image?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )

            if reply == QMessageBox.Save:
                self.saveCorrections()
            elif reply == QMessageBox.Cancel:
                return

        # Navigate to next image (wrap around)
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.loadImage(self.image_files[self.current_image_index])
        self.updateImageCounter()

    # Status updates
    def updateZoomLabel(self, zoom):
        """Update zoom level in status bar"""
        self.zoom_label.setText(f"Zoom: {int(zoom * 100)}%")

    def updateImageCounter(self):
        """Update image counter in status bar"""
        if self.image_files and self.current_image_index >= 0:
            self.image_counter_label.setText(f"Image: {self.current_image_index + 1}/{len(self.image_files)}")
        else:
            self.image_counter_label.setText("Image: -")

    def updatePoseCount(self):
        """Update pose count in status bar"""
        if self.current_poses:
            self.pose_count_label.setText(f"Poses: {len(self.current_poses)}")

    def updatePersonStats(self, person):
        """Update person statistics in status bar"""
        if person:
            # Calculate average confidence
            confidences = [kp.confidence for kp in person.keypoints if kp.visibility > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            self.confidence_label.setText(f"Confidence: {avg_conf:.2f}")

            # Calculate occlusion
            occluded = sum(1 for kp in person.keypoints if kp.visibility == 1)
            missing = sum(1 for kp in person.keypoints if kp.visibility == 0)
            total = len(person.keypoints)
            occlusion_pct = ((occluded + missing) / total * 100) if total > 0 else 0
            self.occlusion_label.setText(f"Occlusion: {occlusion_pct:.1f}%")

            # Count corrections
            corrections = sum(1 for kp in person.keypoints if kp.is_corrected)
            self.corrections_label.setText(f"Corrections: {corrections}")

    def onSplitterMoved(self, pos, index):
        """Handle splitter movement"""
        self.state_manager.set_simple("splitter_sizes", self.main_splitter.sizes())

    def onPersonSelected(self, index):
        """Handle person selection from viewer"""
        # Avoid recursion - only update if different person
        if index != self.current_person_idx:
            self.selectPerson(index)

    def onKeypointSelectedInList(self, keypoint_idx):
        """Handle single keypoint selection from occlusion editor list"""
        try:
            if self.current_person_idx >= 0 and self.current_person_idx < len(self.current_poses):
                logger.debug(f"Selecting keypoint {keypoint_idx} in image viewer for person {self.current_person_idx}")
                self.image_viewer.selectKeypoint(self.current_person_idx, keypoint_idx)
        except Exception as e:
            logger.error(f"Error selecting keypoint {keypoint_idx}: {e}", exc_info=True)

    def onKeypointsSelectedInList(self, keypoint_indices):
        """Handle multiple keypoint selection from occlusion editor list"""
        try:
            if self.current_person_idx >= 0 and self.current_person_idx < len(self.current_poses):
                # Filter out None indices (header items)
                valid_indices = [idx for idx in keypoint_indices if idx is not None and isinstance(idx, int)]
                if valid_indices:
                    logger.debug(f"Selecting {len(valid_indices)} keypoints in image viewer for person {self.current_person_idx}")
                    self.image_viewer.selectMultipleKeypoints(self.current_person_idx, valid_indices)
        except Exception as e:
            logger.error(f"Error selecting multiple keypoints: {e}", exc_info=True)

    def onVisibilityChanged(self, keypoint_idx, new_visibility):
        """Handle visibility change from correction panel"""
        if self.current_person_idx >= 0 and self.current_person_idx < len(self.current_poses):
            person = self.current_poses[self.current_person_idx]
            if keypoint_idx < len(person.keypoints):
                # Update the Person object's keypoint visibility
                person.keypoints[keypoint_idx].visibility = new_visibility
                # Force image viewer to refresh
                self.image_viewer.update()
                logger.debug(f"Updated visibility for keypoint {keypoint_idx} to {new_visibility}")

    def onKeypointMoved(self, person_idx, kp_idx, x, y):
        """Handle keypoint movement"""
        if person_idx < len(self.current_poses):
            person = self.current_poses[person_idx]
            if kp_idx < len(person.keypoints):
                old_pos = (person.keypoints[kp_idx].x, person.keypoints[kp_idx].y)

                # Add to undo stack
                from gui.undo_stack import MoveKeypointCommand
                command = MoveKeypointCommand(person, kp_idx, old_pos, (x, y))
                self.undo_stack.push(command)

                # Update keypoint
                person.keypoints[kp_idx].x = x
                person.keypoints[kp_idx].y = y
                person.keypoints[kp_idx].is_corrected = True

                self.setModified()

    def onLabelsChanged(self, labels):
        """Handle training labels change"""
        if self.current_person_idx < len(self.current_poses):
            person = self.current_poses[self.current_person_idx]
            person.labels = labels
            self.setModified()

    def setModified(self, modified=True):
        """Set document modified state"""
        self.is_modified = modified
        title = self.windowTitle()

        if modified and not title.endswith("*"):
            self.setWindowTitle(title + "*")
        elif not modified and title.endswith("*"):
            self.setWindowTitle(title[:-1])

    def setTheme(self, theme_name):
        """Change application theme"""
        self.theme_manager.setTheme(theme_name)
        self.applyTheme(theme_name)
        self.state_manager.set_simple("theme", theme_name)

        # Update theme action checks
        self.action_theme_light.setChecked(theme_name == "light")
        self.action_theme_dark.setChecked(theme_name == "dark")

    def applyTheme(self, theme_name):
        """Apply theme to application"""
        self.theme_manager.applyTheme(self, theme_name)

    # Progress indication
    def showProgress(self, message, max_value=0):
        """Show progress in status bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(max_value)
        if max_value == 0:
            self.progress_bar.setRange(0, 0)  # Indeterminate
        self.statusBar().showMessage(message)

    def updateProgress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)

    def hideProgress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        self.statusBar().clearMessage()

    # Utility methods
    def showToast(self, message, level=ToastNotification.INFO):
        """Show toast notification"""
        toast = ToastNotification(message, level, self)
        toast.show()

    def openPreferences(self):
        """Open preferences dialog"""
        dialog = PreferencesDialog(self)
        if dialog.exec():
            # Apply settings
            self.setupAutoSave()

    def showShortcuts(self):
        """Show keyboard shortcuts dialog"""
        dialog = ShortcutsDialog(self)
        dialog.exec()

    def openDocumentation(self):
        """Open online documentation"""
        QDesktopServices.openUrl(QUrl("https://posturekit.docs"))

    def showAbout(self):
        """Show about dialog"""
        dialog = AboutDialog(self)
        dialog.exec()

    def loadPoseById(self, pose_id):
        """Load a specific pose by ID"""
        pose = self.db_manager.getPoseById(pose_id)
        if pose and pose.image_path:
            self.loadImage(pose.image_path)

    def loadSettings(self):
        """Load saved application settings"""
        # Restore window state
        self.state_manager.loadWindowState(self)

        # Restore splitter sizes
        if self.state_manager.has("splitter_sizes"):
            self.main_splitter.setSizes(self.state_manager.get_simple("splitter_sizes"))

        # Restore recent files
        self.recent_files = self.state_manager.get_simple("recent_files", [])
        self.updateRecentFilesMenu()

        # Restore view settings
        self.action_show_skeleton.setChecked(
            self.state_manager.get_simple("show_skeleton", True)
        )
        self.action_show_keypoints.setChecked(
            self.state_manager.get_simple("show_keypoints", True)
        )
        self.action_show_confidence.setChecked(
            self.state_manager.get_simple("show_confidence", True)
        )

        # Check for crash recovery
        self.checkCrashRecovery()

    def _terminate_background_threads(self):
        """Gracefully terminate all background threads before shutdown"""
        threads_to_terminate = []

        # Check for operation thread
        if hasattr(self, 'operation_thread') and self.operation_thread is not None:
            if self.operation_thread.isRunning():
                threads_to_terminate.append(('operation_thread', self.operation_thread))

        if not threads_to_terminate:
            logger.debug("No background threads to terminate")
            return

        logger.info(f"Terminating {len(threads_to_terminate)} background thread(s)...")

        for thread_name, thread in threads_to_terminate:
            try:
                # Request graceful stop if the thread has a cancel method
                if hasattr(thread, 'cancel'):
                    logger.debug(f"Requesting graceful stop for {thread_name}")
                    thread.cancel()

                # Wait up to 3 seconds for graceful termination
                if not thread.wait(3000):  # 3 seconds timeout
                    logger.warning(f"Thread {thread_name} did not stop gracefully, forcing termination")
                    thread.terminate()
                    # Wait another second for forced termination
                    thread.wait(1000)
                else:
                    logger.debug(f"Thread {thread_name} stopped gracefully")

            except Exception as e:
                logger.error(f"Error terminating thread {thread_name}: {e}")

        logger.info("Background thread termination complete")

    def _on_database_changed(self, selected_text: str):
        """
        Handle database profile switching.

        Args:
            selected_text: The selected dropdown text ("IRL", "2D Illustrations", "3D Illustrations")
        """
        # Map display text to profile key
        text_to_profile = {
            "IRL": "irl",
            "2D Illustrations": "2d",
            "3D Illustrations": "3d"
        }
        new_profile = text_to_profile.get(selected_text)

        if not new_profile or new_profile == self.current_db_profile:
            return  # No change needed

        logger.info(f"Database profile switching: {self.current_db_profile} → {new_profile}")

        # Check for unsaved changes
        if self.is_modified:
            reply = QMessageBox.question(
                self,
                'Unsaved Changes',
                f'You have unsaved changes. Save before switching to {selected_text} database?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )

            if reply == QMessageBox.Cancel:
                # Revert dropdown selection
                self.database_selector.blockSignals(True)
                profile_to_text = {"irl": "IRL", "2d": "2D Illustrations", "3d": "3D Illustrations"}
                self.database_selector.setCurrentText(profile_to_text[self.current_db_profile])
                self.database_selector.blockSignals(False)
                return
            elif reply == QMessageBox.Save:
                self.saveCorrections()

        # Close current database connections
        try:
            logger.info("Closing similarity engine...")
            if hasattr(self, 'similarity_engine') and self.similarity_engine:
                self.similarity_engine.close()

            logger.info("Closing database manager...")
            if hasattr(self, 'db_manager') and self.db_manager:
                self.db_manager.close()

        except Exception as e:
            logger.error(f"Error closing connections during switch: {e}")

        # Reinitialize database manager with new profile
        try:
            logger.info(f"Initializing database manager for profile: {new_profile}")
            self.db_manager = DatabaseManager(database_profile=new_profile)

            # Reinitialize pose detector
            self.pose_detector = PoseDetector(storage_manager=self.db_manager.storage)

            # Reinitialize similarity engine with new profile
            logger.info(f"Initializing similarity engine for profile: {new_profile}")
            self.similarity_engine = SimilarityEngine(
                storage_manager=self.db_manager.storage,
                database_profile=new_profile
            )

        except Exception as e:
            logger.error(f"Error reinitializing database for {new_profile}: {e}")
            QMessageBox.critical(
                self,
                "Database Switch Failed",
                f"Failed to switch to {selected_text} database:\n{str(e)}\n\nPlease check database connection."
            )
            return

        # Clear current state
        self.current_image_path = None
        self.current_poses = []
        self.current_person_idx = 0
        self.is_modified = False
        self.image_files = []
        self.current_image_index = -1

        # Clear image viewer
        if hasattr(self, 'image_viewer'):
            self.image_viewer.clearImage()

        # Update current profile
        self.current_db_profile = new_profile

        # Save to settings
        from src.gui_legacy.state_manager import SettingsGroup
        self.state_manager.set_simple(SettingsGroup.DATABASE_PROFILE.key, new_profile)
        self.state_manager.save()

        # Clear all GUI panels that show database-specific data
        logger.debug("Clearing GUI panels for database switch...")

        if hasattr(self, 'correction_panel'):
            self.correction_panel.clear()
            logger.debug("Correction panel cleared")

        if hasattr(self, 'similarity_panel'):
            self.similarity_panel.clear_results()
            logger.debug("Similarity panel cleared")

        # Clear undo/redo stack to avoid operations on stale data
        if hasattr(self, 'undo_stack'):
            self.undo_stack.clear()
            logger.debug("Undo stack cleared")

        # Reset all status bar labels to default state
        self.pose_count_label.setText("Poses: 0")
        self.image_counter_label.setText("Image: -")
        self.confidence_label.setText("Confidence: -")
        self.occlusion_label.setText("Occlusion: -")
        self.corrections_label.setText("Corrections: 0")
        logger.debug("Status bar labels reset")

        # Update window title and status bar
        self._update_window_title_for_database()

        # Check database connection status and update indicator
        self.checkDatabaseConnection()
        logger.debug("Database connection status refreshed")

        # Show toast notification
        from src.gui_legacy.widgets.toast_notification import ToastNotification
        toast = ToastNotification(f"Switched to {selected_text} database", parent=self)
        toast.show()

        logger.info(f"Successfully switched to database profile: {new_profile}")

    def _update_window_title_for_database(self):
        """Update window title and status bar to reflect active database."""
        profile_names = {"irl": "IRL", "2d": "2D Illustrations", "3d": "3D Illustrations"}
        db_name = profile_names.get(self.current_db_profile, "IRL")

        # Update window title
        self.setWindowTitle(f"PostureKit - {db_name}")

        # Update status bar
        try:
            # Get database statistics
            stats = self.db_manager.storage.get_statistics()
            total_images = stats.get('total_images', 0)
            total_poses = stats.get('total_poses', 0)

            status_msg = f"Database: {db_name} | {total_images} images | {total_poses} poses indexed"
            self.statusBar().showMessage(status_msg)

            logger.info(f"Window title updated: {db_name} ({total_images} images, {total_poses} poses)")

        except Exception as e:
            logger.warning(f"Failed to update status bar statistics: {e}")
            self.statusBar().showMessage(f"Database: {db_name}")

    def closeEvent(self, event: QCloseEvent):
        """Handle application close with comprehensive resource cleanup"""
        # Check for unsaved changes
        if self.is_modified:
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'You have unsaved changes. Do you want to save before closing?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )

            if reply == QMessageBox.Save:
                self.saveCorrections()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        logger.info("Application closing - starting cleanup sequence")

        # Stop auto-save timer immediately
        if self.auto_save_timer.isActive():
            self.auto_save_timer.stop()
            logger.debug("Auto-save timer stopped")

        # Terminate any running background threads
        self._terminate_background_threads()

        # Save settings
        self.state_manager.saveWindowState(self)
        self.state_manager.set_simple("splitter_sizes", self.main_splitter.sizes())
        self.state_manager.save()

        # Clean up auto-save recovery file
        recovery_path = self.state_manager.getRecoveryPath()
        if os.path.exists(recovery_path):
            try:
                os.remove(recovery_path)
                logger.debug("Auto-save recovery file removed")
            except Exception as e:
                logger.warning(f"Failed to remove recovery file: {e}")

        # Close similarity engine (saves index, clears caches, releases locks)
        try:
            if hasattr(self, 'similarity_engine') and self.similarity_engine:
                logger.info("Closing similarity engine...")
                self.similarity_engine.close()
                logger.info("Similarity engine closed")
        except Exception as e:
            logger.error(f"Failed to close similarity engine: {e}")

        # Close database connections and release PostgreSQL engine pool
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                logger.info("Closing database connections...")
                self.db_manager.close()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Failed to close database: {e}")

        # Dispose shared PostgreSQL connection pool to release SSD lock
        try:
            from src.storage.storage_manager import StorageManager
            logger.info("Disposing shared database engine pool...")
            StorageManager.close_all_engines()
            logger.info("Database engine pool disposed - SSD lock released")
        except Exception as e:
            logger.error(f"Failed to dispose engine pool: {e}")

        # Stop PostgreSQL server
        try:
            logger.info("Stopping PostgreSQL server...")
            stop_postgres()
            logger.info("PostgreSQL server stopped")
        except Exception as e:
            logger.error(f"Failed to stop PostgreSQL: {e}")

        logger.info("Application cleanup complete")
        event.accept()
