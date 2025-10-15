"""
Splash screen for PostureKit application startup.
Modern design with progress tracking and smooth animations.
"""
from PySide6.QtWidgets import QSplashScreen, QProgressBar, QLabel
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, QLinearGradient, QPen
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SplashScreen(QSplashScreen):
    """
    Modern splash screen with gradient design and progress tracking.
    
    Design Features:
    - Gradient background (blue theme)
    - Large app title with subtitle
    - Animated progress bar
    - Status messages for each initialization step
    - Smooth fade-out transition
    
    Size: 700×450px
    """
    
    progress_updated = Signal(int, str)
    
    def __init__(self):
        """Initialize splash screen with modern design."""
        splash_pixmap = self._create_splash_image()
        super().__init__(splash_pixmap, Qt.WindowStaysOnTopHint)
        
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint | 
            Qt.SplashScreen
        )
        
        self._setup_ui()
        self.progress_updated.connect(self._update_progress)
        
        logger.debug("Splash screen initialized")
    
    def _create_splash_image(self) -> QPixmap:
        """
        Create modern splash screen with gradient background.
        
        Returns:
            QPixmap: 700×450px splash image
        """
        width, height = 700, 450
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        # Gradient background (blue theme from spec)
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0.0, QColor("#1E3A8A"))    # Dark blue
        gradient.setColorAt(0.5, QColor("#2563EB"))    # Accent blue
        gradient.setColorAt(1.0, QColor("#1E40AF"))    # Medium blue
        painter.fillRect(0, 0, width, height, gradient)
        
        # Subtle border
        painter.setPen(QPen(QColor(255, 255, 255, 50), 2))
        painter.drawRect(1, 1, width-2, height-2)
        
        # App Title
        title_font = QFont("-apple-system, BlinkMacSystemFont, Segoe UI", 56, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = pixmap.rect().adjusted(0, 80, 0, 0)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop, "PostureKit")
        
        # Subtitle
        subtitle_font = QFont("-apple-system, BlinkMacSystemFont, Segoe UI", 16)
        painter.setFont(subtitle_font)
        painter.setPen(QColor(255, 255, 255, 200))
        subtitle_rect = pixmap.rect().adjusted(0, 160, 0, 0)
        painter.drawText(
            subtitle_rect, 
            Qt.AlignHCenter | Qt.AlignTop, 
            "Multi-Person Pose Estimation & Dataset Curation"
        )
        
        # Technology stack (small text)
        tech_font = QFont("-apple-system, BlinkMacSystemFont, Segoe UI", 11)
        painter.setFont(tech_font)
        painter.setPen(QColor(255, 255, 255, 150))
        tech_rect = pixmap.rect().adjusted(0, 195, 0, 0)
        painter.drawText(
            tech_rect,
            Qt.AlignHCenter | Qt.AlignTop,
            "MMPose RTMW-L • YOLOv8 • 133 Keypoints • PostgreSQL + FAISS"
        )
        
        # Version (bottom right)
        version_font = QFont("-apple-system, BlinkMacSystemFont, Segoe UI", 10)
        painter.setFont(version_font)
        painter.setPen(QColor(255, 255, 255, 120))
        version_rect = pixmap.rect().adjusted(0, 0, -20, -20)
        painter.drawText(version_rect, Qt.AlignRight | Qt.AlignBottom, "v1.0.0-beta")
        
        # Copyright (bottom left)
        painter.drawText(
            pixmap.rect().adjusted(20, 0, 0, -20),
            Qt.AlignLeft | Qt.AlignBottom,
            "© 2025 PostureKit"
        )
        
        painter.end()
        return pixmap
    
    def _setup_ui(self):
        """Setup progress bar and status label with modern styling."""
        # Status label (centered above progress bar)
        self.status_label = QLabel("Initializing...", self)
        self.status_label.setGeometry(50, 280, 600, 30)
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: 500;
                background: transparent;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Progress bar (modern flat design)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(100, 320, 500, 8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: rgba(255, 255, 255, 0.2);
            }
            QProgressBar::chunk {
                background-color: white;
                border-radius: 4px;
            }
        """)
        self.progress_bar.setValue(0)
        
        # Percentage label (below progress bar)
        self.percent_label = QLabel("0%", self)
        self.percent_label.setGeometry(50, 335, 600, 20)
        self.percent_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 11px;
                background: transparent;
            }
        """)
        self.percent_label.setAlignment(Qt.AlignCenter)
    
    def _update_progress(self, percentage: int, message: str):
        """
        Update progress bar and status message.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message to display
        """
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.percent_label.setText(f"{percentage}%")
        
        # Update splash message (appears at bottom)
        self.showMessage(
            message,
            Qt.AlignBottom | Qt.AlignCenter,
            QColor(255, 255, 255, 200)
        )
        
        logger.debug(f"Splash progress: {percentage}% - {message}")
    
    def update_progress(self, percentage: int, message: str):
        """
        Public method to update progress (thread-safe via signal).
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message to display
        """
        self.progress_updated.emit(percentage, message)
    
    def finish_with_main_window(self, main_window):
        """
        Smoothly transition from splash to main window.
        
        Args:
            main_window: MainWindow instance to show
        """
        self.update_progress(100, "Ready!")
        
        # Brief pause at 100% so user sees completion
        QTimer.singleShot(500, lambda: self.finish(main_window))
        logger.info("Splash screen completed, transitioning to main window")


class InitializationWorker:
    """
    Handles application initialization tasks with progress reporting.
    
    Performs heavyweight initialization operations:
    - Loading configuration and validating settings
    - Connecting to PostgreSQL database
    - Initializing correction learner models
    - Setting up FAISS vector index
    - Loading theme system
    
    Each step reports progress to the splash screen.
    """
    
    def __init__(self, splash: SplashScreen):
        """
        Initialize worker with splash screen reference.
        
        Args:
            splash: SplashScreen to update with progress
        """
        self.splash = splash
        self.components = {}
    
    def initialize_all(self):
        """
        Run all initialization steps with progress updates.
        
        Returns:
            Dictionary of initialized components
        
        Raises:
            Exception: If any initialization step fails
        """
        steps = [
            (5, "Loading configuration...", self._load_config),
            (15, "Initializing database connection...", self._init_database),
            (30, "Loading correction learner...", self._load_correction_learner),
            (50, "Initializing multimodal fusion...", self._init_multimodal),
            (70, "Loading theme system...", self._load_theme),
            (85, "Setting up logging...", self._setup_logging),
            (95, "Finalizing setup...", self._finalize)
        ]
        
        for percentage, message, func in steps:
            self.splash.update_progress(percentage, message)
            try:
                func()
            except Exception as e:
                logger.error(f"Initialization failed at '{message}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize: {message}") from e
        
        logger.info("Application initialization complete")
        return self.components
    
    def _load_config(self):
        """Load and validate application configuration."""
        from src.config.settings import settings, validate_settings
        validate_settings()
        self.components['settings'] = settings
        logger.debug("Configuration loaded and validated")
    
    def _init_database(self):
        """Initialize database connection and create tables."""
        from src.storage.storage_manager import StorageManager
        storage_manager = StorageManager()
        storage_manager.initialize_database()
        self.components['storage_manager'] = storage_manager
        logger.debug("Database initialized successfully")
    
    def _load_correction_learner(self):
        """Load correction learner models."""
        from src.learning.correction_learner import CorrectionLearner
        learner = CorrectionLearner(self.components['storage_manager'])
        self.components['correction_learner'] = learner
        logger.debug("Correction learner loaded")
    
    def _init_multimodal(self):
        """Initialize multimodal correction learner."""
        from src.learning.multimodal_correction_learner import MultiModalCorrectionLearner
        mm_learner = MultiModalCorrectionLearner(self.components['storage_manager'])
        self.components['multimodal_learner'] = mm_learner
        logger.debug("Multimodal fusion initialized")
    
    def _load_theme(self):
        """Load and apply theme system."""
        from gui.theme import ThemeManager
        theme_manager = ThemeManager.instance()
        theme_manager.load_saved_theme()
        self.components['theme_manager'] = theme_manager
        logger.debug("Theme system loaded")
    
    def _setup_logging(self):
        """Configure application logging."""
        from src.utils.logging_config import get_logger
        settings = self.components['settings']
        # Logging is already configured by logging_config module
        logger.debug(f"Logging configured at {settings.LOG_LEVEL} level")
    
    def _finalize(self):
        """Final setup tasks and cleanup."""
        # Any final initialization tasks
        logger.debug("Initialization finalized")


# Example usage
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Show splash
    splash = SplashScreen()
    splash.show()
    
    # Simulate initialization
    def simulate_init():
        steps = [
            (10, "Loading configuration..."),
            (25, "Initializing database..."),
            (40, "Loading pose models..."),
            (60, "Loading correction learner..."),
            (75, "Initializing similarity search..."),
            (90, "Finalizing setup..."),
            (100, "Ready!")
        ]
        
        import time
        for percentage, message in steps:
            splash.update_progress(percentage, message)
            time.sleep(0.5)
            app.processEvents()
        
        splash.close()
    
    QTimer.singleShot(100, simulate_init)
    
    sys.exit(app.exec())
