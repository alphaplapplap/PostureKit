"""
Theme management for PostureKit.
Provides consistent visual styling across the application.
"""
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from enum import Enum
import sys
import logging

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"  # Follow system preference


class PostureKitTheme:
    """
    Centralized theme management for PostureKit.

    This class encapsulates all the visual styling decisions in one place,
    making it easy to maintain consistency and support theme switching.
    The colors are carefully chosen to provide good contrast for viewing
    pose keypoints while being comfortable for extended use.
    """

    # Color palette - these colors are chosen for accessibility and aesthetics
    # The primary color (blue) is used for interactive elements and conveys trust
    # The accent color (orange) draws attention to important actions
    # Grays provide hierarchy without overwhelming the user

    COLORS = {
        'primary': '#2563eb',      # Blue - trustworthy, professional
        'primary_dark': '#1e40af',
        'primary_light': '#60a5fa',

        'accent': '#f59e0b',       # Orange - energetic, calls to action
        'accent_dark': '#d97706',
        'accent_light': '#fbbf24',

        'success': '#10b981',      # Green - positive feedback
        'warning': '#f59e0b',      # Orange - caution
        'error': '#ef4444',        # Red - errors and dangers

        'text_primary': '#1f2937',
        'text_secondary': '#6b7280',
        'text_disabled': '#9ca3af',

        'background': '#ffffff',
        'background_secondary': '#f3f4f6',
        'border': '#e5e7eb',

        # Dark theme equivalents
        'dark_text_primary': '#f9fafb',
        'dark_text_secondary': '#d1d5db',
        'dark_background': '#111827',
        'dark_background_secondary': '#1f2937',
        'dark_border': '#374151',
    }

    @staticmethod
    def apply_light_theme(app: QApplication):
        """
        Apply light theme to the application.

        Light themes work well in bright environments and are preferred by
        many users during daytime work. The high contrast helps with precision
        tasks like adjusting keypoint positions.
        """
        palette = QPalette()

        # Set up the color palette for light theme
        # We're being explicit about every color role to ensure consistency

        # Window colors (main background)
        palette.setColor(QPalette.ColorRole.Window, QColor(PostureKitTheme.COLORS['background']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(PostureKitTheme.COLORS['text_primary']))

        # Base colors (for input fields and content areas)
        palette.setColor(QPalette.ColorRole.Base, QColor('#ffffff'))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PostureKitTheme.COLORS['background_secondary']))

        # Text colors
        palette.setColor(QPalette.ColorRole.Text, QColor(PostureKitTheme.COLORS['text_primary']))
        palette.setColor(QPalette.ColorRole.BrightText, QColor('#000000'))

        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(PostureKitTheme.COLORS['background_secondary']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(PostureKitTheme.COLORS['text_primary']))

        # Highlight colors (for selected items)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(PostureKitTheme.COLORS['primary']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))

        # Link colors
        palette.setColor(QPalette.ColorRole.Link, QColor(PostureKitTheme.COLORS['primary']))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(PostureKitTheme.COLORS['primary_dark']))

        app.setPalette(palette)

        # Apply stylesheet for additional styling that palette doesn't cover
        # This includes things like borders, padding, and hover states
        stylesheet = PostureKitTheme._get_light_stylesheet()
        app.setStyleSheet(stylesheet)

        logger.info("Applied light theme")

    @staticmethod
    def apply_dark_theme(app: QApplication):
        """
        Apply dark theme to the application.

        Dark themes reduce eye strain in low-light environments and are
        increasingly popular among users who work long hours. The reduced
        brightness also helps when working with multiple monitors.
        """
        palette = QPalette()

        # Dark theme uses inverted brightness while maintaining readability
        palette.setColor(QPalette.ColorRole.Window, QColor(PostureKitTheme.COLORS['dark_background']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(PostureKitTheme.COLORS['dark_text_primary']))

        palette.setColor(QPalette.ColorRole.Base, QColor(PostureKitTheme.COLORS['dark_background_secondary']))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor('#0f172a'))

        palette.setColor(QPalette.ColorRole.Text, QColor(PostureKitTheme.COLORS['dark_text_primary']))
        palette.setColor(QPalette.ColorRole.BrightText, QColor('#ffffff'))

        palette.setColor(QPalette.ColorRole.Button, QColor(PostureKitTheme.COLORS['dark_background_secondary']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(PostureKitTheme.COLORS['dark_text_primary']))

        palette.setColor(QPalette.ColorRole.Highlight, QColor(PostureKitTheme.COLORS['primary_light']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#000000'))

        palette.setColor(QPalette.ColorRole.Link, QColor(PostureKitTheme.COLORS['primary_light']))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(PostureKitTheme.COLORS['primary']))

        app.setPalette(palette)

        stylesheet = PostureKitTheme._get_dark_stylesheet()
        app.setStyleSheet(stylesheet)

        logger.info("Applied dark theme")

    @staticmethod
    def _get_light_stylesheet() -> str:
        """
        Get stylesheet for light theme.

        Stylesheets in Qt are similar to CSS and allow us to style widgets
        beyond what the palette system provides. This includes hover states,
        borders, rounded corners, and spacing.
        """
        return f"""
            /* Main window styling */
            QMainWindow {{
                background-color: {PostureKitTheme.COLORS['background']};
            }}

            /* Toolbar styling - subtle background to separate from content */
            QToolBar {{
                background-color: {PostureKitTheme.COLORS['background_secondary']};
                border-bottom: 1px solid {PostureKitTheme.COLORS['border']};
                spacing: 8px;
                padding: 4px;
            }}

            /* Menu bar styling */
            QMenuBar {{
                background-color: {PostureKitTheme.COLORS['background']};
                border-bottom: 1px solid {PostureKitTheme.COLORS['border']};
            }}

            QMenuBar::item:selected {{
                background-color: {PostureKitTheme.COLORS['primary_light']};
                color: white;
            }}

            /* Push button styling - modern, flat design with hover feedback */
            QPushButton {{
                background-color: {PostureKitTheme.COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}

            QPushButton:hover {{
                background-color: {PostureKitTheme.COLORS['primary_dark']};
            }}

            QPushButton:pressed {{
                background-color: {PostureKitTheme.COLORS['primary_dark']};
                padding-top: 9px;
                padding-bottom: 7px;
            }}

            QPushButton:disabled {{
                background-color: {PostureKitTheme.COLORS['border']};
                color: {PostureKitTheme.COLORS['text_disabled']};
            }}

            /* Accent buttons for important actions */
            QPushButton[accent="true"] {{
                background-color: {PostureKitTheme.COLORS['accent']};
            }}

            QPushButton[accent="true"]:hover {{
                background-color: {PostureKitTheme.COLORS['accent_dark']};
            }}

            /* Group box styling - clear visual grouping */
            QGroupBox {{
                border: 1px solid {PostureKitTheme.COLORS['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }}

            /* Input field styling - clean and minimal */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid {PostureKitTheme.COLORS['border']};
                border-radius: 4px;
                padding: 6px 10px;
                background-color: white;
            }}

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {PostureKitTheme.COLORS['primary']};
            }}

            /* Progress bar styling - clear visual feedback */
            QProgressBar {{
                border: 1px solid {PostureKitTheme.COLORS['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {PostureKitTheme.COLORS['background_secondary']};
            }}

            QProgressBar::chunk {{
                background-color: {PostureKitTheme.COLORS['primary']};
                border-radius: 3px;
            }}

            /* Dock widget styling - clear separation */
            QDockWidget {{
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }}

            QDockWidget::title {{
                background-color: {PostureKitTheme.COLORS['background_secondary']};
                padding: 6px;
                border-bottom: 1px solid {PostureKitTheme.COLORS['border']};
            }}

            /* Status bar styling */
            QStatusBar {{
                background-color: {PostureKitTheme.COLORS['background_secondary']};
                border-top: 1px solid {PostureKitTheme.COLORS['border']};
            }}

            /* List widget styling */
            QListWidget {{
                border: 1px solid {PostureKitTheme.COLORS['border']};
                border-radius: 4px;
                background-color: white;
            }}

            QListWidget::item:selected {{
                background-color: {PostureKitTheme.COLORS['primary_light']};
                color: white;
            }}

            QListWidget::item:hover {{
                background-color: {PostureKitTheme.COLORS['background_secondary']};
            }}
        """

    @staticmethod
    def _get_dark_stylesheet() -> str:
        """Get stylesheet for dark theme."""
        return f"""
            QMainWindow {{
                background-color: {PostureKitTheme.COLORS['dark_background']};
            }}

            QToolBar {{
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                border-bottom: 1px solid {PostureKitTheme.COLORS['dark_border']};
                spacing: 8px;
                padding: 4px;
            }}

            QMenuBar {{
                background-color: {PostureKitTheme.COLORS['dark_background']};
                border-bottom: 1px solid {PostureKitTheme.COLORS['dark_border']};
            }}

            QMenuBar::item:selected {{
                background-color: {PostureKitTheme.COLORS['primary']};
                color: white;
            }}

            QPushButton {{
                background-color: {PostureKitTheme.COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}

            QPushButton:hover {{
                background-color: {PostureKitTheme.COLORS['primary_light']};
            }}

            QPushButton:disabled {{
                background-color: {PostureKitTheme.COLORS['dark_border']};
                color: {PostureKitTheme.COLORS['text_disabled']};
            }}

            QPushButton[accent="true"] {{
                background-color: {PostureKitTheme.COLORS['accent']};
            }}

            QPushButton[accent="true"]:hover {{
                background-color: {PostureKitTheme.COLORS['accent_light']};
            }}

            QGroupBox {{
                border: 1px solid {PostureKitTheme.COLORS['dark_border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid {PostureKitTheme.COLORS['dark_border']};
                border-radius: 4px;
                padding: 6px 10px;
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {PostureKitTheme.COLORS['primary_light']};
            }}

            QProgressBar {{
                border: 1px solid {PostureKitTheme.COLORS['dark_border']};
                border-radius: 4px;
                text-align: center;
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QProgressBar::chunk {{
                background-color: {PostureKitTheme.COLORS['primary_light']};
                border-radius: 3px;
            }}

            QDockWidget::title {{
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                padding: 6px;
                border-bottom: 1px solid {PostureKitTheme.COLORS['dark_border']};
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QStatusBar {{
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                border-top: 1px solid {PostureKitTheme.COLORS['dark_border']};
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QListWidget {{
                border: 1px solid {PostureKitTheme.COLORS['dark_border']};
                border-radius: 4px;
                background-color: {PostureKitTheme.COLORS['dark_background_secondary']};
                color: {PostureKitTheme.COLORS['dark_text_primary']};
            }}

            QListWidget::item:selected {{
                background-color: {PostureKitTheme.COLORS['primary']};
                color: white;
            }}

            QListWidget::item:hover {{
                background-color: {PostureKitTheme.COLORS['dark_border']};
            }}
        """

    @staticmethod
    def set_app_font(app: QApplication):
        """
        Set application-wide font for consistency and readability.

        Font choice significantly impacts readability and user comfort.
        We use system fonts when available because they're optimized for
        the user's operating system and provide the most native feel.
        """
        # Try to use system fonts for the most native look
        if sys.platform == 'darwin':  # macOS
            font = QFont('.AppleSystemUIFont', 13)
        elif sys.platform == 'win32':  # Windows
            font = QFont('Segoe UI', 9)
        else:  # Linux and others
            font = QFont('Ubuntu', 10)

        app.setFont(font)

    # Backward compatibility instance methods
    def __init__(self):
        """Initialize theme manager."""
        self._current_theme = "dark"

    def setTheme(self, theme_name: str):
        """Set the current theme (backward compatibility)."""
        self._current_theme = theme_name

    def applyTheme(self, app_or_widget, theme_name: str):
        """
        Apply theme to application or widget (backward compatibility).

        Args:
            app_or_widget: QApplication or QWidget to apply theme to
            theme_name: Theme name ("light" or "dark")
        """
        from PySide6.QtWidgets import QApplication

        # Get the application instance
        if isinstance(app_or_widget, QApplication):
            app = app_or_widget
        else:
            app = QApplication.instance()

        if app is None:
            return

        # Apply theme
        if theme_name == "light":
            self.apply_light_theme(app)
        else:
            self.apply_dark_theme(app)

        self._current_theme = theme_name


# Alias for backward compatibility
ThemeManager = PostureKitTheme
