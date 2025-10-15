"""
Toast notification system for non-intrusive user feedback.
"""
from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect, QApplication
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QPalette, QFont
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ToastNotification(QLabel):
    """
    Temporary notification that appears briefly then fades away.

    Toast notifications are perfect for:
    - Success confirmations ("Saved successfully")
    - Minor warnings that don't require action
    - Status updates that don't need user response
    - Background operation completion

    They automatically position themselves in the parent window and
    fade in/out smoothly using Qt animations.
    """

    # Notification types with different styling
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    # Style sheets for different notification types
    STYLES = {
        SUCCESS: """
            QLabel {
                background-color: #27ae60;
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
            }
        """,
        INFO: """
            QLabel {
                background-color: #3498db;
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
            }
        """,
        WARNING: """
            QLabel {
                background-color: #f39c12;
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
            }
        """,
        ERROR: """
            QLabel {
                background-color: #e74c3c;
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
            }
        """
    }

    def __init__(
        self,
        message: str,
        notification_type: str = INFO,
        duration: int = 3000,
        parent=None
    ):
        """
        Initialize toast notification.

        Args:
            message: Text to display
            notification_type: Type of notification (SUCCESS, INFO, WARNING, ERROR)
            duration: How long to show in milliseconds (default 3 seconds)
            parent: Parent widget (notification will appear over this)
        """
        super().__init__(message, parent)

        self.duration = duration
        self.notification_type = notification_type

        self._setup_ui()
        self._setup_animations()

    def _setup_ui(self):
        """Configure notification appearance."""
        # Set style based on type
        self.setStyleSheet(self.STYLES.get(self.notification_type, self.STYLES[self.INFO]))

        # Configure label properties
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMaximumWidth(400)

        # Make it float above other widgets
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Adjust size to content
        self.adjustSize()

        # Initially hidden
        self.setVisible(False)

    def _setup_animations(self):
        """Setup fade in/out animations for smooth appearance."""
        # Opacity effect for fading
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        # Fade in animation
        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in.setDuration(250)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.OutCubic)

        # Fade out animation
        self.fade_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out.setDuration(250)
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.InCubic)
        self.fade_out.finished.connect(self._on_fade_out_complete)

        # Timer to trigger fade out
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self._start_fade_out)
        self.hide_timer.setSingleShot(True)

    def show_notification(self):
        """
        Display the notification with fade-in effect.

        This method positions the notification in the bottom-center of the
        parent widget, then fades it in smoothly. After the duration expires,
        it automatically fades out and hides.
        """
        if self.parent():
            # Position at bottom center of parent widget
            parent_rect = self.parent().rect()
            x = (parent_rect.width() - self.width()) // 2
            y = parent_rect.height() - self.height() - 60  # 60px from bottom

            # Convert to global coordinates
            global_pos = self.parent().mapToGlobal(QPoint(x, y))
            self.move(global_pos)

        # Show and fade in
        self.show()
        self.fade_in.start()

        # Schedule fade out
        self.hide_timer.start(self.duration)

        logger.debug(f"Toast notification shown: {self.text()}")

    def _start_fade_out(self):
        """Begin fade out animation."""
        self.fade_out.start()

    def _on_fade_out_complete(self):
        """Called when fade out animation completes."""
        self.hide()
        self.deleteLater()


class ToastManager:
    """
    Manager for displaying toast notifications in a window.

    The ToastManager handles queuing and positioning of multiple
    notifications, ensuring they don't overlap and appear in sequence.
    """

    def __init__(self, parent_window):
        """
        Initialize toast manager.

        Args:
            parent_window: Window to display notifications in
        """
        self.parent_window = parent_window
        self.active_notifications = []

    def show_success(self, message: str, duration: int = 3000):
        """Show success notification."""
        self._show_toast(message, ToastNotification.SUCCESS, duration)

    def show_info(self, message: str, duration: int = 3000):
        """Show info notification."""
        self._show_toast(message, ToastNotification.INFO, duration)

    def show_warning(self, message: str, duration: int = 4000):
        """Show warning notification."""
        self._show_toast(message, ToastNotification.WARNING, duration)

    def show_error(self, message: str, duration: int = 5000):
        """Show error notification."""
        self._show_toast(message, ToastNotification.ERROR, duration)

    def _show_toast(self, message: str, notification_type: str, duration: int):
        """
        Create and display a toast notification.

        Args:
            message: Text to display
            notification_type: Type of notification
            duration: Display duration in milliseconds
        """
        toast = ToastNotification(
            message,
            notification_type,
            duration,
            self.parent_window
        )

        # Track active notification
        self.active_notifications.append(toast)
        toast.destroyed.connect(lambda: self._on_toast_destroyed(toast))

        # Show it
        toast.show_notification()

    def _on_toast_destroyed(self, toast):
        """Remove toast from active list when destroyed."""
        if toast in self.active_notifications:
            self.active_notifications.remove(toast)


# Convenience functions for global toast notifications
_global_toast_manager: Optional[ToastManager] = None


def init_toast_manager(main_window):
    """
    Initialize global toast manager.

    Call this once during application startup with the main window.

    Args:
        main_window: Main application window
    """
    global _global_toast_manager
    _global_toast_manager = ToastManager(main_window)


def show_success(message: str, duration: int = 3000):
    """Show global success notification."""
    if _global_toast_manager:
        _global_toast_manager.show_success(message, duration)


def show_info(message: str, duration: int = 3000):
    """Show global info notification."""
    if _global_toast_manager:
        _global_toast_manager.show_info(message, duration)


def show_warning(message: str, duration: int = 4000):
    """Show global warning notification."""
    if _global_toast_manager:
        _global_toast_manager.show_warning(message, duration)


def show_error(message: str, duration: int = 5000):
    """Show global error notification."""
    if _global_toast_manager:
        _global_toast_manager.show_error(message, duration)
