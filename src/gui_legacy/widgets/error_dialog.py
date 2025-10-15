"""Error handling dialogs and notifications for PostureKit."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QWidget, QGraphicsOpacityEffect, QApplication
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QPoint
from PySide6.QtGui import QIcon, QFont, QTextCursor
from enum import Enum
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorDialog(QDialog):
    """Blocking error dialog for critical errors."""

    def __init__(
        self,
        title: str,
        message: str,
        details: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        actions: Optional[dict[str, Callable]] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize error dialog.

        Args:
            title: Dialog title
            message: Main error message
            details: Optional detailed error information (expandable)
            severity: Error severity level
            actions: Optional dict of {button_text: callback_function}
            parent: Parent widget
        """
        super().__init__(parent)
        self.severity = severity
        self.details = details
        self.actions = actions or {}
        self.details_expanded = False

        self.setup_ui(title, message)

        # Log the error
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        logger.log(log_level, f"{title}: {message}")
        if details:
            logger.debug(f"Error details: {details}")

    def setup_ui(self, title: str, message: str):
        """Initialize the user interface."""
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header with icon and message
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        # Severity icon
        icon_label = QLabel()
        icon_label.setTextFormat(Qt.TextFormat.RichText)
        icon_text, icon_color = self._get_severity_style()
        icon_label.setText(icon_text)
        icon_label.setStyleSheet(f"font-size: 32pt; color: {icon_color};")
        icon_label.setFixedSize(48, 48)
        header_layout.addWidget(icon_label)

        # Message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextFormat(Qt.TextFormat.PlainText)
        message_font = QFont()
        message_font.setPointSize(10)
        message_label.setFont(message_font)
        header_layout.addWidget(message_label, 1)

        layout.addLayout(header_layout)

        # Details section (collapsible)
        if self.details:
            self.details_widget = QWidget()
            details_layout = QVBoxLayout(self.details_widget)
            details_layout.setContentsMargins(0, 0, 0, 0)
            details_layout.setSpacing(8)

            # Details text area
            self.details_text = QTextEdit()
            self.details_text.setPlainText(self.details)
            self.details_text.setReadOnly(True)
            self.details_text.setMaximumHeight(200)
            self.details_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #E5E7EB;
                    border-radius: 4px;
                    background-color: #F9FAFB;
                    padding: 8px;
                    font-family: 'SF Mono', Monaco, 'Courier New', monospace;
                    font-size: 9pt;
                    color: #1F2937;
                }
            """)
            details_layout.addWidget(self.details_text)

            self.details_widget.setVisible(False)
            layout.addWidget(self.details_widget)

            # Show/Hide Details button
            self.details_btn = QPushButton("Show Details")
            self.details_btn.clicked.connect(self.toggle_details)
            self.details_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #2563EB;
                    border: none;
                    padding: 4px 8px;
                    text-align: left;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    text-decoration: underline;
                }
            """)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # Copy to Clipboard button (if details exist)
        if self.details:
            copy_btn = QPushButton("Copy Details")
            copy_btn.clicked.connect(self.copy_details)
            copy_btn.setStyleSheet(self._get_secondary_button_style())
            button_layout.addWidget(copy_btn)

        # Show Details button (if details exist)
        if self.details:
            button_layout.addWidget(self.details_btn)

        button_layout.addStretch()

        # Custom action buttons
        for action_text, callback in self.actions.items():
            action_btn = QPushButton(action_text)
            action_btn.clicked.connect(lambda checked, cb=callback: self._handle_action(cb))
            action_btn.setStyleSheet(self._get_secondary_button_style())
            button_layout.addWidget(action_btn)

        # Close button (primary)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        close_btn.setStyleSheet(self._get_primary_button_style())
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        # Apply dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
        """)

    def toggle_details(self):
        """Toggle visibility of details section."""
        self.details_expanded = not self.details_expanded
        self.details_widget.setVisible(self.details_expanded)
        self.details_btn.setText("Hide Details" if self.details_expanded else "Show Details")

        # Adjust dialog size
        if self.details_expanded:
            self.adjustSize()

    def copy_details(self):
        """Copy error details to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.details)

        # Brief feedback
        original_text = self.sender().text()
        self.sender().setText("Copied!")
        QTimer.singleShot(1500, lambda: self.sender().setText(original_text))

    def _handle_action(self, callback: Callable):
        """Handle custom action button click."""
        self.accept()
        callback()

    def _get_severity_style(self) -> tuple[str, str]:
        """Get icon and color for severity level."""
        styles = {
            ErrorSeverity.INFO: ("ℹ️", "#2563EB"),      # Blue
            ErrorSeverity.WARNING: ("⚠️", "#F59E0B"),  # Orange
            ErrorSeverity.ERROR: ("❌", "#EF4444"),     # Red
            ErrorSeverity.CRITICAL: ("🚨", "#DC2626")  # Dark Red
        }
        return styles[self.severity]

    def _get_primary_button_style(self) -> str:
        """Get primary button stylesheet."""
        return """
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 10pt;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
        """

    def _get_secondary_button_style(self) -> str:
        """Get secondary button stylesheet."""
        return """
            QPushButton {
                background-color: transparent;
                color: #2563EB;
                border: 1px solid #2563EB;
                padding: 7px 15px;
                border-radius: 6px;
                font-size: 10pt;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(37, 99, 235, 0.1);
            }
        """


class ErrorToast(QWidget):
    """Non-blocking toast notification for minor errors."""

    closed = Signal()

    # Class variable to track toast positions
    _active_toasts: list['ErrorToast'] = []
    _MARGIN = 16
    _SPACING = 12

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        duration: int = 5000,
        actions: Optional[dict[str, Callable]] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize error toast.

        Args:
            message: Toast message
            severity: Error severity level
            duration: Auto-dismiss duration in milliseconds (0 = no auto-dismiss)
            actions: Optional dict of {button_text: callback_function}
            parent: Parent widget
        """
        super().__init__(parent)
        self.severity = severity
        self.duration = duration
        self.actions = actions or {}

        self.setup_ui(message)
        self.setup_animations()

        # Position toast
        self._position_toast()
        ErrorToast._active_toasts.append(self)

        # Auto-dismiss timer
        if duration > 0:
            QTimer.singleShot(duration, self.dismiss)

        # Log
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        logger.log(log_level, f"Toast: {message}")

    def setup_ui(self, message: str):
        """Initialize the user interface."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Severity icon
        icon_label = QLabel()
        icon_text, icon_color = self._get_severity_style()
        icon_label.setText(icon_text)
        icon_label.setStyleSheet(f"font-size: 20pt;")
        layout.addWidget(icon_label)

        # Message
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setMaximumWidth(300)
        message_label.setStyleSheet("color: #1F2937; font-size: 10pt;")
        layout.addWidget(message_label, 1)

        # Action buttons
        for action_text, callback in self.actions.items():
            action_btn = QPushButton(action_text)
            action_btn.clicked.connect(lambda checked, cb=callback: self._handle_action(cb))
            action_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #2563EB;
                    border: 1px solid #2563EB;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 9pt;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: rgba(37, 99, 235, 0.1);
                }
            """)
            layout.addWidget(action_btn)

        # Dismiss button
        dismiss_btn = QPushButton("✕")
        dismiss_btn.clicked.connect(self.dismiss)
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: none;
                border-radius: 12px;
                font-size: 16pt;
                padding: 0;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
                color: #1F2937;
            }
        """)
        layout.addWidget(dismiss_btn)

        # Background color based on severity
        bg_colors = {
            ErrorSeverity.INFO: "#EFF6FF",      # Light blue
            ErrorSeverity.WARNING: "#FEF3C7",  # Light yellow
            ErrorSeverity.ERROR: "#FEE2E2",     # Light red
            ErrorSeverity.CRITICAL: "#FEE2E2"   # Light red
        }

        self.setStyleSheet(f"""
            ErrorToast {{
                background-color: {bg_colors[self.severity]};
                border: 1px solid {self._get_severity_style()[1]};
                border-radius: 8px;
            }}
        """)

        self.adjustSize()

    def setup_animations(self):
        """Setup fade in/out animations."""
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        self.fade_in_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_anim.setDuration(200)
        self.fade_in_anim.setStartValue(0.0)
        self.fade_in_anim.setEndValue(1.0)
        self.fade_in_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.fade_out_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out_anim.setDuration(200)
        self.fade_out_anim.setStartValue(1.0)
        self.fade_out_anim.setEndValue(0.0)
        self.fade_out_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out_anim.finished.connect(self._on_fade_out_finished)

    def showEvent(self, event):
        """Override show event to trigger fade in."""
        super().showEvent(event)
        self.fade_in_anim.start()

    def dismiss(self):
        """Dismiss the toast with fade out animation."""
        self.fade_out_anim.start()

    def _on_fade_out_finished(self):
        """Handle fade out completion."""
        if self in ErrorToast._active_toasts:
            ErrorToast._active_toasts.remove(self)
        self._reposition_toasts()
        self.closed.emit()
        self.deleteLater()

    def _position_toast(self):
        """Position toast in bottom-right corner."""
        if not self.parent():
            # No parent, use screen geometry
            screen = QApplication.primaryScreen().geometry()
            parent_rect = screen
        else:
            parent_rect = self.parent().geometry()

        # Calculate position considering other toasts
        y_offset = self._MARGIN
        for toast in ErrorToast._active_toasts:
            if toast != self:
                y_offset += toast.height() + self._SPACING

        x = parent_rect.width() - self.width() - self._MARGIN
        y = parent_rect.height() - self.height() - y_offset

        self.move(x, y)

    @classmethod
    def _reposition_toasts(cls):
        """Reposition all active toasts after one is dismissed."""
        for i, toast in enumerate(cls._active_toasts):
            y_offset = cls._MARGIN
            for j in range(i):
                y_offset += cls._active_toasts[j].height() + cls._SPACING

            if toast.parent():
                parent_rect = toast.parent().geometry()
            else:
                parent_rect = QApplication.primaryScreen().geometry()

            new_y = parent_rect.height() - toast.height() - y_offset
            toast.move(toast.x(), new_y)

    def _handle_action(self, callback: Callable):
        """Handle action button click."""
        self.dismiss()
        callback()

    def _get_severity_style(self) -> tuple[str, str]:
        """Get icon and color for severity level."""
        styles = {
            ErrorSeverity.INFO: ("ℹ️", "#2563EB"),
            ErrorSeverity.WARNING: ("⚠️", "#F59E0B"),
            ErrorSeverity.ERROR: ("❌", "#EF4444"),
            ErrorSeverity.CRITICAL: ("🚨", "#DC2626")
        }
        return styles[self.severity]


# Convenience functions
def show_error(
    title: str,
    message: str,
    details: Optional[str] = None,
    parent: Optional[QWidget] = None,
    **kwargs
) -> ErrorDialog:
    """Show blocking error dialog."""
    dialog = ErrorDialog(
        title=title,
        message=message,
        details=details,
        severity=ErrorSeverity.ERROR,
        parent=parent,
        **kwargs
    )
    dialog.exec()
    return dialog


def show_warning(
    title: str,
    message: str,
    details: Optional[str] = None,
    parent: Optional[QWidget] = None,
    **kwargs
) -> ErrorDialog:
    """Show blocking warning dialog."""
    dialog = ErrorDialog(
        title=title,
        message=message,
        details=details,
        severity=ErrorSeverity.WARNING,
        parent=parent,
        **kwargs
    )
    dialog.exec()
    return dialog


def show_toast(
    message: str,
    severity: ErrorSeverity = ErrorSeverity.WARNING,
    duration: int = 5000,
    parent: Optional[QWidget] = None,
    **kwargs
) -> ErrorToast:
    """Show non-blocking toast notification."""
    toast = ErrorToast(
        message=message,
        severity=severity,
        duration=duration,
        parent=parent,
        **kwargs
    )
    toast.show()
    return toast


if __name__ == "__main__":
    # Test the dialogs
    import sys

    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Error Dialog Test")
    layout = QVBoxLayout(window)

    def test_error():
        show_error(
            "Database Error",
            "Failed to connect to database. Please check your connection settings.",
            details="PostgreSQL connection failed:\npsycopg2.OperationalError: could not connect to server: Connection refused\n\tIs the server running on host 'localhost' and accepting TCP/IP connections on port 5432?",
            parent=window,
            actions={
                "Retry": lambda: print("Retrying..."),
                "Configure": lambda: print("Opening settings...")
            }
        )

    def test_toast():
        show_toast(
            "Failed to save annotations. Click retry to try again.",
            severity=ErrorSeverity.ERROR,
            parent=window,
            actions={
                "Retry": lambda: print("Retrying..."),
                "Dismiss": lambda: print("Dismissed")
            }
        )

    error_btn = QPushButton("Show Error Dialog")
    error_btn.clicked.connect(test_error)
    layout.addWidget(error_btn)

    toast_btn = QPushButton("Show Toast")
    toast_btn.clicked.connect(test_toast)
    layout.addWidget(toast_btn)

    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())
