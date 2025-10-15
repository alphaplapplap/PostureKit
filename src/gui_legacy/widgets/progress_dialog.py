"""Progress dialogs and widgets for long-running operations."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QTextEdit, QWidget
)
from PySide6.QtCore import Qt, Signal, QTimer, QMutex, QMutexLocker
from PySide6.QtGui import QFont, QPainter, QPen, QColor
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import time


class ProgressState(Enum):
    """Progress dialog states."""
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    total_items: int
    completed: int = 0
    failed: int = 0
    start_time: float = 0.0
    eta_seconds: float = 0.0
    current_item: str = ""


class IndeterminateSpinner(QWidget):
    """
    Spinning indicator for very short operations (<2s).
    Lightweight alternative to progress bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.setFixedSize(32, 32)

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._rotate)
        self.timer.start(50)  # 20fps

    def _rotate(self):
        """Rotate spinner."""
        self.angle = (self.angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        """Custom paint for spinner."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw spinning arc
        pen = QPen(QColor("#2563EB"), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.drawArc(rect, self.angle * 16, 120 * 16)

    def stop(self):
        """Stop animation."""
        self.timer.stop()


class SimpleProgressBar(QWidget):
    """
    Simple progress bar for short operations (2-10s).
    No ETA, just percentage and status.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._setup_ui(title)
        self._position_widget()

    def _setup_ui(self, title: str):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Container with background
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E5E7EB;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(12)
        container_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        container_layout.addWidget(title_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                text-align: center;
                height: 24px;
                background-color: #F3F4F6;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 3px;
            }
        """)
        container_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 10pt;")
        container_layout.addWidget(self.status_label)

        layout.addWidget(container)

    def _position_widget(self):
        """Position widget in bottom-right corner."""
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.right() - self.width() - 20
            y = parent_rect.bottom() - self.height() - 20
            self.move(x, y)

    def update_progress(self, value: int, status: str = ""):
        """Update progress value and status."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)


class ProgressDialog(QDialog):
    """
    Comprehensive progress dialog for long operations (>10s).

    Features:
    - Thread-safe progress updates with mutex
    - ETA calculation using exponential moving average
    - Pause/Resume/Cancel controls
    - Detailed error log with auto-scroll
    - Success/Failed counters with color coding
    - Item-level progress tracking
    """

    cancelled = Signal()
    paused = Signal()
    resumed = Signal()

    def __init__(
        self,
        title: str,
        total_items: int,
        description: str = "",
        allow_pause: bool = True,
        parent=None
    ):
        super().__init__(parent)

        self.stats = ProgressStats(total_items=total_items, start_time=time.time())
        self.state = ProgressState.RUNNING
        self.mutex = QMutex()

        # ETA calculation (exponential moving average)
        self._ema_alpha = 0.3  # Smoothing factor
        self._last_update_time = time.time()
        self._items_per_second = 0.0

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(600)

        self._setup_ui(description, allow_pause)
        self._update_display()

    def _setup_ui(self, description: str, allow_pause: bool):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Description
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #1F2937; font-size: 11pt;")
            layout.addWidget(desc_label)

        # Progress info row
        info_layout = QHBoxLayout()

        # Item counter
        self.counter_label = QLabel()
        self.counter_label.setStyleSheet("color: #1F2937; font-size: 10pt; font-weight: 500;")
        info_layout.addWidget(self.counter_label)

        info_layout.addStretch()

        # ETA
        self.eta_label = QLabel()
        self.eta_label.setStyleSheet("color: #6B7280; font-size: 10pt;")
        info_layout.addWidget(self.eta_label)

        layout.addLayout(info_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.stats.total_items)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E5E7EB;
                border-radius: 6px;
                text-align: center;
                height: 28px;
                background-color: #F3F4F6;
                font-size: 10pt;
                font-weight: 500;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Current item status
        self.current_item_label = QLabel()
        self.current_item_label.setStyleSheet("color: #6B7280; font-size: 10pt;")
        layout.addWidget(self.current_item_label)

        # Success/Failed counters
        counters_layout = QHBoxLayout()

        self.success_label = QLabel()
        self.success_label.setStyleSheet("color: #10B981; font-size: 10pt; font-weight: 500;")
        counters_layout.addWidget(self.success_label)

        counters_layout.addStretch()

        self.failed_label = QLabel()
        self.failed_label.setStyleSheet("color: #EF4444; font-size: 10pt; font-weight: 500;")
        counters_layout.addWidget(self.failed_label)

        layout.addLayout(counters_layout)

        # Error log (collapsible)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setMaximumHeight(120)
        self.error_log.setStyleSheet("""
            QTextEdit {
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                background-color: #FEF2F2;
                padding: 8px;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 9pt;
                color: #DC2626;
            }
        """)
        self.error_log.hide()
        layout.addWidget(self.error_log)

        # Show/Hide errors button
        self.toggle_errors_btn = QPushButton("Show Errors (0)")
        self.toggle_errors_btn.clicked.connect(self._toggle_errors)
        self.toggle_errors_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: 1px solid #E5E7EB;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #F3F4F6;
            }
        """)
        layout.addWidget(self.toggle_errors_btn)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Pause/Resume button
        if allow_pause:
            self.pause_btn = QPushButton("Pause")
            self.pause_btn.clicked.connect(self._toggle_pause)
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F59E0B;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 10pt;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #D97706;
                }
                QPushButton:disabled {
                    background-color: #9CA3AF;
                }
            """)
            button_layout.addWidget(self.pause_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 10pt;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Dialog styling
        self.setStyleSheet("QDialog { background-color: #FFFFFF; }")

    def update_progress(
        self,
        completed: int = None,
        current_item: str = "",
        error: str = None
    ):
        """
        Thread-safe progress update.

        Args:
            completed: Number of items completed (incremental)
            current_item: Description of current item being processed
            error: Error message to log (if any)
        """
        with QMutexLocker(self.mutex):
            if self.state == ProgressState.CANCELLED:
                return

            # Update completed count
            if completed is not None:
                self.stats.completed = completed

            # Update current item
            if current_item:
                self.stats.current_item = current_item

            # Log error
            if error:
                self.stats.failed += 1
                self._append_error(error)

            # Calculate ETA using exponential moving average
            if self.stats.completed > 0 and self.state == ProgressState.RUNNING:
                current_time = time.time()
                elapsed = current_time - self._last_update_time

                if elapsed > 0:
                    instant_rate = 1.0 / elapsed
                    # Exponential moving average
                    self._items_per_second = (
                        self._ema_alpha * instant_rate +
                        (1 - self._ema_alpha) * self._items_per_second
                    )

                    remaining = self.stats.total_items - self.stats.completed
                    if self._items_per_second > 0:
                        self.stats.eta_seconds = remaining / self._items_per_second

                self._last_update_time = current_time

            # Update UI
            self._update_display()

    def _update_display(self):
        """Update all UI elements with current stats."""
        # Counter
        self.counter_label.setText(
            f"{self.stats.completed} / {self.stats.total_items} items"
        )

        # Progress bar
        self.progress_bar.setValue(self.stats.completed)

        # ETA
        if self.state == ProgressState.RUNNING and self.stats.eta_seconds > 0:
            eta_str = self._format_eta(self.stats.eta_seconds)
            self.eta_label.setText(f"ETA: {eta_str}")
        elif self.state == ProgressState.PAUSED:
            self.eta_label.setText("⏸ Paused")
        else:
            self.eta_label.setText("")

        # Current item
        if self.stats.current_item:
            # Truncate long paths
            display_item = self.stats.current_item
            if len(display_item) > 60:
                display_item = "..." + display_item[-57:]
            self.current_item_label.setText(f"Processing: {display_item}")

        # Success/Failed counters
        success_count = self.stats.completed - self.stats.failed
        self.success_label.setText(f"✓ {success_count} succeeded")

        if self.stats.failed > 0:
            self.failed_label.setText(f"✗ {self.stats.failed} failed")
            self.toggle_errors_btn.setText(f"Show Errors ({self.stats.failed})")
        else:
            self.failed_label.setText("")

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

    def _append_error(self, error: str):
        """Append error to error log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.error_log.append(f"[{timestamp}] {error}")

        # Auto-scroll to bottom
        scrollbar = self.error_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _toggle_errors(self):
        """Toggle error log visibility."""
        if self.error_log.isVisible():
            self.error_log.hide()
            self.toggle_errors_btn.setText(f"Show Errors ({self.stats.failed})")
        else:
            self.error_log.show()
            self.toggle_errors_btn.setText(f"Hide Errors ({self.stats.failed})")
        self.adjustSize()

    def _toggle_pause(self):
        """Toggle pause/resume state."""
        with QMutexLocker(self.mutex):
            if self.state == ProgressState.RUNNING:
                self.state = ProgressState.PAUSED
                self.pause_btn.setText("Resume")
                self.pause_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #10B981;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 10pt;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
                self.paused.emit()
            elif self.state == ProgressState.PAUSED:
                self.state = ProgressState.RUNNING
                self._last_update_time = time.time()  # Reset for ETA
                self.pause_btn.setText("Pause")
                self.pause_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #F59E0B;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 10pt;
                        font-weight: 500;
                    }
                    QPushButton:hover {
                        background-color: #D97706;
                    }
                """)
                self.resumed.emit()

            self._update_display()

    def _on_cancel(self):
        """Handle cancel button click."""
        with QMutexLocker(self.mutex):
            self.state = ProgressState.CANCELLED
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")

            if hasattr(self, 'pause_btn'):
                self.pause_btn.setEnabled(False)

            self.cancelled.emit()

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        with QMutexLocker(self.mutex):
            return self.state == ProgressState.CANCELLED

    def is_paused(self) -> bool:
        """Check if operation is paused."""
        with QMutexLocker(self.mutex):
            return self.state == ProgressState.PAUSED

    def finish(self, success: bool = True, message: str = ""):
        """
        Mark operation as complete.

        Args:
            success: Whether operation completed successfully
            message: Optional completion message
        """
        with QMutexLocker(self.mutex):
            if success:
                self.state = ProgressState.COMPLETED
                self.progress_bar.setValue(self.stats.total_items)

                if not message:
                    message = f"✓ Completed {self.stats.completed} items"

                self.current_item_label.setText(message)
                self.eta_label.setText("")
            else:
                self.state = ProgressState.ERROR
                self.current_item_label.setText(message or "Operation failed")
                self.current_item_label.setStyleSheet("color: #EF4444; font-size: 10pt; font-weight: 500;")

            self.cancel_btn.setText("Close")
            self.cancel_btn.setEnabled(True)
            self.cancel_btn.clicked.disconnect()
            self.cancel_btn.clicked.connect(self.accept)

            if hasattr(self, 'pause_btn'):
                self.pause_btn.setEnabled(False)
