"""
Training Labels Input Widget for PostureKit
Handles category, tags, and notes input for pose annotations.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QCheckBox, QPushButton,
    QCompleter, QDialog, QDialogButtonBox, QGroupBox,
    QMessageBox, QFrame
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QStringListModel,
    QTimer, QPropertyAnimation, QEasingCurve,
    QRect, Property
)
from PySide6.QtGui import (
    QPalette, QColor, QFont, QFontMetrics,
    QKeyEvent, QPainter, QPaintEvent
)


@dataclass
class TrainingLabels:
    """Data class for training labels."""
    category: str = ""
    tags: List[str] = None
    notes: str = ""
    include_in_training: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def is_empty(self) -> bool:
        """Check if labels are empty."""
        return not self.category and not self.tags and not self.notes

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        return {
            'category': self.category,
            'tags': self.tags,
            'notes': self.notes,
            'include_in_training': self.include_in_training
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingLabels':
        """Create from dictionary."""
        return cls(
            category=data.get('category', ''),
            tags=data.get('tags', []),
            notes=data.get('notes', ''),
            include_in_training=data.get('include_in_training', True)
        )

    def __eq__(self, other) -> bool:
        """Check equality for change detection."""
        if not isinstance(other, TrainingLabels):
            return False
        return (self.category == other.category and
                self.tags == other.tags and
                self.notes == other.notes and
                self.include_in_training == other.include_in_training)


class TagInput(QLineEdit):
    """Custom line edit for tag input with visual feedback."""

    tags_changed = Signal(list)  # List of tags

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tags: List[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        self.setPlaceholderText("Enter tags separated by commas (e.g., yoga, balance, indoor)")
        self.textChanged.connect(self._on_text_changed)

        # Style for visual feedback
        self.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                padding: 5px;
            }
        """)

    def _on_text_changed(self, text: str) -> None:
        """Process text changes to extract tags."""
        # Parse tags from comma-separated text
        if text:
            raw_tags = [t.strip() for t in text.split(',')]
            # Filter out empty strings and clean up
            self.tags = [tag for tag in raw_tags if tag and tag != '']
        else:
            self.tags = []

    def get_tags(self) -> List[str]:
        """Get the list of cleaned tags."""
        # Final parse when getting tags
        text = self.text()
        if text:
            raw_tags = [t.strip() for t in text.split(',')]
            return [tag.lower() for tag in raw_tags if tag]  # Normalize to lowercase
        return []

    def set_tags(self, tags: List[str]) -> None:
        """Set tags from a list."""
        self.tags = tags
        self.setText(', '.join(tags))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle special key events."""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Emit tags when Enter is pressed
            self.tags_changed.emit(self.get_tags())
        super().keyPressEvent(event)


class CategoryInput(QLineEdit):
    """Category input with autocomplete from existing categories."""

    category_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.existing_categories: Set[str] = set()
        self.completer: Optional[QCompleter] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        self.setPlaceholderText("Enter category (e.g., standing, sitting, action)")

        # Style
        self.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                padding: 5px;
            }
        """)

        # Setup completer for autocomplete
        self._setup_completer()

        # Connect signal
        self.textChanged.connect(self.category_changed.emit)

    def _setup_completer(self) -> None:
        """Setup the autocomplete completer."""
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(10)

        # Style the completion popup
        self.completer.popup().setStyleSheet("""
            QListView {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 2px;
                background-color: white;
                selection-background-color: #3b82f6;
            }
            QListView::item {
                padding: 4px;
            }
            QListView::item:hover {
                background-color: #eff6ff;
            }
        """)

        self.setCompleter(self.completer)

    def update_categories(self, categories: List[str]) -> None:
        """Update the list of existing categories for autocomplete."""
        self.existing_categories = set(categories)

        # Update completer model
        if self.completer:
            model = QStringListModel(sorted(self.existing_categories))
            self.completer.setModel(model)

    def add_category(self, category: str) -> None:
        """Add a new category to the autocomplete list."""
        if category and category not in self.existing_categories:
            self.existing_categories.add(category)
            self.update_categories(list(self.existing_categories))

    def get_category(self) -> str:
        """Get the current category."""
        return self.text().strip().lower()  # Normalize to lowercase


class NotesInput(QTextEdit):
    """Multi-line text input for notes."""

    notes_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._last_text = ""

        # Setup change detection timer
        self.change_timer = QTimer()
        self.change_timer.setInterval(500)  # 500ms delay
        self.change_timer.setSingleShot(True)
        self.change_timer.timeout.connect(self._emit_change)

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        self.setPlaceholderText("Add any additional notes about this pose...")
        self.setMaximumHeight(80)
        self.setMinimumHeight(60)

        # Style
        self.setStyleSheet("""
            QTextEdit {
                padding: 6px;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                font-size: 10pt;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #e0e0e0;
                background-color: #2a2a2a;
            }
            QTextEdit:focus {
                border: 2px solid #3b82f6;
                padding: 5px;
            }
        """)

        # Connect text change with debouncing
        self.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self) -> None:
        """Handle text changes with debouncing."""
        self.change_timer.stop()
        self.change_timer.start()

    def _emit_change(self) -> None:
        """Emit change signal if text actually changed."""
        current_text = self.toPlainText()
        if current_text != self._last_text:
            self._last_text = current_text
            self.notes_changed.emit(current_text)

    def get_notes(self) -> str:
        """Get the current notes text."""
        return self.toPlainText().strip()

    def set_notes(self, notes: str) -> None:
        """Set the notes text."""
        self.setPlainText(notes)
        self._last_text = notes


class LabelChangeDialog(QDialog):
    """Dialog for confirming label changes when overwriting existing labels."""

    def __init__(self,
                 current_labels: TrainingLabels,
                 new_labels: TrainingLabels,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_labels = current_labels
        self.new_labels = new_labels
        self.setWindowTitle("Confirm Label Changes")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Warning message
        warning_label = QLabel("⚠️ This pose already has labels.")
        warning_label.setStyleSheet("""
            QLabel {
                color: #f59e0b;
                font-weight: bold;
                font-size: 11pt;
                padding: 8px;
                background-color: #fef3c7;
                border: 1px solid #fbbf24;
                border-radius: 4px;
            }
        """)
        layout.addWidget(warning_label)

        # Current labels
        current_group = QGroupBox("Current Labels:")
        current_layout = QVBoxLayout()
        current_layout.setSpacing(4)

        if self.current_labels.category:
            cat_label = QLabel(f"Category: {self.current_labels.category}")
            cat_label.setStyleSheet("color: #374151;")
            current_layout.addWidget(cat_label)

        if self.current_labels.tags:
            tags_label = QLabel(f"Tags: {', '.join(self.current_labels.tags)}")
            tags_label.setStyleSheet("color: #374151;")
            tags_label.setWordWrap(True)
            current_layout.addWidget(tags_label)

        if self.current_labels.notes:
            notes_label = QLabel(f"Notes: {self.current_labels.notes[:100]}...")
            notes_label.setStyleSheet("color: #374151;")
            notes_label.setWordWrap(True)
            current_layout.addWidget(notes_label)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # New labels
        new_group = QGroupBox("New Labels:")
        new_layout = QVBoxLayout()
        new_layout.setSpacing(4)

        if self.new_labels.category:
            cat_label = QLabel(f"Category: {self.new_labels.category}")
            cat_label.setStyleSheet("color: #047857; font-weight: bold;")
            new_layout.addWidget(cat_label)

        if self.new_labels.tags:
            tags_label = QLabel(f"Tags: {', '.join(self.new_labels.tags)}")
            tags_label.setStyleSheet("color: #047857; font-weight: bold;")
            tags_label.setWordWrap(True)
            new_layout.addWidget(tags_label)

        if self.new_labels.notes:
            notes_label = QLabel(f"Notes: {self.new_labels.notes[:100]}...")
            notes_label.setStyleSheet("color: #047857; font-weight: bold;")
            notes_label.setWordWrap(True)
            new_layout.addWidget(notes_label)

        new_group.setLayout(new_layout)
        layout.addWidget(new_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Ok
        )
        button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Overwrite")
        button_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

        self.setFixedWidth(400)


class TrainingLabelsWidget(QWidget):
    """Main training labels input widget."""

    # Signals
    labels_changed = Signal(TrainingLabels)
    save_requested = Signal(TrainingLabels)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Add alias for backward compatibility
        self.label_changed = self.labels_changed

        self.current_labels = TrainingLabels()
        self.has_unsaved_changes = False
        self.existing_categories: List[str] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Title with unsaved indicator
        title_layout = QHBoxLayout()

        title_label = QLabel("Training Labels")
        title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        title_layout.addWidget(title_label)

        self.unsaved_indicator = QLabel("*")
        self.unsaved_indicator.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 12pt;")
        self.unsaved_indicator.setVisible(False)
        self.unsaved_indicator.setToolTip("Unsaved changes")
        title_layout.addWidget(self.unsaved_indicator)

        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Category input
        category_layout = QVBoxLayout()
        category_layout.setSpacing(4)

        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: 500;")
        category_layout.addWidget(category_label)

        self.category_input = CategoryInput()
        category_layout.addWidget(self.category_input)

        autocomplete_hint = QLabel("(Autocomplete from existing categories)")
        autocomplete_hint.setStyleSheet("color: #6b7280; font-size: 9pt;")
        category_layout.addWidget(autocomplete_hint)

        layout.addLayout(category_layout)

        # Tags input
        tags_layout = QVBoxLayout()
        tags_layout.setSpacing(4)

        tags_label = QLabel("Tags:")
        tags_label.setStyleSheet("font-weight: 500;")
        tags_layout.addWidget(tags_label)

        self.tags_input = TagInput()
        tags_layout.addWidget(self.tags_input)

        tags_hint = QLabel("(Comma-separated, e.g., indoor, yoga, balance)")
        tags_hint.setStyleSheet("color: #6b7280; font-size: 9pt;")
        tags_layout.addWidget(tags_hint)

        layout.addLayout(tags_layout)

        # Notes input
        notes_layout = QVBoxLayout()
        notes_layout.setSpacing(4)

        notes_label = QLabel("Notes:")
        notes_label.setStyleSheet("font-weight: 500;")
        notes_layout.addWidget(notes_label)

        self.notes_input = NotesInput()
        notes_layout.addWidget(self.notes_input)

        layout.addLayout(notes_layout)

        # Include in training checkbox
        self.include_checkbox = QCheckBox("Include in training dataset")
        self.include_checkbox.setChecked(True)
        self.include_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 10pt;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        layout.addWidget(self.include_checkbox)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear all fields")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e5e7eb;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d1d5db;
            }
        """)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()

        self.save_button = QPushButton("Save Labels")
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
            }
        """)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Add stretch at bottom
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Connect input changes
        self.category_input.category_changed.connect(self._on_content_changed)
        self.tags_input.textChanged.connect(self._on_content_changed)
        self.notes_input.notes_changed.connect(self._on_content_changed)
        self.include_checkbox.toggled.connect(self._on_content_changed)

        # Connect buttons
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)

    def _on_content_changed(self) -> None:
        """Handle any content change."""
        # Get current state
        new_labels = self.get_labels()

        # Check if changed from original
        self.has_unsaved_changes = (new_labels != self.current_labels)

        # Update UI
        self.unsaved_indicator.setVisible(self.has_unsaved_changes)
        self.save_button.setEnabled(self.has_unsaved_changes and not new_labels.is_empty())

        # Emit change signal
        self.labels_changed.emit(new_labels)

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        # Confirm if there's content
        if not self.get_labels().is_empty():
            reply = QMessageBox.question(
                self,
                "Clear Labels",
                "Are you sure you want to clear all labels?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        # Clear all fields
        self.category_input.clear()
        self.tags_input.clear()
        self.notes_input.clear()
        self.include_checkbox.setChecked(True)

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        new_labels = self.get_labels()

        # Check if overwriting existing labels
        if not self.current_labels.is_empty() and new_labels != self.current_labels:
            dialog = LabelChangeDialog(self.current_labels, new_labels, self)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

        # Save the labels
        self.current_labels = new_labels
        self.has_unsaved_changes = False
        self.unsaved_indicator.setVisible(False)
        self.save_button.setEnabled(False)

        # Add category to autocomplete if new
        if new_labels.category:
            self.category_input.add_category(new_labels.category)

        # Emit save signal
        self.save_requested.emit(new_labels)

        # Show confirmation
        self._show_save_confirmation()

    def _show_save_confirmation(self) -> None:
        """Show a brief confirmation that labels were saved."""
        # Create a temporary label for feedback
        confirmation = QLabel("✓ Labels saved", self)
        confirmation.setStyleSheet("""
            QLabel {
                background-color: #10b981;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        confirmation.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Position at top-center of widget
        confirmation.move(
            (self.width() - confirmation.sizeHint().width()) // 2,
            10
        )
        confirmation.show()

        # Fade out after 2 seconds
        QTimer.singleShot(2000, confirmation.deleteLater)

    def get_labels(self) -> TrainingLabels:
        """Get the current label values."""
        return TrainingLabels(
            category=self.category_input.get_category(),
            tags=self.tags_input.get_tags(),
            notes=self.notes_input.get_notes(),
            include_in_training=self.include_checkbox.isChecked()
        )

    def set_labels(self, labels: TrainingLabels) -> None:
        """Set label values from a TrainingLabels object."""
        # Block signals during update
        self.category_input.blockSignals(True)
        self.tags_input.blockSignals(True)
        self.notes_input.blockSignals(True)
        self.include_checkbox.blockSignals(True)

        # Update fields
        self.category_input.setText(labels.category)
        self.tags_input.set_tags(labels.tags)
        self.notes_input.set_notes(labels.notes)
        self.include_checkbox.setChecked(labels.include_in_training)

        # Restore signals
        self.category_input.blockSignals(False)
        self.tags_input.blockSignals(False)
        self.notes_input.blockSignals(False)
        self.include_checkbox.blockSignals(False)

        # Update state
        self.current_labels = labels
        self.has_unsaved_changes = False
        self.unsaved_indicator.setVisible(False)
        self.save_button.setEnabled(False)

    def update_categories(self, categories: List[str]) -> None:
        """Update the list of available categories for autocomplete."""
        self.existing_categories = categories
        self.category_input.update_categories(categories)

    def reset(self) -> None:
        """Reset to empty state."""
        self.set_labels(TrainingLabels())

    def has_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self.has_unsaved_changes

    def prompt_save_if_needed(self) -> bool:
        """
        Prompt to save if there are unsaved changes.

        Returns:
            True if safe to proceed, False if user cancelled
        """
        if not self.has_unsaved_changes:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved label changes. Do you want to save them?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save
        )

        if reply == QMessageBox.StandardButton.Save:
            self._on_save_clicked()
            return True
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:  # Cancel
            return False


# Legacy compatibility - map old LabelInput to new TrainingLabelsWidget
LabelInput = TrainingLabelsWidget


# Example usage and testing
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Training Labels Test")
    window.resize(400, 500)

    # Create labels widget
    labels_widget = TrainingLabelsWidget()

    # Set some test categories for autocomplete
    test_categories = ['standing', 'sitting', 'yoga', 'running', 'jumping']
    labels_widget.update_categories(test_categories)

    # Set initial labels for testing
    initial_labels = TrainingLabels(
        category='standing',
        tags=['indoor', 'portrait'],
        notes='Good quality pose for training',
        include_in_training=True
    )
    labels_widget.set_labels(initial_labels)

    # Connect signals
    labels_widget.labels_changed.connect(
        lambda labels: print(f"Labels changed: {labels.category}, {labels.tags}")
    )
    labels_widget.save_requested.connect(
        lambda labels: print(f"Save requested: {labels.to_dict()}")
    )

    window.setCentralWidget(labels_widget)
    window.show()

    sys.exit(app.exec())
