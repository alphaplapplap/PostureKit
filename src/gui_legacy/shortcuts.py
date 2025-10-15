"""Keyboard shortcut management system for PostureKit."""

from PySide6.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem
from PySide6.QtCore import Qt, QObject, Signal, QSettings, QKeyCombination
from PySide6.QtGui import QKeySequence, QShortcut, QAction, QColor
from enum import Enum
from typing import Optional, Dict, List, Callable, Set
from dataclasses import dataclass
import sys
import logging

logger = logging.getLogger(__name__)


class ShortcutContext(Enum):
    """Shortcut activation contexts."""
    GLOBAL = "global"  # Active everywhere
    VIEWER = "viewer"  # Only in image viewer
    EDITING = "editing"  # Only when editing keypoints
    NAVIGATION = "navigation"  # Only when navigating
    TEXT_INPUT = "text_input"  # Only in text fields (usually for overriding)


@dataclass
class ShortcutDefinition:
    """Definition of a keyboard shortcut."""
    action_id: str
    description: str
    default_sequence: str  # Platform-agnostic (e.g., "Ctrl+O")
    context: ShortcutContext
    category: str  # For grouping in UI (File, Edit, View, etc.)
    customizable: bool = True

    def get_platform_sequence(self) -> str:
        """Get platform-appropriate key sequence."""
        if sys.platform == "darwin":
            # macOS: Replace Ctrl with Cmd, except for specific cases
            sequence = self.default_sequence.replace("Ctrl+", "Cmd+")
            # Some shortcuts should remain Ctrl on macOS (text editing)
            if self.context == ShortcutContext.TEXT_INPUT:
                sequence = self.default_sequence
            return sequence
        return self.default_sequence


class ShortcutManager(QObject):
    """
    Central keyboard shortcut management system.

    Handles shortcut registration, conflict detection, customization,
    and context-aware activation.
    """

    shortcut_triggered = Signal(str)  # action_id

    _instance: Optional['ShortcutManager'] = None

    # Default shortcut definitions (from spec)
    DEFAULT_SHORTCUTS = [
        # File Operations
        ShortcutDefinition("file.open_image", "Open Image", "Ctrl+O", ShortcutContext.GLOBAL, "File"),
        ShortcutDefinition("file.open_directory", "Open Directory", "Ctrl+Shift+O", ShortcutContext.GLOBAL, "File"),
        ShortcutDefinition("file.save", "Save Corrections", "Ctrl+S", ShortcutContext.GLOBAL, "File"),
        ShortcutDefinition("file.export_pose", "Export Current Pose", "Ctrl+E", ShortcutContext.GLOBAL, "File"),
        ShortcutDefinition("file.export_dataset", "Export Training Dataset", "Ctrl+Shift+E", ShortcutContext.GLOBAL, "File"),
        ShortcutDefinition("file.quit", "Quit Application", "Ctrl+Q", ShortcutContext.GLOBAL, "File"),

        # Edit Operations
        ShortcutDefinition("edit.undo", "Undo", "Ctrl+Z", ShortcutContext.EDITING, "Edit"),
        ShortcutDefinition("edit.redo", "Redo", "Ctrl+Shift+Z", ShortcutContext.EDITING, "Edit"),
        ShortcutDefinition("edit.reset", "Reset Corrections", "Ctrl+R", ShortcutContext.EDITING, "Edit"),
        ShortcutDefinition("edit.preferences", "Preferences", "Ctrl+,", ShortcutContext.GLOBAL, "Edit"),

        # View Controls
        ShortcutDefinition("view.zoom_in", "Zoom In", "Ctrl++", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.zoom_out", "Zoom Out", "Ctrl+-", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.fit_window", "Fit to Window", "Ctrl+0", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.actual_size", "Actual Size (100%)", "Ctrl+1", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.zoom_person", "Zoom to Person", "Ctrl+2", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.toggle_skeleton", "Toggle Skeleton", "Ctrl+K", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.toggle_keypoints", "Toggle Keypoints", "Ctrl+J", ShortcutContext.VIEWER, "View"),
        ShortcutDefinition("view.toggle_panel", "Toggle Correction Panel", "Ctrl+P", ShortcutContext.GLOBAL, "View"),
        ShortcutDefinition("view.fullscreen", "Toggle Fullscreen", "F11", ShortcutContext.GLOBAL, "View"),

        # Process
        ShortcutDefinition("process.detect", "Detect Poses", "Ctrl+D", ShortcutContext.GLOBAL, "Process"),
        ShortcutDefinition("process.batch", "Batch Process", "Ctrl+B", ShortcutContext.GLOBAL, "Process"),

        # Database
        ShortcutDefinition("database.delete_pose", "Delete Pose", "Ctrl+Shift+D", ShortcutContext.GLOBAL, "Database"),
        ShortcutDefinition("database.statistics", "View Statistics", "Ctrl+I", ShortcutContext.GLOBAL, "Database"),

        # Navigation
        ShortcutDefinition("nav.next_person", "Next Person", "Tab", ShortcutContext.VIEWER, "Navigation"),
        ShortcutDefinition("nav.prev_person", "Previous Person", "Shift+Tab", ShortcutContext.VIEWER, "Navigation"),

        # Keypoint Editing (non-customizable, context-specific)
        ShortcutDefinition("edit.nudge_up", "Nudge Keypoint Up", "Up", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_down", "Nudge Keypoint Down", "Down", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_left", "Nudge Keypoint Left", "Left", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_right", "Nudge Keypoint Right", "Right", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_up_large", "Nudge Up (10px)", "Shift+Up", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_down_large", "Nudge Down (10px)", "Shift+Down", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_left_large", "Nudge Left (10px)", "Shift+Left", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.nudge_right_large", "Nudge Right (10px)", "Shift+Right", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.delete_keypoint", "Delete Keypoint", "Delete", ShortcutContext.EDITING, "Editing", customizable=False),
        ShortcutDefinition("edit.deselect", "Deselect", "Escape", ShortcutContext.EDITING, "Editing", customizable=False),

        # Help
        ShortcutDefinition("help.shortcuts", "Keyboard Shortcuts", "F1", ShortcutContext.GLOBAL, "Help"),
    ]

    def __init__(self):
        """Initialize shortcut manager (singleton)."""
        if ShortcutManager._instance is not None:
            raise RuntimeError("ShortcutManager is a singleton. Use ShortcutManager.instance()")

        super().__init__()

        # Shortcut registry
        self.definitions: Dict[str, ShortcutDefinition] = {}
        self.shortcuts: Dict[str, QShortcut] = {}  # action_id -> QShortcut
        self.active_contexts: Set[ShortcutContext] = {ShortcutContext.GLOBAL}

        # Settings persistence
        self.settings = QSettings("PostureKit", "PostureKit")

        # Register default shortcuts
        for definition in self.DEFAULT_SHORTCUTS:
            self.definitions[definition.action_id] = definition

        # Load custom shortcuts
        self._load_custom_shortcuts()

        ShortcutManager._instance = self

        logger.info(f"ShortcutManager initialized with {len(self.definitions)} shortcuts")

    @classmethod
    def instance(cls) -> 'ShortcutManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ShortcutManager()
        return cls._instance

    def _load_custom_shortcuts(self):
        """Load customized shortcuts from settings."""
        self.settings.beginGroup("shortcuts")
        for action_id in self.definitions.keys():
            custom_sequence = self.settings.value(action_id)
            if custom_sequence:
                logger.debug(f"Loaded custom shortcut for {action_id}: {custom_sequence}")
                # Store custom sequence (will be applied when registered)
                self.definitions[action_id].default_sequence = custom_sequence
        self.settings.endGroup()

    def register_shortcuts(self, parent: QWidget):
        """
        Register all shortcuts for a parent widget.

        Args:
            parent: Parent widget (typically MainWindow)
        """
        for action_id, definition in self.definitions.items():
            sequence = QKeySequence(definition.get_platform_sequence())

            shortcut = QShortcut(sequence, parent)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda aid=action_id: self._on_shortcut_activated(aid))

            self.shortcuts[action_id] = shortcut

            logger.debug(f"Registered shortcut {action_id}: {sequence.toString()}")

    def register_action_shortcut(self, action: QAction, action_id: str):
        """
        Register shortcut for a QAction.

        Args:
            action: QAction to bind shortcut to
            action_id: Action identifier
        """
        if action_id not in self.definitions:
            logger.warning(f"Unknown action ID: {action_id}")
            return

        definition = self.definitions[action_id]
        sequence = QKeySequence(definition.get_platform_sequence())
        action.setShortcut(sequence)

        # Store reference
        self.shortcuts[action_id] = action

    def _on_shortcut_activated(self, action_id: str):
        """Handle shortcut activation with context checking."""
        definition = self.definitions.get(action_id)
        if not definition:
            return

        # Check if shortcut is active in current context
        if definition.context not in self.active_contexts and definition.context != ShortcutContext.GLOBAL:
            logger.debug(f"Shortcut {action_id} ignored (context {definition.context} not active)")
            return

        logger.debug(f"Shortcut activated: {action_id}")
        self.shortcut_triggered.emit(action_id)

    def set_active_contexts(self, contexts: Set[ShortcutContext]):
        """
        Set currently active shortcut contexts.

        Args:
            contexts: Set of active contexts (GLOBAL is always included)
        """
        self.active_contexts = contexts | {ShortcutContext.GLOBAL}
        logger.debug(f"Active contexts: {[c.value for c in self.active_contexts]}")

    def add_context(self, context: ShortcutContext):
        """Add a context to active contexts."""
        self.active_contexts.add(context)
        logger.debug(f"Added context: {context.value}")

    def remove_context(self, context: ShortcutContext):
        """Remove a context from active contexts."""
        if context != ShortcutContext.GLOBAL:
            self.active_contexts.discard(context)
            logger.debug(f"Removed context: {context.value}")

    def customize_shortcut(self, action_id: str, new_sequence: str) -> bool:
        """
        Customize a shortcut.

        Args:
            action_id: Action to customize
            new_sequence: New key sequence

        Returns:
            True if successful, False if conflict or error
        """
        if action_id not in self.definitions:
            logger.error(f"Unknown action ID: {action_id}")
            return False

        definition = self.definitions[action_id]
        if not definition.customizable:
            logger.error(f"Shortcut {action_id} is not customizable")
            return False

        # Check for conflicts
        conflict = self._check_conflict(action_id, new_sequence)
        if conflict:
            logger.error(f"Shortcut conflict: {new_sequence} already used by {conflict}")
            return False

        # Update definition
        old_sequence = definition.default_sequence
        definition.default_sequence = new_sequence

        # Update registered shortcut
        if action_id in self.shortcuts:
            shortcut = self.shortcuts[action_id]
            if isinstance(shortcut, QShortcut):
                shortcut.setKey(QKeySequence(new_sequence))
            elif isinstance(shortcut, QAction):
                shortcut.setShortcut(QKeySequence(new_sequence))

        # Save to settings
        self.settings.beginGroup("shortcuts")
        self.settings.setValue(action_id, new_sequence)
        self.settings.endGroup()

        logger.info(f"Customized shortcut {action_id}: {old_sequence} -> {new_sequence}")
        return True

    def reset_shortcut(self, action_id: str):
        """Reset shortcut to default."""
        if action_id not in self.definitions:
            return

        # Find default in DEFAULT_SHORTCUTS
        for default in self.DEFAULT_SHORTCUTS:
            if default.action_id == action_id:
                self.customize_shortcut(action_id, default.default_sequence)

                # Remove from settings
                self.settings.beginGroup("shortcuts")
                self.settings.remove(action_id)
                self.settings.endGroup()

                logger.info(f"Reset shortcut {action_id} to default")
                break

    def reset_all_shortcuts(self):
        """Reset all shortcuts to defaults."""
        for action_id in self.definitions.keys():
            self.reset_shortcut(action_id)

    def _check_conflict(self, action_id: str, sequence: str) -> Optional[str]:
        """
        Check if sequence conflicts with existing shortcuts.

        Args:
            action_id: Action being customized (excluded from conflict check)
            sequence: Sequence to check

        Returns:
            Conflicting action_id if conflict exists, None otherwise
        """
        for other_id, definition in self.definitions.items():
            if other_id == action_id:
                continue

            if definition.default_sequence == sequence:
                # Check if contexts overlap
                if (definition.context == ShortcutContext.GLOBAL or
                    self.definitions[action_id].context == ShortcutContext.GLOBAL or
                    definition.context == self.definitions[action_id].context):
                    return other_id

        return None

    def get_shortcut_text(self, action_id: str) -> str:
        """Get display text for shortcut."""
        if action_id not in self.definitions:
            return ""

        definition = self.definitions[action_id]
        sequence = definition.get_platform_sequence()

        # Format for display (e.g., "Ctrl+O" or "⌘O" on macOS)
        if sys.platform == "darwin":
            sequence = sequence.replace("Cmd+", "⌘")
            sequence = sequence.replace("Shift+", "⇧")
            sequence = sequence.replace("Alt+", "⌥")
            sequence = sequence.replace("Ctrl+", "⌃")

        return sequence

    def get_shortcuts_by_category(self) -> Dict[str, List[ShortcutDefinition]]:
        """Get shortcuts grouped by category."""
        categories: Dict[str, List[ShortcutDefinition]] = {}

        for definition in self.definitions.values():
            if definition.category not in categories:
                categories[definition.category] = []
            categories[definition.category].append(definition)

        # Sort by action_id within each category
        for category in categories:
            categories[category].sort(key=lambda d: d.action_id)

        return categories


class ShortcutsDialog(QDialog):
    """Dialog showing all keyboard shortcuts."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.manager = ShortcutManager.instance()
        self.setup_ui()

    def setup_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(False)  # Non-blocking
        self.setMinimumSize(700, 600)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter shortcuts...")
        self.search_input.textChanged.connect(self._filter_shortcuts)
        search_layout.addWidget(self.search_input)

        clear_btn = QPushButton("✕")
        clear_btn.setFixedSize(24, 24)
        clear_btn.clicked.connect(lambda: self.search_input.clear())
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14pt;
                color: #6B7280;
            }
            QPushButton:hover {
                color: #1F2937;
            }
        """)
        search_layout.addWidget(clear_btn)

        layout.addLayout(search_layout)

        # Shortcuts tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Action", "Shortcut"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 400)
        layout.addWidget(self.tree)

        # Populate tree
        self._populate_tree()

        # Info label
        info_label = QLabel(
            "Non-customizable shortcuts are shown in gray. "
            "Context-specific shortcuts only work in certain situations."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #6B7280; font-size: 9pt;")
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        close_btn.setStyleSheet("""
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
        """)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _populate_tree(self):
        """Populate shortcuts tree."""
        self.tree.clear()

        categories = self.manager.get_shortcuts_by_category()

        for category, shortcuts in sorted(categories.items()):
            category_item = QTreeWidgetItem(self.tree, [category, ""])
            font = category_item.font(0)
            font.setBold(True)
            category_item.setFont(0, font)

            for definition in shortcuts:
                shortcut_text = self.manager.get_shortcut_text(definition.action_id)
                item = QTreeWidgetItem(category_item, [
                    definition.description,
                    shortcut_text
                ])

                # Gray out non-customizable shortcuts
                if not definition.customizable:
                    for col in range(2):
                        item.setForeground(col, QColor("#9CA3AF"))

                # Store action_id for filtering
                item.setData(0, Qt.ItemDataRole.UserRole, definition.action_id)

        self.tree.expandAll()

    def _filter_shortcuts(self, text: str):
        """Filter shortcuts by search text."""
        text = text.lower()

        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category_visible = False

            for j in range(category_item.childCount()):
                shortcut_item = category_item.child(j)
                description = shortcut_item.text(0).lower()
                shortcut = shortcut_item.text(1).lower()

                # Show if text matches description or shortcut
                visible = text in description or text in shortcut
                shortcut_item.setHidden(not visible)

                if visible:
                    category_visible = True

            # Hide category if no children visible
            category_item.setHidden(not category_visible)


# Convenience function
def get_shortcuts() -> ShortcutManager:
    """Get global shortcut manager instance."""
    return ShortcutManager.instance()


if __name__ == "__main__":
    # Test shortcuts system
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget

    app = QApplication(sys.argv)

    # Initialize manager
    manager = ShortcutManager.instance()

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)

    # Status label
    status = QLabel("Press a keyboard shortcut...")
    status.setStyleSheet("font-size: 12pt; padding: 10px;")
    layout.addWidget(status)

    # Connect to shortcut signals
    def on_shortcut(action_id: str):
        definition = manager.definitions[action_id]
        status.setText(f"Triggered: {definition.description} ({action_id})")

    manager.shortcut_triggered.connect(on_shortcut)

    # Register shortcuts
    manager.register_shortcuts(window)

    # Test context switching
    context_label = QLabel(f"Active contexts: {[c.value for c in manager.active_contexts]}")
    layout.addWidget(context_label)

    def toggle_editing():
        if ShortcutContext.EDITING in manager.active_contexts:
            manager.remove_context(ShortcutContext.EDITING)
        else:
            manager.add_context(ShortcutContext.EDITING)
        context_label.setText(f"Active contexts: {[c.value for c in manager.active_contexts]}")

    toggle_btn = QPushButton("Toggle Editing Context")
    toggle_btn.clicked.connect(toggle_editing)
    layout.addWidget(toggle_btn)

    # Show shortcuts dialog
    def show_shortcuts():
        dialog = ShortcutsDialog(window)
        dialog.exec()

    shortcuts_btn = QPushButton("Show Shortcuts (F1)")
    shortcuts_btn.clicked.connect(show_shortcuts)
    layout.addWidget(shortcuts_btn)

    window.setCentralWidget(central)
    window.resize(600, 400)
    window.show()

    sys.exit(app.exec())
