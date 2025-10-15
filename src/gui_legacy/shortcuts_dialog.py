"""
Keyboard Shortcuts Reference Dialog
Comprehensive quick-reference guide for all application shortcuts
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLineEdit, QPushButton,
    QHeaderView, QLabel, QWidget, QScrollArea, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QKeySequence, QIcon
from typing import List, Dict, Tuple


class ShortcutCategory:
    """Shortcut category with actions"""

    def __init__(self, name: str):
        self.name = name
        self.shortcuts: List[Tuple[str, str, str]] = []  # (action, shortcut, description)

    def add(self, action: str, shortcut: str, description: str = ""):
        """Add a shortcut"""
        self.shortcuts.append((action, shortcut, description))


class ShortcutsDialog(QDialog):
    """
    Keyboard Shortcuts Reference Dialog
    Size: 700×600px, Non-modal quick reference
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(False)  # Non-blocking
        self.setMinimumSize(700, 600)

        # Build shortcuts database
        self.categories = self._build_shortcuts()
        self.all_shortcuts = self._get_all_shortcuts()

        self._setup_ui()
        self._populate_tables()

    def _setup_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)

        # Search bar
        search_layout = QHBoxLayout()

        search_label = QLabel("Search:")
        search_layout.addWidget(search_label)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter shortcuts...")
        self.search_edit.textChanged.connect(self._filter_shortcuts)
        search_layout.addWidget(self.search_edit)

        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(30)
        clear_btn.clicked.connect(self.search_edit.clear)
        clear_btn.setToolTip("Clear search")
        search_layout.addWidget(clear_btn)

        layout.addLayout(search_layout)

        # Tab widget for categories
        self.tab_widget = QTabWidget()

        # Create tabs
        self.file_table = self._create_shortcuts_table()
        self.edit_table = self._create_shortcuts_table()
        self.view_table = self._create_shortcuts_table()
        self.process_table = self._create_shortcuts_table()
        self.database_table = self._create_shortcuts_table()
        self.help_table = self._create_shortcuts_table()
        self.all_table = self._create_shortcuts_table()

        self.tab_widget.addTab(self.file_table, "File")
        self.tab_widget.addTab(self.edit_table, "Edit")
        self.tab_widget.addTab(self.view_table, "View")
        self.tab_widget.addTab(self.process_table, "Process")
        self.tab_widget.addTab(self.database_table, "Database")
        self.tab_widget.addTab(self.help_table, "Help")
        self.tab_widget.addTab(self.all_table, "All")

        layout.addWidget(self.tab_widget)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Map tables to categories
        self.table_map = {
            'File': self.file_table,
            'Edit': self.edit_table,
            'View': self.view_table,
            'Process': self.process_table,
            'Database': self.database_table,
            'Help': self.help_table,
            'All': self.all_table
        }

    def _create_shortcuts_table(self) -> QTableWidget:
        """Create a shortcuts table widget"""
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])

        # Configure table
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)

        # Column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        # Font
        font = table.font()
        font.setPointSize(10)
        table.setFont(font)

        return table

    def _build_shortcuts(self) -> Dict[str, ShortcutCategory]:
        """Build shortcuts database from spec"""
        categories = {}

        # File Operations
        file_cat = ShortcutCategory("File")
        file_cat.add("Open Image", "Ctrl+O", "Open single image file")
        file_cat.add("Open Directory", "Ctrl+Shift+O", "Open directory browser")
        file_cat.add("Save Corrections", "Ctrl+S", "Save edits to database")
        file_cat.add("Export Corrections", "Ctrl+E", "Export current pose to JSON")
        file_cat.add("Export Dataset", "Ctrl+Shift+E", "Full dataset export dialog")
        file_cat.add("Quit", "Ctrl+Q", "Exit application")
        categories['File'] = file_cat

        # Edit Operations
        edit_cat = ShortcutCategory("Edit")
        edit_cat.add("Undo", "Ctrl+Z", "Undo last keypoint edit")
        edit_cat.add("Redo", "Ctrl+Shift+Z", "Redo undone edit")
        edit_cat.add("Reset Corrections", "Ctrl+R", "Restore original detection")
        edit_cat.add("Preferences", "Ctrl+,", "Open preferences dialog")
        edit_cat.add("Nudge Keypoint", "Arrow Keys", "Move selected keypoint 1px")
        edit_cat.add("Nudge 10px", "Shift+Arrows", "Move selected keypoint 10px")
        edit_cat.add("Nudge 0.1px", "Ctrl+Arrows", "Move selected keypoint 0.1px")
        edit_cat.add("Move Keypoint", "Shift+Drag", "Drag keypoint to new position")
        edit_cat.add("Toggle Visibility", "Right-Click", "Cycle visibility state")
        edit_cat.add("Delete Keypoint", "Delete", "Set keypoint to missing")
        edit_cat.add("Deselect", "Escape", "Deselect current keypoint")
        categories['Edit'] = edit_cat

        # View Controls
        view_cat = ShortcutCategory("View")
        view_cat.add("Zoom In", "Ctrl++", "Increase zoom 20%")
        view_cat.add("Zoom Out", "Ctrl+-", "Decrease zoom 20%")
        view_cat.add("Fit to Window", "Ctrl+0", "Fit image to viewport")
        view_cat.add("Actual Size (100%)", "Ctrl+1", "Reset zoom to 100%")
        view_cat.add("Zoom to Person", "Ctrl+2", "Zoom to selected person's bbox")
        view_cat.add("Toggle Skeleton", "Ctrl+K", "Show/hide skeleton lines")
        view_cat.add("Toggle Keypoints", "Ctrl+J", "Show/hide keypoint dots")
        view_cat.add("Toggle Correction Panel", "Ctrl+P", "Show/hide right panel")
        view_cat.add("Reset View", "Space", "Reset zoom and pan")
        view_cat.add("Fullscreen", "F11", "Toggle fullscreen mode")
        categories['View'] = view_cat

        # Process Operations
        process_cat = ShortcutCategory("Process")
        process_cat.add("Detect Poses", "Ctrl+D", "Run pose detection on current image")
        process_cat.add("Batch Process", "Ctrl+B", "Batch process directory dialog")
        process_cat.add("Next Person", "Tab", "Switch to next detected person")
        process_cat.add("Previous Person", "Shift+Tab", "Switch to previous person")
        categories['Process'] = process_cat

        # Database Operations
        db_cat = ShortcutCategory("Database")
        db_cat.add("Delete Pose", "Ctrl+Shift+D", "Delete pose from database")
        db_cat.add("View Statistics", "Ctrl+I", "Show database stats dialog")
        categories['Database'] = db_cat

        # Help
        help_cat = ShortcutCategory("Help")
        help_cat.add("Keyboard Shortcuts", "F1", "Show this dialog")
        categories['Help'] = help_cat

        return categories

    def _get_all_shortcuts(self) -> List[Tuple[str, str, str]]:
        """Get all shortcuts combined"""
        all_shortcuts = []
        for category in self.categories.values():
            all_shortcuts.extend(category.shortcuts)
        return sorted(all_shortcuts, key=lambda x: x[0])  # Sort by action name

    def _populate_tables(self):
        """Populate all tables with shortcuts"""
        # Populate category tables
        for cat_name, category in self.categories.items():
            if cat_name in self.table_map:
                self._populate_table(self.table_map[cat_name], category.shortcuts)

        # Populate "All" table
        self._populate_table(self.all_table, self.all_shortcuts)

    def _populate_table(self, table: QTableWidget, shortcuts: List[Tuple[str, str, str]]):
        """Populate a table with shortcuts"""
        table.setRowCount(len(shortcuts))

        for row, (action, shortcut, description) in enumerate(shortcuts):
            # Action column
            action_item = QTableWidgetItem(action)
            action_item.setToolTip(description if description else action)
            table.setItem(row, 0, action_item)

            # Shortcut column
            shortcut_item = QTableWidgetItem(shortcut)
            shortcut_item.setToolTip(description if description else action)

            # Make shortcut bold
            font = shortcut_item.font()
            font.setBold(True)
            shortcut_item.setFont(font)

            table.setItem(row, 1, shortcut_item)

    def _filter_shortcuts(self, search_text: str):
        """Filter shortcuts based on search text"""
        search_lower = search_text.lower()

        # Filter each table
        for table in [self.file_table, self.edit_table, self.view_table,
                     self.process_table, self.database_table, self.help_table, self.all_table]:
            for row in range(table.rowCount()):
                action = table.item(row, 0).text().lower()
                shortcut = table.item(row, 1).text().lower()

                # Show row if search text in action or shortcut
                matches = search_lower in action or search_lower in shortcut
                table.setRowHidden(row, not matches)


class ShortcutsDialogCompact(QDialog):
    """
    Compact version with better grouping and visual hierarchy
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(False)
        self.setMinimumSize(700, 600)

        self._setup_ui()

    def _setup_ui(self):
        """Setup compact UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Keyboard Shortcuts Quick Reference")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Search
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search shortcuts...")
        self.search_edit.textChanged.connect(self._filter_content)
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        # Tabs
        self.tabs = QTabWidget()

        # Create category tabs
        self._add_file_tab()
        self._add_edit_tab()
        self._add_view_tab()
        self._add_process_tab()
        self._add_all_tab()

        layout.addWidget(self.tabs)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        layout.addLayout(close_layout)

    def _create_shortcuts_widget(self, shortcuts: List[Tuple[str, str]]) -> QWidget:
        """Create a widget displaying shortcuts"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(8)

        for action, shortcut in shortcuts:
            if not action:  # Spacer
                spacer = QLabel()
                spacer.setFixedHeight(10)
                layout.addWidget(spacer)
                continue

            row_layout = QHBoxLayout()
            row_layout.setSpacing(20)

            # Action (left-aligned, dots to fill space)
            action_label = QLabel(action)
            row_layout.addWidget(action_label, 1)

            # Shortcut (right-aligned, monospace, bold)
            shortcut_label = QLabel(shortcut)
            font = shortcut_label.font()
            font.setFamily("monospace")
            font.setBold(True)
            shortcut_label.setFont(font)
            shortcut_label.setAlignment(Qt.AlignRight)
            shortcut_label.setStyleSheet("color: #2563EB; background: #EFF6FF; padding: 2px 8px; border-radius: 3px;")
            row_layout.addWidget(shortcut_label)

            layout.addLayout(row_layout)

        layout.addStretch()
        return widget

    def _add_file_tab(self):
        """Add File operations tab"""
        shortcuts = [
            ("Open Image", "Ctrl+O"),
            ("Open Directory", "Ctrl+Shift+O"),
            ("Save Corrections", "Ctrl+S"),
            ("Export Corrections", "Ctrl+E"),
            ("Export Dataset", "Ctrl+Shift+E"),
            ("Quit", "Ctrl+Q"),
        ]
        self.tabs.addTab(self._create_shortcuts_widget(shortcuts), "File")

    def _add_edit_tab(self):
        """Add Edit operations tab"""
        shortcuts = [
            ("Undo", "Ctrl+Z"),
            ("Redo", "Ctrl+Shift+Z"),
            ("Reset Corrections", "Ctrl+R"),
            ("Preferences", "Ctrl+,"),
            ("", ""),  # Spacer
            ("Nudge Keypoint 1px", "Arrow Keys"),
            ("Nudge Keypoint 10px", "Shift+Arrows"),
            ("Nudge Keypoint 0.1px", "Ctrl+Arrows"),
            ("Move Keypoint", "Shift+Drag"),
            ("Toggle Visibility", "Right-Click"),
            ("Delete Keypoint", "Delete"),
            ("Deselect Keypoint", "Escape"),
        ]
        self.tabs.addTab(self._create_shortcuts_widget(shortcuts), "Edit")

    def _add_view_tab(self):
        """Add View controls tab"""
        shortcuts = [
            ("Zoom In", "Ctrl++"),
            ("Zoom Out", "Ctrl+-"),
            ("Fit to Window", "Ctrl+0"),
            ("Actual Size (100%)", "Ctrl+1"),
            ("Zoom to Person", "Ctrl+2"),
            ("", ""),  # Spacer
            ("Toggle Skeleton", "Ctrl+K"),
            ("Toggle Keypoints", "Ctrl+J"),
            ("Toggle Correction Panel", "Ctrl+P"),
            ("Reset View", "Space"),
            ("Fullscreen", "F11"),
        ]
        self.tabs.addTab(self._create_shortcuts_widget(shortcuts), "View")

    def _add_process_tab(self):
        """Add Process operations tab"""
        shortcuts = [
            ("Detect Poses", "Ctrl+D"),
            ("Batch Process", "Ctrl+B"),
            ("", ""),  # Spacer
            ("Next Person", "Tab"),
            ("Previous Person", "Shift+Tab"),
            ("", ""),  # Spacer
            ("Delete Pose", "Ctrl+Shift+D"),
            ("View Statistics", "Ctrl+I"),
        ]
        self.tabs.addTab(self._create_shortcuts_widget(shortcuts), "Process")

    def _add_all_tab(self):
        """Add All shortcuts tab"""
        all_shortcuts = [
            # File
            ("Open Image", "Ctrl+O"),
            ("Open Directory", "Ctrl+Shift+O"),
            ("Save Corrections", "Ctrl+S"),
            ("Export Corrections", "Ctrl+E"),
            ("Export Dataset", "Ctrl+Shift+E"),
            ("Quit", "Ctrl+Q"),
            # Edit
            ("Undo", "Ctrl+Z"),
            ("Redo", "Ctrl+Shift+Z"),
            ("Reset Corrections", "Ctrl+R"),
            ("Preferences", "Ctrl+,"),
            # View
            ("Zoom In", "Ctrl++"),
            ("Zoom Out", "Ctrl+-"),
            ("Fit to Window", "Ctrl+0"),
            ("Actual Size", "Ctrl+1"),
            ("Zoom to Person", "Ctrl+2"),
            ("Toggle Skeleton", "Ctrl+K"),
            ("Toggle Keypoints", "Ctrl+J"),
            ("Toggle Correction Panel", "Ctrl+P"),
            ("Fullscreen", "F11"),
            # Process
            ("Detect Poses", "Ctrl+D"),
            ("Batch Process", "Ctrl+B"),
            ("Next Person", "Tab"),
            ("Previous Person", "Shift+Tab"),
            # Help
            ("Keyboard Shortcuts", "F1"),
        ]

        # Sort alphabetically
        all_shortcuts.sort(key=lambda x: x[0])

        self.tabs.addTab(self._create_shortcuts_widget(all_shortcuts), "All")

    def _filter_content(self, text: str):
        """Filter shortcuts based on search"""
        # This is a simplified version - full implementation would filter visible items
        pass


class PrintableShortcutsDialog(QDialog):
    """
    Printable/exportable version with cleaner layout
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Keyboard Shortcuts Reference")
        self.setModal(False)
        self.setMinimumSize(800, 650)

        self._setup_ui()

    def _setup_ui(self):
        """Setup printable UI"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("PostureKit Keyboard Shortcuts")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Print button
        print_btn = QPushButton("Print")
        print_btn.clicked.connect(self._print_shortcuts)
        header_layout.addWidget(print_btn)

        layout.addLayout(header_layout)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # Add sections
        self._add_section(content_layout, "File Operations", [
            ("Open Image", "Ctrl+O"),
            ("Open Directory", "Ctrl+Shift+O"),
            ("Save Corrections", "Ctrl+S"),
            ("Export Corrections", "Ctrl+E"),
            ("Export Dataset", "Ctrl+Shift+E"),
            ("Quit", "Ctrl+Q"),
        ])

        self._add_section(content_layout, "Edit Operations", [
            ("Undo", "Ctrl+Z"),
            ("Redo", "Ctrl+Shift+Z"),
            ("Reset Corrections", "Ctrl+R"),
            ("Preferences", "Ctrl+,"),
            ("Nudge Keypoint", "Arrow Keys"),
            ("Nudge 10px", "Shift+Arrows"),
            ("Move Keypoint", "Shift+Drag"),
            ("Toggle Visibility", "Right-Click"),
        ])

        self._add_section(content_layout, "View Controls", [
            ("Zoom In", "Ctrl++"),
            ("Zoom Out", "Ctrl+-"),
            ("Fit to Window", "Ctrl+0"),
            ("Actual Size", "Ctrl+1"),
            ("Zoom to Person", "Ctrl+2"),
            ("Toggle Skeleton", "Ctrl+K"),
            ("Toggle Keypoints", "Ctrl+J"),
            ("Toggle Panel", "Ctrl+P"),
        ])

        self._add_section(content_layout, "Processing", [
            ("Detect Poses", "Ctrl+D"),
            ("Batch Process", "Ctrl+B"),
            ("Next Person", "Tab"),
            ("Previous Person", "Shift+Tab"),
        ])

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _add_section(self, layout: QVBoxLayout, title: str, shortcuts: List[Tuple[str, str]]):
        """Add a shortcuts section"""
        # Section title
        title_label = QLabel(title)
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        # Grid for shortcuts
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(0, 1)

        for i, (action, shortcut) in enumerate(shortcuts):
            action_label = QLabel(action)
            shortcut_label = QLabel(shortcut)

            shortcut_font = shortcut_label.font()
            shortcut_font.setBold(True)
            shortcut_font.setFamily("monospace")
            shortcut_label.setFont(shortcut_font)

            grid.addWidget(action_label, i, 0)
            grid.addWidget(shortcut_label, i, 1, Qt.AlignRight)

        layout.addLayout(grid)

    def _print_shortcuts(self):
        """Print shortcuts reference"""
        from PySide6.QtPrintSupport import QPrinter, QPrintDialog

        printer = QPrinter()
        dialog = QPrintDialog(printer, self)

        if dialog.exec() == QDialog.Accepted:
            # TODO: Implement printing
            QMessageBox.information(self, "Print", "Printing functionality would be implemented here")


# Example usage
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Show standard version
    dialog = ShortcutsDialog()
    dialog.show()

    sys.exit(app.exec())
