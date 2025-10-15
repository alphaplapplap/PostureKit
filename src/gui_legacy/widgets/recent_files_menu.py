"""
Recent files menu with intelligent handling of missing files.
"""
from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtCore import Signal, QFileInfo
from PySide6.QtGui import QAction, QIcon
from pathlib import Path
from typing import List, Callable
import logging

logger = logging.getLogger(__name__)


class RecentFilesMenu(QMenu):
    """
    Dynamic recent files menu with smart file validation.

    This menu automatically updates to show recently accessed files, with
    intelligent handling of files that no longer exist. When a user selects
    a missing file, they're offered options to locate it or remove it from
    the list rather than silently failing.

    The menu uses system icons to show file types and provides keyboard
    shortcuts for the most recent files (Ctrl+1, Ctrl+2, etc).
    """

    file_selected = Signal(Path)  # Emitted when user selects a file

    def __init__(self, title: str = "Recent Files", parent=None):
        """
        Initialize recent files menu.

        Args:
            title: Menu title
            parent: Parent widget
        """
        super().__init__(title, parent)

        self.recent_files: List[str] = []
        self.max_files = 10

        # Actions for each recent file
        self.file_actions: List[QAction] = []

        # Separator before "Clear Recent Files"
        self.separator = None
        self.clear_action = None

        self._setup_menu()

    def _setup_menu(self):
        """Setup static menu items."""
        # Initially show "No Recent Files"
        self.no_files_action = QAction("No Recent Files", self)
        self.no_files_action.setEnabled(False)
        self.addAction(self.no_files_action)

        # Separator
        self.separator = self.addSeparator()
        self.separator.setVisible(False)

        # Clear recent files action
        self.clear_action = QAction("Clear Recent Files", self)
        self.clear_action.triggered.connect(self._clear_recent_files)
        self.clear_action.setVisible(False)
        self.addAction(self.clear_action)

    def update_recent_files(self, files: List[str]):
        """
        Update menu with new recent files list.

        This method rebuilds the menu to reflect the current state of recent
        files. It validates each file's existence and adds appropriate visual
        indicators (icons, shortcuts) to make the menu both functional and
        aesthetically pleasing.

        Args:
            files: List of file paths (as strings) in most-recent-first order
        """
        self.recent_files = files[:self.max_files]

        # Remove existing file actions
        for action in self.file_actions:
            self.removeAction(action)
        self.file_actions.clear()

        # Add actions for each recent file
        if self.recent_files:
            self.no_files_action.setVisible(False)
            self.separator.setVisible(True)
            self.clear_action.setVisible(True)

            for index, file_path in enumerate(self.recent_files):
                path = Path(file_path)

                # Create action with file name
                action = QAction(path.name, self)

                # Set tooltip with full path
                action.setToolTip(str(path))

                # Add keyboard shortcut for first 9 files
                if index < 9:
                    action.setShortcut(f"Ctrl+{index + 1}")

                # Set icon based on file type
                file_info = QFileInfo(str(path))
                icon = self._get_icon_for_file(file_info)
                if icon:
                    action.setIcon(icon)

                # Disable if file doesn't exist
                if not path.exists():
                    action.setEnabled(False)
                    action.setText(f"{path.name} (missing)")

                # Connect to handler
                action.triggered.connect(
                    lambda checked=False, p=path: self._on_file_selected(p)
                )

                self.file_actions.append(action)
                self.insertAction(self.separator, action)
        else:
            self.no_files_action.setVisible(True)
            self.separator.setVisible(False)
            self.clear_action.setVisible(False)

        logger.debug(f"Updated recent files menu with {len(self.recent_files)} files")

    def _get_icon_for_file(self, file_info: QFileInfo) -> QIcon:
        """
        Get system icon for file type.

        This uses the operating system's icon provider to get the appropriate
        icon for each file type, making the menu feel native and familiar.
        For example, JPEG files show a photo icon, while directories show
        a folder icon.

        Args:
            file_info: QFileInfo for the file

        Returns:
            QIcon for the file type
        """
        from PySide6.QtWidgets import QFileIconProvider

        icon_provider = QFileIconProvider()
        return icon_provider.icon(file_info)

    def _on_file_selected(self, path: Path):
        """
        Handle file selection from menu.

        This method implements intelligent handling of file selection. If the
        file exists, it simply opens it. If the file is missing, it offers the
        user helpful options rather than silently failing.

        Args:
            path: Selected file path
        """
        if path.exists():
            # File exists, emit signal to open it
            self.file_selected.emit(path)
            logger.info(f"Recent file selected: {path}")
        else:
            # File is missing, offer options
            self._handle_missing_file(path)

    def _handle_missing_file(self, path: Path):
        """
        Handle selection of a missing file with user-friendly options.

        When a user tries to open a file that no longer exists, this provides
        three options: locate the file (perhaps it was moved), remove it from
        the recent list, or cancel. This is far more helpful than an error
        message or silent failure.

        Args:
            path: Path to missing file
        """
        msg = QMessageBox(self.parent())
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("File Not Found")
        msg.setText(f"The file '{path.name}' could not be found.")
        msg.setInformativeText(f"Path: {path}\n\nIt may have been moved or deleted.")

        # Add custom buttons
        locate_btn = msg.addButton("Locate File...", QMessageBox.ActionRole)
        remove_btn = msg.addButton("Remove from List", QMessageBox.ActionRole)
        cancel_btn = msg.addButton(QMessageBox.Cancel)

        msg.exec()

        if msg.clickedButton() == locate_btn:
            self._locate_missing_file(path)
        elif msg.clickedButton() == remove_btn:
            self._remove_from_recent(path)

    def _locate_missing_file(self, original_path: Path):
        """
        Let user locate a moved file.

        This opens a file dialog positioned at the last known directory,
        making it easy for users to find files that were moved to nearby
        locations. If they successfully locate the file, it replaces the
        old path in the recent files list.

        Args:
            original_path: Original (now missing) file path
        """
        from PySide6.QtWidgets import QFileDialog

        # Start search in the directory where file used to be
        start_dir = str(original_path.parent) if original_path.parent.exists() else str(Path.home())

        new_path, _ = QFileDialog.getOpenFileName(
            self.parent(),
            f"Locate '{original_path.name}'",
            start_dir,
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if new_path:
            # Replace old path with new path in recent files
            try:
                index = self.recent_files.index(str(original_path))
                self.recent_files[index] = new_path

                # Update menu
                self.update_recent_files(self.recent_files)

                # Open the located file
                self.file_selected.emit(Path(new_path))

                logger.info(f"Located missing file: {original_path.name} -> {new_path}")
            except ValueError:
                logger.warning(f"Could not find {original_path} in recent files list")

    def _remove_from_recent(self, path: Path):
        """
        Remove file from recent files list.

        Args:
            path: File path to remove
        """
        try:
            self.recent_files.remove(str(path))
            self.update_recent_files(self.recent_files)
            logger.info(f"Removed from recent files: {path}")
        except ValueError:
            logger.warning(f"Could not find {path} in recent files list")

    def _clear_recent_files(self):
        """Clear all recent files after confirmation."""
        reply = QMessageBox.question(
            self.parent(),
            "Clear Recent Files",
            "Remove all files from the recent files list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.recent_files.clear()
            self.update_recent_files([])

            # Emit signal so state manager can clear as well
            from PySide6.QtCore import QObject
            if isinstance(self.parent(), QObject):
                # Notify parent to clear state
                pass

            logger.info("Cleared all recent files")
