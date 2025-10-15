"""
Enhanced Similarity Integration - Adds live panel and toolbar actions.

This module extends the basic similarity integration with a dockable panel
and enhanced toolbar/context menu integration for a more streamlined workflow.
"""

import logging
from PySide6.QtWidgets import (
    QMessageBox, QProgressDialog, QToolBar, QMenu, QDockWidget
)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QAction, QKeySequence

from src.intelligence.similarity_engine import SimilarityEngine
from src.gui_legacy.similarity_search_dialog import SimilaritySearchDialog
from src.gui_legacy.similarity_panel import SimilarityPanel

logger = logging.getLogger(__name__)


class EnhancedSimilarityMixin:
    """
    Enhanced mixin that adds similarity panel and toolbar integration.

    Use this instead of SimilaritySearchMixin for full integration:
        class MainWindow(QMainWindow, EnhancedSimilarityMixin):
    """

    def init_enhanced_similarity(self):
        """
        Initialize enhanced similarity components.
        Call this in MainWindow.__init__() after initializing storage and detector.
        """
        # Initialize similarity engine
        self.similarity_engine = SimilarityEngine(self.storage_manager)

        # Try to load existing index
        if not self.similarity_engine.load_index():
            logger.info("No FAISS index found - will need to build on first use")
        else:
            stats = self.similarity_engine.get_statistics()
            logger.info(f"Loaded FAISS index with {stats['total_poses']} poses")

        # Create similarity panel
        self.similarity_panel = SimilarityPanel(
            self.similarity_engine,
            self.storage_manager,
            parent=self
        )

        # Connect panel signals
        self.similarity_panel.pose_selected.connect(self.on_similar_pose_selected)
        self.similarity_panel.search_requested.connect(self.open_similarity_search)

        # Create dock widget for panel
        self.similarity_dock = QDockWidget("Similar Poses", self)
        self.similarity_dock.setWidget(self.similarity_panel)
        self.similarity_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Add dock to main window (right side by default)
        self.addDockWidget(Qt.RightDockWidgetArea, self.similarity_dock)

        # Initially hidden - user can show via menu
        self.similarity_dock.setVisible(False)

    def create_enhanced_similarity_menu(self, menu_bar):
        """
        Create enhanced similarity menu items with panel toggle.

        Args:
            menu_bar: QMenu or QMenuBar to add actions to
        """
        # Main search dialog
        search_action = menu_bar.addAction("Find Similar Poses...")
        search_action.setShortcut(QKeySequence("Ctrl+F"))
        search_action.setStatusTip("Open similarity search dialog")
        search_action.triggered.connect(self.open_similarity_search)

        # Toggle similarity panel
        toggle_panel_action = menu_bar.addAction("Show Similar Poses Panel")
        toggle_panel_action.setCheckable(True)
        toggle_panel_action.setChecked(False)
        toggle_panel_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        toggle_panel_action.setStatusTip("Toggle live similarity panel")
        toggle_panel_action.toggled.connect(self.similarity_dock.setVisible)

        # Keep action in sync with dock visibility
        self.similarity_dock.visibilityChanged.connect(toggle_panel_action.setChecked)

        menu_bar.addSeparator()

        # Find similar to current
        find_current_action = menu_bar.addAction("Find Similar to Current Pose")
        find_current_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        find_current_action.setStatusTip("Search for poses similar to currently displayed pose")
        find_current_action.triggered.connect(self.find_similar_to_current)

        menu_bar.addSeparator()

        # Rebuild index
        rebuild_action = menu_bar.addAction("Rebuild Search Index")
        rebuild_action.setStatusTip("Rebuild FAISS similarity index from database")
        rebuild_action.triggered.connect(self.rebuild_similarity_index)

        return search_action, toggle_panel_action, find_current_action, rebuild_action

    def create_similarity_toolbar(self) -> QToolBar:
        """
        Create a toolbar with similarity search actions.

        Returns:
            QToolBar with similarity actions
        """
        toolbar = QToolBar("Similarity Tools")
        toolbar.setObjectName("SimilarityToolbar")

        # Find similar dialog
        search_action = toolbar.addAction("🔍 Find Similar")
        search_action.setToolTip("Open similarity search dialog (Ctrl+F)")
        search_action.triggered.connect(self.open_similarity_search)

        # Toggle panel
        panel_action = toolbar.addAction("📋 Similar Panel")
        panel_action.setToolTip("Toggle similar poses panel (Ctrl+Shift+F)")
        panel_action.setCheckable(True)
        panel_action.toggled.connect(self.similarity_dock.setVisible)
        self.similarity_dock.visibilityChanged.connect(panel_action.setChecked)

        toolbar.addSeparator()

        # Find similar to current
        current_action = toolbar.addAction("⚡ Similar to Current")
        current_action.setToolTip("Find poses similar to current (Ctrl+Shift+S)")
        current_action.triggered.connect(self.find_similar_to_current)

        return toolbar

    def add_similarity_context_menu(self, context_menu: QMenu, pose_id: str):
        """
        Add similarity actions to image viewer context menu.

        Args:
            context_menu: QMenu to add actions to
            pose_id: UUID of pose being right-clicked
        """
        context_menu.addSeparator()

        # Find similar
        similar_action = context_menu.addAction("Find Similar Poses")
        similar_action.triggered.connect(lambda: self.find_similar_to_pose(pose_id))

        # Show in panel
        show_panel_action = context_menu.addAction("Show Similar in Panel")
        show_panel_action.triggered.connect(lambda: self.show_similar_in_panel(pose_id))

        return similar_action, show_panel_action

    @Slot()
    def open_similarity_search(self):
        """Open full similarity search dialog."""
        try:
            dialog = SimilaritySearchDialog(
                similarity_engine=self.similarity_engine,
                pose_detector=self.pose_detector,
                storage_manager=self.storage_manager,
                parent=self
            )
            dialog.pose_selected.connect(self.on_similar_pose_selected)
            dialog.exec()

        except Exception as e:
            logger.error(f"Failed to open similarity search: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open similarity search:\n{str(e)}"
            )

    @Slot()
    def rebuild_similarity_index(self):
        """Rebuild FAISS index from database."""
        reply = QMessageBox.question(
            self,
            "Rebuild Index",
            "Rebuild the similarity search index from all poses in the database?\n\n"
            "This may take a few minutes for large databases.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                progress = QProgressDialog(
                    "Building similarity index...",
                    "Cancel",
                    0, 3,
                    self
                )
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()

                def update_progress(current, total, message):
                    progress.setValue(current)
                    progress.setLabelText(message)
                    progress.setMaximum(total)

                self.similarity_engine.build_index(
                    force_rebuild=True,
                    progress_callback=update_progress
                )

                progress.close()

                stats = self.similarity_engine.get_statistics()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Index rebuilt successfully!\n\n"
                    f"Total poses indexed: {stats['total_poses']}"
                )

                # Refresh panel if visible
                if self.similarity_dock.isVisible():
                    self.similarity_panel.refresh_results()

            except Exception as e:
                logger.error(f"Failed to rebuild index: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to rebuild index:\n{str(e)}"
                )

    @Slot(str, str)
    def on_similar_pose_selected(self, pose_id: str, image_path: str):
        """
        Handle selection of a similar pose from panel or dialog.

        Args:
            pose_id: UUID of selected pose
            image_path: Path to image containing the pose
        """
        logger.info(f"Loading similar pose: {pose_id}")

        try:
            from uuid import UUID
            pose_uuid = UUID(pose_id)

            # Get pose from database
            with self.storage_manager.session_scope() as session:
                from src.storage.models import PoseDetection
                pose = session.query(PoseDetection).filter(
                    PoseDetection.id == pose_uuid
                ).first()

                if not pose:
                    QMessageBox.warning(self, "Error", "Pose not found in database")
                    return

            # Load the image
            self.loadImage(image_path)

            # Update panel with new current pose
            if self.similarity_dock.isVisible():
                self.similarity_panel.set_current_pose(pose_uuid)

            self.statusBar().showMessage(f"Loaded similar pose from {image_path}", 3000)

        except Exception as e:
            logger.error(f"Failed to load similar pose: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load pose:\n{str(e)}"
            )

    @Slot()
    def find_similar_to_current(self):
        """Find poses similar to currently displayed pose."""
        # Get current pose ID
        current_pose_id = self.get_current_pose_id()

        if not current_pose_id:
            QMessageBox.information(
                self,
                "No Pose",
                "No pose is currently loaded.\n\n"
                "Load an image first to find similar poses."
            )
            return

        self.find_similar_to_pose(str(current_pose_id))

    @Slot(str)
    def find_similar_to_pose(self, pose_id: str):
        """
        Open search dialog pre-populated with results for specified pose.

        Args:
            pose_id: UUID of reference pose
        """
        try:
            from uuid import UUID

            # Check if index exists
            stats = self.similarity_engine.get_statistics()
            if not stats['index_exists'] or stats['total_poses'] == 0:
                reply = QMessageBox.question(
                    self,
                    "No Index",
                    "Search index not found. Build index now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    self.rebuild_similarity_index()
                return

            # Open dialog
            dialog = SimilaritySearchDialog(
                similarity_engine=self.similarity_engine,
                pose_detector=self.pose_detector,
                storage_manager=self.storage_manager,
                parent=self
            )
            dialog.pose_selected.connect(self.on_similar_pose_selected)
            dialog.show()

            # Perform search immediately
            results = self.similarity_engine.search_by_pose_id(
                UUID(pose_id),
                k=20,
                min_confidence=0.3
            )

            dialog.current_results = results
            dialog.display_results(results)
            dialog.results_info_label.setText(f"Found {len(results)} similar poses")

        except Exception as e:
            logger.error(f"Failed to search similar poses: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to search:\n{str(e)}"
            )

    @Slot(str)
    def show_similar_in_panel(self, pose_id: str):
        """
        Show similar poses in the side panel.

        Args:
            pose_id: UUID of reference pose
        """
        from uuid import UUID

        # Show panel if hidden
        if not self.similarity_dock.isVisible():
            self.similarity_dock.setVisible(True)

        # Update panel with this pose
        self.similarity_panel.set_current_pose(UUID(pose_id), debounce_ms=0)

    def on_image_changed(self, image_path: str, pose_id: str):
        """
        Call this when the current image/pose changes.
        Updates the similarity panel automatically if visible.

        Args:
            image_path: Path to new image
            pose_id: UUID of current pose
        """
        from uuid import UUID

        # Update panel if visible
        if self.similarity_dock.isVisible():
            self.similarity_panel.set_current_pose(UUID(pose_id))

    def get_current_pose_id(self):
        """
        Get the currently displayed pose ID.
        Override this method in your MainWindow to return the actual current pose.

        Returns:
            UUID of current pose, or None if no pose loaded
        """
        # This is a placeholder - override in your MainWindow
        # Example implementation:
        # if hasattr(self, 'current_pose') and self.current_pose:
        #     return self.current_pose.id
        # return None
        raise NotImplementedError(
            "Override get_current_pose_id() in your MainWindow to return current pose UUID"
        )

    def save_similarity_state(self) -> dict:
        """Save similarity panel state for persistence."""
        return {
            'panel_visible': self.similarity_dock.isVisible(),
            'panel_settings': self.similarity_panel.get_state(),
            'dock_area': self.dockWidgetArea(self.similarity_dock)
        }

    def restore_similarity_state(self, state: dict):
        """Restore similarity panel state."""
        if 'panel_visible' in state:
            self.similarity_dock.setVisible(state['panel_visible'])
        if 'panel_settings' in state:
            self.similarity_panel.set_state(state['panel_settings'])
        if 'dock_area' in state:
            # Re-dock to saved area
            self.addDockWidget(state['dock_area'], self.similarity_dock)


# ============================================================================
# SIMPLE FUNCTION INTEGRATION (Alternative to Mixin)
# ============================================================================

def add_enhanced_similarity_to_mainwindow(main_window):
    """
    Add enhanced similarity features to MainWindow with single function call.

    Usage in MainWindow.__init__():
        from src.gui_legacy.similarity_panel_integration import add_enhanced_similarity_to_mainwindow
        add_enhanced_similarity_to_mainwindow(self)

    Args:
        main_window: MainWindow instance
    """
    # Get storage_manager (handle both direct and db_manager.storage patterns)
    storage_manager = getattr(main_window, 'storage_manager', None)
    if storage_manager is None and hasattr(main_window, 'db_manager'):
        storage_manager = main_window.db_manager.storage

    if storage_manager is None:
        raise AttributeError("MainWindow must have either 'storage_manager' or 'db_manager.storage'")

    # Initialize engine (skip if already exists)
    if not hasattr(main_window, 'similarity_engine'):
        main_window.similarity_engine = SimilarityEngine(storage_manager)
        main_window.similarity_engine.load_index()

    # Create similarity panel
    main_window.similarity_panel = SimilarityPanel(
        main_window.similarity_engine,
        storage_manager,
        parent=main_window
    )

    # Create dock widget
    main_window.similarity_dock = QDockWidget("Similar Poses", main_window)
    main_window.similarity_dock.setWidget(main_window.similarity_panel)
    main_window.similarity_dock.setAllowedAreas(
        Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
    )
    main_window.addDockWidget(Qt.RightDockWidgetArea, main_window.similarity_dock)
    main_window.similarity_dock.setVisible(False)

    # Connect panel signals
    main_window.similarity_panel.pose_selected.connect(
        lambda pose_id, img_path: main_window.loadImage(img_path)
    )

    # Add search dialog opener
    def open_search():
        dialog = SimilaritySearchDialog(
            main_window.similarity_engine,
            main_window.pose_detector,
            storage_manager,
            main_window
        )
        dialog.pose_selected.connect(
            lambda pose_id, img_path: main_window.loadImage(img_path)
        )
        dialog.exec()

    main_window.similarity_panel.search_requested.connect(open_search)

    # Add to menu
    if hasattr(main_window, 'menuBar'):
        tools_menu = None
        for action in main_window.menuBar().actions():
            if action.text() == "Tools":
                tools_menu = action.menu()
                break

        if tools_menu is None:
            tools_menu = main_window.menuBar().addMenu("Tools")

        # Add actions
        search_action = tools_menu.addAction("Find Similar Poses...")
        search_action.setShortcut(QKeySequence("Ctrl+F"))
        search_action.triggered.connect(open_search)

        toggle_action = tools_menu.addAction("Show Similar Poses Panel")
        toggle_action.setCheckable(True)
        toggle_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        toggle_action.toggled.connect(main_window.similarity_dock.setVisible)
        main_window.similarity_dock.visibilityChanged.connect(toggle_action.setChecked)

        tools_menu.addSeparator()

        # Add rebuild index action
        def rebuild_index():
            from PySide6.QtWidgets import QProgressDialog
            reply = QMessageBox.question(
                main_window,
                "Rebuild Index",
                "Rebuild the similarity search index from all poses in the database?\n\n"
                "This may take a few minutes for large databases.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    progress = QProgressDialog(
                        "Building similarity index...", "Cancel", 0, 3, main_window
                    )
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.show()

                    def update_progress(current, total, message):
                        progress.setValue(current)
                        progress.setLabelText(message)
                        progress.setMaximum(total)

                    main_window.similarity_engine.build_index(
                        force_rebuild=True, progress_callback=update_progress
                    )
                    progress.close()

                    stats = main_window.similarity_engine.get_statistics()
                    QMessageBox.information(
                        main_window,
                        "Success",
                        f"Index rebuilt successfully!\n\n"
                        f"Total poses indexed: {stats['total_poses']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to rebuild index: {e}", exc_info=True)
                    QMessageBox.critical(
                        main_window, "Error", f"Failed to rebuild index:\n{str(e)}"
                    )

        rebuild_action = tools_menu.addAction("Rebuild Search Index")
        rebuild_action.triggered.connect(rebuild_index)

        # Add directory indexing action
        def open_directory_indexer():
            from src.gui_legacy.directory_index_dialog import DirectoryIndexDialog
            from src.core.geometric_feature_extractor import GeometricFeatureExtractor

            feature_extractor = GeometricFeatureExtractor()

            dialog = DirectoryIndexDialog(
                main_window.pose_detector,
                feature_extractor,
                storage_manager,
                main_window.similarity_engine,
                main_window.state_manager,
                main_window.current_db_profile,
                parent=main_window
            )
            dialog.index_built.connect(lambda count: main_window.statusBar().showMessage(
                f"Index built with {count} poses", 5000
            ))
            dialog.exec()

        index_dir_action = tools_menu.addAction("Index Directory...")
        index_dir_action.setStatusTip("Detect poses in a directory and build search index")
        index_dir_action.triggered.connect(open_directory_indexer)

    logger.info("Enhanced similarity search integrated successfully")
