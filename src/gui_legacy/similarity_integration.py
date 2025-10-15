"""
Similarity Search Integration - Add to MainWindow.

This module provides integration code to add similarity search functionality
to your existing PostureKit MainWindow.

Integration Instructions:
1. Add this code to your MainWindow.__init__() method
2. Add the menu item creation to your menu setup
3. Connect the similarity search dialog signals
"""

from PySide6.QtWidgets import QMessageBox, QProgressDialog
from PySide6.QtCore import Slot, Qt

from src.intelligence.similarity_engine import SimilarityEngine
from src.gui_legacy.similarity_search_dialog import SimilaritySearchDialog
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SimilaritySearchMixin:
    """
    Mixin class to add similarity search functionality to MainWindow.

    Add this to your MainWindow class inheritance:
        class MainWindow(QMainWindow, SimilaritySearchMixin):
    """

    def init_similarity_search(self):
        """
        Initialize similarity search components.
        Call this in MainWindow.__init__() after initializing storage and detector.
        """
        # Initialize similarity engine
        self.similarity_engine = SimilarityEngine(self.storage_manager)

        # Try to load existing index
        if not self.similarity_engine.load_index():
            # No index exists - user will need to build it
            logger.info("No FAISS index found - will need to build on first use")
        else:
            stats = self.similarity_engine.get_statistics()
            logger.info(f"Loaded FAISS index with {stats['total_poses']} poses")

    def create_similarity_menu(self, menu_bar):
        """
        Create similarity search menu items.
        Call this in your menu setup code.

        Example:
            tools_menu = menu_bar.addMenu("Tools")
            self.create_similarity_menu(tools_menu)
        """
        similarity_action = menu_bar.addAction("Find Similar Poses...")
        similarity_action.setShortcut("Ctrl+F")
        similarity_action.setStatusTip("Search for poses similar to current or reference image")
        similarity_action.triggered.connect(self.open_similarity_search)

        rebuild_index_action = menu_bar.addAction("Rebuild Search Index")
        rebuild_index_action.setStatusTip("Rebuild FAISS similarity index from database")
        rebuild_index_action.triggered.connect(self.rebuild_similarity_index)

        return similarity_action, rebuild_index_action

    @Slot()
    def open_similarity_search(self):
        """Open similarity search dialog."""
        try:
            # Create dialog
            dialog = SimilaritySearchDialog(
                similarity_engine=self.similarity_engine,
                pose_detector=self.pose_detector,
                storage_manager=self.storage_manager,
                parent=self
            )

            # Connect signals
            dialog.pose_selected.connect(self.on_similar_pose_selected)

            # Show dialog
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
                # Show progress dialog
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

                # Build index
                self.similarity_engine.build_index(
                    force_rebuild=True,
                    progress_callback=update_progress
                )

                progress.close()

                # Show success
                stats = self.similarity_engine.get_statistics()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Index rebuilt successfully!\n\n"
                    f"Total poses indexed: {stats['total_poses']}"
                )

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
        Handle selection of a similar pose from search results.

        Args:
            pose_id: UUID of selected pose
            image_path: Path to image containing the pose
        """
        logger.info(f"Loading similar pose: {pose_id}")

        try:
            # Load the image
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

            # Load the image in main window
            # Adjust this based on your MainWindow's image loading method
            self.loadImage(image_path)

            # Optionally: highlight the specific person if multi-person image
            # self.select_person(pose.person_index)

            self.statusBar().showMessage(f"Loaded similar pose from {image_path}", 3000)

        except Exception as e:
            logger.error(f"Failed to load similar pose: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load pose:\n{str(e)}"
            )

    def add_similarity_to_context_menu(self, context_menu, pose_id):
        """
        Add "Find Similar" action to pose context menu.

        Args:
            context_menu: QMenu to add action to
            pose_id: UUID of current pose
        """
        find_similar_action = context_menu.addAction("Find Similar Poses")
        find_similar_action.triggered.connect(
            lambda: self.find_similar_to_pose(pose_id)
        )
        return find_similar_action

    @Slot(str)
    def find_similar_to_pose(self, pose_id: str):
        """
        Search for poses similar to specified pose ID.

        Args:
            pose_id: UUID of reference pose
        """
        try:
            from uuid import UUID

            # Open search dialog with this pose as reference
            dialog = SimilaritySearchDialog(
                similarity_engine=self.similarity_engine,
                pose_detector=self.pose_detector,
                storage_manager=self.storage_manager,
                parent=self
            )

            # Connect signals
            dialog.pose_selected.connect(self.on_similar_pose_selected)

            # Show dialog
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


# ============================================================================
# INTEGRATION EXAMPLE - Add to your MainWindow
# ============================================================================

"""
# In your main_window.py:

from src.gui_legacy.similarity_integration import SimilaritySearchMixin

class MainWindow(QMainWindow, SimilaritySearchMixin):
    def __init__(self):
        super().__init__()

        # ... your existing initialization ...

        # Initialize similarity search
        self.init_similarity_search()

    def create_menus(self):
        menu_bar = self.menuBar()

        # ... your existing menus ...

        # Add Tools menu with similarity search
        tools_menu = menu_bar.addMenu("Tools")
        self.create_similarity_menu(tools_menu)

    def loadImage(self, image_path):
        '''Your existing image loading method'''
        # ... existing code ...
        pass
"""


# ============================================================================
# ALTERNATIVE: Direct Function Integration (No Mixin)
# ============================================================================

def add_similarity_search_to_mainwindow(main_window):
    """
    Alternative integration method - call this function with your MainWindow instance.

    Usage in MainWindow.__init__():
        from src.gui_legacy.similarity_integration import add_similarity_search_to_mainwindow
        add_similarity_search_to_mainwindow(self)
    """
    # Initialize engine
    main_window.similarity_engine = SimilarityEngine(main_window.storage_manager)
    main_window.similarity_engine.load_index()

    # Add open_similarity_search method
    def open_similarity_search():
        dialog = SimilaritySearchDialog(
            similarity_engine=main_window.similarity_engine,
            pose_detector=main_window.pose_detector,
            storage_manager=main_window.storage_manager,
            parent=main_window
        )
        dialog.pose_selected.connect(
            lambda pose_id, img_path: main_window.loadImage(img_path)
        )
        dialog.exec()

    main_window.open_similarity_search = open_similarity_search

    # Add menu action
    if hasattr(main_window, 'menuBar'):
        tools_menu = None
        for action in main_window.menuBar().actions():
            if action.text() == "Tools":
                tools_menu = action.menu()
                break

        if tools_menu is None:
            tools_menu = main_window.menuBar().addMenu("Tools")

        similarity_action = tools_menu.addAction("Find Similar Poses...")
        similarity_action.setShortcut("Ctrl+F")
        similarity_action.triggered.connect(open_similarity_search)
