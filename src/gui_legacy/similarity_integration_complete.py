"""
Complete Similarity Search Integration for PostureKit MainWindow

This module provides a single function that adds all similarity search functionality
to an existing MainWindow instance, including menu items, dialogs, and event handlers.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QAction, QKeySequence

from src.intelligence.similarity_engine import SimilarityEngine
from src.gui_legacy.directory_index_dialog import DirectoryIndexDialog
from src.gui_legacy.similarity_search_dialog import SimilaritySearchDialog

if TYPE_CHECKING:
    from src.gui_legacy.main_window import MainWindow

logger = logging.getLogger(__name__)


def add_similarity_search_to_mainwindow(main_window: 'MainWindow'):
    """
    Add complete similarity search functionality to MainWindow.

    This function:
    - Initializes GeometricFeatureExtractor if needed
    - Uses existing SimilarityEngine (already initialized)
    - Adds Search menu with all actions
    - Connects signals and slots
    - Handles error cases gracefully

    Args:
        main_window: The MainWindow instance to enhance
    """
    # Initialize feature extractor if it doesn't exist
    if not hasattr(main_window, 'feature_extractor'):
        from src.core.geometric_feature_extractor import GeometricFeatureExtractor
        main_window.feature_extractor = GeometricFeatureExtractor()
        logger.info("Geometric feature extractor initialized")

    # Get storage manager from db_manager if needed
    if not hasattr(main_window, 'storage_manager'):
        if hasattr(main_window, 'db_manager') and hasattr(main_window.db_manager, 'storage'):
            main_window.storage_manager = main_window.db_manager.storage
            logger.info("Using storage from db_manager")
        else:
            logger.error("No storage_manager or db_manager.storage found")
            return

    # Check if similarity_engine exists (should already be initialized in MainWindow.__init__)
    if not hasattr(main_window, 'similarity_engine'):
        logger.error("similarity_engine not found - should be initialized in MainWindow.__init__")
        return

    # Add create_similarity_menu method to MainWindow
    def create_similarity_menu(self, menu):
        """Create Search menu with similarity search actions."""
        # Index Directory action
        index_dir_action = QAction("Index Directory...", self)
        index_dir_action.setShortcut(QKeySequence("Ctrl+Shift+I"))
        index_dir_action.setStatusTip("Index all images in a directory for similarity search")
        index_dir_action.triggered.connect(lambda: open_directory_indexing_dialog(self))
        menu.addAction(index_dir_action)

        # Rebuild Index action
        rebuild_action = QAction("Rebuild Search Index", self)
        rebuild_action.setStatusTip("Rebuild FAISS index from database")
        rebuild_action.triggered.connect(lambda: rebuild_search_index(self))
        menu.addAction(rebuild_action)

        menu.addSeparator()

        # Find Similar Poses action
        find_similar_action = QAction("Find Similar Poses...", self)
        find_similar_action.setShortcut(QKeySequence("Ctrl+F"))
        find_similar_action.setStatusTip("Search for poses similar to a reference image")
        find_similar_action.triggered.connect(lambda: open_similarity_search_dialog(self))
        menu.addAction(find_similar_action)

        # Find Similar to Current action
        find_current_action = QAction("Find Similar to Current", self)
        find_current_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        find_current_action.setStatusTip("Search for poses similar to current image")
        find_current_action.triggered.connect(lambda: find_similar_to_current(self))
        menu.addAction(find_current_action)

        menu.addSeparator()

        # Index Statistics action
        stats_action = QAction("Index Statistics", self)
        stats_action.setStatusTip("Show similarity search index statistics")
        stats_action.triggered.connect(lambda: show_index_statistics(self))
        menu.addAction(stats_action)

    # Bind method to instance
    import types
    main_window.create_similarity_menu = types.MethodType(create_similarity_menu, main_window)

    # Add helper method for handling result selection
    def on_similar_pose_selected(self, pose_id: str, image_path: str):
        """Handle selection of similar pose from search results."""
        try:
            logger.info(f"Loading similar pose: {image_path}")
            # Use existing loadImage method
            if hasattr(self, 'loadImage'):
                self.loadImage(image_path)
            else:
                logger.warning("MainWindow does not have loadImage method")
        except Exception as e:
            logger.error(f"Failed to load similar pose: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Load Error",
                f"Failed to load image: {str(e)}"
            )

    main_window.on_similar_pose_selected = types.MethodType(on_similar_pose_selected, main_window)

    logger.info("Similarity search integration complete")


def open_directory_indexing_dialog(main_window: 'MainWindow'):
    """Open directory indexing dialog."""
    if not main_window.similarity_engine:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Similarity engine is not initialized."
        )
        return

    try:
        dialog = DirectoryIndexDialog(
            main_window.pose_detector,
            main_window.feature_extractor,
            main_window.storage_manager,
            main_window.similarity_engine,
            main_window.state_manager,
            main_window.current_db_profile,
            parent=main_window
        )

        # Connect signal to handle index completion
        dialog.index_built.connect(
            lambda count: on_index_built(main_window, count)
        )

        dialog.exec()

    except Exception as e:
        logger.error(f"Failed to open directory indexing dialog: {e}", exc_info=True)
        QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to open indexing dialog: {str(e)}"
        )


def rebuild_search_index(main_window: 'MainWindow'):
    """Rebuild FAISS search index from database."""
    if not main_window.similarity_engine:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Similarity engine is not initialized."
        )
        return

    reply = QMessageBox.question(
        main_window,
        "Rebuild Index",
        "Rebuild the similarity search index from the database?\n\n"
        "This will take a few seconds depending on the number of poses.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    if reply == QMessageBox.No:
        return

    try:
        main_window.similarity_engine.build_index(force_rebuild=True)

        stats = main_window.similarity_engine.get_statistics()

        QMessageBox.information(
            main_window,
            "Index Rebuilt",
            f"Search index rebuilt successfully!\n\n"
            f"Total poses indexed: {stats['total_poses']}\n"
            f"Feature dimension: {stats['feature_dim']}"
        )

    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}", exc_info=True)
        QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to rebuild search index: {str(e)}"
        )


def open_similarity_search_dialog(main_window: 'MainWindow'):
    """Open similarity search dialog."""
    if not main_window.similarity_engine:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Similarity engine is not initialized."
        )
        return

    # Check if index exists
    if main_window.similarity_engine.index is None:
        reply = QMessageBox.question(
            main_window,
            "No Index Found",
            "No similarity search index found.\n\n"
            "Would you like to index a directory now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            open_directory_indexing_dialog(main_window)
        return

    try:
        dialog = SimilaritySearchDialog(
            main_window.pose_detector,
            main_window.feature_extractor,
            main_window.similarity_engine,
            parent=main_window
        )

        # Connect result selection signal
        dialog.pose_selected.connect(
            lambda pose_id, image_path: main_window.on_similar_pose_selected(pose_id, image_path)
        )

        dialog.exec()

    except Exception as e:
        logger.error(f"Failed to open similarity search dialog: {e}", exc_info=True)
        QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to open search dialog: {str(e)}"
        )


def find_similar_to_current(main_window: 'MainWindow'):
    """Search for poses similar to currently loaded image."""
    if not main_window.similarity_engine:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Similarity engine is not initialized."
        )
        return

    # Check if index exists
    if main_window.similarity_engine.index is None:
        QMessageBox.warning(
            main_window,
            "No Index",
            "No similarity search index found.\n\n"
            "Please index a directory first via Search → Index Directory."
        )
        return

    # Get current image path
    current_image = None
    if hasattr(main_window, 'current_image_path') and main_window.current_image_path:
        current_image = main_window.current_image_path
    elif hasattr(main_window, 'image_path') and main_window.image_path:
        current_image = main_window.image_path
    else:
        QMessageBox.warning(
            main_window,
            "No Image",
            "No image is currently loaded.\n\n"
            "Please load an image first."
        )
        return

    try:
        # Open search dialog with pre-loaded reference image
        dialog = SimilaritySearchDialog(
            main_window.pose_detector,
            main_window.feature_extractor,
            main_window.similarity_engine,
            reference_image=current_image,
            parent=main_window
        )

        # Connect result selection signal
        dialog.pose_selected.connect(
            lambda pose_id, image_path: main_window.on_similar_pose_selected(pose_id, image_path)
        )

        dialog.exec()

    except Exception as e:
        logger.error(f"Failed to search for similar poses: {e}", exc_info=True)
        QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to search for similar poses: {str(e)}"
        )


def show_index_statistics(main_window: 'MainWindow'):
    """Display similarity search index statistics."""
    if not main_window.similarity_engine:
        QMessageBox.warning(
            main_window,
            "Not Available",
            "Similarity engine is not initialized."
        )
        return

    try:
        stats = main_window.similarity_engine.get_statistics()

        message = f"""Similarity Search Index Statistics

Total Poses: {stats['total_poses']:,}
Feature Dimension: {stats['dimension']}
Index Type: {stats.get('index_type', 'IndexFlatL2')}
Index Loaded: {'Yes' if stats['index_exists'] else 'No'}

Index File: {stats.get('index_path', 'N/A')}
"""

        if stats['total_poses'] == 0:
            message += "\n⚠️ No poses indexed. Use 'Index Directory' to add poses."

        QMessageBox.information(
            main_window,
            "Index Statistics",
            message
        )

    except Exception as e:
        logger.error(f"Failed to get index statistics: {e}", exc_info=True)
        QMessageBox.critical(
            main_window,
            "Error",
            f"Failed to retrieve statistics: {str(e)}"
        )


def on_index_built(main_window: 'MainWindow', count: int):
    """Handle completion of index building."""
    logger.info(f"Search index built with {count} poses")

    # Optional: Update UI or status bar
    if hasattr(main_window, 'statusBar'):
        main_window.statusBar().showMessage(
            f"Search index ready with {count} poses",
            5000
        )
