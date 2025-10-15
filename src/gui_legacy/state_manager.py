"""Settings persistence and state management for PostureKit."""

from PySide6.QtCore import QSettings, QByteArray, QObject, Signal, QMutex, QMutexLocker
from PySide6.QtWidgets import QWidget, QSplitter
from typing import Any, Optional, Dict, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from src.utils.thread_safe_cache import ThreadSafeOrderedDict
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SettingsVersion(Enum):
    """Settings schema versions for migration."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    CURRENT = V1_1  # Update this when schema changes


@dataclass
class SettingDefinition(Generic[T]):
    """Definition of a setting with validation."""
    key: str
    default_value: T
    description: str
    validator: Optional[Callable[[Any], bool]] = None
    migrator: Optional[Callable[[Any, str], T]] = None  # (old_value, from_version) -> new_value
    persistent: bool = True  # False for session-only settings


class SettingsGroup:
    """Organized group of related settings."""

    # Window/UI settings
    WINDOW_GEOMETRY = SettingDefinition("window/geometry", QByteArray(), "Main window geometry")
    WINDOW_STATE = SettingDefinition("window/state", QByteArray(), "Main window state")
    SPLITTER_MAIN = SettingDefinition("window/splitter_main", [650, 350], "Main splitter sizes")
    SPLITTER_RIGHT = SettingDefinition("window/splitter_right", [80, 200, 100, 100, 100], "Right panel splitter sizes")
    PANEL_VISIBLE = SettingDefinition("window/panel_visible", True, "Right panel visibility")

    # Display settings
    SHOW_SKELETON = SettingDefinition("display/show_skeleton", True, "Show skeleton lines")
    SHOW_KEYPOINTS = SettingDefinition("display/show_keypoints", True, "Show keypoint dots")
    SHOW_CONFIDENCE = SettingDefinition("display/show_confidence", True, "Show confidence colors")
    USE_COLORED_SKELETON = SettingDefinition("display/use_colored_skeleton", True, "Use anatomical colors")
    KEYPOINT_RADIUS = SettingDefinition(
        "display/keypoint_radius",
        4.0,
        "Keypoint dot radius",
        validator=lambda v: isinstance(v, (int, float)) and 2.0 <= v <= 10.0
    )
    LINE_WIDTH = SettingDefinition(
        "display/line_width",
        2.5,
        "Skeleton line width",
        validator=lambda v: isinstance(v, (int, float)) and 1.0 <= v <= 5.0
    )
    FACE_DISPLAY_MODE = SettingDefinition("display/face_display_mode", "dots", "Face landmarks display mode")

    # Detection settings
    PERSON_CONFIDENCE = SettingDefinition(
        "detection/person_confidence",
        0.5,
        "Person detection confidence threshold",
        validator=lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0
    )
    POSE_CONFIDENCE = SettingDefinition(
        "detection/pose_confidence",
        0.3,
        "Pose detection confidence threshold",
        validator=lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0
    )
    MAX_PERSONS = SettingDefinition(
        "detection/max_persons",
        10,
        "Maximum persons per image",
        validator=lambda v: isinstance(v, int) and 1 <= v <= 50
    )
    DEVICE = SettingDefinition("detection/device", "auto", "Computation device")

    # General settings
    THEME = SettingDefinition("general/theme", "light", "UI theme")
    SHOW_SPLASH = SettingDefinition("general/show_splash", True, "Show splash screen")
    CRASH_RECOVERY_ENABLED = SettingDefinition("general/crash_recovery", True, "Enable crash recovery")
    CRASH_RECOVERY_INTERVAL = SettingDefinition(
        "general/crash_recovery_interval",
        2,
        "Crash recovery interval (minutes)",
        validator=lambda v: isinstance(v, int) and 1 <= v <= 30
    )
    CHECK_UPDATES = SettingDefinition("general/check_updates", False, "Check for updates on startup")
    LANGUAGE = SettingDefinition("general/language", "en", "Interface language")
    DATABASE_PROFILE = SettingDefinition(
        "general/database_profile",
        "irl",
        "Active database profile (irl/2d/3d)",
        validator=lambda v: v in ["irl", "2d", "3d"]
    )

    # File handling
    RECENT_FILES_COUNT = SettingDefinition(
        "files/recent_count",
        10,
        "Number of recent files",
        validator=lambda v: isinstance(v, int) and 1 <= v <= 20
    )
    DEFAULT_IMAGE_DIR = SettingDefinition("files/default_image_dir", "", "Default image directory")
    DEFAULT_EXPORT_DIR = SettingDefinition("files/default_export_dir", "", "Default export directory")

    # Indexing - Last directory per database profile
    LAST_INDEXED_DIR_IRL = SettingDefinition("indexing/last_dir_irl", "", "Last indexed directory for IRL profile")
    LAST_INDEXED_DIR_2D = SettingDefinition("indexing/last_dir_2d", "", "Last indexed directory for 2D profile")
    LAST_INDEXED_DIR_3D = SettingDefinition("indexing/last_dir_3d", "", "Last indexed directory for 3D profile")

    # Performance settings
    BATCH_THREADS = SettingDefinition(
        "performance/batch_threads",
        4,
        "Batch processing threads",
        validator=lambda v: isinstance(v, int) and 1 <= v <= 16
    )
    IMAGE_CACHE_SIZE = SettingDefinition(
        "performance/image_cache_mb",
        512,
        "Image cache size (MB)",
        validator=lambda v: isinstance(v, int) and 100 <= v <= 2000
    )
    MAX_POSES_MEMORY = SettingDefinition(
        "performance/max_poses_memory",
        50,
        "Max poses in memory",
        validator=lambda v: isinstance(v, int) and 10 <= v <= 500
    )
    USE_GPU = SettingDefinition("performance/use_gpu", True, "Use GPU acceleration")
    TARGET_FPS = SettingDefinition(
        "performance/target_fps",
        60,
        "Target FPS",
        validator=lambda v: v in [30, 60, 120]
    )

    # Session-only settings (not persisted across app restarts)
    LAST_IMAGE_PATH = SettingDefinition("session/last_image", "", "Last opened image", persistent=False)
    LAST_PERSON_INDEX = SettingDefinition("session/last_person_index", 0, "Last selected person", persistent=False)
    ZOOM_LEVEL = SettingDefinition("session/zoom_level", 1.0, "Current zoom level", persistent=False)
    PAN_POSITION = SettingDefinition("session/pan_position", [0, 0], "Current pan position", persistent=False)

    # Recent files list (special handling)
    RECENT_FILES = SettingDefinition("files/recent_files", [], "Recent file paths")

    # Database
    DB_CONNECTION = SettingDefinition("database/connection", "postgresql://localhost/posturekit", "Database connection string")


class StateManager(QObject):
    """
    Central settings and state management system.

    Handles settings persistence, validation, migration, and recovery.
    """

    setting_changed = Signal(str, object)  # key, new_value

    _instance: Optional['StateManager'] = None

    def __init__(self, organization: str = "PostureKit", application: str = "PostureKit"):
        """
        Initialize state manager (singleton).

        Args:
            organization: Organization name for QSettings
            application: Application name for QSettings
        """
        if StateManager._instance is not None:
            raise RuntimeError("StateManager is a singleton. Use StateManager.instance()")

        super().__init__()

        self.settings = QSettings(organization, application)
        self.mutex = QMutex()  # Thread safety

        # Cache for frequently accessed settings (thread-safe LRU with memory limit)
        self.MAX_CACHE_SIZE = 100  # Reasonable limit for settings cache
        self._cache = ThreadSafeOrderedDict[str, Any](
            max_size=self.MAX_CACHE_SIZE,
            max_memory_mb=10  # 10MB cache limit
        )

        # Track which settings have been modified (for atomic save)
        self._modified: set = set()

        # Backup settings on init
        self._create_backup()

        # Check schema version and migrate if needed
        self._check_and_migrate()

        StateManager._instance = self

        logger.info(f"StateManager initialized (version {SettingsVersion.CURRENT.value})")
        logger.info(f"Settings location: {self.settings.fileName()}")

    @classmethod
    def instance(cls) -> 'StateManager':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:  # Fast path without lock
            import threading
            if not hasattr(cls, '_instance_lock'):
                cls._instance_lock = threading.Lock()
            with cls._instance_lock:
                if cls._instance is None:  # Double-check inside lock
                    cls._instance = StateManager()
        return cls._instance

    def _create_backup(self):
        """Create backup of settings file."""
        settings_file = Path(self.settings.fileName())
        if not settings_file.exists():
            return

        backup_dir = settings_file.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Keep last 5 backups
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"settings_backup_{timestamp}.ini"

        try:
            shutil.copy2(settings_file, backup_file)
            logger.info(f"Created settings backup: {backup_file}")

            # Clean old backups (keep last 5)
            backups = sorted(backup_dir.glob("settings_backup_*.ini"))
            for old_backup in backups[:-5]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            logger.error(f"Failed to create settings backup: {e}")

    def _check_and_migrate(self):
        """Check settings version and migrate if needed."""
        stored_version = self.settings.value("_schema_version", "1.0")

        if stored_version == SettingsVersion.CURRENT.value:
            return

        logger.info(f"Migrating settings from {stored_version} to {SettingsVersion.CURRENT.value}")

        try:
            self._migrate_settings(stored_version, SettingsVersion.CURRENT.value)
            self.settings.setValue("_schema_version", SettingsVersion.CURRENT.value)
            self.settings.sync()
            logger.info("Settings migration completed successfully")
        except Exception as e:
            logger.error(f"Settings migration failed: {e}")
            # Restore from backup if migration fails
            self._restore_from_backup()

    def _migrate_settings(self, from_version: str, to_version: str):
        """
        Migrate settings between versions.

        Args:
            from_version: Source version
            to_version: Target version
        """
        # Example migration: v1.0 -> v1.1
        if from_version == "1.0" and to_version == "1.1":
            # In v1.1, we split theme into theme_mode (light/dark/auto)
            old_theme = self.settings.value("general/theme", "light")
            if old_theme in ["light", "dark"]:
                self.settings.setValue("general/theme_mode", old_theme)
            else:
                self.settings.setValue("general/theme_mode", "light")

        # Add more migration paths as needed
        # if from_version == "1.1" and to_version == "1.2":
        #     ...

    def _restore_from_backup(self):
        """Restore settings from most recent backup."""
        settings_file = Path(self.settings.fileName())
        backup_dir = settings_file.parent / "backups"

        if not backup_dir.exists():
            logger.error("No backups available for restore")
            return

        backups = sorted(backup_dir.glob("settings_backup_*.ini"))
        if not backups:
            logger.error("No backup files found")
            return

        latest_backup = backups[-1]

        try:
            shutil.copy2(latest_backup, settings_file)
            logger.info(f"Restored settings from backup: {latest_backup}")

            # Reload settings
            self.settings = QSettings(self.settings.organizationName(), self.settings.applicationName())
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")

    def get(self, definition: SettingDefinition[T], use_cache: bool = True) -> T:
        """
        Get setting value with validation.

        Args:
            definition: Setting definition
            use_cache: Whether to use cached value

        Returns:
            Setting value (validated and typed)
        """
        with QMutexLocker(self.mutex):
            # Check cache first (thread-safe LRU)
            if use_cache:
                cached = self._cache.get(definition.key)
                if cached is not None:
                    return cached

            # Get from QSettings
            value = self.settings.value(definition.key, definition.default_value)

            # Type coercion for complex types
            if isinstance(definition.default_value, list) and not isinstance(value, list):
                # QSettings sometimes returns comma-separated strings
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        value = definition.default_value

            # Validate
            if definition.validator and not definition.validator(value):
                logger.warning(
                    f"Invalid value for {definition.key}: {value}, "
                    f"using default: {definition.default_value}"
                )
                value = definition.default_value

            # Cache with automatic LRU eviction (thread-safe)
            self._cache.set(definition.key, value)

            return value

    def set(self, definition: SettingDefinition[T], value: T, emit_signal: bool = True):
        """
        Set setting value with validation.

        Args:
            definition: Setting definition
            value: New value
            emit_signal: Whether to emit setting_changed signal
        """
        with QMutexLocker(self.mutex):
            # Validate
            if definition.validator and not definition.validator(value):
                logger.error(f"Validation failed for {definition.key}: {value}")
                raise ValueError(f"Invalid value for {definition.key}: {value}")

            # Convert complex types to JSON for storage
            if isinstance(value, (list, dict)):
                storage_value = json.dumps(value)
            else:
                storage_value = value

            # Save to QSettings (only if persistent)
            if definition.persistent:
                self.settings.setValue(definition.key, storage_value)
                self._modified.add(definition.key)

            # Update cache
            self._cache[definition.key] = value

            logger.debug(f"Set {definition.key} = {value}")

        # Emit signal outside mutex lock to avoid deadlock
        if emit_signal:
            self.setting_changed.emit(definition.key, value)

    def sync(self):
        """Force synchronization of settings to disk (atomic write)."""
        with QMutexLocker(self.mutex):
            if self._modified:
                self.settings.sync()
                self._modified.clear()
                logger.debug("Settings synchronized to disk")

    def reset(self, definition: SettingDefinition[T]):
        """Reset setting to default value."""
        self.set(definition, definition.default_value)

    def reset_all(self):
        """Reset all settings to defaults."""
        with QMutexLocker(self.mutex):
            self.settings.clear()
            self._cache.clear()
            self._modified.clear()

            # Set schema version
            self.settings.setValue("_schema_version", SettingsVersion.CURRENT.value)
            self.settings.sync()

            logger.info("All settings reset to defaults")

    def export_settings(self, file_path: Path) -> bool:
        """
        Export settings to file.

        Args:
            file_path: Path to export file

        Returns:
            True if successful
        """
        try:
            # Get all settings as dict
            settings_dict = {}
            self.settings.beginGroup("")
            for key in self.settings.allKeys():
                settings_dict[key] = self.settings.value(key)
            self.settings.endGroup()

            # Add metadata
            export_data = {
                "version": SettingsVersion.CURRENT.value,
                "exported_at": datetime.now().isoformat(),
                "settings": settings_dict
            }

            # Write to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Settings exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return False

    def import_settings(self, file_path: Path) -> bool:
        """
        Import settings from file.

        Args:
            file_path: Path to import file

        Returns:
            True if successful
        """
        try:
            with open(file_path, 'r') as f:
                export_data = json.load(f)

            # Check version compatibility
            import_version = export_data.get("version", "1.0")
            if import_version != SettingsVersion.CURRENT.value:
                logger.warning(f"Importing settings from version {import_version}")
                # Could trigger migration here if needed

            # Import settings
            settings_dict = export_data.get("settings", {})
            with QMutexLocker(self.mutex):
                for key, value in settings_dict.items():
                    self.settings.setValue(key, value)

                self._cache.clear()
                self.settings.sync()

            logger.info(f"Settings imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False

    # Convenience methods for common operations

    def save_window_state(self, widget: QWidget):
        """Save window geometry and state."""
        if hasattr(widget, 'saveGeometry'):
            geometry = widget.saveGeometry()
            self.set(SettingsGroup.WINDOW_GEOMETRY, geometry)

        if hasattr(widget, 'saveState'):
            state = widget.saveState()
            self.set(SettingsGroup.WINDOW_STATE, state)

    def restore_window_state(self, widget: QWidget):
        """Restore window geometry and state."""
        geometry = self.get(SettingsGroup.WINDOW_GEOMETRY)
        if geometry and hasattr(widget, 'restoreGeometry'):
            widget.restoreGeometry(geometry)

        state = self.get(SettingsGroup.WINDOW_STATE)
        if state and hasattr(widget, 'restoreState'):
            widget.restoreState(state)

    def save_splitter_state(self, splitter: QSplitter, definition: SettingDefinition):
        """Save splitter sizes."""
        sizes = splitter.sizes()
        self.set(definition, sizes)

    def restore_splitter_state(self, splitter: QSplitter, definition: SettingDefinition):
        """Restore splitter sizes."""
        sizes = self.get(definition)
        if sizes:
            splitter.setSizes(sizes)

    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        recent = self.get(SettingsGroup.RECENT_FILES)

        # Remove if already exists
        if file_path in recent:
            recent.remove(file_path)

        # Add to front
        recent.insert(0, file_path)

        # Limit to max count
        max_count = self.get(SettingsGroup.RECENT_FILES_COUNT)
        recent = recent[:max_count]

        self.set(SettingsGroup.RECENT_FILES, recent)

    def get_recent_files(self) -> List[str]:
        """Get recent files list."""
        return self.get(SettingsGroup.RECENT_FILES)

    def clear_recent_files(self):
        """Clear recent files list."""
        self.set(SettingsGroup.RECENT_FILES, [])

    # Backward compatibility methods for simple string-based API
    def get_simple(self, key: str, default: Any = None) -> Any:
        """
        Get setting value using simple string key (backward compatibility).

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        return self.settings.value(key, default)

    def set_simple(self, key: str, value: Any):
        """
        Set setting value using simple string key (backward compatibility).

        Args:
            key: Setting key
            value: Value to set
        """
        self.settings.setValue(key, value)

    def has(self, key: str) -> bool:
        """Check if a setting key exists."""
        return self.settings.contains(key)

    def getRecoveryPath(self) -> str:
        """Get path to crash recovery file."""
        return str(Path.home() / ".posturekit" / "recovery.json")

    def loadWindowState(self, window):
        """
        Load window geometry and state (backward compatibility).

        Args:
            window: QMainWindow to restore state to
        """
        geometry = self.settings.value("window/geometry")
        if geometry:
            window.restoreGeometry(geometry)

        state = self.settings.value("window/state")
        if state:
            window.restoreState(state)

    def saveWindowState(self, window):
        """
        Save window geometry and state (backward compatibility).

        Args:
            window: QMainWindow to save state from
        """
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.setValue("window/state", window.saveState())

    def save(self):
        """Save settings to disk."""
        self.sync()


# Convenience function
def get_state() -> StateManager:
    """Get global state manager instance."""
    return StateManager.instance()


if __name__ == "__main__":
    # Test state manager
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel

    app = QApplication(sys.argv)

    # Initialize state manager
    state = StateManager.instance()

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)

    # Display some settings
    status = QLabel()
    def update_status():
        theme = state.get(SettingsGroup.THEME)
        radius = state.get(SettingsGroup.KEYPOINT_RADIUS)
        recent = state.get_recent_files()
        status.setText(
            f"Theme: {theme}\n"
            f"Keypoint Radius: {radius}\n"
            f"Recent Files: {len(recent)}"
        )

    update_status()
    layout.addWidget(status)

    # Test buttons
    def toggle_theme():
        current = state.get(SettingsGroup.THEME)
        new_theme = "dark" if current == "light" else "light"
        state.set(SettingsGroup.THEME, new_theme)
        update_status()

    btn = QPushButton("Toggle Theme")
    btn.clicked.connect(toggle_theme)
    layout.addWidget(btn)

    def add_file():
        state.add_recent_file(f"/path/to/file_{len(state.get_recent_files())}.jpg")
        update_status()

    btn2 = QPushButton("Add Recent File")
    btn2.clicked.connect(add_file)
    layout.addWidget(btn2)

    def test_export():
        from pathlib import Path
        export_path = Path.home() / "posturekit_settings_export.json"
        if state.export_settings(export_path):
            status.setText(f"Exported to {export_path}")

    btn3 = QPushButton("Export Settings")
    btn3.clicked.connect(test_export)
    layout.addWidget(btn3)

    window.setCentralWidget(central)
    window.resize(400, 300)
    window.show()

    sys.exit(app.exec())
