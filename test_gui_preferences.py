"""
Test GUI preferences for model selection.

Demonstrates:
1. Preferences dialog with model selection
2. Settings persistence
3. Conversion to PostureKitBridge parameters
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox
from PySide6.QtCore import QSettings
from gui_legacy.dialogs.preferences_dialog import PreferencesDialog


class TestWindow(QMainWindow):
    """Simple test window to demonstrate preferences."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PostureKit Preferences Test")
        self.setGeometry(100, 100, 600, 400)

        # Central widget
        central = QWidget()
        layout = QVBoxLayout()

        # Info label
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.update_info()
        layout.addWidget(self.info_label)

        # Open preferences button
        prefs_btn = QPushButton("Open Preferences")
        prefs_btn.clicked.connect(self.open_preferences)
        layout.addWidget(prefs_btn)

        # Show current config button
        config_btn = QPushButton("Show Current Config")
        config_btn.clicked.connect(self.show_config)
        layout.addWidget(config_btn)

        # Test bridge initialization button
        test_btn = QPushButton("Test PostureKitBridge Initialization")
        test_btn.clicked.connect(self.test_bridge_init)
        layout.addWidget(test_btn)

        layout.addStretch()

        central.setLayout(layout)
        self.setCentralWidget(central)

    def update_info(self):
        """Update info label with current settings."""
        config = PreferencesDialog.get_pose_detector_config()

        pose_models = config['pose_models']
        if isinstance(pose_models, list):
            model_text = f"Ensemble: {' + '.join([m.upper() for m in pose_models])}"
        else:
            model_text = pose_models.upper()

        info = f"""
<h3>Current Settings:</h3>
<p><b>Pose Model:</b> {model_text}</p>
<p><b>Fusion Method:</b> {config['fusion_method']}</p>
<p><b>Two-Stage Detection:</b> {config['use_two_stage']}</p>

<p style='color: #666;'>
<i>These settings are persisted and will be used when initializing PostureKitBridge.</i>
</p>
        """
        self.info_label.setText(info)

    def open_preferences(self):
        """Open preferences dialog."""
        dialog = PreferencesDialog(self)
        dialog.settings_changed.connect(self.on_settings_changed)
        dialog.exec()

    def on_settings_changed(self, changed):
        """Handle settings changes."""
        print(f"Settings changed: {changed.keys()}")
        self.update_info()

    def show_config(self):
        """Show current configuration as code."""
        config = PreferencesDialog.get_pose_detector_config()

        code = f"""
from src.swift_bridge import PostureKitBridge

# Initialize with current preferences:
bridge = PostureKitBridge(
    pose_models={repr(config['pose_models'])},
    fusion_method='{config['fusion_method']}',
    use_two_stage={config['use_two_stage']}
)
        """

        QMessageBox.information(
            self,
            "Current Configuration",
            f"<h3>PostureKitBridge Parameters:</h3><pre>{code}</pre>"
        )

    def test_bridge_init(self):
        """Test PostureKitBridge initialization with current settings."""
        try:
            from swift_bridge import PostureKitBridge

            config = PreferencesDialog.get_pose_detector_config()

            QMessageBox.information(
                self,
                "Initializing Bridge",
                f"Initializing PostureKitBridge with:\n"
                f"  pose_models: {config['pose_models']}\n"
                f"  fusion_method: {config['fusion_method']}\n"
                f"  use_two_stage: {config['use_two_stage']}\n\n"
                f"This may take a few seconds..."
            )

            # Initialize bridge
            bridge = PostureKitBridge(**config)

            QMessageBox.information(
                self,
                "Success",
                f"✅ PostureKitBridge initialized successfully!\n\n"
                f"Configuration:\n"
                f"  pose_models: {config['pose_models']}\n"
                f"  fusion_method: {config['fusion_method']}\n"
                f"  use_two_stage: {config['use_two_stage']}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to initialize PostureKitBridge:\n{e}"
            )


def main():
    """Run test application."""
    app = QApplication(sys.argv)

    # Set application name for QSettings
    app.setOrganizationName("PostureKit")
    app.setApplicationName("PostureKit")

    window = TestWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
