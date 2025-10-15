"""
About dialog with application information and credits.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QTextBrowser, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, QDesktopServices
import platform
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AboutDialog(QDialog):
    """
    Professional about dialog with application information.

    This dialog serves multiple purposes:
    1. Brand communication - Presents PostureKit's identity
    2. Version tracking - Shows exact version for bug reports
    3. Credits - Acknowledges contributors and dependencies
    4. Legal compliance - Displays license information
    5. System info - Helps with troubleshooting

    The tabbed interface keeps information organized while ensuring
    users can easily find what they need.
    """

    # Version information (would typically come from package metadata)
    VERSION = "0.1.0"
    BUILD_DATE = "2024-01-15"

    def __init__(self, parent=None):
        """
        Initialize about dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("About PostureKit")
        self.setMinimumSize(600, 500)

        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog layout and content."""
        layout = QVBoxLayout()

        # Header with logo and title
        header = self._create_header()
        layout.addWidget(header)

        # Tabbed content
        tabs = QTabWidget()
        tabs.addTab(self._create_about_tab(), "About")
        tabs.addTab(self._create_credits_tab(), "Credits")
        tabs.addTab(self._create_license_tab(), "License")
        tabs.addTab(self._create_system_tab(), "System Info")

        layout.addWidget(tabs)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_header(self) -> QWidget:
        """
        Create header with application logo and title.

        The header establishes visual identity and immediately communicates
        what application the user is looking at. The logo and title are
        prominently displayed using the same styling as the splash screen
        for brand consistency.

        Returns:
            Header widget
        """
        header = QWidget()
        header.setFixedHeight(120)
        header.setStyleSheet("background-color: #2d3444;")

        layout = QHBoxLayout()

        # Logo (would be replaced with actual logo image)
        logo_label = QLabel()
        logo_pixmap = self._create_logo(100)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label)

        # Title and version
        text_layout = QVBoxLayout()

        title_label = QLabel("PostureKit")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white;")
        text_layout.addWidget(title_label)

        subtitle_label = QLabel("Pose Estimation Training Platform")
        subtitle_label.setStyleSheet("color: #b4b4b4; font-size: 14px;")
        text_layout.addWidget(subtitle_label)

        version_label = QLabel(f"Version {self.VERSION}")
        version_label.setStyleSheet("color: #888; font-size: 12px;")
        text_layout.addWidget(version_label)

        text_layout.addStretch()

        layout.addLayout(text_layout)
        layout.addStretch()

        header.setLayout(layout)
        return header

    def _create_logo(self, size: int) -> QPixmap:
        """
        Create application logo.

        In a production application, this would load an actual logo image.
        For now, we create a simple placeholder that matches the brand colors.

        Args:
            size: Logo size in pixels

        Returns:
            Logo pixmap
        """
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw simple logo placeholder (circle with "PK")
        painter.setBrush(QColor(52, 152, 219))  # Blue
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)

        # Draw "PK" text
        painter.setPen(Qt.white)
        font = QFont()
        font.setPointSize(size // 3)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "PK")

        painter.end()
        return pixmap

    def _create_about_tab(self) -> QWidget:
        """
        Create about tab with description and key information.

        This tab provides an overview of what PostureKit does and why it exists.
        It's written in accessible language that helps new users understand the
        application's purpose without requiring deep technical knowledge.

        Returns:
            About tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()

        # Description
        desc_label = QLabel()
        desc_label.setWordWrap(True)
        desc_label.setText(
            "PostureKit is a professional desktop application for training pose estimation models "
            "through manual correction and intelligent learning. Built specifically for macOS, "
            "it leverages Apple Silicon GPU acceleration to provide real-time pose detection, "
            "correction, and active learning capabilities.\n\n"

            "Key Features:\n"
            "• Real-time 133-keypoint pose detection using MMPose RTMW-L\n"
            "• Interactive pose correction with visual feedback\n"
            "• Automatic viewpoint estimation and geometric analysis\n"
            "• Active learning that improves accuracy from your corrections\n"
            "• Advanced similarity search for organizing large datasets\n"
            "• Native macOS experience with Metal GPU acceleration\n\n"

            "PostureKit is designed for researchers, annotators, and machine learning engineers "
            "who need precise control over pose estimation training data."
        )
        layout.addWidget(desc_label)

        layout.addStretch()

        # Links
        links_layout = QHBoxLayout()

        website_btn = QPushButton("Website")
        website_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://posturekit.ai"))
        )
        links_layout.addWidget(website_btn)

        docs_btn = QPushButton("Documentation")
        docs_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://docs.posturekit.ai"))
        )
        links_layout.addWidget(docs_btn)

        github_btn = QPushButton("GitHub")
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/posturekit/posturekit"))
        )
        links_layout.addWidget(github_btn)

        links_layout.addStretch()

        layout.addLayout(links_layout)

        tab.setLayout(layout)
        return tab

    def _create_credits_tab(self) -> QWidget:
        """
        Create credits tab acknowledging contributors and dependencies.

        Proper attribution is both legally important and professionally courteous.
        This tab acknowledges everyone who contributed to making PostureKit possible,
        from core developers to the maintainers of critical dependencies.

        Returns:
            Credits tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()

        # Create scrollable text area for credits
        credits_text = QTextBrowser()
        credits_text.setOpenExternalLinks(True)
        credits_text.setHtml(self._get_credits_html())

        layout.addWidget(credits_text)

        tab.setLayout(layout)
        return tab

    def _get_credits_html(self) -> str:
        """
        Generate HTML for credits display.

        This method builds formatted HTML that properly credits all contributors
        and dependencies. Using HTML allows for rich formatting with links to
        project pages and proper visual hierarchy.

        Returns:
            HTML string for credits
        """
        return """
        <h3>Core Development</h3>
        <p>
        PostureKit is developed and maintained by the PostureKit team.<br>
        Project Lead: [Your Name]<br>
        Contributors: [List contributors]
        </p>

        <h3>Built With</h3>
        <p>
        PostureKit is built on top of excellent open-source projects:
        </p>

        <h4>Core Dependencies</h4>
        <ul>
        <li><b>MMPose</b> - Pose estimation framework by OpenMMLab<br>
        <a href="https://github.com/open-mmlab/mmpose">https://github.com/open-mmlab/mmpose</a></li>

        <li><b>PyTorch</b> - Deep learning framework<br>
        <a href="https://pytorch.org">https://pytorch.org</a></li>

        <li><b>PySide6</b> - Python bindings for Qt<br>
        <a href="https://wiki.qt.io/Qt_for_Python">https://wiki.qt.io/Qt_for_Python</a></li>

        <li><b>SQLAlchemy</b> - SQL toolkit and ORM<br>
        <a href="https://www.sqlalchemy.org">https://www.sqlalchemy.org</a></li>

        <li><b>PostgreSQL</b> - Advanced open-source database<br>
        <a href="https://www.postgresql.org">https://www.postgresql.org</a></li>

        <li><b>OpenCV</b> - Computer vision library<br>
        <a href="https://opencv.org">https://opencv.org</a></li>

        <li><b>NumPy</b> - Numerical computing library<br>
        <a href="https://numpy.org">https://numpy.org</a></li>

        <li><b>scikit-learn</b> - Machine learning library<br>
        <a href="https://scikit-learn.org">https://scikit-learn.org</a></li>
        </ul>

        <h4>Additional Libraries</h4>
        <ul>
        <li>SciPy - Scientific computing</li>
        <li>python-dotenv - Environment configuration</li>
        <li>psycopg2 - PostgreSQL adapter</li>
        </ul>

        <h3>Special Thanks</h3>
        <p>
        We are grateful to the open-source community and the researchers who
        developed the algorithms and models that power PostureKit.
        </p>

        <p>
        Special recognition to the MMPose team for their excellent RTM pose
        estimation models and comprehensive documentation.
        </p>
        """

    def _create_license_tab(self) -> QWidget:
        """
        Create license tab with legal information.

        License information is essential for legal compliance and helps users
        understand their rights and obligations. This tab displays PostureKit's
        license along with information about third-party licenses.

        Returns:
            License tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()

        license_text = QTextBrowser()
        license_text.setOpenExternalLinks(True)
        license_text.setHtml(self._get_license_html())

        layout.addWidget(license_text)

        tab.setLayout(layout)
        return tab

    def _get_license_html(self) -> str:
        """
        Generate HTML for license information.

        Returns:
            HTML string with license text
        """
        return """
        <h3>PostureKit License</h3>
        <p>
        PostureKit is released under the MIT License.
        </p>

        <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
MIT License

Copyright (c) 2024 PostureKit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
        </pre>

        <h3>Third-Party Licenses</h3>
        <p>
        PostureKit includes or depends on software released under various licenses:
        </p>

        <ul>
        <li><b>MMPose</b> - Apache License 2.0</li>
        <li><b>PyTorch</b> - BSD-style License</li>
        <li><b>Qt/PySide6</b> - LGPL v3</li>
        <li><b>PostgreSQL</b> - PostgreSQL License</li>
        <li><b>OpenCV</b> - Apache License 2.0</li>
        </ul>

        <p>
        Full license texts for all dependencies are available in the
        <code>licenses/</code> directory of the installation.
        </p>
        """

    def _create_system_tab(self) -> QWidget:
        """
        Create system information tab for troubleshooting.

        When users report bugs or performance issues, system information is
        critical for diagnosis. This tab collects all relevant details about
        the runtime environment in a format that's easy to copy and paste
        into bug reports or support requests.

        Returns:
            System info tab widget
        """
        tab = QWidget()
        layout = QVBoxLayout()

        info_text = QTextBrowser()
        info_text.setPlainText(self._get_system_info())

        layout.addWidget(info_text)

        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self._copy_system_info(info_text.toPlainText()))
        layout.addWidget(copy_btn)

        tab.setLayout(layout)
        return tab

    def _get_system_info(self) -> str:
        """
        Collect system information for troubleshooting.

        This gathers comprehensive information about the runtime environment,
        including OS version, Python version, library versions, and hardware
        details. All information that could be relevant for debugging is included.

        Returns:
            Formatted system information string
        """
        import torch
        import cv2
        import sqlalchemy

        info = []
        info.append(f"PostureKit Version: {self.VERSION}")
        info.append(f"Build Date: {self.BUILD_DATE}")
        info.append("")

        info.append("System Information:")
        info.append(f"  OS: {platform.system()} {platform.release()}")
        info.append(f"  Architecture: {platform.machine()}")
        info.append(f"  Python: {sys.version}")
        info.append("")

        info.append("Libraries:")
        info.append(f"  PyTorch: {torch.__version__}")
        info.append(f"  OpenCV: {cv2.__version__}")
        info.append(f"  SQLAlchemy: {sqlalchemy.__version__}")

        try:
            import mmpose
            info.append(f"  MMPose: {mmpose.__version__}")
        except:
            info.append("  MMPose: Not found")

        info.append("")

        info.append("Hardware:")
        info.append(f"  Processor: {platform.processor()}")

        # GPU information
        if torch.backends.mps.is_available():
            info.append("  GPU: Apple Silicon (MPS available)")
        elif torch.cuda.is_available():
            info.append(f"  GPU: {torch.cuda.get_device_name(0)}")
            info.append(f"  CUDA: {torch.version.cuda}")
        else:
            info.append("  GPU: None (CPU only)")

        info.append("")

        # Database info
        try:
            from src.config.settings import settings
            info.append("Database:")
            info.append(f"  Host: {settings.DB_HOST}")
            info.append(f"  Port: {settings.DB_PORT}")
            info.append(f"  Database: {settings.DB_NAME}")
        except:
            info.append("Database: Configuration not loaded")

        return "\n".join(info)

    def _copy_system_info(self, text: str):
        """
        Copy system information to clipboard.

        Args:
            text: System info text to copy
        """
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        from gui.widgets.toast_notification import show_success
        show_success("System information copied to clipboard")

        logger.info("System information copied to clipboard")
