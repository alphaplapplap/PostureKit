"""
Statistics dialog showing database and dataset statistics
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QPen
from PySide6.QtCharts import QChart, QChartView, QPieSeries, QBarSeries, QBarSet

from src.storage.storage_manager import StorageManager


class StatisticsDialog(QDialog):
    """Dialog showing comprehensive database statistics"""

    def __init__(self, db_manager: StorageManager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.setWindowTitle("Database Statistics")
        self.setModal(False)  # Non-blocking
        self.resize(700, 600)
        self.initUI()
        self.loadStatistics()

    def initUI(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # Overview tab
        self.overview_tab = self.createOverviewTab()
        self.tabs.addTab(self.overview_tab, "Overview")

        # Quality tab
        self.quality_tab = self.createQualityTab()
        self.tabs.addTab(self.quality_tab, "Quality")

        # Categories tab
        self.categories_tab = self.createCategoriesTab()
        self.tabs.addTab(self.categories_tab, "Categories")

        # Timeline tab
        self.timeline_tab = self.createTimelineTab()
        self.tabs.addTab(self.timeline_tab, "Timeline")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("Export Report (CSV)")
        export_btn.clicked.connect(self.exportReport)
        button_layout.addWidget(export_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def createOverviewTab(self):
        """Create overview statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Stats grid
        stats_group = QGroupBox("Dataset Overview")
        stats_layout = QGridLayout()

        # Create stat cards
        self.total_images_label = self.createStatCard("Total Images", "0")
        stats_layout.addWidget(self.total_images_label, 0, 0)

        self.total_poses_label = self.createStatCard("Total Poses", "0")
        stats_layout.addWidget(self.total_poses_label, 0, 1)

        self.avg_poses_label = self.createStatCard("Avg Poses/Image", "0.0")
        stats_layout.addWidget(self.avg_poses_label, 0, 2)

        self.corrected_label = self.createStatCard("Corrected", "0 (0%)")
        stats_layout.addWidget(self.corrected_label, 1, 0)

        self.labeled_label = self.createStatCard("Labeled", "0 (0%)")
        stats_layout.addWidget(self.labeled_label, 1, 1)

        self.training_ready_label = self.createStatCard("Training-Ready", "0 (0%)")
        stats_layout.addWidget(self.training_ready_label, 1, 2)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Quality metrics
        quality_group = QGroupBox("Quality Metrics")
        quality_layout = QGridLayout()

        self.avg_confidence_label = self.createStatCard("Avg Confidence", "0.00")
        quality_layout.addWidget(self.avg_confidence_label, 0, 0)

        self.avg_occlusion_label = self.createStatCard("Avg Occlusion", "0.0%")
        quality_layout.addWidget(self.avg_occlusion_label, 0, 1)

        self.high_quality_label = self.createStatCard("High Quality (>0.7)", "0 (0%)")
        quality_layout.addWidget(self.high_quality_label, 1, 0)

        self.medium_quality_label = self.createStatCard("Medium (0.4-0.7)", "0 (0%)")
        quality_layout.addWidget(self.medium_quality_label, 1, 1)

        self.low_quality_label = self.createStatCard("Low (<0.4)", "0 (0%)")
        quality_layout.addWidget(self.low_quality_label, 1, 2)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def createQualityTab(self):
        """Create quality analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Confidence distribution chart placeholder
        chart_label = QLabel("Confidence Distribution Chart")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_label.setMinimumHeight(300)
        chart_label.setStyleSheet("""
            QLabel {
                background-color: #1a1f2e;
                border: 1px solid #2a3341;
                border-radius: 8px;
                color: #9ca3af;
                font-size: 14px;
            }
        """)
        layout.addWidget(chart_label)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def createCategoriesTab(self):
        """Create categories breakdown tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Categories table
        self.categories_table = QTableWidget()
        self.categories_table.setColumnCount(3)
        self.categories_table.setHorizontalHeaderLabels(["Category", "Count", "Percentage"])
        self.categories_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.categories_table)

        # Top tags
        self.tags_label = QLabel("Top Tags: Loading...")
        self.tags_label.setWordWrap(True)
        layout.addWidget(self.tags_label)

        widget.setLayout(layout)
        return widget

    def createTimelineTab(self):
        """Create timeline analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Timeline chart placeholder
        chart_label = QLabel("Poses Added Over Time")
        chart_label.setAlignment(Qt.AlignCenter)
        chart_label.setMinimumHeight(400)
        chart_label.setStyleSheet("""
            QLabel {
                background-color: #1a1f2e;
                border: 1px solid #2a3341;
                border-radius: 8px;
                color: #9ca3af;
                font-size: 14px;
            }
        """)
        layout.addWidget(chart_label)

        widget.setLayout(layout)
        return widget

    def createStatCard(self, title, value):
        """Create a statistics card widget"""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #1a1f2e;
                border: 1px solid #2a3341;
                border-radius: 8px;
                padding: 12px;
            }
            QWidget:hover {
                border-color: #4a9eff;
            }
        """)

        layout = QVBoxLayout()

        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 12px;
            color: #9ca3af;
            text-transform: uppercase;
        """)
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName(f"{title.lower().replace(' ', '_')}_value")
        value_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #4a9eff;
        """)
        layout.addWidget(value_label)

        widget.setLayout(layout)
        return widget

    def loadStatistics(self):
        """Load statistics from database"""
        try:
            stats = self.db_manager.getStatistics()

            # Update overview tab
            self.updateStatCard(self.total_images_label, str(stats.get('total_images', 0)))
            self.updateStatCard(self.total_poses_label, str(stats.get('total_poses', 0)))

            # Calculate averages
            if stats.get('total_images', 0) > 0:
                avg_poses = stats['total_poses'] / stats['total_images']
                self.updateStatCard(self.avg_poses_label, f"{avg_poses:.1f}")

            # Calculate percentages
            total = stats.get('total_poses', 0)
            if total > 0:
                corrected = stats.get('corrected_poses', 0)
                labeled = stats.get('labeled_poses', 0)

                self.updateStatCard(self.corrected_label,
                                  f"{corrected} ({corrected/total*100:.0f}%)")
                self.updateStatCard(self.labeled_label,
                                  f"{labeled} ({labeled/total*100:.0f}%)")

            # Update confidence
            avg_conf = stats.get('average_confidence', 0)
            self.updateStatCard(self.avg_confidence_label, f"{avg_conf:.2f}")

            # Update categories
            categories = stats.get('categories', [])
            self.updateCategoriesTable(categories)

        except Exception as e:
            print(f"Failed to load statistics: {e}")

    def updateStatCard(self, card_widget, value):
        """Update a stat card's value"""
        value_label = card_widget.findChild(QLabel)
        if value_label and value_label.objectName():
            value_label.setText(value)

    def updateCategoriesTable(self, categories):
        """Update categories table"""
        self.categories_table.setRowCount(len(categories))

        total = sum(cat['count'] for cat in categories)

        for i, cat in enumerate(categories):
            self.categories_table.setItem(i, 0, QTableWidgetItem(cat['category']))
            self.categories_table.setItem(i, 1, QTableWidgetItem(str(cat['count'])))

            if total > 0:
                percentage = cat['count'] / total * 100
                self.categories_table.setItem(i, 2, QTableWidgetItem(f"{percentage:.1f}%"))

    def exportReport(self):
        """Export statistics to CSV file"""
        # Implementation for CSV export
        pass
