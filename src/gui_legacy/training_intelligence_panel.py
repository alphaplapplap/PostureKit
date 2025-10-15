"""
Training Intelligence Panel for PostureKit
Active learning panel that suggests which poses to annotate next for optimal training dataset.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QProgressBar,
    QScrollArea, QFrame, QGridLayout, QDockWidget,
    QGroupBox, QListWidget, QListWidgetItem, QMessageBox,
    QTextEdit
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QThread, QObject,
    QRect, QSize, QTimer, QPointF
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor,
    QPen, QBrush, QFont, QIcon, QPolygonF
)

try:
    from intelligence.training_intelligence import (
        TrainingIntelligence,
        SamplingStrategy as BackendSamplingStrategy,
        TrainingCandidate
    )
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    HAS_BACKEND = True
except ImportError:
    logger = None
    HAS_BACKEND = False


class SamplingStrategy(Enum):
    """Active learning sampling strategies."""
    UNCERTAINTY = "Uncertainty - Low confidence poses"
    DIVERSITY = "Diversity - Underrepresented poses"
    HYBRID = "Hybrid - Balanced approach"
    MARGIN = "Margin - Decision boundary cases"
    ENTROPY = "Entropy - Maximum information gain"
    # Map to backend strategies if available
    CLUSTER_CENTERS = "Cluster Centers - Representatives"
    BOUNDARY = "Decision Boundary - Refine edges"


@dataclass
class DatasetStats:
    """Statistics about the current dataset."""
    total_poses: int = 0
    labeled: int = 0
    corrected: int = 0
    high_quality: int = 0
    categories: Dict[str, int] = field(default_factory=dict)

    @property
    def labeled_percentage(self) -> float:
        """Calculate percentage of labeled poses."""
        return (self.labeled / self.total_poses * 100) if self.total_poses > 0 else 0

    @property
    def corrected_percentage(self) -> float:
        """Calculate percentage of corrected poses."""
        return (self.corrected / self.total_poses * 100) if self.total_poses > 0 else 0

    @property
    def high_quality_percentage(self) -> float:
        """Calculate percentage of high quality poses."""
        return (self.high_quality / self.total_poses * 100) if self.total_poses > 0 else 0


@dataclass
class CandidatePose:
    """A candidate pose for annotation."""
    pose_id: int
    score: float  # 0.0 to 1.0, higher = more valuable to annotate
    reason: str
    file_path: str
    thumbnail: Optional[QPixmap] = None
    confidence: float = 0.0
    category: Optional[str] = None
    is_priority: bool = False  # Show star indicator

    def __lt__(self, other):
        """Enable sorting by score."""
        return self.score > other.score  # Higher scores first


class ThumbnailWidget(QWidget):
    """Widget for displaying pose thumbnail with overlay info."""

    def __init__(self, size: int = 80, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.size = size
        self.pixmap: Optional[QPixmap] = None
        self.score: float = 0.0
        self.is_priority: bool = False
        self.setFixedSize(size, size)

    def set_thumbnail(self, pixmap: Optional[QPixmap]) -> None:
        """Set the thumbnail image."""
        if pixmap:
            self.pixmap = pixmap.scaled(
                self.size, self.size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            self.pixmap = None
        self.update()

    def set_score(self, score: float, is_priority: bool = False) -> None:
        """Set the score and priority status."""
        self.score = score
        self.is_priority = is_priority
        self.update()

    def paintEvent(self, event) -> None:
        """Custom painting for thumbnail with overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        if self.pixmap:
            # Draw centered thumbnail
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            painter.drawPixmap(x, y, self.pixmap)
        else:
            # Draw placeholder
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.drawRect(1, 1, self.width() - 2, self.height() - 2)

            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image")

        # Draw score bar at bottom
        bar_height = 4
        bar_width = int(self.width() * self.score)

        # Choose color based on score
        if self.score > 0.8:
            color = QColor(16, 185, 129)  # Green
        elif self.score > 0.5:
            color = QColor(245, 158, 11)  # Yellow
        else:
            color = QColor(239, 68, 68)   # Red

        painter.fillRect(0, self.height() - bar_height,
                        bar_width, bar_height, color)

        # Draw priority star if needed
        if self.is_priority:
            painter.setPen(QPen(QColor(255, 193, 7), 2))
            painter.setBrush(QBrush(QColor(255, 235, 59)))

            # Draw star in top-left corner
            star_points = [
                QPointF(8, 2), QPointF(10, 7), QPointF(15, 7), QPointF(11, 10),
                QPointF(13, 15), QPointF(8, 12), QPointF(3, 15), QPointF(5, 10),
                QPointF(1, 7), QPointF(6, 7)
            ]

            star = QPolygonF(star_points)
            painter.drawPolygon(star)


class CandidateListItem(QWidget):
    """Custom list item widget for candidate poses."""

    clicked = Signal(object)  # CandidatePose
    double_clicked = Signal(object)  # CandidatePose

    def __init__(self, candidate: CandidatePose, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.candidate = candidate
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Thumbnail
        self.thumbnail = ThumbnailWidget(60)
        self.thumbnail.set_thumbnail(self.candidate.thumbnail)
        self.thumbnail.set_score(self.candidate.score, self.candidate.is_priority)
        layout.addWidget(self.thumbnail)

        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        # Title with priority indicator
        title_layout = QHBoxLayout()
        title_layout.setSpacing(4)

        if self.candidate.is_priority:
            star_label = QLabel("⭐")
            title_layout.addWidget(star_label)

        title = QLabel(f"Pose #{self.candidate.pose_id}")
        title.setStyleSheet("font-weight: bold;")
        title_layout.addWidget(title)

        score_label = QLabel(f"(Score: {self.candidate.score:.2f})")
        score_label.setStyleSheet("color: #666;")
        title_layout.addWidget(score_label)

        title_layout.addStretch()
        info_layout.addLayout(title_layout)

        # Reason
        reason_label = QLabel(f"Reason: {self.candidate.reason}")
        reason_label.setStyleSheet("font-size: 9pt; color: #444;")
        reason_label.setWordWrap(True)
        info_layout.addWidget(reason_label)

        # Additional info if available
        if self.candidate.category:
            cat_label = QLabel(f"Category: {self.candidate.category}")
            cat_label.setStyleSheet("font-size: 8pt; color: #666;")
            info_layout.addWidget(cat_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QWidget:hover {
                background-color: #f0f9ff;
                border-color: #3b82f6;
            }
        """)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.candidate)

    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.candidate)


class ActiveLearningWorker(QObject):
    """Background worker for computing active learning scores."""

    candidates_found = Signal(list)  # List[CandidatePose]
    progress_updated = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, training_intelligence=None):
        super().__init__()
        self.training_intelligence = training_intelligence
        self.strategy = SamplingStrategy.HYBRID
        self.num_candidates = 10
        self.dataset_stats = DatasetStats()

    @Slot(list, object, int)
    def find_candidates(self,
                       poses_data: List[Dict],
                       strategy: SamplingStrategy,
                       num_candidates: int) -> None:
        """
        Find candidate poses using the selected active learning strategy.

        Args:
            poses_data: List of pose dictionaries with metadata
            strategy: Selected sampling strategy
            num_candidates: Number of candidates to return
        """
        try:
            # If we have backend integration, use it
            if HAS_BACKEND and self.training_intelligence:
                self._find_with_backend(strategy, num_candidates)
            else:
                # Use fallback implementation
                candidates = self._find_with_fallback(poses_data, strategy, num_candidates)
                self.candidates_found.emit(candidates)

        except Exception as e:
            if logger:
                logger.error(f"Candidate search failed: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

    def _find_with_backend(self, strategy: SamplingStrategy, n: int):
        """Use backend TrainingIntelligence."""
        # Map UI strategy to backend strategy
        strategy_map = {
            SamplingStrategy.UNCERTAINTY: BackendSamplingStrategy.UNCERTAINTY,
            SamplingStrategy.DIVERSITY: BackendSamplingStrategy.DIVERSITY,
            SamplingStrategy.HYBRID: BackendSamplingStrategy.HYBRID,
            SamplingStrategy.CLUSTER_CENTERS: BackendSamplingStrategy.CLUSTER_CENTERS,
            SamplingStrategy.BOUNDARY: BackendSamplingStrategy.BOUNDARY,
        }

        backend_strategy = strategy_map.get(strategy, BackendSamplingStrategy.HYBRID)

        # Get candidates from backend
        backend_candidates = self.training_intelligence.identify_training_candidates(
            n_candidates=n,
            strategy=backend_strategy,
            exclude_labeled=True
        )

        # Convert to CandidatePose objects
        candidates = []
        for bc in backend_candidates:
            candidate = CandidatePose(
                pose_id=bc.pose_id,
                score=bc.priority_score,
                reason=bc.reasoning,
                file_path=getattr(bc, 'file_path', ''),
                confidence=getattr(bc, 'confidence', 0.0),
                category=getattr(bc, 'category', None),
                is_priority=(bc.priority_score > 0.7)
            )
            candidates.append(candidate)

        self.candidates_found.emit(candidates)

    def _find_with_fallback(self, poses_data: List[Dict], strategy: SamplingStrategy, n: int) -> List[CandidatePose]:
        """Fallback implementation without backend."""
        if strategy == SamplingStrategy.UNCERTAINTY:
            return self._uncertainty_sampling(poses_data, n)
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._diversity_sampling(poses_data, n)
        elif strategy == SamplingStrategy.MARGIN:
            return self._margin_sampling(poses_data, n)
        elif strategy == SamplingStrategy.ENTROPY:
            return self._entropy_sampling(poses_data, n)
        else:  # HYBRID
            return self._hybrid_sampling(poses_data, n)

    def _uncertainty_sampling(self, poses_data: List[Dict], n: int) -> List[CandidatePose]:
        """Select poses with lowest confidence scores."""
        candidates = []
        sorted_poses = sorted(poses_data, key=lambda p: p.get('confidence', 1.0))

        for i, pose in enumerate(sorted_poses[:n]):
            score = 1.0 - pose.get('confidence', 0.5)

            candidate = CandidatePose(
                pose_id=pose.get('id', i),
                score=score,
                reason="Low detection confidence",
                file_path=pose.get('file_path', ''),
                confidence=pose.get('confidence', 0.0),
                category=pose.get('category'),
                is_priority=(score > 0.9)
            )
            candidates.append(candidate)
            self.progress_updated.emit(int((i + 1) / n * 100))

        return candidates

    def _diversity_sampling(self, poses_data: List[Dict], n: int) -> List[CandidatePose]:
        """Select poses that increase dataset diversity."""
        candidates = []
        category_counts = {}
        for pose in poses_data:
            cat = pose.get('category', 'uncategorized')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        avg_count = sum(category_counts.values()) / len(category_counts) if category_counts else 1
        scored_poses = []

        for pose in poses_data:
            cat = pose.get('category', 'uncategorized')
            cat_count = category_counts.get(cat, 0)
            diversity_score = max(0, 1.0 - (cat_count / avg_count))
            scored_poses.append((pose, diversity_score))

        scored_poses.sort(key=lambda x: x[1], reverse=True)

        for i, (pose, score) in enumerate(scored_poses[:n]):
            candidate = CandidatePose(
                pose_id=pose.get('id', i),
                score=score,
                reason="Underrepresented pose type",
                file_path=pose.get('file_path', ''),
                category=pose.get('category'),
                is_priority=(score > 0.8)
            )
            candidates.append(candidate)
            self.progress_updated.emit(int((i + 1) / n * 100))

        return candidates

    def _margin_sampling(self, poses_data: List[Dict], n: int) -> List[CandidatePose]:
        """Select poses near decision boundaries."""
        candidates = []
        for i, pose in enumerate(poses_data[:n]):
            confidence = pose.get('confidence', 0.5)
            margin_score = 1.0 - abs(confidence - 0.5) * 2

            candidate = CandidatePose(
                pose_id=pose.get('id', i),
                score=margin_score,
                reason="Near decision boundary",
                file_path=pose.get('file_path', ''),
                confidence=confidence,
                is_priority=(margin_score > 0.85)
            )
            candidates.append(candidate)
            self.progress_updated.emit(int((i + 1) / n * 100))

        candidates.sort()
        return candidates[:n]

    def _entropy_sampling(self, poses_data: List[Dict], n: int) -> List[CandidatePose]:
        """Select poses with maximum information gain."""
        candidates = []
        for i, pose in enumerate(poses_data[:n]):
            keypoint_confidences = pose.get('keypoint_confidences', [])
            if keypoint_confidences:
                variance = np.var(keypoint_confidences)
                entropy_score = min(1.0, variance * 2)
            else:
                entropy_score = 0.5

            candidate = CandidatePose(
                pose_id=pose.get('id', i),
                score=entropy_score,
                reason="High information content",
                file_path=pose.get('file_path', ''),
                is_priority=(entropy_score > 0.75)
            )
            candidates.append(candidate)
            self.progress_updated.emit(int((i + 1) / n * 100))

        candidates.sort()
        return candidates[:n]

    def _hybrid_sampling(self, poses_data: List[Dict], n: int) -> List[CandidatePose]:
        """Combine multiple strategies for balanced selection."""
        strategies_n = n // 3 + 1

        uncertainty = self._uncertainty_sampling(poses_data, strategies_n)
        diversity = self._diversity_sampling(poses_data, strategies_n)
        margin = self._margin_sampling(poses_data, strategies_n)

        all_candidates = uncertainty + diversity + margin
        seen_ids = set()
        candidates = []

        for candidate in all_candidates:
            if candidate.pose_id not in seen_ids:
                candidate.reason = "Balanced selection"
                candidates.append(candidate)
                seen_ids.add(candidate.pose_id)

                if len(candidates) >= n:
                    break

        return candidates[:n]


class TrainingIntelligencePanel(QDockWidget):
    """Main training intelligence panel widget."""

    # Signals
    candidate_selected = Signal(object)  # CandidatePose
    load_pose_requested = Signal(int)  # pose_id
    marked_as_done = Signal(int)  # pose_id
    # Legacy signal for backward compatibility
    load_candidate_requested = Signal(str)  # pose_id
    poseSelected = Signal(int)  # Backward compatibility - pose ID selected

    def __init__(self, training_intelligence=None, parent: Optional[QWidget] = None):
        super().__init__("Training Intelligence", parent)

        self.training_intelligence = training_intelligence
        self.stats = DatasetStats()
        self.candidates: List[CandidatePose] = []
        self.selected_candidate: Optional[CandidatePose] = None
        self.target_annotations = 100
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ActiveLearningWorker] = None

        self._setup_ui()
        # self._setup_worker()  # TODO: implement worker setup

        # Update bias status on initialization
        self._update_bias_status()

        if logger:
            logger.info("TrainingIntelligencePanel initialized")

    def _setup_ui(self):
        """Create the user interface layout."""
        # QDockWidget needs a central widget, not direct layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Dataset Statistics Section
        # This gives users a bird's eye view of their annotation progress
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("No statistics available")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self._refresh_statistics)
        stats_layout.addWidget(refresh_stats_btn)

        # Bias correction training button
        self.train_corrections_btn = QPushButton("Train from Corrections")
        self.train_corrections_btn.setToolTip(
            "Manually trigger training from corrections (auto-trains every 10 corrections)"
        )
        self.train_corrections_btn.clicked.connect(self._train_from_corrections)
        self.train_corrections_btn.setEnabled(False)  # Disabled until we have enough corrections
        stats_layout.addWidget(self.train_corrections_btn)

        # Bias status label
        self.bias_status_label = QLabel("Bias corrections: Not trained")
        self.bias_status_label.setWordWrap(True)
        self.bias_status_label.setStyleSheet("color: gray; font-size: 10pt;")
        stats_layout.addWidget(self.bias_status_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Candidate Search Section
        # This is where users configure and trigger the intelligent sampling
        search_group = QGroupBox("Find Training Candidates")
        search_layout = QVBoxLayout()

        # Strategy selection with helpful tooltips
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Diversity (Cover feature space)", SamplingStrategy.DIVERSITY)
        self.strategy_combo.addItem("Uncertainty (Hard examples)", SamplingStrategy.UNCERTAINTY)
        self.strategy_combo.addItem("Hybrid (Best of both)", SamplingStrategy.HYBRID)
        self.strategy_combo.addItem("Cluster Centers (Representatives)", SamplingStrategy.CLUSTER_CENTERS)
        self.strategy_combo.addItem("Decision Boundary (Refine edges)", SamplingStrategy.BOUNDARY)

        # Default to hybrid - it's the most generally useful
        self.strategy_combo.setCurrentIndex(2)

        # Add tooltip explaining the current strategy
        self.strategy_combo.currentIndexChanged.connect(self._update_strategy_tooltip)
        self._update_strategy_tooltip()

        strategy_layout.addWidget(self.strategy_combo)
        search_layout.addLayout(strategy_layout)

        # Number of candidates to find
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Number of candidates:"))

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(20)
        count_layout.addWidget(self.count_spin)

        search_layout.addLayout(count_layout)

        # Search button with progress bar
        self.search_btn = QPushButton("Find Candidates")
        self.search_btn.clicked.connect(self._start_candidate_search)
        search_layout.addWidget(self.search_btn)

        self.search_progress = QProgressBar()
        self.search_progress.setVisible(False)
        search_layout.addWidget(self.search_progress)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Results Section
        # Shows the ranked list of candidates with their priority scores
        results_group = QGroupBox("Recommended Poses")
        results_layout = QVBoxLayout()

        self.candidates_list = QListWidget()
        self.candidates_list.itemDoubleClicked.connect(self._on_candidate_selected)
        results_layout.addWidget(self.candidates_list)

        # Helper text
        help_label = QLabel("Double-click a pose to load it for annotation")
        help_label.setStyleSheet("color: gray; font-style: italic;")
        results_layout.addWidget(help_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Recommendations Section
        # Shows AI-generated advice based on current dataset state
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout()

        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(100)
        rec_layout.addWidget(self.recommendations_text)

        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Set stretch factors so results list gets most space
        layout.setStretch(2, 1)  # Search section
        layout.setStretch(3, 3)  # Results section gets 3x space

        # Set the central widget on the QDockWidget
        self.setWidget(central_widget)

    def _update_strategy_tooltip(self):
        """Update tooltip based on selected strategy."""
        # This helps users understand what each strategy does
        tooltips = {
            SamplingStrategy.DIVERSITY:
                "Selects poses that are far apart in feature space.\n"
                "Best for: Initial dataset building, exploring variety",

            SamplingStrategy.UNCERTAINTY:
                "Selects poses the classifier is least confident about.\n"
                "Best for: Refining model after initial training",

            SamplingStrategy.HYBRID:
                "Combines diversity and uncertainty.\n"
                "Best for: General purpose active learning",

            SamplingStrategy.CLUSTER_CENTERS:
                "Selects representative poses from each natural cluster.\n"
                "Best for: Understanding dataset structure",

            SamplingStrategy.BOUNDARY:
                "Selects poses near decision boundaries.\n"
                "Best for: Fine-tuning classification boundaries"
        }

        current_strategy = self.strategy_combo.currentData()
        self.strategy_combo.setToolTip(tooltips.get(current_strategy, ""))

    def _refresh_statistics(self):
        """Update dataset statistics display."""
        logger.info("Refreshing dataset statistics")

        try:
            # Get coverage analysis from training intelligence
            analysis = self.training_intelligence.analyze_dataset_coverage()

            # Format statistics for display
            stats_text = f"""
<b>Dataset Overview:</b><br>
- Total poses: {analysis['total_poses']}<br>
- Labeled: {analysis['labeled_poses']}<br>
- Unlabeled: {analysis['unlabeled_poses']}<br>
- Coverage: {analysis['coverage_percentage']:.1f}%<br>
<br>
<b>Category Distribution:</b><br>
"""

            # Show category breakdown
            for category, count in analysis['category_distribution'].items():
                pct = (count / analysis['labeled_poses'] * 100) if analysis['labeled_poses'] > 0 else 0
                stats_text += f"• {category}: {count} ({pct:.1f}%)<br>"

            # Highlight underrepresented categories
            if analysis['underrepresented_categories']:
                stats_text += f"<br><b style='color: orange;'>Underrepresented:</b> "
                stats_text += ", ".join(analysis['underrepresented_categories'])

            self.stats_label.setText(stats_text)

            # Update recommendations
            if analysis['recommendations']:
                rec_text = "<b>AI Recommendations:</b><br><br>"
                for i, rec in enumerate(analysis['recommendations'], 1):
                    rec_text += f"{i}. {rec}<br>"
                self.recommendations_text.setHtml(rec_text)
            else:
                self.recommendations_text.setHtml("<i>No recommendations at this time</i>")

            logger.info("Statistics refreshed successfully")

        except Exception as e:
            logger.error(f"Failed to refresh statistics: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Statistics Error",
                f"Failed to load statistics:\n{str(e)}"
            )

        # Always update bias status when refreshing statistics
        self._update_bias_status()

    def _start_candidate_search(self):
        """Start background search for training candidates."""
        logger.info("Starting candidate search")

        # Get search parameters
        n_candidates = self.count_spin.value()
        strategy = self.strategy_combo.currentData()

        # Disable controls during search
        self.search_btn.setEnabled(False)
        self.search_progress.setVisible(True)
        self.search_progress.setValue(0)

        # Create and start worker thread
        self.search_worker = CandidateSearchWorker(
            self.training_intelligence,
            n_candidates,
            strategy
        )

        # Connect signals
        self.search_worker.progress.connect(self._on_search_progress)
        self.search_worker.finished.connect(self._on_search_finished)
        self.search_worker.error.connect(self._on_search_error)

        # Start the search
        self.search_worker.start()

    def _on_search_progress(self, message: str, percentage: int):
        """Update progress during search."""
        self.search_progress.setValue(percentage)
        self.search_progress.setFormat(f"{message} ({percentage}%)")

    def _on_search_finished(self, candidates: list):
        """Handle search completion."""
        logger.info(f"Search completed with {len(candidates)} candidates")

        # Re-enable controls
        self.search_btn.setEnabled(True)
        self.search_progress.setVisible(False)

        # Store candidates
        self.current_candidates = candidates

        # Populate results list
        self.candidates_list.clear()

        for i, candidate in enumerate(candidates, 1):
            # Create informative list item
            # Format: "1. [Score: 0.95] Pose ABC123 - High diversity"
            item_text = (
                f"{i}. [Score: {candidate.priority_score:.2f}] "
                f"Pose {candidate.pose_id[:8]}... - {candidate.reasoning}"
            )

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, candidate.pose_id)

            # Color code by priority (high priority = green, low = yellow)
            if candidate.priority_score > 0.7:
                item.setForeground(Qt.GlobalColor.darkGreen)
            elif candidate.priority_score > 0.4:
                item.setForeground(Qt.GlobalColor.darkYellow)

            self.candidates_list.addItem(item)

        # Show success message
        QMessageBox.information(
            self,
            "Search Complete",
            f"Found {len(candidates)} training candidates.\n\n"
            f"Double-click any pose to load it for annotation."
        )

    def _on_search_error(self, error_message: str):
        """Handle search errors."""
        logger.error(f"Search error: {error_message}")

        # Re-enable controls
        self.search_btn.setEnabled(True)
        self.search_progress.setVisible(False)

        # Show error message
        QMessageBox.critical(
            self,
            "Search Failed",
            f"Failed to find training candidates:\n\n{error_message}"
        )

    def _on_candidate_selected(self, item: QListWidgetItem):
        """Handle user selecting a candidate to annotate."""
        pose_id = item.data(Qt.ItemDataRole.UserRole)
        logger.info(f"User selected candidate: {pose_id}")

        # Emit signal to main window to load this pose
        self.load_candidate_requested.emit(pose_id)

    def _train_from_corrections(self):
        """Train bias correction model from manual corrections."""
        try:
            # Get BiasCorrector from main window's pose_detector
            main_window = self.parent()
            if not hasattr(main_window, 'pose_detector'):
                QMessageBox.warning(
                    self,
                    "Training Failed",
                    "Could not access pose detector"
                )
                return

            bias_corrector = main_window.pose_detector.bias_corrector
            if bias_corrector is None:
                QMessageBox.warning(
                    self,
                    "Training Failed",
                    "Bias corrector not available"
                )
                return

            # Get correction count
            db_manager = main_window.db_manager
            correction_count = db_manager.storage.count_corrected_poses()

            if correction_count < 10:
                QMessageBox.warning(
                    self,
                    "Insufficient Data",
                    f"Need at least 10 corrections to train. Currently have {correction_count}."
                )
                return

            # Confirm training
            reply = QMessageBox.question(
                self,
                "Train Bias Correction",
                f"Train from {correction_count} manual corrections?\n\n"
                "This will update the bias correction model used for future pose detections.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Train model
            logger.info(f"Training bias correction model from {correction_count} corrections")
            stats = bias_corrector.train()

            # Show results
            QMessageBox.information(
                self,
                "Training Complete",
                f"Successfully trained bias correction model:\n\n"
                f"• Total corrections processed: {stats['total_corrections']}\n"
                f"• Biases learned: {stats['biases_learned']}\n"
                f"• Skipped (insufficient data): {stats['skipped_insufficient_data']}\n\n"
                "New detections will now use these learned corrections."
            )

            # Update status display
            self._update_bias_status()

            logger.info(f"Bias correction training complete: {stats}")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Training Failed",
                f"Failed to train bias correction model:\n\n{str(e)}"
            )

    def _update_bias_status(self):
        """Update bias correction status display."""
        try:
            # Get BiasCorrector from main window
            main_window = self.parent()
            if not hasattr(main_window, 'pose_detector'):
                return

            bias_corrector = main_window.pose_detector.bias_corrector
            db_manager = main_window.db_manager

            if bias_corrector is None:
                self.bias_status_label.setText("Bias corrections: Not available")
                self.bias_status_label.setStyleSheet("color: gray; font-size: 10pt;")
                self.train_corrections_btn.setEnabled(False)
                return

            # Get correction count
            correction_count = db_manager.storage.count_corrected_poses()

            # Get bias summary
            bias_summary = bias_corrector.get_bias_summary()
            num_biases = bias_summary['total_biases']

            # Update button state (but show that auto-training happens every 10)
            self.train_corrections_btn.setEnabled(correction_count >= 10)

            # Update status label
            if num_biases > 0:
                next_train = ((correction_count // 10) + 1) * 10  # Next multiple of 10
                status_text = (
                    f"Bias corrections: {num_biases} biases learned | "
                    f"{correction_count} corrections (next auto-train at {next_train}) | "
                    f"Avg confidence: {bias_summary['avg_confidence']:.2f}"
                )
                self.bias_status_label.setText(status_text)
                self.bias_status_label.setStyleSheet("color: green; font-size: 10pt;")
            else:
                status_text = f"Bias corrections: Not trained | {correction_count} corrections (auto-trains at 10)"
                self.bias_status_label.setText(status_text)
                if correction_count >= 10:
                    self.bias_status_label.setStyleSheet("color: orange; font-size: 10pt;")
                else:
                    self.bias_status_label.setStyleSheet("color: gray; font-size: 10pt;")

        except Exception as e:
            logger.error(f"Failed to update bias status: {e}", exc_info=True)
            self.bias_status_label.setText("Bias corrections: Error checking status")
            self.bias_status_label.setStyleSheet("color: red; font-size: 10pt;")
