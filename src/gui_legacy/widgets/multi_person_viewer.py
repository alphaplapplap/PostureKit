"""
Multi-person pose viewer with interactive selection.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage
import numpy as np
from typing import List, Optional, Tuple


class PersonNavigator(QWidget):
    """Simple navigation bar for switching between detected persons."""

    personChanged = Signal(int)  # Emits person index when selection changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_person = 0
        self._total_persons = 0
        self._setup_ui()

    def _setup_ui(self):
        """Setup the navigation UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Previous button
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._on_previous)
        layout.addWidget(self.prev_btn)

        # Person label
        self.label = QLabel("Person 0 / 0")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label, 1)

        # Next button
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._on_next)
        layout.addWidget(self.next_btn)

        self._update_buttons()

    def setCurrentPerson(self, current: int, total: int):
        """
        Set current person and total count.

        Args:
            current: Current person number (1-indexed)
            total: Total number of persons
        """
        self._current_person = current - 1  # Convert to 0-indexed
        self._total_persons = total
        self.label.setText(f"Person {current} / {total}")
        self._update_buttons()

    def _update_buttons(self):
        """Update button enabled state."""
        self.prev_btn.setEnabled(self._current_person > 0)
        self.next_btn.setEnabled(self._current_person < self._total_persons - 1)

    def _on_previous(self):
        """Handle previous button click."""
        if self._current_person > 0:
            self._current_person -= 1
            self.personChanged.emit(self._current_person)
            self.label.setText(f"Person {self._current_person + 1} / {self._total_persons}")
            self._update_buttons()

    def _on_next(self):
        """Handle next button click."""
        if self._current_person < self._total_persons - 1:
            self._current_person += 1
            self.personChanged.emit(self._current_person)
            self.label.setText(f"Person {self._current_person + 1} / {self._total_persons}")
            self._update_buttons()


class MultiPersonViewer(QWidget):
    """
    Displays image with multiple pose overlays.

    Features:
    - Shows all detected people with bounding boxes
    - Color-coded skeletons for each person
    - Click to select person
    - Hover highlights

    Signals:
        person_selected: Emitted when user clicks a person (person_index)
    """

    person_selected = Signal(int)
    image_dropped = Signal(object)  # Path
    open_image_requested = Signal()  # Request to open file dialog
    zoom_changed = Signal(float)  # Zoom level (1.0 = 100%)
    keypoint_moved = Signal(int, float, float)  # (keypoint_index, new_x, new_y)

    # Color palette for different people
    PERSON_COLORS = [
        QColor(0, 255, 0),      # Green
        QColor(0, 150, 255),    # Blue
        QColor(255, 165, 0),    # Orange
        QColor(255, 0, 255),    # Magenta
        QColor(0, 255, 255),    # Cyan
        QColor(255, 255, 0),    # Yellow
    ]

    # MMPose skeleton connections (COCO format)
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2),  # Head
        (1, 3), (2, 4),  # Face
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image: Optional[np.ndarray] = None
        self.people_data: List[Tuple] = []  # [(person_det, pose, viewpoint, features), ...]
        self.selected_person: int = 0
        self.hovered_person: Optional[int] = None

        self.pixmap: Optional[QPixmap] = None
        self.scale_factor = 1.0
        self.zoom_level = 1.0  # User-controlled zoom (separate from auto-fit)
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # Pan offset for dragging
        self.pan_offset = QPointF(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPointF(0, 0)

        # Keypoint editing
        self.selected_keypoint: Optional[int] = None
        self.hovered_keypoint: Optional[int] = None
        self.is_dragging_keypoint = False
        self.edit_mode = True  # Always allow keypoint editing

        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Set cursor to indicate clickability when empty
        self._update_cursor()

    def load_image_and_poses(
        self,
        image: np.ndarray,
        people_data: List[Tuple]
    ):
        """
        Load image with multiple people and their poses.

        Args:
            image: RGB image array (H, W, 3)
            people_data: List of (PersonDetection, PoseResult, ViewpointEstimate, GeometricFeatures)
        """
        self.image = image
        self.people_data = people_data
        self.selected_person = 0 if people_data else None

        # Convert to QPixmap with explicit cleanup of temporary QImage
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)

        # Explicit cleanup of temporary QImage to prevent memory accumulation
        del q_image

        # Calculate scale to fit widget
        self._update_scale()
        self._update_cursor()
        self.update()

    def _update_scale(self):
        """Calculate scale factor to fit image in widget."""
        if self.pixmap is None:
            return

        widget_width = self.width()
        widget_height = self.height()
        pixmap_width = self.pixmap.width()
        pixmap_height = self.pixmap.height()

        scale_w = widget_width / pixmap_width
        scale_h = widget_height / pixmap_height
        self.scale_factor = min(scale_w, scale_h, 1.0)  # Don't scale up

    def resizeEvent(self, event):
        """Handle widget resize."""
        self._update_scale()
        super().resizeEvent(event)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        """Convert image coordinates to widget coordinates."""
        return QPointF(
            x * self.scale_factor + self.pan_offset.x(),
            y * self.scale_factor + self.pan_offset.y()
        )

    def _widget_to_image(self, pos: QPointF) -> Tuple[float, float]:
        """Convert widget coordinates to image coordinates."""
        return (
            (pos.x() - self.pan_offset.x()) / self.scale_factor,
            (pos.y() - self.pan_offset.y()) / self.scale_factor
        )

    def _find_person_at(self, pos: QPointF) -> Optional[int]:
        """Find person index at widget position."""
        img_x, img_y = self._widget_to_image(pos)

        for i, (person_det, _, _, _) in enumerate(self.people_data):
            bbox = person_det.bbox  # [x, y, w, h]
            if (bbox[0] <= img_x <= bbox[0] + bbox[2] and
                bbox[1] <= img_y <= bbox[1] + bbox[3]):
                return i

        return None

    def _find_keypoint_at(self, pos: QPointF) -> Optional[int]:
        """Find keypoint index at widget position for selected person."""
        if not self.people_data or self.selected_person >= len(self.people_data):
            return None

        _, pose_result, _, _ = self.people_data[self.selected_person]
        keypoints = pose_result.keypoints

        img_x, img_y = self._widget_to_image(pos)

        # Hit radius scales with zoom for easier selection
        hit_radius = max(8 / self.scale_factor, 5)

        for i in range(len(keypoints)):
            kp_x, kp_y, conf = keypoints[i]
            if conf < 0.3:
                continue

            dx = img_x - kp_x
            dy = img_y - kp_y
            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= hit_radius:
                return i

        return None

    def mousePressEvent(self, event):
        """Handle mouse press - select keypoint, person, or start pan."""
        if event.button() == Qt.MouseButton.LeftButton:
            # If no image loaded, clicking opens file dialog
            if self.pixmap is None:
                self.open_image_requested.emit()
                return

            # Check for keypoint click first (higher priority)
            keypoint_idx = self._find_keypoint_at(event.position())
            if keypoint_idx is not None and self.edit_mode:
                self.selected_keypoint = keypoint_idx
                self.is_dragging_keypoint = True
                self._update_cursor()
                self.update()
                return

            # Then check for person selection
            person_idx = self._find_person_at(event.position())
            if person_idx is not None:
                self.selected_person = person_idx
                self.person_selected.emit(person_idx)
                self.selected_keypoint = None
                self.update()

        # Middle button or Space+Left for panning
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_pan_pos = event.position()
            self._update_cursor()

    def mouseReleaseEvent(self, event):
        """Handle mouse release - stop panning or keypoint dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_dragging_keypoint:
                self.is_dragging_keypoint = False
                self._update_cursor()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self._update_cursor()

    def mouseMoveEvent(self, event):
        """Handle mouse move - drag keypoint, pan, or update hover."""
        if self.is_dragging_keypoint and self.selected_keypoint is not None:
            # Drag keypoint
            if self.people_data and self.selected_person < len(self.people_data):
                img_x, img_y = self._widget_to_image(event.position())

                # Update keypoint position
                _, pose_result, _, _ = self.people_data[self.selected_person]
                pose_result.keypoints[self.selected_keypoint, 0] = img_x
                pose_result.keypoints[self.selected_keypoint, 1] = img_y

                # Emit signal
                self.keypoint_moved.emit(self.selected_keypoint, img_x, img_y)
                self.update()

        elif self.is_panning:
            # Pan the view
            delta = event.position() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.position()
            self.update()
        else:
            # Update hover states
            self.hovered_person = self._find_person_at(event.position())
            self.hovered_keypoint = self._find_keypoint_at(event.position())
            self._update_cursor()
            self.update()

    def paintEvent(self, event):
        """Draw image with all pose overlays."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Show empty state if no image loaded
            if self.pixmap is None:
                self._draw_empty_state(painter)
                return

            # Draw image with pan offset
            scaled_width = int(self.pixmap.width() * self.scale_factor)
            scaled_height = int(self.pixmap.height() * self.scale_factor)

            # Apply pan offset
            x_offset = int(self.pan_offset.x())
            y_offset = int(self.pan_offset.y())

            painter.drawPixmap(x_offset, y_offset, scaled_width, scaled_height, self.pixmap)

            # Draw each person
            for i, (person_det, pose_result, _, _) in enumerate(self.people_data):
                color = self.PERSON_COLORS[i % len(self.PERSON_COLORS)]

                # Determine alpha based on selection/hover
                if i == self.selected_person:
                    alpha = 255  # Fully opaque
                    bbox_width = 3
                elif i == self.hovered_person:
                    alpha = 200  # Slightly transparent
                    bbox_width = 2
                else:
                    alpha = 120  # More transparent
                    bbox_width = 1

                color.setAlpha(alpha)

                # Draw bounding box with pan offset
                bbox = person_det.bbox
                bbox_rect = QRectF(
                    bbox[0] * self.scale_factor + x_offset,
                    bbox[1] * self.scale_factor + y_offset,
                    bbox[2] * self.scale_factor,
                    bbox[3] * self.scale_factor
                )
                painter.setPen(QPen(color, bbox_width, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(bbox_rect)

                # Draw person label
                label = f"Person {i + 1}"
                painter.setPen(QPen(color, 1))
                painter.drawText(
                    int(bbox[0] * self.scale_factor + x_offset),
                    int(bbox[1] * self.scale_factor + y_offset - 5),
                    label
                )

                # Draw skeleton
                is_selected = (i == self.selected_person)
                self._draw_skeleton(painter, pose_result.keypoints, color, is_selected)
        finally:
            painter.end()

    def _draw_skeleton(self, painter, keypoints, color, is_selected_person=False):
        """Draw pose skeleton."""
        # Scale line width and keypoint radius with zoom
        line_width = max(2 * self.scale_factor, 1.0)
        keypoint_radius = max(4 * self.scale_factor, 2.0)
        selected_radius = max(8 * self.scale_factor, 4.0)
        hovered_radius = max(6 * self.scale_factor, 3.0)

        # Draw connections
        for start_idx, end_idx in self.SKELETON_CONNECTIONS:
            if (start_idx >= len(keypoints) or end_idx >= len(keypoints)):
                continue

            start_conf = keypoints[start_idx, 2]
            end_conf = keypoints[end_idx, 2]

            if start_conf < 0.3 or end_conf < 0.3:
                continue  # Skip low-confidence connections

            start_pos = self._image_to_widget(keypoints[start_idx, 0], keypoints[start_idx, 1])
            end_pos = self._image_to_widget(keypoints[end_idx, 0], keypoints[end_idx, 1])

            painter.setPen(QPen(color, line_width))
            painter.drawLine(start_pos, end_pos)

        # Draw keypoints
        for i in range(len(keypoints)):
            conf = keypoints[i, 2]
            if conf < 0.3:
                continue

            pos = self._image_to_widget(keypoints[i, 0], keypoints[i, 1])

            # Determine color and radius
            if is_selected_person and i == self.selected_keypoint:
                # Selected keypoint - bright white with outline
                kp_color = QColor(255, 255, 255)
                radius = selected_radius
                painter.setPen(QPen(QColor(0, 0, 0), 2))  # Black outline
                painter.setBrush(QBrush(kp_color))
                painter.drawEllipse(pos, radius, radius)
            elif is_selected_person and i == self.hovered_keypoint:
                # Hovered keypoint - yellow highlight
                kp_color = QColor(255, 255, 0)
                radius = hovered_radius
                painter.setPen(QPen(kp_color, 2))
                painter.setBrush(QBrush(kp_color))
                painter.drawEllipse(pos, radius, radius)
            else:
                # Normal keypoint - color by confidence
                if conf > 0.7:
                    kp_color = color
                elif conf > 0.5:
                    kp_color = QColor(255, 255, 0)  # Yellow for medium
                else:
                    kp_color = QColor(255, 165, 0)  # Orange for low

                painter.setPen(QPen(kp_color, 1))
                painter.setBrush(QBrush(kp_color))
                painter.drawEllipse(pos, keypoint_radius, keypoint_radius)

    def _draw_empty_state(self, painter):
        """Draw empty state with drop zone."""
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Calculate center
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Draw icon
        icon_font = painter.font()
        icon_font.setPointSize(64)
        painter.setFont(icon_font)
        painter.setPen(QColor(100, 100, 100))

        icon_rect = QRectF(center_x - 50, center_y - 120, 100, 80)
        painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, "🖼️")

        # Draw main text
        main_font = painter.font()
        main_font.setPointSize(18)
        main_font.setBold(True)
        painter.setFont(main_font)
        painter.setPen(QColor(150, 150, 150))

        text_rect = QRectF(center_x - 200, center_y - 20, 400, 40)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "Drop image here or click Open")

        # Draw hint text
        hint_font = painter.font()
        hint_font.setPointSize(12)
        hint_font.setBold(False)
        painter.setFont(hint_font)
        painter.setPen(QColor(100, 100, 100))

        hint_rect = QRectF(center_x - 200, center_y + 30, 400, 30)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter, "Supports: JPG, PNG, BMP, TIFF, WebP")

        # Draw dashed border (drop zone indicator)
        pen = QPen(QColor(80, 80, 80), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        margin = 40
        drop_rect = QRectF(
            margin,
            margin,
            self.width() - 2 * margin,
            self.height() - 2 * margin
        )
        painter.drawRoundedRect(drop_rect, 10, 10)

    def dragEnterEvent(self, event):
        """Handle drag enter - accept image files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle file drop."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()

            # Emit signal to parent to load the image
            from pathlib import Path
            self.image_dropped.emit(Path(file_path))

    def _update_cursor(self):
        """Update cursor based on state."""
        if self.pixmap is None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.is_dragging_keypoint:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self.is_panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self.hovered_keypoint is not None:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming toward cursor."""
        if self.pixmap is None:
            return

        # Get mouse position
        mouse_pos = event.position()

        # Get zoom delta (reduced sensitivity)
        delta = event.angleDelta().y()
        zoom_factor = 1.05 if delta > 0 else 0.95

        # Calculate new zoom
        old_zoom = self.zoom_level
        new_zoom = np.clip(
            old_zoom * zoom_factor,
            self.min_zoom,
            self.max_zoom
        )

        if new_zoom == old_zoom:
            return  # No change

        # Convert mouse position to image coordinates BEFORE zoom
        img_x = (mouse_pos.x() - self.pan_offset.x()) / old_zoom
        img_y = (mouse_pos.y() - self.pan_offset.y()) / old_zoom

        # Update zoom
        self.zoom_level = new_zoom
        self.scale_factor = new_zoom

        # Adjust pan offset to keep mouse position fixed in image space
        # new_mouse_screen = img_pos * new_zoom + new_pan
        # We want: new_mouse_screen = mouse_pos (keep mouse at same screen position)
        # So: new_pan = mouse_pos - img_pos * new_zoom
        self.pan_offset.setX(mouse_pos.x() - img_x * new_zoom)
        self.pan_offset.setY(mouse_pos.y() - img_y * new_zoom)

        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom_in(self):
        """Zoom in (Ctrl++)."""
        if self.pixmap is None:
            return

        self.zoom_level = np.clip(self.zoom_level * 1.2, self.min_zoom, self.max_zoom)
        self.scale_factor = self.zoom_level
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom_out(self):
        """Zoom out (Ctrl+-)."""
        if self.pixmap is None:
            return

        self.zoom_level = np.clip(self.zoom_level / 1.2, self.min_zoom, self.max_zoom)
        self.scale_factor = self.zoom_level
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def zoom_to_fit(self):
        """Fit image to window (Ctrl+0)."""
        if self.pixmap is None:
            return

        self._update_scale()
        self.zoom_level = self.scale_factor
        self.pan_offset = QPointF(0, 0)
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def reset_view(self):
        """Reset to 100% zoom (Ctrl+R)."""
        if self.pixmap is None:
            return

        self.zoom_level = 1.0
        self.scale_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.zoom_changed.emit(self.zoom_level)
        self.update()
