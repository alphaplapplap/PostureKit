"""
OpenGL-accelerated image viewer with skeleton overlay and editing capabilities
Complete implementation based on UI/UX requirements specification
"""

import os
import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QMenu, QApplication, QToolTip
)
from PySide6.QtCore import (
    Qt, Signal, QPoint, QPointF, QRect, QRectF, QTimer,
    QSize, Slot, QUrl, QEvent
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    QWheelEvent, QMouseEvent, QKeyEvent, QPaintEvent,
    QContextMenuEvent, QDragEnterEvent, QDropEvent,
    QTransform, QPolygonF, QLinearGradient, QRadialGradient,
    QFontMetrics, QPainterPath, QAction, QNativeGestureEvent
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer

from src.core.models import Person, Keypoint, KEYPOINT_NAMES, SKELETON_CONNECTIONS


class ViewerState:
    """Viewer state management"""
    def __init__(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.rotation = 0.0
        self.fit_mode = False

    def reset(self):
        """Reset to default view"""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.rotation = 0.0
        self.fit_mode = False


class ImageViewer(QOpenGLWidget):
    """OpenGL-accelerated image viewer with pose skeleton overlay"""

    # Signals
    keypointMoved = Signal(int, int, float, float)  # person_idx, kp_idx, new_x, new_y
    keypointVisibilityChanged = Signal(int, int, int)  # person_idx, kp_idx, visibility
    personSelected = Signal(int)  # person_idx
    zoomChanged = Signal(float)  # zoom_level
    keypointSelected = Signal(int, int)  # person_idx, kp_idx
    modificationMade = Signal()  # Any modification
    imageDropped = Signal(str)  # file_path

    def __init__(self, parent=None):
        super().__init__(parent)

        # Image and pose data
        self.image: Optional[QImage] = None
        self.image_path: Optional[str] = None
        self.persons: List[Person] = []
        self.selected_person_idx = -1
        self.selected_keypoint_idx = -1
        self.selected_keypoint_indices = []  # List of keypoint indices for multi-select
        self.hovered_keypoint = None  # (person_idx, kp_idx)

        # Display options
        self.show_skeleton = True
        self.show_keypoints = True
        self.show_bboxes = True
        self.show_confidence_colors = True
        self.use_colored_parts = False
        self.show_occlusion_heatmap = False
        self.keypoint_radius = 6
        self.line_width = 2.5
        self.bbox_line_width = 2

        # Occlusion display filters
        self.show_visible_keypoints = True
        self.show_occluded_keypoints = True
        self.show_missing_keypoints = False

        # View state
        self.view_state = ViewerState()

        # Interaction state
        self.is_panning = False
        self.is_dragging_keypoint = False
        self.drag_start_pos = QPoint()
        self.last_mouse_pos = QPoint()
        self.mouse_world_pos = QPointF()

        # Performance
        self.use_antialiasing = True
        self.target_fps = 60
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update)

        # Skeleton connections organized by body part
        self.body_connections = [
            # Head connections
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arms
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # Torso
            (5, 11), (6, 12), (11, 12),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        self.foot_connections = [
            (15, 17), (15, 18), (15, 19),  # Left foot
            (16, 20), (16, 21), (16, 22)   # Right foot
        ]

        # Hand connections (simplified)
        self.hand_connections = self.generateHandConnections()

        # Face mesh (simplified for performance)
        self.face_connections = self.generateFaceConnections()

        # Person colors for multi-person visualization
        self.person_colors = [
            QColor(255, 77, 77),    # Red
            QColor(77, 255, 77),    # Green
            QColor(77, 144, 255),   # Blue
            QColor(255, 255, 77),   # Yellow
            QColor(255, 77, 255),   # Magenta
            QColor(77, 255, 255),   # Cyan
            QColor(255, 144, 77),   # Orange
            QColor(144, 77, 255),   # Purple
            QColor(144, 255, 77),   # Lime
            QColor(255, 77, 144)    # Pink
        ]

        # Anatomical part colors
        self.part_colors = {
            'head': QColor(156, 39, 176),    # Purple
            'torso': QColor(74, 144, 226),   # Blue
            'left_arm': QColor(76, 175, 80),  # Green
            'right_arm': QColor(255, 152, 0), # Orange
            'left_leg': QColor(0, 188, 212),  # Cyan
            'right_leg': QColor(255, 235, 59), # Yellow
            'left_hand': QColor(139, 195, 74), # Light Green
            'right_hand': QColor(255, 183, 77), # Light Orange
            'feet': QColor(233, 30, 99)       # Pink
        }

        # Initialize settings
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CrossCursor)

        # Enable native gesture events (macOS trackpad)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        # OpenGL settings will be initialized in initializeGL

    def initializeGL(self):
        """Initialize OpenGL settings"""
        # This would contain actual OpenGL initialization
        # For this implementation, we'll use QPainter which handles OpenGL internally
        pass

    def paintGL(self):
        """Main OpenGL rendering method using QPainter"""
        print(f"[DEBUG] paintGL called, zoom={self.view_state.zoom:.2f}, pan=({self.view_state.pan_x:.1f}, {self.view_state.pan_y:.1f})")
        painter = QPainter(self)
        try:
            # Set rendering hints
            if self.use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                painter.setRenderHint(QPainter.TextAntialiasing)

            # Clear background with gradient
            self.drawBackground(painter)

            # Apply view transformations
            painter.save()
            self.applyViewTransform(painter)

            # Draw image if loaded
            if self.image:
                self.drawImage(painter)

                # Draw occlusion heatmap if enabled
                if self.show_occlusion_heatmap:
                    self.drawOcclusionHeatmap(painter)

                # Draw pose overlays
                if self.persons:
                    for i, person in enumerate(self.persons):
                        is_selected = (i == self.selected_person_idx)
                        self.drawPerson(painter, person, i, is_selected)

            painter.restore()

            # Draw UI overlays and empty state (not affected by view transform)
            if not self.image:
                self.drawEmptyState(painter)

            self.drawOverlays(painter)
        finally:
            painter.end()

    def drawBackground(self, painter):
        """Draw viewer background with gradient"""
        rect = self.rect()
        gradient = QRadialGradient(rect.center(), max(rect.width(), rect.height()) / 2)
        gradient.setColorAt(0, QColor(25, 30, 40))
        gradient.setColorAt(1, QColor(15, 20, 25))
        painter.fillRect(rect, gradient)

        # Draw grid pattern for reference
        if not self.image:
            painter.setPen(QPen(QColor(30, 35, 45), 1, Qt.DotLine))
            grid_size = 50

            for x in range(0, rect.width(), grid_size):
                painter.drawLine(x, 0, x, rect.height())
            for y in range(0, rect.height(), grid_size):
                painter.drawLine(0, y, rect.width(), y)

    def applyViewTransform(self, painter):
        """Apply zoom, pan, and rotation transformations"""
        center = self.rect().center()

        # Translate to center, apply transformations, translate back
        painter.translate(center.x() + self.view_state.pan_x,
                         center.y() + self.view_state.pan_y)
        painter.scale(self.view_state.zoom, self.view_state.zoom)
        painter.rotate(self.view_state.rotation)

        if self.image:
            painter.translate(-self.image.width() / 2, -self.image.height() / 2)

    def drawImage(self, painter):
        """Draw the loaded image"""
        if not self.image:
            return

        painter.drawImage(0, 0, self.image)

        # Draw subtle border
        painter.setPen(QPen(QColor(50, 50, 50), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, self.image.width(), self.image.height())

    def drawEmptyState(self, painter):
        """Draw placeholder when no image is loaded"""
        # Use screen coordinates (widget center) instead of image coordinates
        center_x = self.width() / 2
        center_y = self.height() / 2
        rect = QRectF(center_x - 200, center_y - 150, 400, 300)

        # Draw dashed border
        pen = QPen(QColor(100, 100, 100), 3, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(30, 35, 45, 100))
        painter.drawRoundedRect(rect, 20, 20)

        # Draw drop icon
        drop_icon = QRectF(rect.center().x() - 40, rect.center().y() - 60, 80, 80)
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        painter.drawEllipse(drop_icon)

        # Arrow pointing down
        arrow = QPolygonF([
            QPointF(rect.center().x(), rect.center().y() - 20),
            QPointF(rect.center().x() - 15, rect.center().y() - 35),
            QPointF(rect.center().x() + 15, rect.center().y() - 35)
        ])
        painter.setBrush(QColor(150, 150, 150))
        painter.drawPolygon(arrow)

        # Draw text
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Arial", 14, QFont.Normal))
        painter.drawText(rect.adjusted(0, 60, 0, 0), Qt.AlignCenter,
                        "Drop an image here\nor use File → Open Image")

        painter.setFont(QFont("Arial", 11, QFont.Normal))
        painter.setPen(QColor(120, 120, 120))
        painter.drawText(rect.adjusted(0, 100, 0, 0), Qt.AlignCenter,
                        "Ctrl+O")

    def drawPerson(self, painter, person, person_idx, is_selected):
        """Draw a single person's pose"""
        color = self.person_colors[person_idx % len(self.person_colors)]

        # Adjust opacity for non-selected persons
        if not is_selected and self.selected_person_idx >= 0:
            painter.setOpacity(0.5)

        # Draw bounding box
        if self.show_bboxes:
            self.drawBoundingBox(painter, person, color, is_selected, person_idx)

        # Draw skeleton
        if self.show_skeleton:
            self.drawSkeleton(painter, person, color, is_selected)

        # Draw keypoints
        if self.show_keypoints:
            self.drawKeypoints(painter, person, person_idx, color, is_selected)

        painter.setOpacity(1.0)

    def drawBoundingBox(self, painter, person, color, is_selected, person_idx):
        """Draw bounding box around detected person"""
        x, y, w, h = person.bbox

        # Create gradient for selected box
        if is_selected:
            pen = QPen(color, 3, Qt.SolidLine)
            # Add glow effect
            glow_pen = QPen(color.lighter(150), 6)
            glow_pen.setColor(QColor(color.red(), color.green(), color.blue(), 50))
            painter.setPen(glow_pen)
            painter.drawRect(x-2, y-2, w+4, h+4)
        else:
            pen = QPen(color, 2, Qt.DashLine)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x, y, w, h)

        # Draw corner handles for selected person
        if is_selected:
            handle_size = 8
            painter.setBrush(color)
            corners = [
                (x, y), (x + w, y),
                (x, y + h), (x + w, y + h)
            ]
            for cx, cy in corners:
                painter.drawEllipse(QPoint(cx, cy), handle_size, handle_size)

        # Draw person label with background
        label = f"Person {person_idx + 1}"
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)

        fm = QFontMetrics(font)
        text_rect = fm.boundingRect(label)
        label_rect = QRect(x, y - 25, text_rect.width() + 10, 20)

        # Semi-transparent background
        painter.fillRect(label_rect, QColor(0, 0, 0, 180))

        # Draw text
        painter.setPen(Qt.white)
        painter.drawText(label_rect, Qt.AlignCenter, label)

        # Draw confidence badge
        conf_text = f"{person.confidence:.2f}"
        conf_rect = QRect(x + w - 40, y - 25, 40, 20)

        # Color based on confidence
        if person.confidence > 0.7:
            conf_color = QColor(16, 185, 129)
        elif person.confidence > 0.4:
            conf_color = QColor(251, 191, 36)
        else:
            conf_color = QColor(239, 68, 68)

        painter.fillRect(conf_rect, conf_color)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 9))
        painter.drawText(conf_rect, Qt.AlignCenter, conf_text)

    def drawSkeleton(self, painter, person, color, is_selected):
        """Draw skeleton connections between keypoints"""
        base_width = self.line_width * (1.5 if is_selected else 1.0)

        # Draw body connections
        self.drawConnections(painter, person, self.body_connections,
                           color, base_width, 'body')

        # Draw foot connections
        self.drawConnections(painter, person, self.foot_connections,
                           color, base_width * 0.8, 'feet')

        # Draw hand connections if visible
        if len(person.keypoints) > 91 and person.keypoints[91].visibility > 0:  # Left hand wrist
            self.drawConnections(painter, person,
                               [(i, i+1) for i in range(91, 111)],
                               color, base_width * 0.6, 'left_hand')

        if len(person.keypoints) > 112 and person.keypoints[112].visibility > 0:  # Right hand wrist
            self.drawConnections(painter, person,
                               [(i, i+1) for i in range(112, 132)],
                               color, base_width * 0.6, 'right_hand')

        # Draw face connections if enough face points are visible
        visible_face = sum(1 for i in range(23, 91)
                          if i < len(person.keypoints) and person.keypoints[i].visibility > 0)
        if visible_face > 10:
            self.drawFaceMesh(painter, person, color, base_width * 0.4)

    def drawConnections(self, painter, person, connections, color, width, part_name=''):
        """Draw skeleton connections with confidence-based coloring"""
        for start_idx, end_idx in connections:
            if start_idx >= len(person.keypoints) or end_idx >= len(person.keypoints):
                continue

            kp1 = person.keypoints[start_idx]
            kp2 = person.keypoints[end_idx]

            # Skip based on visibility and display settings
            # visibility: 0=missing, 1=occluded, 2=visible
            if kp1.visibility == 0 and not self.show_missing_keypoints:
                continue
            if kp2.visibility == 0 and not self.show_missing_keypoints:
                continue
            if kp1.visibility == 1 and not self.show_occluded_keypoints:
                continue
            if kp2.visibility == 1 and not self.show_occluded_keypoints:
                continue
            if kp1.visibility == 2 and not self.show_visible_keypoints:
                continue
            if kp2.visibility == 2 and not self.show_visible_keypoints:
                continue

            # Determine line color
            if self.show_confidence_colors:
                avg_conf = (kp1.confidence + kp2.confidence) / 2
                line_color = self.getConfidenceColor(avg_conf)
            elif self.use_colored_parts and part_name in self.part_colors:
                line_color = self.part_colors[part_name]
            else:
                line_color = color

            # Create gradient line for visual appeal
            # Apply 40% opacity if either keypoint is occluded
            if kp1.visibility == 1 or kp2.visibility == 1:
                # Semi-transparent colors for occluded connections
                start_color = QColor(line_color.red(), line_color.green(), line_color.blue(), 102)
                end_color = QColor(line_color.darker(120).red(),
                                  line_color.darker(120).green(),
                                  line_color.darker(120).blue(), 102)
            else:
                start_color = line_color
                end_color = line_color.darker(120)

            gradient = QLinearGradient(kp1.x, kp1.y, kp2.x, kp2.y)
            gradient.setColorAt(0, start_color)
            gradient.setColorAt(1, end_color)

            pen = QPen(QBrush(gradient), width)
            pen.setCapStyle(Qt.RoundCap)

            # Dotted line for occluded connections
            if kp1.visibility == 1 or kp2.visibility == 1:
                pen.setStyle(Qt.DotLine)

            painter.setPen(pen)
            painter.drawLine(QPointF(kp1.x, kp1.y), QPointF(kp2.x, kp2.y))

    def drawKeypoints(self, painter, person, person_idx, color, is_selected):
        """Draw individual keypoints with interactive features"""
        for kp_idx, kp in enumerate(person.keypoints):
            # Filter based on visibility and display settings
            # visibility: 0=missing, 1=occluded, 2=visible
            if kp.visibility == 0 and not self.show_missing_keypoints:
                continue
            if kp.visibility == 1 and not self.show_occluded_keypoints:
                continue
            if kp.visibility == 2 and not self.show_visible_keypoints:
                continue

            # Check if this keypoint is hovered or selected
            is_hovered = (self.hovered_keypoint == (person_idx, kp_idx))
            is_selected_kp = (is_selected and (kp_idx == self.selected_keypoint_idx or kp_idx in self.selected_keypoint_indices))

            # Determine keypoint color
            if self.show_confidence_colors:
                kp_color = self.getConfidenceColor(kp.confidence)
            elif self.use_colored_parts:
                kp_color = self.getPartColor(kp_idx)
            else:
                kp_color = color

            # Calculate radius with hover/selection effects
            radius = self.keypoint_radius
            if is_selected_kp:
                radius *= 1.8
            elif is_hovered:
                radius *= 1.5

            # Draw glow effect for selected/hovered keypoints
            if is_selected_kp or is_hovered:
                glow_radius = radius * 2
                gradient = QRadialGradient(kp.x, kp.y, glow_radius)
                gradient.setColorAt(0, QColor(kp_color.red(), kp_color.green(),
                                            kp_color.blue(), 100))
                gradient.setColorAt(1, Qt.transparent)
                painter.setBrush(gradient)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(kp.x, kp.y), glow_radius, glow_radius)

            # Draw keypoint circle with different styles based on visibility
            if kp.visibility == 0:  # Missing - hollow with dotted border
                painter.setPen(QPen(QColor(128, 128, 128), 2, Qt.DotLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(kp.x, kp.y), radius, radius)
            elif kp.visibility == 1:  # Occluded - 40% opacity with patterned fill
                # Create semi-transparent color (40% opacity = 102/255)
                occluded_color = QColor(kp_color.red(), kp_color.green(), kp_color.blue(), 102)
                painter.setPen(QPen(QColor(255, 255, 255, 102), 2))  # Semi-transparent white border
                painter.setBrush(QBrush(occluded_color, Qt.Dense4Pattern))
                painter.drawEllipse(QPointF(kp.x, kp.y), radius, radius)
            else:  # Visible - solid fill
                painter.setPen(QPen(Qt.white, 2))
                painter.setBrush(QBrush(kp_color))
                painter.drawEllipse(QPointF(kp.x, kp.y), radius, radius)

            # Draw correction indicator
            if kp.is_corrected:
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(kp.x, kp.y), radius + 3, radius + 3)

            # Draw keypoint index for debugging (optional)
            if is_selected_kp and self.isVisible():
                painter.setPen(Qt.white)
                painter.setFont(QFont("Arial", 8))
                painter.drawText(kp.x + radius + 5, kp.y - radius,
                               f"{kp_idx}: {KEYPOINT_NAMES.get(kp_idx, 'Unknown')}")

    def drawFaceMesh(self, painter, person, color, width):
        """Draw simplified face mesh"""
        # This is simplified for performance
        # Full implementation would draw all 68 face landmarks
        face_points = []
        for i in range(23, min(91, len(person.keypoints))):
            kp = person.keypoints[i]
            if kp.visibility > 0:
                face_points.append(QPointF(kp.x, kp.y))

        if len(face_points) > 3:
            painter.setPen(QPen(color, width, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            # Draw convex hull or simplified mesh
            # This is a placeholder for actual face mesh rendering

    def drawOcclusionHeatmap(self, painter):
        """Draw occlusion heatmap overlay"""
        if not self.persons:
            return

        # Create heatmap based on occluded keypoints
        painter.setOpacity(0.3)

        for person in self.persons:
            for kp in person.keypoints:
                if kp.visibility == 1:  # Occluded
                    # Draw heat spot
                    gradient = QRadialGradient(kp.x, kp.y, 50)
                    gradient.setColorAt(0, QColor(255, 0, 0, 100))
                    gradient.setColorAt(1, Qt.transparent)
                    painter.setBrush(gradient)
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(QPointF(kp.x, kp.y), 50, 50)

        painter.setOpacity(1.0)

    def drawOverlays(self, painter):
        """Draw UI overlays (zoom indicator, etc.)"""
        # Draw zoom indicator
        if abs(self.view_state.zoom - 1.0) > 0.01:
            zoom_text = f"Zoom: {int(self.view_state.zoom * 100)}%"
            font = QFont("Arial", 11, QFont.Bold)
            painter.setFont(font)

            fm = QFontMetrics(font)
            text_rect = fm.boundingRect(zoom_text)

            # Position in top-right corner
            x = self.width() - text_rect.width() - 20
            y = 20

            # Background
            bg_rect = QRect(x - 10, y, text_rect.width() + 20, text_rect.height() + 10)
            painter.fillRect(bg_rect, QColor(0, 0, 0, 150))

            # Text
            painter.setPen(Qt.white)
            painter.drawText(x, y + text_rect.height(), zoom_text)

        # Draw selection info
        if self.selected_person_idx >= 0:
            # Multi-select display
            if len(self.selected_keypoint_indices) > 1:
                info_text = f"Selected {len(self.selected_keypoint_indices)} keypoints"

                painter.setFont(QFont("Arial", 10))
                painter.setPen(Qt.white)

                # Draw in bottom-left corner
                lines = [info_text]
                y = self.height() - 20 - (len(lines) * 15)

                # Background
                bg_rect = QRect(10, y - 5, 250, len(lines) * 15 + 10)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))

                # Draw text
                painter.drawText(15, y + 15, info_text)

            # Single keypoint display
            elif self.selected_keypoint_idx >= 0:
                person = self.persons[self.selected_person_idx]
                kp = person.keypoints[self.selected_keypoint_idx]

                info_text = f"Keypoint {self.selected_keypoint_idx}: {KEYPOINT_NAMES.get(self.selected_keypoint_idx, 'Unknown')}"
                info_text += f"\nConfidence: {kp.confidence:.3f}"
                info_text += f"\nPosition: ({int(kp.x)}, {int(kp.y)})"
                info_text += f"\nVisibility: {'Visible' if kp.visibility == 2 else 'Occluded' if kp.visibility == 1 else 'Missing'}"

                painter.setFont(QFont("Arial", 10))
                painter.setPen(Qt.white)

                # Draw in bottom-left corner
                lines = info_text.split('\n')
                y = self.height() - 20 - (len(lines) * 15)

                # Background
                bg_rect = QRect(10, y - 5, 250, len(lines) * 15 + 10)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))

                # Draw each line
                for i, line in enumerate(lines):
                    painter.drawText(15, y + (i + 1) * 15, line)

    def getConfidenceColor(self, confidence):
        """Get color based on confidence value"""
        if confidence > 0.7:
            return QColor(16, 185, 129)   # Green
        elif confidence > 0.4:
            return QColor(251, 191, 36)   # Yellow
        else:
            return QColor(239, 68, 68)    # Red

    def getPartColor(self, keypoint_idx):
        """Get anatomical part color for keypoint"""
        if keypoint_idx < 5:
            return self.part_colors['head']
        elif keypoint_idx in [5, 6, 11, 12]:
            return self.part_colors['torso']
        elif keypoint_idx in [7, 9]:
            return self.part_colors['left_arm']
        elif keypoint_idx in [8, 10]:
            return self.part_colors['right_arm']
        elif keypoint_idx in [13, 15]:
            return self.part_colors['left_leg']
        elif keypoint_idx in [14, 16]:
            return self.part_colors['right_leg']
        elif 91 <= keypoint_idx < 112:
            return self.part_colors['left_hand']
        elif 112 <= keypoint_idx < 133:
            return self.part_colors['right_hand']
        elif 17 <= keypoint_idx < 23:
            return self.part_colors['feet']
        else:
            return self.part_colors['head']  # Face points

    def generateHandConnections(self):
        """Generate hand skeleton connections"""
        connections = []
        # Simplified hand connections
        # Would be more complex for full hand skeleton
        for hand_start in [91, 112]:  # Left and right hand
            for finger in range(5):
                base = hand_start + finger * 4
                for i in range(3):
                    connections.append((base + i, base + i + 1))
        return connections

    def generateFaceConnections(self):
        """Generate face mesh connections"""
        # Simplified face connections
        # Full implementation would have detailed face mesh
        return []

    # ============================================
    # Native Gesture Event Handlers (macOS trackpad)
    # ============================================

    def event(self, event: QEvent) -> bool:
        """Handle native gesture events for trackpad support"""
        if event.type() == QEvent.NativeGesture:
            return self.nativeGestureEvent(event)
        return super().event(event)

    def nativeGestureEvent(self, event: QNativeGestureEvent) -> bool:
        """Handle native trackpad gestures (pinch, pan, rotate)"""
        gesture_type = event.gestureType()

        # Pinch to zoom (macOS magnification gesture)
        if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
            zoom_delta = event.value()
            print(f"[DEBUG] Native zoom gesture: delta={zoom_delta:.4f}")

            # value() returns the scale change (typically small values like 0.01)
            # Convert to zoom factor
            zoom_factor = 1.0 + zoom_delta

            # Zoom towards gesture position
            pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
            self.zoomToPoint(pos, zoom_factor)

            event.accept()
            return True

        # Two-finger pan (macOS smart zoom or pan gesture)
        elif gesture_type == Qt.NativeGestureType.PanNativeGesture:
            # Get pan offset from gesture
            # Note: Pan gestures on macOS might not work as expected
            # We rely on wheelEvent for scroll-based panning
            event.accept()
            return True

        # Rotate gesture (optional, could be used for image rotation)
        elif gesture_type == Qt.NativeGestureType.RotateNativeGesture:
            # Rotation value in degrees
            # rotation_delta = event.value()
            # Could implement rotation here if needed
            event.accept()
            return True

        return False

    # ============================================
    # Mouse Event Handlers
    # ============================================

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        self.last_mouse_pos = event.pos()
        self.drag_start_pos = event.pos()

        if event.button() == Qt.LeftButton:
            # Convert to image coordinates
            img_pos = self.screenToImage(event.pos())

            # Check for keypoint under cursor
            hit_keypoint = self.getKeypointAt(img_pos)

            if hit_keypoint:
                person_idx, kp_idx = hit_keypoint

                # Check for multi-select modifiers (Cmd on Mac, Ctrl on others)
                modifiers = event.modifiers()
                is_multi_select = (modifiers & Qt.ControlModifier) or (modifiers & Qt.MetaModifier)

                if is_multi_select:
                    # Multi-select mode: toggle keypoint in selection
                    if kp_idx in self.selected_keypoint_indices:
                        # Deselect this keypoint
                        self.selected_keypoint_indices.remove(kp_idx)
                    else:
                        # Add to selection
                        if self.selected_person_idx != person_idx:
                            # Changed person, clear previous selection
                            self.selected_keypoint_indices = [kp_idx]
                            self.selected_person_idx = person_idx
                        else:
                            # Same person, add to selection
                            self.selected_keypoint_indices.append(kp_idx)

                    self.selected_keypoint_idx = -1  # Clear single selection
                    self.update()
                elif modifiers & Qt.ShiftModifier:
                    # Shift-click: start dragging (works for both single and multi-select)
                    if not self.selected_keypoint_indices:
                        # No multi-select active, select this keypoint first
                        self.selectKeypoint(person_idx, kp_idx)
                    self.is_dragging_keypoint = True
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    # Normal single selection
                    self.selectKeypoint(person_idx, kp_idx)
            else:
                # Check for person selection
                person_idx = self.getPersonAt(img_pos)
                if person_idx >= 0:
                    self.selectPerson(person_idx)
                else:
                    # Start panning
                    self.is_panning = True
                    self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.MiddleButton:
            # Middle button always pans
            self.is_panning = True
            self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.RightButton:
            # Will be handled in contextMenuEvent
            pass

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement"""
        current_pos = event.pos()
        delta = current_pos - self.last_mouse_pos

        # Update world position
        self.mouse_world_pos = self.screenToImage(current_pos)

        if self.is_panning:
            # Pan the view
            self.view_state.pan_x += delta.x()
            self.view_state.pan_y += delta.y()
            # Disable fit mode when user manually pans
            self.view_state.fit_mode = False
            self.update()

        elif self.is_dragging_keypoint:
            # Move selected keypoint(s)
            if self.selected_person_idx >= 0:
                person = self.persons[self.selected_person_idx]
                new_pos = self.screenToImage(current_pos)

                if new_pos:
                    # Calculate delta for multi-select dragging
                    screen_delta_x = delta.x() / self.view_state.zoom
                    screen_delta_y = delta.y() / self.view_state.zoom

                    # Move single or multiple keypoints
                    if self.selected_keypoint_idx >= 0:
                        # Single keypoint mode
                        kp = person.keypoints[self.selected_keypoint_idx]
                        kp.x = new_pos.x()
                        kp.y = new_pos.y()
                        kp.is_corrected = True

                        self.keypointMoved.emit(
                            self.selected_person_idx,
                            self.selected_keypoint_idx,
                            kp.x, kp.y
                        )
                    elif self.selected_keypoint_indices:
                        # Multi-select mode: move all selected keypoints by delta
                        for kp_idx in self.selected_keypoint_indices:
                            if 0 <= kp_idx < len(person.keypoints):
                                kp = person.keypoints[kp_idx]
                                kp.x += screen_delta_x
                                kp.y += screen_delta_y
                                kp.is_corrected = True

                                self.keypointMoved.emit(
                                    self.selected_person_idx,
                                    kp_idx,
                                    kp.x, kp.y
                                )

                    self.modificationMade.emit()
                    self.update()

        else:
            # Update hover state
            img_pos = self.screenToImage(current_pos)
            new_hover = self.getKeypointAt(img_pos)

            if new_hover != self.hovered_keypoint:
                self.hovered_keypoint = new_hover
                if new_hover:
                    self.setCursor(Qt.PointingHandCursor)
                    # Show tooltip
                    person_idx, kp_idx = new_hover
                    kp = self.persons[person_idx].keypoints[kp_idx]
                    tooltip = f"{KEYPOINT_NAMES.get(kp_idx, f'Point {kp_idx}')}\n"
                    tooltip += f"Confidence: {kp.confidence:.2f}"
                    QToolTip.showText(event.globalPos(), tooltip)
                else:
                    self.setCursor(Qt.CrossCursor)
                    QToolTip.hideText()
                self.update()

        self.last_mouse_pos = current_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.is_dragging_keypoint = False
            self.setCursor(Qt.CrossCursor)

        elif event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CrossCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click - zoom to point"""
        if event.button() == Qt.LeftButton:
            # Zoom in centered on click point
            self.zoomToPoint(event.pos(), 2.0)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel and trackpad scroll"""
        # Differentiate between trackpad and mouse wheel
        # Trackpad provides pixelDelta, mouse wheel provides angleDelta
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        print(f"[DEBUG] wheelEvent: pixel_delta=({pixel_delta.x()}, {pixel_delta.y()}), angle_delta=({angle_delta.x()}, {angle_delta.y()})")

        # If pixelDelta is available, it's trackpad scrolling - use for panning
        if not pixel_delta.isNull():
            # Two-finger pan on trackpad
            print(f"[DEBUG] Trackpad pan: delta=({pixel_delta.x()}, {pixel_delta.y()})")
            self.view_state.pan_x += pixel_delta.x()
            self.view_state.pan_y += pixel_delta.y()
            # Disable fit mode when user manually pans with trackpad
            self.view_state.fit_mode = False
            self.update()
            event.accept()
            return

        # Mouse wheel zoom - use angleDelta with reduced sensitivity
        if not angle_delta.isNull():
            delta = angle_delta.y()
            # Reduced sensitivity: 1.05/0.95 instead of 1.1/0.9
            zoom_factor = 1.05 if delta > 0 else 0.95
            print(f"[DEBUG] Mouse wheel zoom: factor={zoom_factor}")

            # Zoom towards mouse position
            self.zoomToPoint(event.position().toPoint(), zoom_factor)
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        """Handle widget resize (e.g., splitter adjustment)"""
        super().resizeEvent(event)

        # If fit mode is enabled, maintain it on resize
        if self.view_state.fit_mode and self.image:
            self.fitToWindow()
        # Otherwise, adjust pan to keep the image centered if it was centered
        elif self.image:
            # Maintain relative position during resize
            old_size = event.oldSize()
            new_size = event.size()

            if old_size.isValid():
                # Calculate the change in viewport center
                delta_x = (new_size.width() - old_size.width()) / 2.0
                delta_y = (new_size.height() - old_size.height()) / 2.0

                # Adjust pan to compensate for size change
                self.view_state.pan_x += delta_x
                self.view_state.pan_y += delta_y

        self.update()

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Show context menu"""
        img_pos = self.screenToImage(event.pos())
        hit_keypoint = self.getKeypointAt(img_pos)

        menu = QMenu(self)

        if hit_keypoint:
            person_idx, kp_idx = hit_keypoint
            kp = self.persons[person_idx].keypoints[kp_idx]

            # Keypoint-specific actions
            menu.addAction(f"Keypoint {kp_idx}: {KEYPOINT_NAMES.get(kp_idx, 'Unknown')}")
            menu.addSeparator()

            # Visibility actions
            visible_action = menu.addAction("Set Visible")
            visible_action.triggered.connect(partial(self.setKeypointVisibility, person_idx, kp_idx, 2))
            visible_action.setEnabled(kp.visibility != 2)

            occluded_action = menu.addAction("Set Occluded")
            occluded_action.triggered.connect(partial(self.setKeypointVisibility, person_idx, kp_idx, 1))
            occluded_action.setEnabled(kp.visibility != 1)

            missing_action = menu.addAction("Set Missing")
            missing_action.triggered.connect(partial(self.setKeypointVisibility, person_idx, kp_idx, 0))
            missing_action.setEnabled(kp.visibility != 0)

            menu.addSeparator()

            # Other actions
            reset_action = menu.addAction("Reset to Original")
            reset_action.triggered.connect(partial(self.resetKeypoint, person_idx, kp_idx))

            delete_action = menu.addAction("Delete Keypoint")
            delete_action.triggered.connect(partial(self.deleteKeypoint, person_idx, kp_idx))

        else:
            # General view actions
            menu.addAction("Zoom In", self.zoomIn)
            menu.addAction("Zoom Out", self.zoomOut)
            menu.addAction("Reset View", self.resetView)
            menu.addSeparator()

            # Person selection submenu
            person_idx = self.getPersonAt(img_pos)
            if person_idx >= 0:
                menu.addAction(f"Select Person {person_idx + 1}",
                             partial(self.selectPerson, person_idx))

        menu.exec(event.globalPos())

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()

        # Keypoint nudging with arrow keys
        if self.selected_keypoint_idx >= 0 and self.selected_person_idx >= 0:
            step = 10 if modifiers & Qt.ShiftModifier else 1
            if modifiers & Qt.ControlModifier:
                step = 0.5

            person = self.persons[self.selected_person_idx]
            kp = person.keypoints[self.selected_keypoint_idx]

            moved = False
            if key == Qt.Key_Left:
                kp.x -= step
                moved = True
            elif key == Qt.Key_Right:
                kp.x += step
                moved = True
            elif key == Qt.Key_Up:
                kp.y -= step
                moved = True
            elif key == Qt.Key_Down:
                kp.y += step
                moved = True

            if moved:
                kp.is_corrected = True
                self.keypointMoved.emit(
                    self.selected_person_idx,
                    self.selected_keypoint_idx,
                    kp.x, kp.y
                )
                self.modificationMade.emit()
                self.update()
                return

        # View controls
        if key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoomIn()
        elif key == Qt.Key_Minus:
            self.zoomOut()
        elif key == Qt.Key_0:
            if modifiers & Qt.ControlModifier:
                self.resetView()
            else:
                self.fitToWindow()
        elif key == Qt.Key_Space:
            self.resetView()
        elif key == Qt.Key_Tab:
            if modifiers & Qt.ShiftModifier:
                self.selectPreviousPerson()
            else:
                self.selectNextPerson()
        elif key == Qt.Key_Escape:
            self.clearSelection()
        elif key == Qt.Key_Delete:
            self.deleteSelectedKeypoint()
        elif key == Qt.Key_Home:
            self.selectFirstKeypoint()
        elif key == Qt.Key_End:
            self.selectLastKeypoint()

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            # Check if any URL is an image
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    self.imageDropped.emit(path)
                    event.acceptProposedAction()
                    return

    # Public methods
    def loadImage(self, image_path: str):
        """Load an image file"""
        print(f"[DEBUG] ImageViewer.loadImage() called with: {image_path}")

        # Explicitly delete old image to free memory (QImage holds C++ memory)
        if self.image is not None:
            del self.image
            self.image = None

        # Clear previous pose data when loading new image
        # Person objects may contain large numpy arrays
        if self.persons:
            for person in self.persons:
                if hasattr(person, 'keypoints') and person.keypoints is not None:
                    del person.keypoints
            self.persons = []

        self.clearSelection()

        self.image = QImage(image_path)

        # Validate image loaded successfully
        if self.image.isNull():
            error_msg = f"Failed to load image: {image_path}. File may be corrupted or unsupported format."
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        self.image_path = image_path
        print(f"[DEBUG] Image loaded successfully: {self.image.width()}x{self.image.height()}")

        # Auto-fit images to window, especially important for high-resolution images
        # Check if image is larger than viewport
        margin = 40
        view_width = self.width() - 2 * margin
        view_height = self.height() - 2 * margin

        if self.image.width() > view_width or self.image.height() > view_height:
            # Image is larger than viewport, fit to window
            print(f"[DEBUG] Image larger than viewport, auto-fitting")
            self.fitToWindow()
        else:
            # Image fits within viewport, center at 1:1 zoom
            print(f"[DEBUG] Image fits in viewport, centering at 1:1 zoom")
            self.resetView()

        self.update()

    def setPoses(self, persons: List[Person]):
        """Set pose data"""
        # Clean up old person data before replacing
        if self.persons:
            for person in self.persons:
                if hasattr(person, 'keypoints') and person.keypoints is not None:
                    del person.keypoints

        self.persons = persons
        self.clearSelection()
        self.update()

    def selectPerson(self, person_idx: int):
        """Select a person by index"""
        if 0 <= person_idx < len(self.persons):
            self.selected_person_idx = person_idx
            self.selected_keypoint_idx = -1
            self.selected_keypoint_indices = []
            self.personSelected.emit(person_idx)
            self.update()

    def selectKeypoint(self, person_idx: int, keypoint_idx: int):
        """Select a specific keypoint"""
        if 0 <= person_idx < len(self.persons):
            person = self.persons[person_idx]
            if 0 <= keypoint_idx < len(person.keypoints):
                self.selected_person_idx = person_idx
                self.selected_keypoint_idx = keypoint_idx
                self.selected_keypoint_indices = []  # Clear multi-select when single selecting
                self.keypointSelected.emit(person_idx, keypoint_idx)
                self.update()

    def selectMultipleKeypoints(self, person_idx: int, keypoint_indices: list):
        """Select multiple keypoints at once"""
        if 0 <= person_idx < len(self.persons):
            person = self.persons[person_idx]
            # Filter to valid keypoint indices
            valid_indices = [idx for idx in keypoint_indices if 0 <= idx < len(person.keypoints)]

            if valid_indices:
                self.selected_person_idx = person_idx
                self.selected_keypoint_idx = -1  # Clear single selection
                self.selected_keypoint_indices = valid_indices
                self.update()

    def clearSelection(self):
        """Clear all selections"""
        self.selected_person_idx = -1
        self.selected_keypoint_idx = -1
        self.selected_keypoint_indices = []
        self.update()

    def selectNextPerson(self):
        """Select next person in list"""
        if self.persons:
            self.selected_person_idx = (self.selected_person_idx + 1) % len(self.persons)
            self.selected_keypoint_idx = -1
            self.selected_keypoint_indices = []
            self.personSelected.emit(self.selected_person_idx)
            self.update()

    def selectPreviousPerson(self):
        """Select previous person in list"""
        if self.persons:
            self.selected_person_idx = (self.selected_person_idx - 1) % len(self.persons)
            self.selected_keypoint_idx = -1
            self.selected_keypoint_indices = []
            self.personSelected.emit(self.selected_person_idx)
            self.update()

    def selectFirstKeypoint(self):
        """Select first visible keypoint of selected person"""
        if self.selected_person_idx >= 0:
            person = self.persons[self.selected_person_idx]
            for i, kp in enumerate(person.keypoints):
                if kp.visibility > 0:
                    self.selectKeypoint(self.selected_person_idx, i)
                    break

    def selectLastKeypoint(self):
        """Select last visible keypoint of selected person"""
        if self.selected_person_idx >= 0:
            person = self.persons[self.selected_person_idx]
            for i in range(len(person.keypoints) - 1, -1, -1):
                if person.keypoints[i].visibility > 0:
                    self.selectKeypoint(self.selected_person_idx, i)
                    break

    # View control methods
    def zoomIn(self):
        """Zoom in by 20% towards viewport center"""
        center = QPoint(self.width() // 2, self.height() // 2)
        self.zoomToPoint(center, 1.2)

    def zoomOut(self):
        """Zoom out by 20% from viewport center"""
        center = QPoint(self.width() // 2, self.height() // 2)
        self.zoomToPoint(center, 0.8)

    def setZoom(self, zoom: float):
        """Set absolute zoom level, maintaining viewport center"""
        if not self.image:
            return

        # Calculate zoom factor
        zoom_factor = zoom / self.view_state.zoom

        # Zoom towards viewport center
        center = QPoint(self.width() // 2, self.height() // 2)
        self.zoomToPoint(center, zoom_factor)

    def setOcclusionDisplay(self, show_visible: bool, show_occluded: bool, show_missing: bool):
        """Set which keypoints to display based on visibility status"""
        self.show_visible_keypoints = show_visible
        self.show_occluded_keypoints = show_occluded
        self.show_missing_keypoints = show_missing
        self.update()

    def zoomToPoint(self, screen_point: QPoint, zoom_factor: float):
        """Zoom towards a specific point"""
        # Get the point in image space before zoom
        img_point_before = self.screenToImage(screen_point)

        # Apply zoom
        self.view_state.zoom *= zoom_factor
        self.view_state.zoom = max(0.1, min(10.0, self.view_state.zoom))
        # Disable fit mode when user manually zooms
        self.view_state.fit_mode = False

        # Get the point in image space after zoom
        img_point_after = self.screenToImage(screen_point)

        # Adjust pan to keep the point stationary
        if img_point_before and img_point_after:
            delta = (img_point_after - img_point_before) * self.view_state.zoom
            self.view_state.pan_x -= delta.x()
            self.view_state.pan_y -= delta.y()

        self.zoomChanged.emit(self.view_state.zoom)
        self.update()

    def zoomToBounds(self, bbox: Tuple[int, int, int, int]):
        """Zoom to fit a bounding box"""
        x, y, w, h = bbox

        # Calculate zoom to fit bbox
        margin = 50
        view_width = self.width() - 2 * margin
        view_height = self.height() - 2 * margin

        zoom_x = view_width / w if w > 0 else 1.0
        zoom_y = view_height / h if h > 0 else 1.0

        self.view_state.zoom = min(zoom_x, zoom_y, 5.0)

        # Center on bbox
        bbox_center = QPointF(x + w/2, y + h/2)
        view_center = QPointF(self.width()/2, self.height()/2)

        self.view_state.pan_x = view_center.x() - bbox_center.x() * self.view_state.zoom
        self.view_state.pan_y = view_center.y() - bbox_center.y() * self.view_state.zoom

        self.zoomChanged.emit(self.view_state.zoom)
        self.update()

    def fitToWindow(self):
        """Fit image to window size"""
        if not self.image:
            return

        # Calculate zoom to fit
        margin = 20
        view_width = self.width() - 2 * margin
        view_height = self.height() - 2 * margin

        zoom_x = view_width / self.image.width()
        zoom_y = view_height / self.image.height()

        self.view_state.zoom = min(zoom_x, zoom_y)
        self.view_state.pan_x = 0
        self.view_state.pan_y = 0
        self.view_state.fit_mode = True

        self.zoomChanged.emit(self.view_state.zoom)
        self.update()

    def resetView(self):
        """Reset view to 1:1 zoom and center the image"""
        if not self.image:
            return

        # Reset zoom to 1:1
        self.view_state.zoom = 1.0
        self.view_state.fit_mode = False

        # Center the image
        if self.image:
            img_center = QPointF(self.image.width() / 2, self.image.height() / 2)
            view_center = QPointF(self.width() / 2, self.height() / 2)

            self.view_state.pan_x = view_center.x() - img_center.x()
            self.view_state.pan_y = view_center.y() - img_center.y()

        self.zoomChanged.emit(self.view_state.zoom)
        self.update()

    # Utility methods
    def screenToImage(self, screen_pos: QPoint) -> Optional[QPointF]:
        """Convert screen coordinates to image coordinates"""
        if not self.image:
            return None

        # Apply inverse transformation
        center = self.rect().center()

        # Remove pan
        x = screen_pos.x() - center.x() - self.view_state.pan_x
        y = screen_pos.y() - center.y() - self.view_state.pan_y

        # Remove zoom
        x /= self.view_state.zoom
        y /= self.view_state.zoom

        # Add image center offset
        x += self.image.width() / 2
        y += self.image.height() / 2

        return QPointF(x, y)

    def imageToScreen(self, img_pos: QPointF) -> QPoint:
        """Convert image coordinates to screen coordinates"""
        center = self.rect().center()

        # Remove image center offset
        x = img_pos.x() - self.image.width() / 2
        y = img_pos.y() - self.image.height() / 2

        # Apply zoom
        x *= self.view_state.zoom
        y *= self.view_state.zoom

        # Apply pan
        x += center.x() + self.view_state.pan_x
        y += center.y() + self.view_state.pan_y

        return QPoint(int(x), int(y))

    def getKeypointAt(self, img_pos: Optional[QPointF]) -> Optional[Tuple[int, int]]:
        """Get keypoint at image position"""
        if not img_pos or not self.persons:
            return None

        # Search from selected person first for better UX
        search_order = []
        if self.selected_person_idx >= 0:
            search_order.append(self.selected_person_idx)
        search_order.extend([i for i in range(len(self.persons))
                           if i != self.selected_person_idx])

        # Check each person's keypoints
        for person_idx in search_order:
            person = self.persons[person_idx]
            for kp_idx, kp in enumerate(person.keypoints):
                if kp.visibility == 0:
                    continue

                # Check distance
                dist = math.sqrt((kp.x - img_pos.x())**2 + (kp.y - img_pos.y())**2)
                threshold = self.keypoint_radius * 2 / self.view_state.zoom

                if dist <= threshold:
                    return (person_idx, kp_idx)

        return None

    def getPersonAt(self, img_pos: Optional[QPointF]) -> int:
        """Get person index at image position"""
        if not img_pos or not self.persons:
            return -1

        for i, person in enumerate(self.persons):
            x, y, w, h = person.bbox
            if (x <= img_pos.x() <= x + w and
                y <= img_pos.y() <= y + h):
                return i

        return -1

    def setKeypointVisibility(self, person_idx: int, kp_idx: int, visibility: int):
        """Set keypoint visibility state"""
        if 0 <= person_idx < len(self.persons):
            person = self.persons[person_idx]
            if 0 <= kp_idx < len(person.keypoints):
                person.keypoints[kp_idx].visibility = visibility
                person.keypoints[kp_idx].is_corrected = True
                self.keypointVisibilityChanged.emit(person_idx, kp_idx, visibility)
                self.modificationMade.emit()
                self.update()

    def resetKeypoint(self, person_idx: int, kp_idx: int):
        """Reset keypoint to original position"""
        if 0 <= person_idx < len(self.persons):
            person = self.persons[person_idx]
            if 0 <= kp_idx < len(person.keypoints):
                person.keypoints[kp_idx].reset()
                self.modificationMade.emit()
                self.update()

    def deleteKeypoint(self, person_idx: int, kp_idx: int):
        """Delete (hide) a keypoint"""
        self.setKeypointVisibility(person_idx, kp_idx, 0)

    def deleteSelectedKeypoint(self):
        """Delete currently selected keypoint"""
        if self.selected_person_idx >= 0 and self.selected_keypoint_idx >= 0:
            self.deleteKeypoint(self.selected_person_idx, self.selected_keypoint_idx)
            self.selected_keypoint_idx = -1

    def resetCorrections(self):
        """Reset all corrections for current image"""
        for person in self.persons:
            person.resetToOriginal()
        self.modificationMade.emit()
        self.update()

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            # Clean up QImage (C++ memory)
            if hasattr(self, "image") and self.image is not None:
                del self.image

            # Clean up Person objects (numpy arrays)
            if hasattr(self, "persons") and self.persons:
                for person in self.persons:
                    if hasattr(person, "keypoints") and person.keypoints is not None:
                        del person.keypoints
                self.persons = []
        except Exception:
            pass  # Suppress errors during cleanup

