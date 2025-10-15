"""
Enhanced Keypoint Editor with OpenGL Acceleration
Includes complete hand/face skeleton connections and GPU-accelerated rendering
"""

from PySide6.QtWidgets import QWidget, QMenu
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer, QPoint
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QCursor,
    QMouseEvent, QKeyEvent, QWheelEvent, QSurfaceFormat
)
from PySide6.QtOpenGL import QOpenGLVersionProfile
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from typing import Optional, List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from enum import Enum


class VisibilityState(Enum):
    """Keypoint visibility states per COCO format"""
    MISSING = 0
    OCCLUDED = 1
    VISIBLE = 2


class ColorScheme(Enum):
    """Available coloring modes"""
    CONFIDENCE = "confidence"
    ANATOMICAL = "anatomical"
    PERSON_ID = "person_id"


@dataclass
class Keypoint:
    """Single keypoint data structure"""
    index: int
    name: str
    x: float
    y: float
    confidence: float
    visibility: VisibilityState
    is_corrected: bool = False
    is_locked: bool = False


@dataclass
class Pose:
    """Complete pose with 133 keypoints"""
    person_id: int
    keypoints: List[Keypoint]
    bbox: Tuple[float, float, float, float]
    overall_confidence: float

    def get_keypoint(self, index: int) -> Optional[Keypoint]:
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return None


class KeypointEditorGL(QOpenGLWidget):
    """
    OpenGL-accelerated keypoint editor with complete skeleton rendering
    """

    # Signals
    keypoint_selected = Signal(int, int)
    keypoint_moved = Signal(int, int, float, float)
    keypoint_visibility_changed = Signal(int, int, int)
    person_selected = Signal(int)
    zoom_changed = Signal(float)
    confidence_changed = Signal(int, int, float)  # person_id, keypoint_idx, new_confidence

    # Complete skeleton definitions
    BODY_CONNECTIONS = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16),
        # Feet
        (15, 17), (15, 18), (15, 19),
        (16, 20), (16, 21), (16, 22),
    ]

    # Hand skeleton (21 points each)
    # Left hand: indices 91-111, Right hand: 112-132
    HAND_CONNECTIONS = [
        # Palm base to fingers
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb (indices 1-4)
        (1, 2), (2, 3), (3, 4),
        # Index finger (indices 5-8)
        (5, 6), (6, 7), (7, 8),
        # Middle finger (indices 9-12)
        (9, 10), (10, 11), (11, 12),
        # Ring finger (indices 13-16)
        (13, 14), (14, 15), (15, 16),
        # Pinky (indices 17-20)
        (17, 18), (18, 19), (19, 20),
    ]

    # Face landmarks (68 points: indices 23-90)
    FACE_CONNECTIONS = [
        # Jaw contour (0-16 -> 23-39)
        *[(i, i+1) for i in range(23, 39)],
        # Left eyebrow (17-21 -> 40-44)
        *[(i, i+1) for i in range(40, 44)],
        # Right eyebrow (22-26 -> 45-49)
        *[(i, i+1) for i in range(45, 49)],
        # Nose bridge (27-30 -> 50-53)
        *[(i, i+1) for i in range(50, 53)],
        # Nose base (31-35 -> 54-58)
        *[(i, i+1) for i in range(54, 58)],
        (58, 54),  # Close nose base
        # Left eye (36-41 -> 59-64)
        *[(i, i+1) for i in range(59, 64)],
        (64, 59),  # Close left eye
        # Right eye (42-47 -> 65-70)
        *[(i, i+1) for i in range(65, 70)],
        (70, 65),  # Close right eye
        # Outer mouth (48-59 -> 71-82)
        *[(i, i+1) for i in range(71, 82)],
        (82, 71),  # Close outer mouth
        # Inner mouth (60-67 -> 83-90)
        *[(i, i+1) for i in range(83, 90)],
        (90, 83),  # Close inner mouth
    ]

    # Anatomical colors
    PART_COLORS = {
        'head': (156, 39, 176),
        'torso': (74, 144, 226),
        'left_arm': (76, 175, 80),
        'right_arm': (255, 152, 0),
        'left_leg': (0, 188, 212),
        'right_leg': (255, 235, 59),
        'left_hand': (139, 195, 74),
        'right_hand': (255, 183, 77),
        'feet': (233, 30, 99),
        'face': (156, 39, 176),
    }

    PERSON_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (128, 255, 0), (255, 0, 128),
    ]

    # OpenGL shaders
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec4 color;

    uniform mat4 projection;
    uniform mat4 view;

    out vec4 fragColor;

    void main() {
        gl_Position = projection * view * vec4(position, 0.0, 1.0);
        fragColor = color;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    in vec4 fragColor;
    out vec4 outColor;

    void main() {
        outColor = fragColor;
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set OpenGL format
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(4)  # 4x MSAA
        self.setFormat(fmt)

        # State
        self.poses: List[Pose] = []
        self.selected_person_id: Optional[int] = None
        self.selected_keypoint_idx: Optional[int] = None
        self.hovered_keypoint: Optional[Tuple[int, int]] = None

        # Display settings
        self.show_skeleton = True
        self.show_keypoints = True
        self.show_confidence_colors = True
        self.use_colored_skeleton = True
        self.color_scheme = ColorScheme.CONFIDENCE
        self.keypoint_radius = 4.0
        self.line_width = 2.5
        self.face_display_mode = "dots_only"  # hidden, dots_only, minimal, full_mesh

        # View transform
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.image_size = (800, 600)

        # Interaction
        self.is_panning = False
        self.is_dragging_keypoint = False
        self.drag_start_pos = QPointF()
        self.last_mouse_pos = QPointF()

        # Undo/Redo
        self.undo_stack: List[Dict] = []
        self.redo_stack: List[Dict] = []
        self.max_undo_size = 100

        # OpenGL resources
        self.shader_program = None
        self.vao_lines = None
        self.vbo_lines = None
        self.vao_points = None
        self.vbo_points = None

        # Vertex buffers
        self.line_vertices = []
        self.point_vertices = []

        # UI
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.ArrowCursor)
        self.setMinimumSize(400, 300)

    def initializeGL(self):
        """Initialize OpenGL resources"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Compile shaders
        self.shader_program = compileProgram(
            compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        # Create VAOs and VBOs for lines
        self.vao_lines = glGenVertexArrays(1)
        self.vbo_lines = glGenBuffers(1)

        glBindVertexArray(self.vao_lines)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)

        # Position attribute (vec2)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Color attribute (vec4)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        # Create VAOs and VBOs for points
        self.vao_points = glGenVertexArrays(1)
        self.vbo_points = glGenBuffers(1)

        glBindVertexArray(self.vao_points)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def resizeGL(self, w, h):
        """Handle resize"""
        glViewport(0, 0, w, h)

    def paintGL(self):
        """Main OpenGL rendering"""
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if not self.poses:
            return

        # Use shader program
        glUseProgram(self.shader_program)

        # Set up projection matrix (orthographic)
        w, h = self.width(), self.height()
        projection = self._get_ortho_matrix(0, w, h, 0, -1, 1)
        view = self._get_view_matrix()

        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        view_loc = glGetUniformLocation(self.shader_program, "view")

        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        # Build vertex buffers
        self._build_vertex_buffers()

        # Draw lines (skeleton)
        # Scale line width inversely with zoom for consistent appearance
        if self.show_skeleton and self.line_vertices:
            glLineWidth(self.line_width / self.zoom_level)
            glBindVertexArray(self.vao_lines)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_lines)
            glBufferData(GL_ARRAY_BUFFER,
                        np.array(self.line_vertices, dtype=np.float32).nbytes,
                        np.array(self.line_vertices, dtype=np.float32),
                        GL_DYNAMIC_DRAW)
            glDrawArrays(GL_LINES, 0, len(self.line_vertices))
            glBindVertexArray(0)

        # Draw points (keypoints)
        # Scale keypoint size inversely with zoom for better visibility at high zoom
        if self.show_keypoints and self.point_vertices:
            glPointSize((self.keypoint_radius / self.zoom_level) * 2)
            glBindVertexArray(self.vao_points)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_points)
            glBufferData(GL_ARRAY_BUFFER,
                        np.array(self.point_vertices, dtype=np.float32).nbytes,
                        np.array(self.point_vertices, dtype=np.float32),
                        GL_DYNAMIC_DRAW)
            glDrawArrays(GL_POINTS, 0, len(self.point_vertices))
            glBindVertexArray(0)

        glUseProgram(0)

        # Use QPainter for UI overlays (bounding boxes, labels)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.pan_offset)
        painter.scale(self.zoom_level, self.zoom_level)

        for pose in self.poses:
            if pose.person_id == self.selected_person_id:
                self._draw_bbox_overlay(painter, pose)

        painter.end()

    def _build_vertex_buffers(self):
        """Build OpenGL vertex buffers from poses"""
        self.line_vertices = []
        self.point_vertices = []

        for pose in self.poses:
            is_selected = (pose.person_id == self.selected_person_id)
            opacity = 1.0 if is_selected else 0.6

            # Build skeleton lines
            if self.show_skeleton:
                self._add_skeleton_lines(pose, is_selected, opacity)

            # Build keypoint points
            if self.show_keypoints:
                self._add_keypoint_points(pose, is_selected, opacity)

    def _add_skeleton_lines(self, pose: Pose, is_selected: bool, opacity: float):
        """Add skeleton line vertices"""
        line_width_mult = 1.5 if is_selected else 1.0

        # Body connections
        for start_idx, end_idx in self.BODY_CONNECTIONS:
            self._add_connection(pose, start_idx, end_idx, opacity)

        # Left hand connections (indices 91-111)
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            self._add_connection(pose, 91 + start_idx, 91 + end_idx, opacity)

        # Right hand connections (indices 112-132)
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            self._add_connection(pose, 112 + start_idx, 112 + end_idx, opacity)

        # Face connections (based on display mode)
        if self.face_display_mode in ["minimal", "full_mesh"]:
            for start_idx, end_idx in self.FACE_CONNECTIONS:
                self._add_connection(pose, start_idx, end_idx, opacity * 0.7)

    def _add_connection(self, pose: Pose, start_idx: int, end_idx: int, opacity: float):
        """Add a single connection line to vertex buffer"""
        start_kpt = pose.get_keypoint(start_idx)
        end_kpt = pose.get_keypoint(end_idx)

        if not start_kpt or not end_kpt:
            return

        if (start_kpt.visibility == VisibilityState.MISSING or
            end_kpt.visibility == VisibilityState.MISSING):
            return

        # Get color
        color = self._get_connection_color(start_kpt, end_kpt, pose, start_idx)
        r, g, b = [c / 255.0 for c in color]

        # Adjust opacity for occluded
        if (start_kpt.visibility == VisibilityState.OCCLUDED or
            end_kpt.visibility == VisibilityState.OCCLUDED):
            opacity *= 0.7
            r, g, b = 1.0, 0.65, 0.0  # Orange tint

        # Add vertices (position + color)
        self.line_vertices.extend([
            start_kpt.x, start_kpt.y, r, g, b, opacity,
            end_kpt.x, end_kpt.y, r, g, b, opacity
        ])

    def _add_keypoint_points(self, pose: Pose, is_selected: bool, opacity: float):
        """Add keypoint point vertices"""
        for kpt in pose.keypoints:
            if kpt.visibility == VisibilityState.MISSING:
                continue

            # Skip face landmarks in dots_only mode if too many
            if (self.face_display_mode == "hidden" and
                23 <= kpt.index <= 90):
                continue

            is_kpt_selected = (is_selected and
                             self.selected_keypoint_idx == kpt.index)
            is_hovered = (self.hovered_keypoint and
                         self.hovered_keypoint == (pose.person_id, kpt.index))

            color = self._get_keypoint_color(kpt, pose)
            r, g, b = [c / 255.0 for c in color]

            # Adjust opacity for visibility
            point_opacity = opacity
            if kpt.visibility == VisibilityState.OCCLUDED:
                point_opacity *= 0.7

            # Reduce opacity for face keypoints (23-90) for better visibility
            # Face landmarks can obscure facial features when zoomed in
            if 23 <= kpt.index <= 90:
                point_opacity *= 0.4

            # Add vertex
            self.point_vertices.extend([
                kpt.x, kpt.y, r, g, b, point_opacity
            ])

    def _draw_bbox_overlay(self, painter: QPainter, pose: Pose):
        """Draw bounding box overlay using QPainter"""
        x, y, w, h = pose.bbox

        color = self.PERSON_COLORS[pose.person_id % len(self.PERSON_COLORS)]
        pen = QPen(QColor(*color), 2.0 / self.zoom_level)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(x, y, w, h))

        # Label
        painter.setBrush(QBrush(QColor(*color)))
        painter.setPen(Qt.NoPen)
        label_rect = QRectF(x, y - 20 / self.zoom_level, 80 / self.zoom_level, 18 / self.zoom_level)
        painter.drawRect(label_rect)

        painter.setPen(QPen(Qt.white))
        font = painter.font()
        font.setPointSizeF(10 / self.zoom_level)
        painter.setFont(font)
        painter.drawText(label_rect, Qt.AlignCenter, f"Person {pose.person_id + 1}")

    @staticmethod
    def _get_ortho_matrix(left, right, bottom, top, near, far):
        """Create orthographic projection matrix"""
        matrix = np.identity(4, dtype=np.float32)
        matrix[0, 0] = 2.0 / (right - left)
        matrix[1, 1] = 2.0 / (top - bottom)
        matrix[2, 2] = -2.0 / (far - near)
        matrix[3, 0] = -(right + left) / (right - left)
        matrix[3, 1] = -(top + bottom) / (top - bottom)
        matrix[3, 2] = -(far + near) / (far - near)
        return matrix

    def _get_view_matrix(self):
        """Create view matrix with zoom and pan"""
        matrix = np.identity(4, dtype=np.float32)
        matrix[0, 0] = self.zoom_level
        matrix[1, 1] = self.zoom_level
        matrix[3, 0] = self.pan_offset.x() * 2.0 / self.width()
        matrix[3, 1] = -self.pan_offset.y() * 2.0 / self.height()
        return matrix

    # Color methods (same as before)
    def _get_keypoint_color(self, kpt: Keypoint, pose: Pose) -> Tuple[int, int, int]:
        if self.color_scheme == ColorScheme.CONFIDENCE:
            return self._get_confidence_color(kpt.confidence)
        elif self.color_scheme == ColorScheme.ANATOMICAL:
            return self._get_anatomical_color(kpt.index)
        elif self.color_scheme == ColorScheme.PERSON_ID:
            return self.PERSON_COLORS[pose.person_id % len(self.PERSON_COLORS)]
        return (128, 128, 128)

    def _get_connection_color(self, start_kpt: Keypoint, end_kpt: Keypoint,
                             pose: Pose, start_idx: int) -> Tuple[int, int, int]:
        if self.color_scheme == ColorScheme.CONFIDENCE:
            avg_conf = (start_kpt.confidence + end_kpt.confidence) / 2
            return self._get_confidence_color(avg_conf)
        elif self.color_scheme == ColorScheme.ANATOMICAL:
            return self._get_anatomical_color(start_idx)
        elif self.color_scheme == ColorScheme.PERSON_ID:
            return self.PERSON_COLORS[pose.person_id % len(self.PERSON_COLORS)]
        return (128, 128, 128)

    @staticmethod
    def _get_confidence_color(confidence: float) -> Tuple[int, int, int]:
        if confidence >= 0.7:
            return (0, 255, 0)
        elif confidence >= 0.4:
            return (255, 255, 0)
        elif confidence >= 0.1:
            return (255, 0, 0)
        return (128, 128, 128)

    def _get_anatomical_color(self, kpt_index: int) -> Tuple[int, int, int]:
        if kpt_index <= 4:
            return self.PART_COLORS['head']
        elif kpt_index <= 6:
            return self.PART_COLORS['torso']
        elif kpt_index in [7, 9]:
            return self.PART_COLORS['left_arm']
        elif kpt_index in [8, 10]:
            return self.PART_COLORS['right_arm']
        elif kpt_index in [11, 13, 15]:
            return self.PART_COLORS['left_leg']
        elif kpt_index in [12, 14, 16]:
            return self.PART_COLORS['right_leg']
        elif 17 <= kpt_index <= 22:
            return self.PART_COLORS['feet']
        elif 23 <= kpt_index <= 90:
            return self.PART_COLORS['face']
        elif 91 <= kpt_index <= 111:
            return self.PART_COLORS['left_hand']
        elif 112 <= kpt_index <= 132:
            return self.PART_COLORS['right_hand']
        return (128, 128, 128)

    # Mouse/keyboard interaction (same as before, but trigger update())
    def mousePressEvent(self, event: QMouseEvent):
        pos = self._screen_to_image(event.position())

        if event.button() == Qt.LeftButton:
            clicked_kpt = self._find_keypoint_at(pos)

            if clicked_kpt:
                person_id, kpt_idx = clicked_kpt
                self.selected_person_id = person_id
                self.selected_keypoint_idx = kpt_idx
                self.keypoint_selected.emit(person_id, kpt_idx)

                if event.modifiers() & Qt.ShiftModifier:
                    self.is_dragging_keypoint = True
                    self.drag_start_pos = pos
                    self.setCursor(Qt.ClosedHandCursor)
            else:
                clicked_person = self._find_person_at(pos)
                if clicked_person is not None:
                    self.selected_person_id = clicked_person
                    self.selected_keypoint_idx = None
                    self.person_selected.emit(clicked_person)
                else:
                    self.is_panning = True
                    self.last_mouse_pos = event.position()
                    self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.RightButton:
            clicked_kpt = self._find_keypoint_at(pos)
            if clicked_kpt:
                self._show_keypoint_context_menu(clicked_kpt, event.globalPosition().toPoint())

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = self._screen_to_image(event.position())

        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.position()
            self.update()

        elif self.is_dragging_keypoint:
            if self.selected_person_id is not None and self.selected_keypoint_idx is not None:
                pose = self.get_selected_pose()
                if pose:
                    kpt = pose.get_keypoint(self.selected_keypoint_idx)
                    if kpt and not kpt.is_locked:
                        kpt.x = pos.x()
                        kpt.y = pos.y()
                        kpt.is_corrected = True
                        self.keypoint_moved.emit(
                            self.selected_person_id,
                            self.selected_keypoint_idx,
                            kpt.x, kpt.y
                        )
                self.update()

        else:
            hovered = self._find_keypoint_at(pos)
            if hovered != self.hovered_keypoint:
                self.hovered_keypoint = hovered
                self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.is_dragging_keypoint = False
            self.setCursor(Qt.ArrowCursor if not self.hovered_keypoint else Qt.PointingHandCursor)

    def wheelEvent(self, event: QWheelEvent):
        old_zoom = self.zoom_level
        delta = event.angleDelta().y() / 120.0
        zoom_factor = 1.1 if delta > 0 else 0.9

        self.zoom_level = np.clip(self.zoom_level * zoom_factor, 0.1, 10.0)

        cursor_pos = event.position()
        zoom_change = self.zoom_level / old_zoom
        self.pan_offset = cursor_pos - (cursor_pos - self.pan_offset) * zoom_change

        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if self.selected_person_id is None or self.selected_keypoint_idx is None:
            return

        pose = self.get_selected_pose()
        if not pose:
            return

        kpt = pose.get_keypoint(self.selected_keypoint_idx)
        if not kpt or kpt.is_locked:
            return

        nudge_amount = 1.0
        if event.modifiers() & Qt.ShiftModifier:
            nudge_amount = 10.0
        elif event.modifiers() & Qt.ControlModifier:
            nudge_amount = 0.1

        if event.key() == Qt.Key_Left:
            kpt.x -= nudge_amount
            kpt.is_corrected = True
        elif event.key() == Qt.Key_Right:
            kpt.x += nudge_amount
            kpt.is_corrected = True
        elif event.key() == Qt.Key_Up:
            kpt.y -= nudge_amount
            kpt.is_corrected = True
        elif event.key() == Qt.Key_Down:
            kpt.y += nudge_amount
            kpt.is_corrected = True
        elif event.key() == Qt.Key_Tab:
            self._select_next_person(reverse=bool(event.modifiers() & Qt.ShiftModifier))
        elif event.key() == Qt.Key_Space:
            self.fit_to_window()
        elif event.key() == Qt.Key_Escape:
            self.selected_keypoint_idx = None
        else:
            event.ignore()
            return

        self.update()
        event.accept()

    # Helper methods (same as before)
    def _screen_to_image(self, screen_pos: QPointF) -> QPointF:
        return (screen_pos - self.pan_offset) / self.zoom_level

    def _find_keypoint_at(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        hit_radius = self.keypoint_radius / self.zoom_level * 2

        if self.selected_person_id is not None:
            pose = self.get_selected_pose()
            if pose:
                for kpt in pose.keypoints:
                    if kpt.visibility == VisibilityState.MISSING:
                        continue
                    dist = np.hypot(kpt.x - pos.x(), kpt.y - pos.y())
                    if dist <= hit_radius:
                        return (pose.person_id, kpt.index)

        for pose in self.poses:
            if pose.person_id == self.selected_person_id:
                continue
            for kpt in pose.keypoints:
                if kpt.visibility == VisibilityState.MISSING:
                    continue
                dist = np.hypot(kpt.x - pos.x(), kpt.y - pos.y())
                if dist <= hit_radius:
                    return (pose.person_id, kpt.index)

        return None

    def _find_person_at(self, pos: QPointF) -> Optional[int]:
        for pose in self.poses:
            x, y, w, h = pose.bbox
            if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                return pose.person_id
        return None

    def _select_next_person(self, reverse=False):
        if not self.poses:
            return

        if self.selected_person_id is None:
            self.selected_person_id = self.poses[0].person_id
        else:
            current_idx = next((i for i, p in enumerate(self.poses)
                              if p.person_id == self.selected_person_id), 0)
            next_idx = (current_idx + (-1 if reverse else 1)) % len(self.poses)
            self.selected_person_id = self.poses[next_idx].person_id

        self.selected_keypoint_idx = None
        self.person_selected.emit(self.selected_person_id)
        self.update()

    def _show_keypoint_context_menu(self, keypoint_info: Tuple[int, int], global_pos: QPoint):
        person_id, kpt_idx = keypoint_info
        pose = next((p for p in self.poses if p.person_id == person_id), None)
        if not pose:
            return

        kpt = pose.get_keypoint(kpt_idx)
        if not kpt:
            return

        menu = QMenu(self)

        visible_action = menu.addAction("Set Visible")
        visible_action.triggered.connect(
            lambda: self._set_keypoint_visibility(person_id, kpt_idx, VisibilityState.VISIBLE)
        )

        occluded_action = menu.addAction("Set Occluded")
        occluded_action.triggered.connect(
            lambda: self._set_keypoint_visibility(person_id, kpt_idx, VisibilityState.OCCLUDED)
        )

        missing_action = menu.addAction("Set Missing")
        missing_action.triggered.connect(
            lambda: self._set_keypoint_visibility(person_id, kpt_idx, VisibilityState.MISSING)
        )

        menu.addSeparator()

        lock_action = menu.addAction("Unlock Position" if kpt.is_locked else "Lock Position")
        lock_action.triggered.connect(lambda: self._toggle_keypoint_lock(person_id, kpt_idx))

        reset_action = menu.addAction("Reset to Original")
        reset_action.setEnabled(kpt.is_corrected)

        menu.exec(global_pos)

    def _set_keypoint_visibility(self, person_id: int, kpt_idx: int, visibility: VisibilityState):
        pose = next((p for p in self.poses if p.person_id == person_id), None)
        if pose:
            kpt = pose.get_keypoint(kpt_idx)
            if kpt:
                kpt.visibility = visibility
                self.keypoint_visibility_changed.emit(person_id, kpt_idx, visibility.value)
                self.update()

    def _toggle_keypoint_lock(self, person_id: int, kpt_idx: int):
        pose = next((p for p in self.poses if p.person_id == person_id), None)
        if pose:
            kpt = pose.get_keypoint(kpt_idx)
            if kpt:
                kpt.is_locked = not kpt.is_locked
                self.update()

    # View controls
    def fit_to_window(self):
        if not self.image_size:
            return

        img_w, img_h = self.image_size
        widget_w = self.width()
        widget_h = self.height()

        zoom_w = widget_w / img_w
        zoom_h = widget_h / img_h
        self.zoom_level = min(zoom_w, zoom_h) * 0.95

        self.pan_offset = QPointF(
            (widget_w - img_w * self.zoom_level) / 2,
            (widget_h - img_h * self.zoom_level) / 2
        )

        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def set_poses(self, poses: List[Pose]):
        self.poses = poses
        if poses and self.selected_person_id is None:
            self.selected_person_id = poses[0].person_id
            self.person_selected.emit(self.selected_person_id)
        self.update()

    def set_image_size(self, width: int, height: int):
        self.image_size = (width, height)
        self.fit_to_window()

    def get_selected_pose(self) -> Optional[Pose]:
        if self.selected_person_id is None:
            return None
        for pose in self.poses:
            if pose.person_id == self.selected_person_id:
                return pose
        return None

    def clear(self):
        """Clear all poses and reset state"""
        self.poses = []
        self.selected_person_id = None
        self.selected_keypoint_idx = None
        self.hovered_keypoint = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.update()

    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.vao_lines:
            glDeleteVertexArrays(1, [self.vao_lines])
        if self.vbo_lines:
            glDeleteBuffers(1, [self.vbo_lines])
        if self.vao_points:
            glDeleteVertexArrays(1, [self.vao_points])
        if self.vbo_points:
            glDeleteBuffers(1, [self.vbo_points])
        if self.shader_program:
            glDeleteProgram(self.shader_program)

# Alias for backwards compatibility
KeypointEditor = KeypointEditorGL
