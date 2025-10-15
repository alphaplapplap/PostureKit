"""Undo/redo system for PostureKit."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack, QUndoCommand
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of undoable commands."""
    KEYPOINT_MOVE = "keypoint_move"
    KEYPOINT_VISIBILITY = "keypoint_visibility"
    KEYPOINT_CONFIDENCE = "keypoint_confidence"
    MACRO = "macro"  # Group of commands


@dataclass
class KeypointState:
    """Compact keypoint state representation."""
    index: int
    x: float
    y: float
    confidence: float
    visibility: int  # 0=missing, 1=occluded, 2=visible

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'index': self.index,
            'x': self.x,
            'y': self.y,
            'confidence': self.confidence,
            'visibility': self.visibility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeypointState':
        """Create from dictionary."""
        return cls(**data)

    def __eq__(self, other: 'KeypointState') -> bool:
        """Check equality."""
        return (
            self.index == other.index and
            abs(self.x - other.x) < 0.01 and
            abs(self.y - other.y) < 0.01 and
            abs(self.confidence - other.confidence) < 0.001 and
            self.visibility == other.visibility
        )


class BaseCommand(QUndoCommand):
    """Base class for all undoable commands."""

    def __init__(
        self,
        person_id: int,
        description: str,
        parent: Optional[QUndoCommand] = None
    ):
        """
        Initialize command.

        Args:
            person_id: ID of person being edited
            description: Human-readable description
            parent: Parent command (for command grouping)
        """
        super().__init__(description, parent)
        self.person_id = person_id
        self._first_redo = True  # Track if this is initial execution

    def redo(self):
        """Execute/redo the command."""
        if self._first_redo:
            # Don't execute on first redo (command already applied)
            self._first_redo = False
        else:
            self._do_redo()

    def undo(self):
        """Undo the command."""
        self._do_undo()

    def _do_redo(self):
        """Actual redo implementation (override in subclasses)."""
        raise NotImplementedError

    def _do_undo(self):
        """Actual undo implementation (override in subclasses)."""
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Check if command can still be executed."""
        return True


class KeypointMoveCommand(BaseCommand):
    """Command for moving a keypoint."""

    def __init__(
        self,
        person_id: int,
        keypoint_index: int,
        old_state: KeypointState,
        new_state: KeypointState,
        callback: Callable[[int, int, float, float], None],
        parent: Optional[QUndoCommand] = None
    ):
        """
        Initialize move command.

        Args:
            person_id: Person ID
            keypoint_index: Index of keypoint
            old_state: Previous keypoint state
            new_state: New keypoint state
            callback: Function to apply state (person_id, kpt_idx, x, y)
            parent: Parent command for grouping
        """
        super().__init__(
            person_id,
            f"Move keypoint {keypoint_index}",
            parent
        )
        self.keypoint_index = keypoint_index
        self.old_state = old_state
        self.new_state = new_state
        self.callback = callback

    def _do_redo(self):
        """Apply new position."""
        self.callback(
            self.person_id,
            self.keypoint_index,
            self.new_state.x,
            self.new_state.y
        )
        logger.debug(f"Redo: Move keypoint {self.keypoint_index} to ({self.new_state.x}, {self.new_state.y})")

    def _do_undo(self):
        """Restore old position."""
        self.callback(
            self.person_id,
            self.keypoint_index,
            self.old_state.x,
            self.old_state.y
        )
        logger.debug(f"Undo: Move keypoint {self.keypoint_index} to ({self.old_state.x}, {self.old_state.y})")

    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Merge consecutive move commands for same keypoint.

        This prevents undo stack explosion during drag operations.
        """
        if not isinstance(other, KeypointMoveCommand):
            return False

        if (self.person_id != other.person_id or
            self.keypoint_index != other.keypoint_index):
            return False

        # Merge: keep original old_state, update to new new_state
        self.new_state = other.new_state
        logger.debug(f"Merged move commands for keypoint {self.keypoint_index}")
        return True

    def id(self) -> int:
        """Return command ID for merging."""
        return hash((CommandType.KEYPOINT_MOVE, self.person_id, self.keypoint_index))


class KeypointVisibilityCommand(BaseCommand):
    """Command for changing keypoint visibility."""

    def __init__(
        self,
        person_id: int,
        keypoint_index: int,
        old_visibility: int,
        new_visibility: int,
        callback: Callable[[int, int, int], None],
        parent: Optional[QUndoCommand] = None
    ):
        """
        Initialize visibility command.

        Args:
            person_id: Person ID
            keypoint_index: Index of keypoint
            old_visibility: Previous visibility (0/1/2)
            new_visibility: New visibility (0/1/2)
            callback: Function to apply visibility
            parent: Parent command
        """
        visibility_names = {0: "missing", 1: "occluded", 2: "visible"}
        super().__init__(
            person_id,
            f"Set keypoint {keypoint_index} {visibility_names[new_visibility]}",
            parent
        )
        self.keypoint_index = keypoint_index
        self.old_visibility = old_visibility
        self.new_visibility = new_visibility
        self.callback = callback

    def _do_redo(self):
        """Apply new visibility."""
        self.callback(self.person_id, self.keypoint_index, self.new_visibility)
        logger.debug(f"Redo: Set keypoint {self.keypoint_index} visibility to {self.new_visibility}")

    def _do_undo(self):
        """Restore old visibility."""
        self.callback(self.person_id, self.keypoint_index, self.old_visibility)
        logger.debug(f"Undo: Set keypoint {self.keypoint_index} visibility to {self.old_visibility}")


class KeypointConfidenceCommand(BaseCommand):
    """Command for adjusting keypoint confidence."""

    def __init__(
        self,
        person_id: int,
        keypoint_index: int,
        old_confidence: float,
        new_confidence: float,
        callback: Callable[[int, int, float], None],
        parent: Optional[QUndoCommand] = None
    ):
        """Initialize confidence command."""
        super().__init__(
            person_id,
            f"Adjust keypoint {keypoint_index} confidence",
            parent
        )
        self.keypoint_index = keypoint_index
        self.old_confidence = old_confidence
        self.new_confidence = new_confidence
        self.callback = callback

    def _do_redo(self):
        """Apply new confidence."""
        self.callback(self.person_id, self.keypoint_index, self.new_confidence)

    def _do_undo(self):
        """Restore old confidence."""
        self.callback(self.person_id, self.keypoint_index, self.old_confidence)


class MacroCommand(BaseCommand):
    """Command that groups multiple commands together."""

    def __init__(
        self,
        person_id: int,
        description: str,
        commands: Optional[List[BaseCommand]] = None
    ):
        """
        Initialize macro command.

        Args:
            person_id: Person ID
            description: Description of grouped operation
            commands: List of commands to execute together
        """
        super().__init__(person_id, description)
        self.commands = commands or []

    def add_command(self, command: BaseCommand):
        """Add command to macro."""
        self.commands.append(command)

    def _do_redo(self):
        """Execute all commands in order."""
        for command in self.commands:
            command._do_redo()
        logger.debug(f"Redo macro: {self.text()} ({len(self.commands)} commands)")

    def _do_undo(self):
        """Undo all commands in reverse order."""
        for command in reversed(self.commands):
            command._do_undo()
        logger.debug(f"Undo macro: {self.text()} ({len(self.commands)} commands)")

    def is_valid(self) -> bool:
        """Macro is valid if all sub-commands are valid."""
        return all(cmd.is_valid() for cmd in self.commands)


class UndoManager(QObject):
    """
    Central undo/redo management system.

    Manages separate undo stacks per person/image with command grouping,
    compression, and validation.
    """

    can_undo_changed = Signal(bool)
    can_redo_changed = Signal(bool)
    clean_changed = Signal(bool)

    def __init__(self):
        """Initialize undo manager."""
        super().__init__()

        # Separate stacks per (image_id, person_id)
        self.stacks: Dict[tuple, QUndoStack] = {}
        self.current_context: Optional[tuple] = None  # (image_id, person_id)

        # Active macro command (for grouping)
        self.active_macro: Optional[MacroCommand] = None

        # Configuration
        self.max_undo_count = 100
        self.merge_timeout_ms = 500  # Merge commands within this time

        logger.info("UndoManager initialized")

    def set_context(self, image_id: int, person_id: int):
        """
        Set current editing context.

        Args:
            image_id: Current image ID
            person_id: Current person ID
        """
        context = (image_id, person_id)

        if context != self.current_context:
            self.current_context = context

            # Create stack if doesn't exist
            if context not in self.stacks:
                stack = QUndoStack(self)
                stack.setUndoLimit(self.max_undo_count)
                stack.canUndoChanged.connect(self.can_undo_changed)
                stack.canRedoChanged.connect(self.can_redo_changed)
                stack.cleanChanged.connect(self.clean_changed)
                self.stacks[context] = stack
                logger.debug(f"Created undo stack for context {context}")

            # Emit state changes
            self._emit_state_changes()

    def clear_context(self, image_id: Optional[int] = None):
        """
        Clear undo stacks for context.

        Args:
            image_id: If provided, clear all stacks for this image.
                     If None, clear current context only.
        """
        if image_id is not None:
            # Clear all stacks for this image
            to_remove = [k for k in self.stacks.keys() if k[0] == image_id]
            for context in to_remove:
                self.stacks[context].clear()
                del self.stacks[context]
            logger.info(f"Cleared {len(to_remove)} undo stacks for image {image_id}")
        elif self.current_context:
            # Clear current context only
            if self.current_context in self.stacks:
                self.stacks[self.current_context].clear()
            logger.info(f"Cleared undo stack for context {self.current_context}")

        self._emit_state_changes()

    def _get_current_stack(self) -> Optional[QUndoStack]:
        """Get undo stack for current context."""
        if self.current_context and self.current_context in self.stacks:
            return self.stacks[self.current_context]
        return None

    def push_command(self, command: BaseCommand):
        """
        Push command to undo stack.

        Args:
            command: Command to execute and add to stack
        """
        stack = self._get_current_stack()
        if not stack:
            logger.warning("No active undo stack")
            return

        # If we're in a macro, add to macro instead
        if self.active_macro:
            self.active_macro.add_command(command)
            logger.debug(f"Added command to active macro: {command.text()}")
            return

        # Push to stack (this will execute the command)
        stack.push(command)
        logger.debug(f"Pushed command: {command.text()}")

    def begin_macro(self, description: str):
        """
        Begin a macro command group.

        Args:
            description: Description of the grouped operation
        """
        if self.active_macro:
            logger.warning("Macro already active, ending previous macro")
            self.end_macro()

        if not self.current_context:
            logger.warning("No active context for macro")
            return

        self.active_macro = MacroCommand(
            person_id=self.current_context[1],
            description=description
        )
        logger.debug(f"Begin macro: {description}")

    def end_macro(self):
        """End current macro command group and push to stack."""
        if not self.active_macro:
            logger.warning("No active macro to end")
            return

        # Only push if macro has commands
        if self.active_macro.commands:
            stack = self._get_current_stack()
            if stack:
                stack.push(self.active_macro)
                logger.debug(
                    f"End macro: {self.active_macro.text()} "
                    f"({len(self.active_macro.commands)} commands)"
                )
        else:
            logger.debug("Discarded empty macro")

        self.active_macro = None

    def undo(self):
        """Undo last command in current context."""
        stack = self._get_current_stack()
        if stack and stack.canUndo():
            stack.undo()
            logger.info(f"Undo: {stack.undoText()}")

    def redo(self):
        """Redo last undone command in current context."""
        stack = self._get_current_stack()
        if stack and stack.canRedo():
            stack.redo()
            logger.info(f"Redo: {stack.redoText()}")

    def can_undo(self) -> bool:
        """Check if undo is available."""
        stack = self._get_current_stack()
        return stack.canUndo() if stack else False

    def can_redo(self) -> bool:
        """Check if redo is available."""
        stack = self._get_current_stack()
        return stack.canRedo() if stack else False

    def undo_text(self) -> str:
        """Get description of next undo command."""
        stack = self._get_current_stack()
        return stack.undoText() if stack else ""

    def redo_text(self) -> str:
        """Get description of next redo command."""
        stack = self._get_current_stack()
        return stack.redoText() if stack else ""

    def set_clean(self):
        """Mark current state as clean (saved)."""
        stack = self._get_current_stack()
        if stack:
            stack.setClean()

    def is_clean(self) -> bool:
        """Check if current state is clean (no unsaved changes)."""
        stack = self._get_current_stack()
        return stack.isClean() if stack else True

    def _emit_state_changes(self):
        """Emit all state change signals."""
        self.can_undo_changed.emit(self.can_undo())
        self.can_redo_changed.emit(self.can_redo())
        self.clean_changed.emit(self.is_clean())

    def get_stats(self) -> Dict[str, Any]:
        """Get undo system statistics."""
        stack = self._get_current_stack()
        if not stack:
            return {
                'active_stack': None,
                'undo_count': 0,
                'redo_count': 0,
                'total_stacks': len(self.stacks)
            }

        return {
            'active_stack': self.current_context,
            'undo_count': stack.count() - stack.index(),
            'redo_count': stack.index(),
            'total_stacks': len(self.stacks),
            'is_clean': stack.isClean()
        }


# Simple UndoStack wrapper for backward compatibility
class UndoStack(QObject):
    """Simple wrapper around QUndoStack for backward compatibility."""

    canUndoChanged = Signal(bool)
    canRedoChanged = Signal(bool)

    def __init__(self, max_size: int = 100, parent=None):
        super().__init__(parent)
        self._stack = QUndoStack(self)
        self._stack.setUndoLimit(max_size)
        self._stack.canUndoChanged.connect(self.canUndoChanged)
        self._stack.canRedoChanged.connect(self.canRedoChanged)

        # Add snake_case aliases for signals
        self.can_undo_changed = self.canUndoChanged
        self.can_redo_changed = self.canRedoChanged

    def push(self, command: QUndoCommand):
        """Push a command onto the stack."""
        self._stack.push(command)

    def undo(self):
        """Undo the last command."""
        self._stack.undo()

    def redo(self):
        """Redo the last undone command."""
        self._stack.redo()

    def canUndo(self) -> bool:
        """Check if undo is available."""
        return self._stack.canUndo()

    def canRedo(self) -> bool:
        """Check if redo is available."""
        return self._stack.canRedo()

    def clear(self):
        """Clear the undo stack."""
        self._stack.clear()

    def getHistory(self) -> List[str]:
        """Get undo history (for auto-save)."""
        return []  # Simplified implementation

    def restoreHistory(self, history: List[str]):
        """Restore undo history (from auto-save)."""
        pass  # Simplified implementation


# Export UndoCommand for convenience
UndoCommand = QUndoCommand


# Singleton instance
_undo_manager: Optional[UndoManager] = None


def get_undo_manager() -> UndoManager:
    """Get global undo manager instance."""
    global _undo_manager
    if _undo_manager is None:
        _undo_manager = UndoManager()
    return _undo_manager


if __name__ == "__main__":
    # Test undo system
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel

    app = QApplication(sys.argv)

    # Initialize manager
    manager = get_undo_manager()
    manager.set_context(image_id=1, person_id=1)

    # Test data
    keypoint_positions = {0: (100.0, 100.0)}

    def move_keypoint(person_id: int, kpt_idx: int, x: float, y: float):
        """Callback to apply keypoint move."""
        keypoint_positions[kpt_idx] = (x, y)
        status_label.setText(f"Keypoint 0: ({x:.1f}, {y:.1f})")

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)

    # Status
    status_label = QLabel(f"Keypoint 0: {keypoint_positions[0]}")
    layout.addWidget(status_label)

    # Undo/Redo info
    info_label = QLabel()
    def update_info():
        stats = manager.get_stats()
        info_label.setText(
            f"Can Undo: {manager.can_undo()} ({manager.undo_text()})\n"
            f"Can Redo: {manager.can_redo()} ({manager.redo_text()})\n"
            f"Clean: {manager.is_clean()}\n"
            f"Stats: {stats}"
        )

    manager.can_undo_changed.connect(update_info)
    manager.can_redo_changed.connect(update_info)
    update_info()
    layout.addWidget(info_label)

    # Test: Single move
    def test_single_move():
        old_state = KeypointState(0, *keypoint_positions[0], 0.9, 2)
        new_state = KeypointState(0, 150.0, 150.0, 0.9, 2)
        cmd = KeypointMoveCommand(1, 0, old_state, new_state, move_keypoint)
        manager.push_command(cmd)
        move_keypoint(1, 0, 150.0, 150.0)
        update_info()

    btn1 = QPushButton("Move Keypoint (Single)")
    btn1.clicked.connect(test_single_move)
    layout.addWidget(btn1)

    # Test: Macro
    def test_macro():
        manager.begin_macro("Move multiple keypoints")

        # Move 1
        old_state = KeypointState(0, *keypoint_positions[0], 0.9, 2)
        new_state = KeypointState(0, 200.0, 200.0, 0.9, 2)
        cmd = KeypointMoveCommand(1, 0, old_state, new_state, move_keypoint)
        manager.push_command(cmd)
        move_keypoint(1, 0, 200.0, 200.0)

        # Move 2
        old_state = KeypointState(0, 200.0, 200.0, 0.9, 2)
        new_state = KeypointState(0, 250.0, 250.0, 0.9, 2)
        cmd = KeypointMoveCommand(1, 0, old_state, new_state, move_keypoint)
        manager.push_command(cmd)
        move_keypoint(1, 0, 250.0, 250.0)

        manager.end_macro()
        update_info()

    btn2 = QPushButton("Move Keypoint (Macro)")
    btn2.clicked.connect(test_macro)
    layout.addWidget(btn2)

    # Undo/Redo buttons
    undo_btn = QPushButton("Undo")
    undo_btn.clicked.connect(lambda: (manager.undo(), update_info()))
    layout.addWidget(undo_btn)

    redo_btn = QPushButton("Redo")
    redo_btn.clicked.connect(lambda: (manager.redo(), update_info()))
    layout.addWidget(redo_btn)

    window.setCentralWidget(central)
    window.resize(400, 400)
    window.show()

    sys.exit(app.exec())
