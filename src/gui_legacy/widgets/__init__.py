"""
GUI widgets package for PostureKit.
"""
from .keypoint_editor import KeypointEditorGL
from .label_input import LabelInput

__all__ = ['KeypointEditorGL', 'LabelInput']

# Alias for backwards compatibility
KeypointEditor = KeypointEditorGL
