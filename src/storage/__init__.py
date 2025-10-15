"""Storage module for PostureKit."""
from src.storage.models import (
    Base,
    Image,
    PoseDetection,
    GeometricFeatures,
    VisualFeatures,
    FusedFeatures,
    TrainingLabel
)

__all__ = [
    'Base',
    'Image',
    'PoseDetection',
    'GeometricFeatures',
    'VisualFeatures',
    'FusedFeatures',
    'TrainingLabel'
]
