"""
Learning module for PostureKit.
Provides bias correction learning from manual corrections.
"""

from src.learning.correction_learner import (
    BiasCorrector,
    BiasEstimate,
    CorrectionLearnerError,
    InsufficientDataError,
)

__all__ = [
    'BiasCorrector',
    'BiasEstimate',
    'CorrectionLearnerError',
    'InsufficientDataError',
]
