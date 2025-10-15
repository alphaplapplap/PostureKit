"""
Logger utility for PostureKit
"""

import logging
from src.utils.logging_config import get_logger as _get_logger


def get_logger(name=None):
    """Get a configured logger instance"""
    return _get_logger(name if name else __name__)


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
