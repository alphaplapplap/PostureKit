"""
Bounding box utilities for PostureKit.
"""

import numpy as np
from typing import Tuple, List, Union

from src.constants import BBOX_DIMENSIONS


class BboxUtils:
    """Bounding box manipulation and validation utilities."""

    @staticmethod
    def clamp_to_image(
        bbox: Union[List[float], np.ndarray],
        image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Clamp bounding box to image boundaries.

        Args:
            bbox: Bounding box as [x, y, width, height]
            image_shape: Image shape as (height, width, channels)

        Returns:
            Clamped (x, y, width, height) as integers
        """
        x, y, w, h = [int(v) for v in bbox[:4]]
        img_height, img_width = image_shape[:2]

        # Clamp x and y to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        # Adjust width and height to not exceed image bounds
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        return x, y, w, h

    @staticmethod
    def area(bbox: Union[List[float], np.ndarray]) -> float:
        """
        Calculate bounding box area.

        Args:
            bbox: Bounding box as [x, y, width, height]

        Returns:
            Area in pixels
        """
        return float(bbox[2] * bbox[3])

    @staticmethod
    def validate(bbox: Union[List[float], np.ndarray]) -> bool:
        """
        Validate bounding box format.

        Args:
            bbox: Bounding box to validate

        Returns:
            True if valid, False otherwise
        """
        if bbox is None:
            return False

        if len(bbox) < BBOX_DIMENSIONS:
            return False

        x, y, w, h = bbox[:4]
        return w > 0 and h > 0

    @staticmethod
    def to_xyxy(bbox: Union[List[float], np.ndarray]) -> Tuple[float, float, float, float]:
        """
        Convert from [x, y, w, h] to [x_min, y_min, x_max, y_max].

        Args:
            bbox: Bounding box as [x, y, width, height]

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        x, y, w, h = bbox[:4]
        return (x, y, x + w, y + h)

    @staticmethod
    def from_xyxy(xyxy: Tuple[float, float, float, float]) -> List[float]:
        """
        Convert from [x_min, y_min, x_max, y_max] to [x, y, w, h].

        Args:
            xyxy: Bounding box as (x_min, y_min, x_max, y_max)

        Returns:
            [x, y, width, height]
        """
        x_min, y_min, x_max, y_max = xyxy
        return [x_min, y_min, x_max - x_min, y_max - y_min]
