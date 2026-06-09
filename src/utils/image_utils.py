"""
Image processing utilities for PostureKit.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from src.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ImageUtils:
    """Image format conversion and validation utilities."""

    @staticmethod
    def normalize_to_rgb(image: np.ndarray, image_path: Optional[Path] = None) -> np.ndarray:
        """
        Convert image to RGB format regardless of input format.

        Handles:
        - Grayscale (H, W)
        - Grayscale with channel (H, W, 1)
        - BGR (H, W, 3)
        - BGRA (H, W, 4)

        Args:
            image: Input image array
            image_path: Optional path for error messages

        Returns:
            RGB image array of shape (H, W, 3)

        Raises:
            ValidationError: If image format is unsupported
        """
        path_str = str(image_path.name) if image_path else "unknown"

        # Handle 2D grayscale
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Handle 3D images
        if len(image.shape) == 3:
            channels = image.shape[2]

            if channels == 1:
                # Grayscale with channel dimension
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                # Standard BGR image
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                # RGBA image
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                raise ValidationError(
                    f"Unsupported channel count: {channels}",
                    operation="normalize_to_rgb",
                    context={'image_path': path_str, 'shape': image.shape}
                )

        # Unexpected shape
        raise ValidationError(
            f"Unexpected image shape: {image.shape}",
            operation="normalize_to_rgb",
            context={'image_path': path_str, 'shape': image.shape}
        )

    @staticmethod
    def load_image(image_path: Path) -> Optional[np.ndarray]:
        """
        Load image from file with error handling.

        Args:
            image_path: Path to image file

        Returns:
            Image array or None if loading failed
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image

    @staticmethod
    def validate_image_array(
        image: np.ndarray,
        expected_channels: int = 3,
        min_size: Tuple[int, int] = (32, 32)
    ) -> None:
        """
        Validate image array meets requirements.

        Args:
            image: Image array to validate
            expected_channels: Expected number of channels
            min_size: Minimum (width, height)

        Raises:
            ValidationError: If image doesn't meet requirements
        """
        if image is None:
            raise ValidationError(
                "Image is None",
                operation="validate_image_array"
            )

        if len(image.shape) not in [2, 3]:
            raise ValidationError(
                f"Invalid image dimensions: {len(image.shape)}",
                operation="validate_image_array",
                context={'shape': image.shape}
            )

        height, width = image.shape[:2]
        if width < min_size[0] or height < min_size[1]:
            raise ValidationError(
                f"Image too small: {width}x{height}, minimum: {min_size}",
                operation="validate_image_array",
                context={'shape': image.shape, 'min_size': min_size}
            )

        if len(image.shape) == 3 and image.shape[2] != expected_channels:
            raise ValidationError(
                f"Wrong number of channels: {image.shape[2]}, expected: {expected_channels}",
                operation="validate_image_array",
                context={'shape': image.shape}
            )
