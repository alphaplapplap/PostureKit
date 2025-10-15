"""
Thumbnail Generator for PostureKit.

Generates optimized 200x200 JPEG thumbnails for database storage.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
import io
from PIL import Image

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ThumbnailGenerator:
    """Generate optimized thumbnails for database storage."""

    def __init__(self, size: int = 200, quality: int = 80):
        """
        Initialize thumbnail generator.

        Args:
            size: Thumbnail size (square canvas)
            quality: JPEG quality (1-100, default 80 for good compression/quality balance)
        """
        self.size = size
        self.quality = quality
        logger.debug(f"ThumbnailGenerator initialized: size={size}x{size}, quality={quality}")

    def generate_from_array(self, image: np.ndarray, format_bgr: bool = True) -> Optional[bytes]:
        """
        Generate thumbnail from numpy array.

        Args:
            image: Image as numpy array (H, W, 3)
            format_bgr: If True, image is in BGR format (OpenCV default),
                       if False, image is in RGB format

        Returns:
            JPEG thumbnail as bytes, or None if generation failed
        """
        try:
            if image is None or image.size == 0:
                logger.error("Cannot generate thumbnail from empty image")
                return None

            # Get original dimensions
            height, width = image.shape[:2]

            # Calculate aspect-ratio preserving dimensions that fit WITHIN canvas
            # (no dimension can exceed self.size - prevents cropping)
            scale = min(self.size / width, self.size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize image (use INTER_AREA for downscaling - best quality)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Create square canvas (black background for letterboxing/pillarboxing)
            canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

            # Center the resized image on canvas (no cropping - use letterboxing/pillarboxing)
            x_offset = (self.size - new_width) // 2
            y_offset = (self.size - new_height) // 2

            # Place on canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

            # OPTIMIZATION: Use cv2.imencode directly instead of PIL conversion
            # This eliminates BGR→RGB→PIL→JPEG pipeline
            # cv2.imencode expects BGR format (OpenCV default)
            if not format_bgr:
                # Convert RGB to BGR for cv2.imencode
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            # Encode directly with OpenCV (faster than PIL)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            success, encoded = cv2.imencode('.jpg', canvas, encode_params)

            if not success:
                logger.error("Failed to encode thumbnail as JPEG")
                return None

            thumbnail_bytes = encoded.tobytes()

            logger.debug(
                f"Generated thumbnail: {width}x{height} → {self.size}x{self.size}, "
                f"{len(thumbnail_bytes)} bytes"
            )

            return thumbnail_bytes

        except Exception as e:
            logger.error(f"Failed to generate thumbnail from array: {e}", exc_info=True)
            return None

    def generate_from_file(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """
        Generate thumbnail from image file.

        Args:
            file_path: Path to image file

        Returns:
            JPEG thumbnail as bytes, or None if generation failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.error(f"Image file not found: {file_path}")
                return None

            # Load image with OpenCV
            image = cv2.imread(str(file_path))

            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return None

            return self.generate_from_array(image)

        except Exception as e:
            logger.error(f"Failed to generate thumbnail from file {file_path}: {e}", exc_info=True)
            return None

    @staticmethod
    def bytes_to_base64(thumbnail_bytes: bytes) -> str:
        """
        Convert thumbnail bytes to base64 string for JSON serialization.

        Args:
            thumbnail_bytes: JPEG bytes

        Returns:
            Base64-encoded string
        """
        import base64
        return base64.b64encode(thumbnail_bytes).decode('utf-8')

    @staticmethod
    def base64_to_bytes(base64_string: str) -> bytes:
        """
        Convert base64 string back to bytes.

        Args:
            base64_string: Base64-encoded thumbnail

        Returns:
            JPEG bytes
        """
        import base64
        return base64.b64decode(base64_string)


# Global instance for convenience
default_generator = ThumbnailGenerator()


def generate_thumbnail(image: Union[np.ndarray, str, Path]) -> Optional[bytes]:
    """
    Convenience function to generate thumbnail.

    Args:
        image: Numpy array or path to image file

    Returns:
        JPEG thumbnail as bytes, or None if generation failed
    """
    if isinstance(image, (str, Path)):
        return default_generator.generate_from_file(image)
    elif isinstance(image, np.ndarray):
        return default_generator.generate_from_array(image)
    else:
        logger.error(f"Unsupported image type: {type(image)}")
        return None
