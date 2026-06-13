"""
Input validation decorators and utilities for PostureKit.
"""

import numpy as np
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Optional, Tuple

from src.exceptions import ValidationError
from src.constants import KEYPOINTS_SIZE, MIN_BBOX_SIZE_PX


def validate_image_array(func: Callable) -> Callable:
    """
    Decorator to validate image array parameter.

    Expects parameter named 'image' or 'image_array'.
    Validates:
    - Not None
    - Shape is (H, W) or (H, W, C)
    - Minimum size 32x32
    - dtype is uint8 or float32
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find image parameter
        image = kwargs.get('image') or kwargs.get('image_array')

        if image is None and len(args) > 0:
            # Check if first positional arg is the image
            if isinstance(args[0], np.ndarray):
                image = args[0]

        if image is None:
            raise ValidationError(
                "Image parameter is required",
                operation=func.__name__,
                context={'function': func.__name__}
            )

        # Validate shape
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"Image must be numpy array, got {type(image)}",
                operation=func.__name__,
                context={'type': str(type(image))}
            )

        if len(image.shape) not in [2, 3]:
            raise ValidationError(
                f"Invalid image dimensions: {len(image.shape)}D",
                operation=func.__name__,
                context={'shape': image.shape}
            )

        # Validate minimum size
        height, width = image.shape[:2]
        if width < 32 or height < 32:
            raise ValidationError(
                f"Image too small: {width}x{height}, minimum 32x32",
                operation=func.__name__,
                context={'width': width, 'height': height}
            )

        # Validate dtype
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValidationError(
                f"Invalid image dtype: {image.dtype}, expected uint8 or float32",
                operation=func.__name__,
                context={'dtype': str(image.dtype)}
            )

        return func(*args, **kwargs)

    return wrapper


def validate_file_path(func: Callable) -> Callable:
    """
    Decorator to validate file path parameter.

    Expects parameter named 'file_path', 'image_path', or 'path'.
    Validates:
    - Path exists
    - Is a file (not directory)
    - Has read permission
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find path parameter
        path = kwargs.get('file_path') or kwargs.get('image_path') or kwargs.get('path')

        if path is None and len(args) > 0:
            # Check if first positional arg is a path
            if isinstance(args[0], (str, Path)):
                path = args[0]

        if path is None:
            raise ValidationError(
                "File path parameter is required",
                operation=func.__name__,
                context={'function': func.__name__}
            )

        path = Path(path)

        # Validate existence
        if not path.exists():
            raise ValidationError(
                f"File does not exist: {path}",
                operation=func.__name__,
                context={'file_path': str(path)}
            )

        # Validate is file
        if not path.is_file():
            raise ValidationError(
                f"Path is not a file: {path}",
                operation=func.__name__,
                context={'file_path': str(path)}
            )

        # Validate readable
        if not path.exists() or not path.is_file():
            raise ValidationError(
                f"Cannot read file: {path}",
                operation=func.__name__,
                context={'file_path': str(path)}
            )

        return func(*args, **kwargs)

    return wrapper


def validate_search_params(func: Callable) -> Callable:
    """
    Decorator to validate search parameters.

    Validates:
    - feature_vector: shape and dtype
    - k: positive integer
    - min_confidence: in [0, 1]
    - query_keypoints: shape if provided
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate feature_vector
        feature_vector = kwargs.get('feature_vector')
        if feature_vector is not None:
            if not isinstance(feature_vector, (list, np.ndarray)):
                raise ValidationError(
                    f"feature_vector must be list or array, got {type(feature_vector)}",
                    operation=func.__name__,
                    context={'type': str(type(feature_vector))}
                )

            vec = np.array(feature_vector)
            if len(vec.shape) != 1:
                raise ValidationError(
                    f"feature_vector must be 1D, got shape {vec.shape}",
                    operation=func.__name__,
                    context={'shape': vec.shape}
                )

            # Common dimensions: 66 (geometric v3) or 642 (fused = 66 + 576)
            if vec.shape[0] not in [66, 642]:
                raise ValidationError(
                    f"Invalid feature dimension: {vec.shape[0]}, expected 66 or 642",
                    operation=func.__name__,
                    context={'dimension': vec.shape[0]}
                )

        # Validate k
        k = kwargs.get('k')
        if k is not None:
            if not isinstance(k, int) or k <= 0:
                raise ValidationError(
                    f"k must be positive integer, got {k}",
                    operation=func.__name__,
                    context={'k': k}
                )

            if k > 1000:
                raise ValidationError(
                    f"k too large: {k}, maximum 1000",
                    operation=func.__name__,
                    context={'k': k}
                )

        # Validate confidence thresholds
        for param_name in ['min_confidence', 'min_feature_confidence', 'min_region_confidence']:
            conf = kwargs.get(param_name)
            if conf is not None:
                if not isinstance(conf, (int, float)):
                    raise ValidationError(
                        f"{param_name} must be numeric, got {type(conf)}",
                        operation=func.__name__,
                        context={param_name: conf}
                    )

                if not 0.0 <= conf <= 1.0:
                    raise ValidationError(
                        f"{param_name} must be in [0, 1], got {conf}",
                        operation=func.__name__,
                        context={param_name: conf}
                    )

        # Validate query_keypoints if provided
        query_keypoints = kwargs.get('query_keypoints')
        if query_keypoints is not None:
            kp = np.array(query_keypoints)
            if kp.shape != (KEYPOINTS_SIZE,):
                raise ValidationError(
                    f"query_keypoints must have shape ({KEYPOINTS_SIZE},), got {kp.shape}",
                    operation=func.__name__,
                    context={'shape': kp.shape, 'expected': (KEYPOINTS_SIZE,)}
                )

        # Validate query_bbox if provided
        query_bbox = kwargs.get('query_bbox')
        if query_bbox is not None:
            bbox = np.array(query_bbox)
            if bbox.shape != (4,):
                raise ValidationError(
                    f"query_bbox must have shape (4,), got {bbox.shape}",
                    operation=func.__name__,
                    context={'shape': bbox.shape}
                )

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                raise ValidationError(
                    f"Invalid bbox dimensions: w={w}, h={h}",
                    operation=func.__name__,
                    context={'bbox': bbox.tolist()}
                )

        return func(*args, **kwargs)

    return wrapper


def validate_directory(func: Callable) -> Callable:
    """
    Decorator to validate directory path parameter.

    Expects parameter named 'directory' or 'directory_path'.
    Validates:
    - Path exists
    - Is a directory
    - Has read permission
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find directory parameter
        directory = kwargs.get('directory') or kwargs.get('directory_path')

        if directory is None and len(args) > 0:
            if isinstance(args[0], (str, Path)):
                directory = args[0]

        if directory is None:
            raise ValidationError(
                "Directory parameter is required",
                operation=func.__name__,
                context={'function': func.__name__}
            )

        directory = Path(directory)

        # Validate existence
        if not directory.exists():
            raise ValidationError(
                f"Directory does not exist: {directory}",
                operation=func.__name__,
                context={'directory': str(directory)}
            )

        # Validate is directory
        if not directory.is_dir():
            raise ValidationError(
                f"Path is not a directory: {directory}",
                operation=func.__name__,
                context={'directory': str(directory)}
            )

        return func(*args, **kwargs)

    return wrapper
