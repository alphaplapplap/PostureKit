"""
Visual Feature Extractor for PostureKit.
Extracts 576-dimensional visual embeddings using MobileNetV3-Small.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging
import weakref
from PIL import Image

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


def _has_mps_module() -> bool:
    """Check if torch.mps module is available (PyTorch 2.1+)."""
    return hasattr(torch, 'mps')


@dataclass
class VisualFeatures:
    """
    Visual features extracted from an image.

    Attributes:
        feature_vector: 576-dimensional visual embedding
        model_name: Name of model used for extraction
        normalization: Normalization method applied
    """
    feature_vector: np.ndarray  # (576,) float32
    model_name: str
    normalization: str = "l2"

    def __post_init__(self):
        """Validate feature dimensions."""
        assert self.feature_vector.shape == (576,), \
            f"Feature vector must be (576,), got {self.feature_vector.shape}"
        assert self.feature_vector.dtype == np.float32, \
            f"Feature vector must be float32, got {self.feature_vector.dtype}"

    @property
    def shape(self):
        """Convenience property to access feature_vector.shape."""
        return self.feature_vector.shape

    @property
    def dtype(self):
        """Convenience property to access feature_vector.dtype."""
        return self.feature_vector.dtype

    def __array__(self, dtype=None):
        """Support numpy operations on this object."""
        if dtype is not None:
            return self.feature_vector.astype(dtype)
        return self.feature_vector

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'feature_vector': self.feature_vector.tolist(),
            'model_name': self.model_name,
            'normalization': self.normalization
        }


class VisualFeatureExtractorError(Exception):
    """Base exception for Visual Feature Extractor errors."""
    pass


class ModelInitializationError(VisualFeatureExtractorError):
    """Raised when model fails to initialize."""
    pass


class FeatureExtractionError(VisualFeatureExtractorError):
    """Raised when feature extraction fails."""
    pass


class VisualFeatureExtractor:
    """
    Extracts 576-dimensional visual features using MobileNetV3-Small.

    Features:
    - Pre-trained on ImageNet (general visual understanding)
    - Apple Silicon MPS optimized
    - L2 normalized outputs for similarity search
    - Lazy loading (model loads on first use)

    Attributes:
        device: Torch device (mps, cuda, or cpu)
        normalize: Whether to L2 normalize feature vectors
    """

    MODEL_NAME = "mobilenet_v3_small"
    FEATURE_DIM = 576

    # Class-level model cache with weak references to prevent memory leaks
    _model_cache = weakref.WeakValueDictionary()

    def __init__(
        self,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize Visual Feature Extractor.

        Args:
            device: Device to use ('mps', 'cuda', 'cpu'). Uses settings default if None.
            normalize: Whether to L2 normalize feature vectors
        """
        self.device = device or settings.DEVICE
        self.normalize = normalize

        # Model loaded lazily
        self._model = None
        self._device_obj = None
        self._transform = None

        logger.info(
            f"VisualFeatureExtractor initialized: device={self.device}, "
            f"model={self.MODEL_NAME}, normalize={self.normalize}"
        )

    def _initialize_device(self) -> torch.device:
        """Initialize and validate torch device."""
        try:
            if self.device == 'mps':
                if not torch.backends.mps.is_available():
                    logger.warning("MPS not available, falling back to CPU")
                    return torch.device('cpu')

                # Enable CPU fallback for unsupported MPS operations (hardsigmoid)
                import os
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

                device_obj = torch.device('mps')
                logger.info("Using Apple Silicon MPS with CPU fallback for unsupported ops")

            elif self.device == 'cuda':
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    return torch.device('cpu')
                device_obj = torch.device('cuda')
                logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")

            else:
                device_obj = torch.device('cpu')
                logger.info("Using CPU")

            return device_obj

        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize device: {e}") from e

    def _load_model(self) -> None:
        """Load MobileNetV3-Small model (lazy loading with caching)."""
        if self._model is not None:
            return  # Already loaded

        # Initialize device first
        self._device_obj = self._initialize_device()

        # Check cache (TOCTOU-safe: use get() to avoid race between check and retrieval)
        cache_key = (self.MODEL_NAME, str(self._device_obj))
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            self._model = cached_model
            logger.info(f"Reusing cached MobileNetV3 model on {str(self._device_obj)}")
            # Still need to initialize transforms
            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            return

        logger.info("Loading MobileNetV3-Small model...")

        try:
            # Load pre-trained model
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = mobilenet_v3_small(weights=weights)

            # Remove classification head to get features
            # MobileNetV3-Small features are 576-dim before classifier
            self._model = nn.Sequential(*list(model.children())[:-1])

            # Move to device and set to eval mode
            self._model = self._model.to(self._device_obj)
            self._model.eval()

            # Cache the model
            self._model_cache[cache_key] = self._model

            # Initialize transforms
            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info(
                f"Model loaded successfully: {self.MODEL_NAME} on {str(self._device_obj)}, "
                f"feature_dim={self.FEATURE_DIM}"
            )

        except Exception as e:
            raise ModelInitializationError(f"Failed to load model: {e}") from e

    def extract(self, image: np.ndarray) -> VisualFeatures:
        """
        Extract 576-dimensional visual features from image.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8

        Returns:
            VisualFeatures object with 576-dim embedding

        Raises:
            FeatureExtractionError: If extraction fails
        """
        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        # Validate input
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise FeatureExtractionError(
                f"Expected RGB image (H,W,3), got {image.shape}"
            )

        logger.debug(f"Extracting visual features from image shape: {image.shape}")

        try:
            # Convert to PIL Image with safe dtype handling
            if image.dtype != np.uint8:
                # Detect float types (float16, float32, float64)
                if np.issubdtype(image.dtype, np.floating):
                    # Float images: check if normalized [0, 1] or [0, 255]
                    if image.max() <= 1.0:
                        # Normalized float [0, 1] -> uint8 [0, 255]
                        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                    else:
                        # Float [0, 255] -> uint8 (clip to prevent overflow)
                        image = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    # Integer types (int8, int16, uint16, etc.)
                    # Safely convert to uint8 range [0, 255]
                    if np.issubdtype(image.dtype, np.signedinteger):
                        # Signed int -> remap to [0, 255]
                        image = np.clip((image.astype(np.float32) + 128), 0, 255).astype(np.uint8)
                    else:
                        # Unsigned int -> clip and cast
                        image = np.clip(image, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Apply transforms
            input_tensor = self._transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)

            # Use pinned memory for faster GPU transfers (CUDA only)
            if self._device_obj.type == 'cuda':
                input_batch = input_batch.pin_memory().to(self._device_obj, non_blocking=True)
            else:
                input_batch = input_batch.to(self._device_obj)

            # Extract features (inference_mode is faster than no_grad)
            with torch.inference_mode():
                features = self._model(input_batch)

            # Convert to numpy
            feature_vector = features.squeeze().detach().cpu().numpy().astype(np.float32)

            # Flatten if needed (MobileNetV3 outputs (576, 1, 1))
            if feature_vector.ndim > 1:
                feature_vector = feature_vector.flatten()

            # Ensure correct shape
            if feature_vector.shape[0] != self.FEATURE_DIM:
                raise FeatureExtractionError(
                    f"Expected {self.FEATURE_DIM} features, got {feature_vector.shape[0]}"
                )

            # L2 normalization
            if self.normalize:
                norm = np.linalg.norm(feature_vector)
                if norm > 0:
                    feature_vector = feature_vector / norm

            logger.debug(
                f"Extracted visual features: shape={feature_vector.shape}, "
                f"norm={float(np.linalg.norm(feature_vector)):.4f}"
            )

            return VisualFeatures(
                feature_vector=feature_vector,
                model_name=self.MODEL_NAME,
                normalization='l2' if self.normalize else 'none'
            )

        except Exception as e:
            if isinstance(e, FeatureExtractionError):
                raise
            raise FeatureExtractionError(f"Feature extraction failed: {e}") from e

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract visual features from batch of images using true batched inference.

        Args:
            images: List of RGB images as numpy arrays (H, W, 3) uint8

        Returns:
            Array of shape (N, 576) with visual features for each image

        Raises:
            FeatureExtractionError: If extraction fails
        """
        if not images:
            raise FeatureExtractionError("Empty image list provided")

        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        logger.debug(f"Extracting visual features from batch of {len(images)} images")

        try:
            # Pre-allocate output array (use zeros to prevent uninitialized memory)
            # Note: Performance difference vs empty() is negligible for this size
            batch_features = np.zeros((len(images), self.FEATURE_DIM), dtype=np.float32)

            # Convert all images to tensors and batch them
            tensor_list = []
            for i, image in enumerate(images):
                # Validate input
                if len(image.shape) != 3 or image.shape[2] != 3:
                    raise FeatureExtractionError(
                        f"Image {i}: Expected RGB image (H,W,3), got {image.shape}"
                    )

                # Convert to PIL Image with safe dtype handling
                if image.dtype != np.uint8:
                    # Detect float types (float16, float32, float64)
                    if np.issubdtype(image.dtype, np.floating):
                        # Float images: check if normalized [0, 1] or [0, 255]
                        if image.max() <= 1.0:
                            # Normalized float [0, 1] -> uint8 [0, 255]
                            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                        else:
                            # Float [0, 255] -> uint8 (clip to prevent overflow)
                            image = np.clip(image, 0, 255).astype(np.uint8)
                    else:
                        # Integer types (int8, int16, uint16, etc.)
                        # Safely convert to uint8 range [0, 255]
                        if np.issubdtype(image.dtype, np.signedinteger):
                            # Signed int -> remap to [0, 255]
                            image = np.clip((image.astype(np.float32) + 128), 0, 255).astype(np.uint8)
                        else:
                            # Unsigned int -> clip and cast
                            image = np.clip(image, 0, 255).astype(np.uint8)

                pil_image = Image.fromarray(image)
                tensor = self._transform(pil_image)
                tensor_list.append(tensor)

            # Stack into batch tensor with pinned memory for faster GPU transfers
            batch_tensor = torch.stack(tensor_list)

            # Use pinned memory for large batch transfers (CUDA only, significant speedup)
            if self._device_obj.type == 'cuda':
                batch_tensor = batch_tensor.pin_memory().to(self._device_obj, non_blocking=True)
            else:
                batch_tensor = batch_tensor.to(self._device_obj)

            # Single batched forward pass (much faster than loop!)
            with torch.inference_mode():
                features = self._model(batch_tensor)

            # Convert to numpy
            features_np = features.squeeze().detach().cpu().numpy().astype(np.float32)

            # Handle single image case (squeeze removes batch dim)
            if len(images) == 1:
                features_np = features_np.reshape(1, -1)

            # Ensure correct shape
            if features_np.shape != (len(images), self.FEATURE_DIM):
                raise FeatureExtractionError(
                    f"Expected shape ({len(images)}, {self.FEATURE_DIM}), got {features_np.shape}"
                )

            # Apply L2 normalization if requested
            if self.normalize:
                for i in range(len(images)):
                    norm = np.linalg.norm(features_np[i])
                    if norm > 0:
                        features_np[i] = features_np[i] / norm

            logger.debug(f"Batch extraction complete: shape={features_np.shape}")

            return features_np

        except Exception as e:
            if isinstance(e, FeatureExtractionError):
                raise
            raise FeatureExtractionError(f"Batch extraction failed: {e}") from e

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._device_obj = None
            self._transform = None

            # Clear GPU cache (synchronize first)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for operations first
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                if _has_mps_module():
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                # Else: PyTorch < 2.1, skip MPS cache cleanup

            logger.info("Visual model unloaded from memory")

    def unload(self) -> None:
        """Alias for unload_model() for backwards compatibility."""
        self.unload_model()
