"""
PyTorch 2.6 compatibility patch.
Must be imported before any mmengine or mmpose modules.
"""
import torch
import sys

# Monkey-patch torch.load to disable weights_only for PyTorch 2.6+ compatibility
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Force weights_only=False for compatibility with numpy-based checkpoints."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Patch safe_load to fallback to unsafe load
_original_safe_load = getattr(torch.serialization, 'safe_load', None)
if _original_safe_load:
    torch.serialization.safe_load = _patched_torch_load

# Add numpy globals to safe_globals list for torch.serialization
try:
    import numpy
    torch.serialization.add_safe_globals([
        numpy.core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype,
        numpy.core.multiarray.scalar,
    ])
except Exception:
    pass

# Patch mmengine's checkpoint loading
def _apply_mmengine_patch():
    try:
        import mmengine.runner.checkpoint as checkpoint

        # Patch torch.load reference in checkpoint module - this is the key!
        # mmengine calls checkpoint.torch.load internally
        checkpoint.torch.load = _patched_torch_load

    except (ImportError, AttributeError) as e:
        pass

# Apply mmengine patch immediately if already imported
if 'mmengine' in sys.modules:
    _apply_mmengine_patch()
