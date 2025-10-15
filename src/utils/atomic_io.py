"""Atomic file I/O operations to prevent corruption."""

import os
import tempfile
import pickle
import json
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def atomic_write(path: Path, data: bytes, mode: str = 'wb') -> None:
    """
    Atomically write data to a file using temp file + rename.

    Args:
        path: Target file path
        data: Data to write
        mode: File mode ('wb' for binary, 'w' for text)

    Raises:
        IOError: If write fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomic rename (POSIX guarantee)
        os.replace(tmp_path, path)
        logger.debug(f"Atomically wrote {len(data)} bytes to {path}")

    except Exception as e:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except:
            pass
        raise IOError(f"Failed to atomically write {path}: {e}")


def atomic_pickle_dump(obj: Any, path: Path, protocol: int = 4) -> None:
    """
    Atomically save Python object using pickle.

    Args:
        obj: Object to pickle
        path: Target file path
        protocol: Pickle protocol version (4 = Python 3.4+ compatible)
    """
    data = pickle.dumps(obj, protocol=protocol)
    atomic_write(path, data, mode='wb')


def atomic_json_dump(obj: Any, path: Path, indent: int = 2) -> None:
    """
    Atomically save JSON data.

    Args:
        obj: JSON-serializable object
        path: Target file path
        indent: JSON indentation
    """
    data = json.dumps(obj, indent=indent).encode('utf-8')
    atomic_write(path, data, mode='wb')


def atomic_multi_write(files: Dict[Path, bytes]) -> None:
    """
    Atomically write multiple files (all or nothing).

    If any write fails, all temp files are cleaned up and original files unchanged.

    Args:
        files: Dict mapping file paths to data

    Raises:
        IOError: If any write fails
    """
    temp_paths = {}

    try:
        # Write all to temp files first
        for path, data in files.items():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp"
            )

            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            temp_paths[path] = tmp_path

        # Atomic rename all at once
        for path, tmp_path in temp_paths.items():
            os.replace(tmp_path, path)

        logger.info(f"Atomically wrote {len(files)} files")

    except Exception as e:
        # Clean up all temp files on failure
        for tmp_path in temp_paths.values():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise IOError(f"Failed to atomically write multiple files: {e}")
