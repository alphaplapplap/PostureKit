"""
Centralized progress reporting for indexing and batch operations.
"""

import json
import sys
from typing import Optional


class ProgressReporter:
    """Handles progress reporting with consistent formatting."""

    @staticmethod
    def send_progress(
        current_file: str,
        images_processed: int,
        total_images: int,
        poses_indexed: int,
        failed_images: int,
        skipped_images: int,
        failed_index_additions: int = 0
    ) -> None:
        """
        Send progress update to stdout for Swift to parse.

        Args:
            current_file: Name of file being processed
            images_processed: Number of images processed so far
            total_images: Total number of images to process
            poses_indexed: Number of poses successfully indexed
            failed_images: Number of images that failed processing
            skipped_images: Number of images skipped
            failed_index_additions: Number of poses that failed to add to index
        """
        progress = {
            'type': 'progress',
            'current_file': str(current_file),
            'images_processed': images_processed,
            'total_images': total_images,
            'poses_indexed': poses_indexed,
            'failed_images': failed_images,
            'skipped_images': skipped_images,
            'failed_index_additions': failed_index_additions,
            'progress': images_processed / total_images if total_images > 0 else 0
        }

        # Send to Swift via stdout
        print(f"PROGRESS:{json.dumps(progress)}", flush=True)

    @staticmethod
    def send_thumbnail_progress(
        images_processed: int,
        images_updated: int,
        images_failed: int,
        images_skipped: int,
        total_images: Optional[int] = None
    ) -> None:
        """
        Send thumbnail generation progress.

        Args:
            images_processed: Number of images processed
            images_updated: Number successfully updated
            images_failed: Number that failed
            images_skipped: Number skipped
            total_images: Total to process (if known)
        """
        progress = {
            'type': 'thumbnail_progress',
            'images_processed': images_processed,
            'images_updated': images_updated,
            'images_failed': images_failed,
            'images_skipped': images_skipped
        }

        if total_images is not None:
            progress['total_images'] = total_images
            progress['progress'] = images_processed / total_images if total_images > 0 else 0

        print(json.dumps(progress), flush=True)
