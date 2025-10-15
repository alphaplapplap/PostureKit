"""
Database Manager - GUI adapter for StorageManager.
Provides backward compatibility for GUI code.
"""
from typing import List, Optional
from pathlib import Path

from src.storage.storage_manager import StorageManager
from src.core.models import Pose
from src.utils.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class DummySignal:
    """Dummy signal for non-QObject classes."""
    def connect(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


class DatabaseManager:
    """
    GUI adapter wrapping StorageManager.

    Provides backward-compatible interface for existing GUI code
    while using the new StorageManager backend.
    """

    def __init__(self, connection_string: Optional[str] = None, database_profile: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            connection_string: PostgreSQL connection string (optional)
            database_profile: Database profile (irl/2d/3d). Overrides connection_string if provided.
        """
        self.storage = StorageManager(connection_string, database_profile=database_profile)

        # Dummy signal for GUI compatibility
        self.connectionChanged = DummySignal()

        logger.info("DatabaseManager initialized (wrapping StorageManager)")

    def isConnected(self) -> bool:
        """Check if database is connected."""
        try:
            # Try a simple query
            stats = self.storage.get_statistics()
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def getPosesForImage(self, file_path: str) -> List[Pose]:
        """
        Get all poses for a given image.

        Args:
            file_path: Path to image file

        Returns:
            List containing single Pose object with all persons
        """
        try:
            poses_data = self.storage.get_poses_by_image(Path(file_path))

            if not poses_data:
                return []

            # Convert each database row to a Person object
            from core.models import Person, Keypoint
            import numpy as np

            persons = []
            for data in poses_data:
                # Validate required NOT NULL database fields
                if 'keypoints' not in data:
                    logger.error(f"Missing required field 'keypoints' in pose data")
                    continue
                if 'person_id' not in data:
                    logger.error(f"Missing required field 'person_id' in pose data")
                    continue
                if 'confidence' not in data:
                    logger.error(f"Missing required field 'confidence' in pose data")
                    continue

                # Extract keypoints (NOT NULL field)
                keypoints_flat = data['keypoints']
                expected_dim = settings.KEYPOINTS_FLAT_DIM

                if not isinstance(keypoints_flat, list) or len(keypoints_flat) != expected_dim:
                    logger.error(f"Invalid keypoints array: expected {expected_dim} elements, got {len(keypoints_flat) if isinstance(keypoints_flat, list) else 'non-list'}")
                    continue

                # Reshape to (NUM_KEYPOINTS, KEYPOINT_COORDINATES_DIM)
                kp_array = np.array(keypoints_flat).reshape(
                    settings.NUM_KEYPOINTS,
                    settings.KEYPOINT_COORDINATES_DIM
                )

                # Extract visibility (nullable field)
                visibility_data = data.get('keypoint_visibility') or data.get('visibility')

                from core.models import KEYPOINT_NAMES
                # COCO visibility constants: 0=missing, 1=occluded, 2=visible
                VISIBILITY_VISIBLE = 2

                keypoints = []
                for kp_idx, kp in enumerate(kp_array):
                    # Validate keypoint name exists in KEYPOINT_NAMES
                    if kp_idx not in KEYPOINT_NAMES:
                        raise ValueError(f"Invalid keypoint index {kp_idx}: not in KEYPOINT_NAMES")

                    # Determine visibility: use database value if available, otherwise COCO standard VISIBLE
                    if visibility_data is not None and kp_idx < len(visibility_data):
                        visibility = int(visibility_data[kp_idx])
                    else:
                        # NULL visibility in DB means keypoint is VISIBLE per COCO standard
                        visibility = VISIBILITY_VISIBLE

                    keypoints.append(Keypoint(
                        index=kp_idx,
                        name=KEYPOINT_NAMES[kp_idx],
                        x=float(kp[0]),
                        y=float(kp[1]),
                        confidence=float(kp[2]),
                        visibility=visibility
                    ))

                # Extract bbox (nullable field - calculate from keypoints if NULL)
                bbox_data = data.get('bbox')
                if bbox_data and len(bbox_data) == 4:
                    bbox_tuple = tuple(bbox_data)
                else:
                    # Calculate from visible keypoints
                    xs = [kp.x for kp in keypoints if kp.visibility > 0]
                    ys = [kp.y for kp in keypoints if kp.visibility > 0]
                    if xs and ys:
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        bbox_tuple = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                    else:
                        logger.warning(f"No visible keypoints for person_id={data['person_id']}, cannot calculate bbox")
                        continue  # Skip persons with no visible keypoints and no bbox

                # Extract labels (optional JSONB field)
                labels = {}
                if 'labels' in data and data['labels']:
                    labels_dict = data['labels']
                    # Only include keys that exist in the database JSONB
                    if 'category' in labels_dict:
                        labels['category'] = labels_dict['category']
                    if 'tags' in labels_dict:
                        labels['tags'] = labels_dict['tags']
                    if 'user_notes' in labels_dict:
                        labels['notes'] = labels_dict['user_notes']

                person = Person(
                    id=data['person_id'],  # Use database person_id
                    bbox=bbox_tuple,
                    keypoints=keypoints,
                    confidence=data['confidence'],  # NOT NULL field
                    total_count=len(poses_data),
                    labels=labels
                )
                persons.append(person)

            # Create single Pose object containing all persons
            # is_corrected is NOT NULL with DEFAULT FALSE in database - must always be present
            pose = Pose(
                image_path=file_path,
                persons=persons,
                is_modified=any(data['is_corrected'] for data in poses_data)
            )

            return [pose]  # Return list for compatibility

        except Exception as e:
            logger.error(f"Failed to get poses for image {file_path}: {e}")
            return []

    def getPoseById(self, pose_id: str) -> Optional[Pose]:
        """
        Get pose by ID.

        Args:
            pose_id: UUID of pose

        Returns:
            Pose object or None
        """
        try:
            from uuid import UUID
            data = self.storage.get_pose_by_id(UUID(pose_id))

            if data:
                return self._convert_to_pose(data)
            return None

        except Exception as e:
            logger.error(f"Failed to get pose {pose_id}: {e}")
            return None

    def getAllPoses(self) -> List[Pose]:
        """
        Get all poses in database.

        Returns:
            List of all Pose objects
        """
        try:
            stats = self.storage.get_statistics()
            # For now, return empty list - this would need pagination for real use
            logger.warning("getAllPoses() called - not implemented for large datasets")
            return []

        except Exception as e:
            logger.error(f"Failed to get all poses: {e}")
            return []

    def updatePose(self, pose: Pose) -> bool:
        """
        Update pose with corrections.

        Args:
            pose: Pose object with updated data

        Returns:
            True if successful
        """
        try:
            from uuid import UUID
            import numpy as np

            # Extract keypoints and visibility if available
            keypoints = None
            visibility = None
            if hasattr(pose, 'keypoints') and pose.keypoints:
                # Convert Pose keypoints to (133, 3) ndarray
                keypoints = np.array([[kp.x, kp.y, kp.confidence] for kp in pose.keypoints])
                # Extract visibility as (133,) ndarray
                visibility = np.array([kp.visibility for kp in pose.keypoints])

            # Extract category/tags/notes if available
            category = getattr(pose, 'category', None)
            tags = getattr(pose, 'tags', None)
            user_notes = getattr(pose, 'notes', None)

            success = self.storage.update_pose_correction(
                pose_id=UUID(str(pose.id)),
                keypoints=keypoints,
                visibility=visibility,
                category=category,
                tags=tags,
                user_notes=user_notes
            )

            return success

        except Exception as e:
            logger.error(f"Failed to update pose: {e}")
            return False

    def deletePose(self, pose_id: str) -> bool:
        """
        Delete pose from database.

        Args:
            pose_id: UUID of pose to delete

        Returns:
            True if successful
        """
        try:
            from uuid import UUID
            return self.storage.delete_pose(UUID(pose_id))

        except Exception as e:
            logger.error(f"Failed to delete pose {pose_id}: {e}")
            return False

    def getPoseCount(self) -> int:
        """
        Get total number of poses in database.

        Returns:
            Count of poses
        """
        try:
            stats = self.storage.get_statistics()
            if 'total_poses' not in stats:
                raise ValueError("Missing 'total_poses' in statistics result")
            return stats['total_poses']

        except Exception as e:
            logger.error(f"Failed to get pose count: {e}")
            raise

    def _convert_to_pose(self, data: dict) -> Optional[Pose]:
        """
        Convert storage format to Pose object.

        Args:
            data: Dict from StorageManager (single person)

        Returns:
            Pose object containing single Person
        """
        try:
            from core.models import Keypoint, Person
            import numpy as np

            # Validate required fields (NOT NULL in database)
            if 'keypoints' not in data:
                raise ValueError("Missing required field: keypoints")
            if 'person_id' not in data:
                raise ValueError("Missing required field: person_id")
            if 'confidence' not in data:
                raise ValueError("Missing required field: confidence")

            # Extract keypoints (NOT NULL field)
            keypoints_flat = data['keypoints']
            expected_dim = settings.KEYPOINTS_FLAT_DIM

            if not isinstance(keypoints_flat, list) or len(keypoints_flat) != expected_dim:
                raise ValueError(f"Invalid keypoints array: expected {expected_dim} elements, got {len(keypoints_flat) if isinstance(keypoints_flat, list) else 'non-list'}")

            # Reshape to (NUM_KEYPOINTS, KEYPOINT_COORDINATES_DIM)
            kp_array = np.array(keypoints_flat).reshape(
                settings.NUM_KEYPOINTS,
                settings.KEYPOINT_COORDINATES_DIM
            )

            # Extract visibility (nullable field)
            visibility_data = data.get('keypoint_visibility') or data.get('visibility')

            from core.models import KEYPOINT_NAMES
            # COCO visibility constants: 0=missing, 1=occluded, 2=visible
            VISIBILITY_VISIBLE = 2

            keypoints = []
            for idx, kp in enumerate(kp_array):
                # Validate keypoint name exists in KEYPOINT_NAMES
                if idx not in KEYPOINT_NAMES:
                    raise ValueError(f"Invalid keypoint index {idx}: not in KEYPOINT_NAMES")

                # Determine visibility: use database value if available, otherwise COCO standard VISIBLE
                if visibility_data is not None and idx < len(visibility_data):
                    visibility = int(visibility_data[idx])
                else:
                    # NULL visibility in DB means keypoint is VISIBLE per COCO standard
                    visibility = VISIBILITY_VISIBLE

                keypoints.append(Keypoint(
                    index=idx,
                    name=KEYPOINT_NAMES[idx],
                    x=float(kp[0]),
                    y=float(kp[1]),
                    confidence=float(kp[2]),
                    visibility=visibility
                ))

            # Extract bbox (nullable field - calculate from keypoints if NULL)
            bbox_data = data.get('bbox')
            if bbox_data and len(bbox_data) == 4:
                bbox_tuple = tuple(bbox_data)
            else:
                # Calculate bbox from visible keypoints
                xs = [kp.x for kp in keypoints if kp.visibility > 0]
                ys = [kp.y for kp in keypoints if kp.visibility > 0]
                if not xs or not ys:
                    raise ValueError(f"Cannot calculate bbox: no visible keypoints for person_id={data['person_id']}")
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bbox_tuple = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

            # Extract labels (optional JSONB field)
            labels = {}
            if 'labels' in data and data['labels']:
                labels_dict = data['labels']
                # Only include keys that exist in the database JSONB
                if 'category' in labels_dict:
                    labels['category'] = labels_dict['category']
                if 'tags' in labels_dict:
                    labels['tags'] = labels_dict['tags']
                if 'user_notes' in labels_dict:
                    labels['notes'] = labels_dict['user_notes']

            # Get person_id from database (NOT NULL field)
            person_id = data['person_id']

            # Get image_path from database (always present from JOIN)
            if 'image_path' not in data:
                raise ValueError("Missing required field: image_path")
            image_path = data['image_path']

            try:
                all_poses_in_image = self.storage.get_poses_by_image(Path(image_path))
                total_count = len(all_poses_in_image)
            except Exception as e:
                logger.error(f"Failed to get total person count for {image_path}: {e}")
                raise

            person = Person(
                id=person_id,
                bbox=bbox_tuple,
                keypoints=keypoints,
                confidence=data['confidence'],
                total_count=total_count,
                labels=labels
            )

            # Validate is_corrected field (NOT NULL with DEFAULT FALSE in database)
            if 'is_corrected' not in data:
                raise ValueError("Missing required field: is_corrected")

            # Create Pose object with single person
            pose = Pose(
                image_path=image_path,
                persons=[person],
                is_modified=data['is_corrected']
            )

            return pose

        except Exception as e:
            logger.error(f"Failed to convert pose data: {e}")
            return None

    def close(self):
        """Close database connections."""
        if self.storage:
            self.storage.close()
