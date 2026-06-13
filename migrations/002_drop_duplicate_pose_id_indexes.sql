-- Migration: Drop duplicate pose_id indexes
-- Date: 2026-06-13
-- Description:
--   geometric_features, visual_features, fused_features, and
--   correction_statistics each declared pose_id with unique=True (backed by a
--   UNIQUE index named *_pose_id_key) AND a redundant plain btree index named
--   idx_*_pose_id. The two are strict duplicates on the same single column;
--   the unique index serves every pose_id lookup, join, and .in_() fetch.
--
--   This drops the redundant plain indexes. It is idempotent (IF EXISTS) and a
--   safe no-op on the empty 2d/3d profiles. The unique constraint indexes
--   (*_pose_id_key) are intentionally NOT touched -- they back the unique
--   constraints and must remain.
--
--   Run per DB_PROFILE via scripts/run_drop_duplicate_indexes_migration.py.

DROP INDEX IF EXISTS idx_geometric_features_pose_id;
DROP INDEX IF EXISTS idx_visual_features_pose_id;
DROP INDEX IF EXISTS idx_fused_features_pose_id;
DROP INDEX IF EXISTS idx_correction_statistics_pose_id;
