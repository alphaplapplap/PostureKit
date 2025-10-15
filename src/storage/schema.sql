-- PostureKit Database Schema
-- PostgreSQL 14+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pose detections table with correction support
CREATE TABLE IF NOT EXISTS pose_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID REFERENCES images(id) ON DELETE CASCADE,
    keypoints FLOAT8[399] NOT NULL,  -- Current keypoints (133 * 3: x,y,conf)
    keypoint_visibility SMALLINT[133],  -- 0=not labeled, 1=occluded, 2=visible
    bbox FLOAT8[4],
    overall_confidence FLOAT8 NOT NULL,
    person_id INTEGER NOT NULL,

    -- Correction tracking
    is_corrected BOOLEAN DEFAULT FALSE,
    original_keypoints FLOAT8[399],  -- Frozen on first correction, never changes
    original_keypoint_visibility SMALLINT[133],  -- Original visibility flags
    first_corrected_at TIMESTAMPTZ,
    correction_count INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Geometric features table
CREATE TABLE IF NOT EXISTS geometric_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pose_id UUID REFERENCES pose_detections(id) ON DELETE CASCADE,
    feature_vector FLOAT8[52] NOT NULL,  -- 52-dimensional feature vector
    joint_angles JSONB,                  -- JSON object with joint angle values
    limb_ratios JSONB,                   -- JSON object with limb ratio values
    body_angles JSONB,                   -- JSON object with body angle values
    symmetry_scores JSONB,               -- JSON object with symmetry scores
    occlusion_pattern FLOAT8[7],         -- 7-dimensional occlusion encoding
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT geometric_features_unique_pose UNIQUE(pose_id)
);

-- Visual features table
CREATE TABLE IF NOT EXISTS visual_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pose_id UUID REFERENCES pose_detections(id) ON DELETE CASCADE,
    feature_vector FLOAT8[576] NOT NULL,  -- 576-dimensional visual feature vector from MobileNetV3
    model_name TEXT NOT NULL DEFAULT 'mobilenet_v3_small',
    normalization TEXT DEFAULT 'l2',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT visual_features_unique_pose UNIQUE(pose_id)
);

-- Fused multimodal features table
CREATE TABLE IF NOT EXISTS fused_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pose_id UUID REFERENCES pose_detections(id) ON DELETE CASCADE,
    geometric_feature_id UUID REFERENCES geometric_features(id) ON DELETE CASCADE,
    visual_feature_id UUID REFERENCES visual_features(id) ON DELETE CASCADE,
    fused_vector FLOAT8[628] NOT NULL,   -- 628-dimensional fused vector (52 geometric + 576 visual)
    geometric_vector FLOAT8[52] NOT NULL, -- Original 52-dim geometric features
    visual_vector FLOAT8[576] NOT NULL,   -- Original 576-dim visual features
    fusion_method TEXT NOT NULL DEFAULT 'concatenate',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fused_features_unique_pose UNIQUE(pose_id),
    CONSTRAINT fusion_method_valid CHECK (fusion_method IN ('concatenate', 'weighted', 'normalized'))
);

-- Training labels table
CREATE TABLE IF NOT EXISTS training_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pose_id UUID REFERENCES pose_detections(id) ON DELETE CASCADE,
    category TEXT,                       -- User-defined category (e.g., "standing", "sitting")
    difficulty TEXT,                     -- Annotation difficulty (easy, medium, hard)
    tags TEXT[],                         -- Array of user tags
    user_notes TEXT,                     -- Free-form notes
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Correction statistics table
CREATE TABLE IF NOT EXISTS correction_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pose_id UUID UNIQUE REFERENCES pose_detections(id) ON DELETE CASCADE,
    keypoints_corrected_count INTEGER NOT NULL,
    avg_correction_distance FLOAT8,
    max_correction_distance FLOAT8,
    corrected_keypoint_types TEXT[],
    avg_confidence_before FLOAT8,
    avg_confidence_after FLOAT8,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Correction events log (append-only history)
CREATE TABLE IF NOT EXISTS correction_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pose_id UUID REFERENCES pose_detections(id) ON DELETE CASCADE,
    correction_number INTEGER NOT NULL,
    changed_keypoint_indices SMALLINT[],  -- Which keypoints moved
    max_displacement FLOAT8,  -- Largest movement in pixels
    avg_displacement FLOAT8,  -- Average movement across changed keypoints
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_pose_detections_image_id ON pose_detections(image_id);
CREATE INDEX IF NOT EXISTS idx_pose_detections_corrected ON pose_detections(is_corrected) WHERE is_corrected = true;
CREATE INDEX IF NOT EXISTS idx_geometric_features_pose_id ON geometric_features(pose_id);
CREATE INDEX IF NOT EXISTS idx_visual_features_pose_id ON visual_features(pose_id);
CREATE INDEX IF NOT EXISTS idx_fused_features_pose_id ON fused_features(pose_id);
CREATE INDEX IF NOT EXISTS idx_fused_features_geometric_id ON fused_features(geometric_feature_id);
CREATE INDEX IF NOT EXISTS idx_fused_features_visual_id ON fused_features(visual_feature_id);
CREATE INDEX IF NOT EXISTS idx_training_labels_pose_id ON training_labels(pose_id);
CREATE INDEX IF NOT EXISTS idx_training_labels_category ON training_labels(category);
CREATE INDEX IF NOT EXISTS idx_correction_statistics_pose_id ON correction_statistics(pose_id);
CREATE INDEX IF NOT EXISTS idx_correction_statistics_keypoint_types ON correction_statistics USING GIN(corrected_keypoint_types);
CREATE INDEX IF NOT EXISTS idx_correction_events_pose_id ON correction_events(pose_id);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
CREATE INDEX IF NOT EXISTS idx_pose_detections_created_at ON pose_detections(created_at);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_pose_detections_updated_at
    BEFORE UPDATE ON pose_detections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_visual_features_updated_at
    BEFORE UPDATE ON visual_features
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_labels_updated_at
    BEFORE UPDATE ON training_labels
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE images IS 'Stores metadata about processed images';
COMMENT ON TABLE pose_detections IS 'Stores detected poses with 133 keypoints from RTMW-L model';
COMMENT ON TABLE geometric_features IS 'Stores 52-dimensional geometric feature vectors for pose similarity';
COMMENT ON TABLE visual_features IS 'Stores 576-dimensional visual features from MobileNetV3-Small';
COMMENT ON TABLE fused_features IS 'Stores 628-dimensional fused multimodal features combining geometric and visual';
COMMENT ON TABLE training_labels IS 'Stores user annotations and corrections for training';
COMMENT ON TABLE correction_statistics IS 'Stores statistics about manual corrections made to poses for training analysis';

COMMENT ON COLUMN pose_detections.keypoints IS 'Flattened array of 133 keypoints, each with [x, y, confidence]';
COMMENT ON COLUMN pose_detections.is_corrected IS 'True if user has manually corrected this pose';
COMMENT ON COLUMN geometric_features.feature_vector IS '52-dim vector: 12 joint angles + 10 limb ratios + 15 body angles + 8 symmetry + 7 occlusion';
COMMENT ON COLUMN visual_features.feature_vector IS '576-dim visual features from MobileNetV3-Small';
COMMENT ON COLUMN fused_features.fused_vector IS '628-dim fused features: 52 geometric + 576 visual';
COMMENT ON COLUMN fused_features.fusion_method IS 'Fusion strategy used: concatenate, weighted, or normalized';
