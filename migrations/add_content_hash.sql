-- Migration: Add content_hash column for duplicate detection
-- Date: 2025-10-09

-- Add content_hash column to images table
ALTER TABLE images
ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- Create index for fast duplicate detection
CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);

-- Add comment
COMMENT ON COLUMN images.content_hash IS 'SHA-256 hash of image file content for deduplication';
