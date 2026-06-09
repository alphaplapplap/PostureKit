-- Migration: Add excluded_folders table
-- Date: 2025-10-18
-- Description: Adds excluded_folders table for filtering search results and auto-removing moved files

-- Create excluded_folders table
CREATE TABLE IF NOT EXISTS excluded_folders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    folder_path TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- Add index for fast lookups
CREATE INDEX IF NOT EXISTS idx_excluded_folders_path ON excluded_folders(folder_path);

-- Add comment for documentation
COMMENT ON TABLE excluded_folders IS 'Excluded folders for filtering search results and auto-removal of moved files';
COMMENT ON COLUMN excluded_folders.folder_path IS 'Absolute path to excluded folder';
COMMENT ON COLUMN excluded_folders.notes IS 'Optional user notes about why folder is excluded';
