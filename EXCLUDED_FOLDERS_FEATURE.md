# Excluded Folders Feature

## Overview
This feature allows you to exclude specific folders from search results and automatically removes images from the database when they are moved to excluded folders.

## Features

### 1. Folder Exclusion Management
- **Add Excluded Folders**: Select folders that should be excluded from search results
- **Remove Excluded Folders**: Remove folders from the exclusion list
- **View Excluded Folders**: See all currently excluded folders with their paths and notes
- **Optional Notes**: Add notes to remember why a folder was excluded

### 2. Automatic Database Updates
- **Auto-Removal on Move**: When you move files to an excluded folder, they are automatically removed from the database
- **Real-time Monitoring**: The folder watcher detects file moves in real-time
- **Batch Processing**: Multiple file moves are debounced and processed efficiently

### 3. Search Result Filtering
- **Transparent Filtering**: Excluded folders are automatically filtered from all search results
- **No Manual Intervention**: Once a folder is excluded, you never see results from it

## Implementation Details

### Database Schema
**New Table: `excluded_folders`**
- `id`: UUID primary key
- `folder_path`: Absolute path to excluded folder (unique)
- `created_at`: Timestamp when folder was added
- `notes`: Optional user notes

**Migration**: `migrations/001_add_excluded_folders.sql`

### Backend Components

#### 1. Storage Manager (src/storage/storage_manager.py)
New methods:
- `add_excluded_folder(folder_path, notes=None)`: Add folder to exclusion list
- `remove_excluded_folder(folder_path)`: Remove folder from exclusion list
- `get_excluded_folders()`: Get list of all excluded folders
- `is_path_excluded(file_path)`: Check if a file path is in an excluded folder
- `delete_images_in_excluded_folders()`: Delete all images in excluded folders

#### 2. Folder Watcher (src/storage/folder_watcher.py)
Enhanced `ImageFileHandler`:
- Accepts `storage_manager` parameter for exclusion checks
- Modified `on_moved()` to detect moves to excluded folders
- Automatically queues files for deletion when moved to excluded folders

#### 3. Search Engine (src/intelligence/similarity_engine.py)
Enhanced `search_by_feature()`:
- Added exclusion check before returning results (line 772-775)
- Filters out images from excluded folders automatically

### Frontend Components

#### 1. Settings View (PostureKit/SettingsView.swift)
New sections:
- **ExcludedFoldersSection**: Displays list of excluded folders with add/remove controls
- **AddExcludedFolderDialog**: Modal dialog for selecting folders to exclude

#### 2. Settings View Model
New properties:
- `excludedFolders`: Array of excluded folder objects
- `isLoadingExcludedFolders`: Loading state indicator

New methods:
- `loadExcludedFolders()`: Load excluded folders from database
- `addExcludedFolder(folderPath, notes)`: Add folder and delete existing images
- `removeExcludedFolder(folderPath)`: Remove folder from exclusion list

## Usage

### Adding an Excluded Folder
1. Open Settings (⚙️ icon in top-right)
2. Scroll to "Excluded Folders" section
3. Click "Add Folder" button
4. Browse and select the folder you want to exclude
5. Optionally add notes explaining why
6. Click "Add"

**What happens:**
- Folder is added to exclusion list
- All existing images in that folder are immediately removed from the database
- Future searches will never show results from this folder

### Removing an Excluded Folder
1. Open Settings
2. Scroll to "Excluded Folders" section
3. Find the folder in the list
4. Click the ❌ button next to it

**What happens:**
- Folder is removed from exclusion list
- Future searches CAN show results from this folder again
- Note: You'll need to re-index the folder if you want those images back in the database

### Automatic File Move Detection
When you move a file to an excluded folder (via Finder or any app):
1. Folder watcher detects the move event
2. Checks if destination is in an excluded folder
3. If yes, automatically deletes the image from the database
4. This happens in the background with a 10-second debounce

## Testing

### Test Scenario 1: Add Excluded Folder
1. Add a folder to exclusions
2. Verify existing images from that folder disappear from search results immediately
3. Check Settings UI shows the folder in the list

### Test Scenario 2: Move File to Excluded Folder
1. Have folder watching enabled
2. Add a folder to exclusions
3. Move an image file to that folder
4. Wait 10-15 seconds for debounce
5. Search for that image - it should not appear in results
6. Check database - image record should be deleted

### Test Scenario 3: Remove Excluded Folder
1. Remove a folder from exclusions
2. The folder should disappear from Settings UI
3. Note: Images won't reappear until you re-index the folder

## Database Migration

To add the excluded_folders table to existing databases:

```bash
# Run the migration SQL script
psql -U your_user -d posturekit_irl < migrations/001_add_excluded_folders.sql
psql -U your_user -d posturekit_2d < migrations/001_add_excluded_folders.sql
psql -U your_user -d posturekit_3d < migrations/001_add_excluded_folders.sql
```

Or use the SQLAlchemy ORM approach (automatically handles it):
```python
from src.storage.models import Base
from src.storage.storage_manager import StorageManager

storage = StorageManager()
Base.metadata.create_all(storage.engine)
```

## Performance Considerations

1. **Exclusion Check Performance**: Uses indexed lookups on `folder_path`
2. **Search Filtering**: Minimal overhead - single string comparison per result
3. **Batch Deletion**: Efficient batch operations when adding excluded folders
4. **Debouncing**: 10-second debounce prevents excessive database operations

## Architecture Decisions

### Why exclude instead of delete immediately?
- **Reversible**: Users can remove folders from exclusions later
- **Flexible**: Can re-index folders if needed
- **Safe**: Prevents accidental data loss

### Why auto-delete on move?
- **User Intent**: Moving to excluded folder signals intent to remove from database
- **Consistency**: Keeps database in sync with user's file organization
- **Storage**: Prevents database bloat from files user explicitly excluded

### Why 10-second debounce on deletes?
- **Safety**: Gives users time to undo moves
- **Efficiency**: Batches multiple moves together
- **UX**: Reduces database churn during bulk operations

## Future Enhancements

Potential improvements:
1. **Regex Patterns**: Support excluding folders by pattern (e.g., `*/trash/*`)
2. **Temporary Exclusions**: Time-limited exclusions that expire
3. **Exclusion Statistics**: Show count of images excluded per folder
4. **Import/Export**: Share exclusion lists across profiles
5. **Nested Exclusions**: Exclude/include specific subfolders within excluded folders

## Troubleshooting

### Images not being filtered
- Check that folder path is exact (case-sensitive)
- Verify folder was added successfully (check Settings)
- Check database: `SELECT * FROM excluded_folders;`

### Files not auto-deleting on move
- Ensure folder watching is enabled
- Check that storage_manager is passed to FolderWatcher
- Wait 10-15 seconds for debounce
- Check logs for error messages

### Cannot add folder
- Ensure folder path exists and is absolute
- Check database permissions
- Verify migration was run successfully
