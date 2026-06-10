#!/usr/bin/env python3
"""
Persistent search server for PostureKit.
Keeps the PostureKitBridge loaded to avoid re-initialization overhead.
Accepts JSON commands via stdin, returns JSON responses via stdout.
"""
import sys
import json
import logging
import traceback
import signal
import atexit
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging to stderr (stdout is for JSON responses)
logging.basicConfig(
    level=logging.INFO,
    format='[SEARCH SERVER] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Import bridge
try:
    from src.swift_bridge import PostureKitBridge
    logger.info("Swift bridge imported successfully")
except ImportError as e:
    logger.error(f"Failed to import swift_bridge: {e}")
    logger.error(f"sys.path: {sys.path}")
    sys.exit(1)

# CRITICAL: Disable FAISS multithreading to prevent OpenMP pthread_mutex conflicts
# When running as subprocess alongside main app, OpenMP runtimes conflict
# Single searches don't benefit from parallelism anyway (I/O bound)
try:
    import faiss
    faiss.omp_set_num_threads(1)
    logger.info("FAISS threading disabled (num_threads=1) to prevent OpenMP conflicts")
except Exception as e:
    logger.warning(f"Failed to configure FAISS threading (non-fatal): {e}")

# Global bridge instance (initialized on first command)
bridge = None


def cleanup_gracefully():
    """
    Graceful cleanup of database connections and resources.
    Called on:
    - Explicit 'shutdown' command
    - SIGTERM/SIGINT signals
    - Normal exit (atexit)
    - Process termination
    """
    global bridge

    if bridge is None:
        logger.info("Cleanup: Bridge never initialized, nothing to clean")
        return

    logger.info("=" * 70)
    logger.info("GRACEFUL CLEANUP - Releasing database connections")
    logger.info("=" * 70)

    try:
        # Close similarity engine (saves index, clears caches)
        if hasattr(bridge, 'similarity_engine') and bridge.similarity_engine:
            logger.info("Closing similarity engine...")
            bridge.similarity_engine.close()
            logger.info("✓ Similarity engine closed")

        # Close storage manager (closes database sessions)
        if hasattr(bridge, 'storage_manager') and bridge.storage_manager:
            logger.info("Closing storage manager...")
            bridge.storage_manager.close()
            logger.info("✓ Storage manager closed")

        # Dispose ALL shared connection pools
        logger.info("Disposing all shared database engines...")
        from src.storage.storage_manager import StorageManager
        StorageManager.close_all_engines()
        logger.info("✓ All connection pools disposed")

        logger.info("✓ Cleanup complete - external drives safe to eject")

    except Exception as e:
        logger.error(f"⚠ Cleanup error: {e}")
        # Continue anyway - best effort cleanup

    finally:
        logger.info("=" * 70)


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT gracefully."""
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    logger.info(f"Received {signal_name} - initiating graceful shutdown")
    cleanup_gracefully()
    sys.exit(0)


def initialize_bridge(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the bridge with given configuration."""
    global bridge
    try:
        logger.info(f"Initializing bridge with config: {config}")
        # Skip loading pose detection models (400MB+) - we only need search
        bridge = PostureKitBridge(
            num_threads=config.get('num_threads', 4),
            device=config.get('device', 'cpu'),
            skip_models=True  # CRITICAL: Avoid loading 400MB+ pose models for search-only
        )
        logger.info("Bridge initialized successfully (search-only mode)")
        return {
            'status': 'success',
            'message': 'Bridge initialized (search-only)'
        }
    except Exception as e:
        logger.error(f"Bridge initialization failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a search request."""
    global bridge

    # Initialize bridge on first search if not already initialized
    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        # Extract search parameters
        feature_vector = params['feature_vector']
        k = params.get('k', 20)
        min_confidence = params.get('min_confidence', 0.5)
        query_confidence = params.get('query_confidence')
        min_feature_confidence = params.get('min_feature_confidence', 0.35)
        min_valid_overlap = params.get('min_valid_overlap', 12)
        required_regions = params.get('required_regions')
        min_region_confidence = params.get('min_region_confidence', 0.3)
        min_similarity = params.get('min_similarity', 0.0)  # Similarity floor (0-1); 0 = top-k mode
        deduplicate_images = params.get('deduplicate_images', False)  # Default: show all people
        # Optional OKS / flip-search inputs (absent in older requests → unchanged behavior)
        query_keypoints = params.get('query_keypoints')  # nested 133×3 [x, y, conf]
        query_bbox = params.get('query_bbox')            # [x, y, w, h]
        enable_flip_search = bool(params.get('include_flipped', False))

        logger.info(f"Searching with k={k}, min_confidence={min_confidence}, min_similarity={min_similarity}, "
                    f"deduplicate_images={deduplicate_images}, has_keypoints={query_keypoints is not None}, "
                    f"has_bbox={query_bbox is not None}, flip={enable_flip_search}")
        logger.info(f"About to call bridge.search_similar()...")

        # Perform search
        results = bridge.search_similar(
            feature_vector=feature_vector,
            k=k,
            min_confidence=min_confidence,
            query_confidence=query_confidence,
            min_feature_confidence=min_feature_confidence,
            min_valid_overlap=min_valid_overlap,
            required_regions=required_regions,
            min_region_confidence=min_region_confidence,
            min_similarity=min_similarity,
            deduplicate_images=deduplicate_images,
            query_keypoints=query_keypoints,
            query_bbox=query_bbox,
            enable_flip_search=enable_flip_search
        )

        logger.info(f"Search complete, found {len(results)} results")

        return {
            'status': 'success',
            'results': results
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_statistics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a statistics request."""
    global bridge

    # Initialize bridge if not already initialized
    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        stats = bridge.similarity_engine.get_statistics()
        return {
            'status': 'success',
            'statistics': stats
        }
    except Exception as e:
        logger.error(f"Statistics failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_excluded_folders(params: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Handle excluded-folder management (list / add / remove)."""
    global bridge

    # Initialize bridge if not already initialized
    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        storage = bridge.storage_manager
        if action == 'list':
            folders = storage.get_excluded_folders()
            return {'status': 'success', 'folders': folders}
        elif action == 'add':
            folder_path = params['folder_path']
            entry = storage.add_excluded_folder(folder_path, notes=params.get('notes'))
            # Cached search results may contain newly-excluded images — drop them
            bridge.similarity_engine.clear_search_cache()
            logger.info(f"Excluded folder added: {entry['folder_path']}")
            return {'status': 'success', 'folder': entry}
        elif action == 'remove':
            folder_path = params['folder_path']
            removed = storage.remove_excluded_folder(folder_path)
            bridge.similarity_engine.clear_search_cache()
            logger.info(f"Excluded folder removed: {folder_path} (found={removed})")
            return {'status': 'success', 'removed': removed}
        else:
            return {'status': 'error', 'message': f'Unknown excluded-folders action: {action}'}
    except Exception as e:
        logger.error(f"Excluded folders {action} failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_browse_body_parts(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a browse by body parts request."""
    global bridge

    # Initialize bridge if not already initialized
    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        # Extract browse parameters
        required_regions = params.get('required_regions', [])
        category_thresholds = params.get('category_thresholds', {})
        k = params.get('k', 50)
        sort_by = params.get('sort_by', 'confidence')
        seed = params.get('seed')

        logger.info(f"Browsing by body parts: regions={required_regions}, k={k}, sort={sort_by}")
        logger.info(f"Category thresholds: {category_thresholds}")

        # Perform browse
        results = bridge.browse_by_body_parts(
            required_regions=required_regions if required_regions else None,
            category_thresholds=category_thresholds if category_thresholds else None,
            k=k,
            sort_by=sort_by,
            seed=seed
        )

        logger.info(f"Browse complete, found {len(results)} results")

        return {
            'status': 'success',
            'results': results,
            'count': len(results)
        }

    except Exception as e:
        logger.error(f"Browse by body parts failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def main():
    """Main server loop - read commands from stdin, write responses to stdout."""
    logger.info("Search server starting...")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    # Register graceful cleanup handlers
    # This ensures database connections are closed even if process is terminated
    signal.signal(signal.SIGTERM, signal_handler)  # Handle Swift's .terminate()
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    atexit.register(cleanup_gracefully)            # Handle normal exit
    logger.info("✓ Cleanup handlers registered (SIGTERM, SIGINT, atexit)")

    # Signal ready
    print(json.dumps({'status': 'ready'}), flush=True)

    while True:
        try:
            # Read command from stdin (blocking)
            line = sys.stdin.readline()
            if not line:
                logger.info("EOF received, shutting down")
                break

            line = line.strip()
            if not line:
                continue

            # Parse command
            try:
                command = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                response = {
                    'status': 'error',
                    'message': f'Invalid JSON: {e}'
                }
                print(json.dumps(response), flush=True)
                continue

            # Handle command
            cmd_type = command.get('command')
            params = command.get('params', {})

            if cmd_type == 'search':
                response = handle_search(params)
            elif cmd_type == 'statistics':
                response = handle_statistics(params)
            elif cmd_type == 'browse_body_parts':
                response = handle_browse_body_parts(params)
            elif cmd_type == 'list_excluded_folders':
                response = handle_excluded_folders(params, 'list')
            elif cmd_type == 'add_excluded_folder':
                response = handle_excluded_folders(params, 'add')
            elif cmd_type == 'remove_excluded_folder':
                response = handle_excluded_folders(params, 'remove')
            elif cmd_type == 'shutdown':
                logger.info("Shutdown command received - cleaning up gracefully")
                cleanup_gracefully()
                response = {'status': 'success', 'message': 'Shutdown complete'}
                print(json.dumps(response), flush=True)
                break
            else:
                response = {
                    'status': 'error',
                    'message': f'Unknown command: {cmd_type}'
                }

            # Send response
            print(json.dumps(response), flush=True)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received - cleaning up")
            cleanup_gracefully()
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            traceback.print_exc(file=sys.stderr)
            response = {
                'status': 'error',
                'message': f'Server error: {e}'
            }
            print(json.dumps(response), flush=True)

    logger.info("Search server shutting down")


if __name__ == '__main__':
    main()
