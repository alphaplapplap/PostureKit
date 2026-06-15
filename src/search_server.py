#!/usr/bin/env python3
"""
Persistent search server for PostureKit.
Keeps the PostureKitBridge loaded to avoid re-initialization overhead.
Accepts JSON commands via stdin, returns JSON responses via stdout.
"""
import sys
import os
import json
import logging
import traceback
import signal
import atexit
import threading
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

# FAISS OpenMP thread count. Default 1. The old "dual-libomp" worry doesn't apply
# here: this server runs faiss-ONLY (bridge built with skip_models=True; verified
# torch never enters sys.modules in this process), so there is a single OpenMP
# runtime and raising the count would be SAFE. It is just not WORTH it: measured on
# the 74,907-pose `irl` profile, the IndexFlatL2 scan is ~34 ms (~0.5% of a ~7 s
# uncached threshold search; the rest is DB hydration + the confidence/OKS rerank).
# N=8 gave only ~1.16x on the scan and ~1.0x end-to-end — within noise. The real
# bottleneck is hydration + per-search O(N) Python overhead, not the scan. The
# FAISS_SEARCH_THREADS knob is kept ONLY as an escape hatch for a future, much
# larger corpus where the flat scan could come to dominate; raising it at today's
# scale buys nothing. IndexFlatL2 output is thread-count-deterministic (verified
# bit-identical), so N>1 changes speed, not results.
try:
    import faiss
    _faiss_threads = max(1, int(os.getenv('FAISS_SEARCH_THREADS', '1')))
    faiss.omp_set_num_threads(_faiss_threads)
    logger.info(f"FAISS threads set to {_faiss_threads} (env FAISS_SEARCH_THREADS, default 1)")
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


def handle_search_by_pose_id(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a search-by-stored-pose request ("Find more poses like this")."""
    global bridge

    # Initialize bridge on first search if not already initialized
    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        pose_id = params['pose_id']
        k = params.get('k', 20)
        min_confidence = params.get('min_confidence', 0.5)
        min_feature_confidence = params.get('min_feature_confidence', 0.35)
        min_valid_overlap = params.get('min_valid_overlap', 12)
        required_regions = params.get('required_regions')
        min_region_confidence = params.get('min_region_confidence', 0.3)
        min_similarity = params.get('min_similarity', 0.0)  # Similarity floor (0-1); 0 = top-k mode
        deduplicate_images = params.get('deduplicate_images', False)
        enable_flip_search = bool(params.get('include_flipped', False))

        logger.info(f"Searching by pose id {pose_id} with k={k}, min_similarity={min_similarity}, "
                    f"deduplicate_images={deduplicate_images}, flip={enable_flip_search}")

        results = bridge.search_similar_by_pose_id(
            pose_id=pose_id,
            k=k,
            min_confidence=min_confidence,
            min_feature_confidence=min_feature_confidence,
            min_valid_overlap=min_valid_overlap,
            required_regions=required_regions,
            min_region_confidence=min_region_confidence,
            min_similarity=min_similarity,
            deduplicate_images=deduplicate_images,
            enable_flip_search=enable_flip_search
        )

        logger.info(f"Search by pose id complete, found {len(results)} results")

        return {
            'status': 'success',
            'results': results
        }

    except Exception as e:
        logger.error(f"Search by pose id failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_fetch_details(params: Dict[str, Any]) -> Dict[str, Any]:
    """On-demand heavy-field fetch for the visible page of a deferred-detail search
    (finding 25). Searches now ship lightweight rows for large result sets; Swift
    calls this with the visible page's pose_ids to get their thumbnails, normalized
    keypoints and detailed-region breakdowns. The server already holds the DB hot."""
    global bridge

    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        pose_ids = params.get('pose_ids', [])
        details = bridge.fetch_result_details(pose_ids)
        logger.info(f"fetch_details: returned heavy fields for {len(details)}/{len(pose_ids)} poses")
        return {
            'status': 'success',
            'details': details
        }
    except Exception as e:
        logger.error(f"fetch_details failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e)
        }


def handle_update_image_paths(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a batched stored-path update after files were moved on disk."""
    global bridge

    if bridge is None:
        config = params.get('config', {})
        init_result = initialize_bridge(config)
        if init_result['status'] != 'success':
            return init_result

    try:
        moves = params.get('moves', [])
        result = bridge.update_image_paths(moves)
        if result.get('error'):
            return {'status': 'error', 'message': result['error']}
        return {'status': 'success', **result}

    except Exception as e:
        logger.error(f"Path update failed: {e}")
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

    # Finding 43: eagerly build the bridge (DB connect + create_all + FAISS load) BEFORE
    # signalling ready, so the first user search does not pay that startup cost while they wait.
    # The heavier confidence-aware corpus-cache warm (~2s for a large index) is deferred to a
    # BACKGROUND daemon thread AFTER ready, so it can never push boot past the Swift-side 10s
    # ready timeout; _ensure_corpus_cache is lock-guarded and idempotent, so a first search that
    # arrives mid-warm simply waits on the lock. Failures here are non-fatal — an empty/missing
    # index is valid and the lazy per-command path still initializes on demand — but we must
    # always print the ready line afterwards or the Swift launch sequence hangs.
    try:
        init_result = initialize_bridge({})
        if init_result.get('status') == 'success' and bridge is not None:
            def _warm_corpus_cache():
                try:
                    bridge.similarity_engine.reload_if_stale()
                    bridge.similarity_engine._ensure_corpus_cache()
                    logger.info("✓ Corpus cache warmed (background)")
                    # Plausibility cache (search-speed): static per pose, ~15s for a large index.
                    # Kicking it here (it spawns its own daemon builder and returns immediately)
                    # means dense threshold searches hit the fast lightweight-hydration path soon
                    # after boot; until it's ready, searches fall back to exact inline recompute.
                    bridge.similarity_engine._ensure_plausibility_cache()
                except Exception as warm_err:
                    logger.warning(f"Corpus cache warm skipped: {warm_err}")
            threading.Thread(target=_warm_corpus_cache, name="corpus-warm", daemon=True).start()
            logger.info("✓ Bridge initialized at boot (eager init); corpus cache warming in background")
        else:
            logger.info("Eager bridge init did not complete; will initialize lazily on first command")
    except Exception as e:
        logger.warning(f"Eager bridge init failed ({e}); falling back to lazy init")

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
            elif cmd_type == 'search_by_pose_id':
                response = handle_search_by_pose_id(params)
            elif cmd_type == 'fetch_details':
                response = handle_fetch_details(params)
            elif cmd_type == 'update_image_paths':
                response = handle_update_image_paths(params)
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
