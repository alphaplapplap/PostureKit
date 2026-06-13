#!/usr/bin/env python3
"""
Persistent DETECTION server for PostureKit.

Mirrors src/search_server.py, but for the query-image detection path. Loads a FULL
PostureKitBridge (skip_models=False) ONCE and reuses it for every request, so the
~6s interpreter+import+model-load cost is paid once per app session instead of on
every query-image drop (findings 22 + 23).

Protocol (newline-delimited JSON on stdin, ONE JSON line per response on stdout):

  {"type":"detect_all","image_path":"/abs/path"}
      -> {"status":"ok","persons":[ <person dict>, ... ]}
      Each person dict is exactly the shape the one-shot detectAllPoses script
      returns (keypoints/visibility/bbox/confidence/person_id) PLUS a folded-in
      "features" dict (geometric feature_vector + feature_confidence + fused_vector +
      joint_angles/limb_ratios/body_angles/symmetry_scores/occlusion_pattern), so no
      separate extract_features round trip is needed (finding 23).

  {"type":"detect_pose","image_path":"/abs/path","bbox":[x1,y1,x2,y2]}
      -> {"status":"ok","person":{ <person dict with features> }}
      Returns a single person (the one whose bbox best matches the supplied bbox, or
      the highest-confidence person when bbox is omitted), or person=null if none.

  {"type":"extract_features","image_path":"/abs/path","person_index":N}
      -> {"status":"ok","features":[...],"confidence":[...]}
      Back-compat: detect_all already includes features, but this re-detects and
      returns the Nth person's geometric vector + confidence on its own.

  {"type":"shutdown"} -> exits cleanly.

On any error: {"status":"error","error":"...message..."} as the single stdout JSON
line, with the traceback to stderr.

STDOUT DISCIPLINE: rtmlib / mmpose / onnxruntime print banners to stdout during lazy
model init (e.g. "load ...onnx with onnxruntime backend"). Swift's detect-server
client reads one JSON line per response, so any stray stdout line would corrupt the
protocol. We therefore redirect sys.stdout -> sys.stderr for the ENTIRE server
lifetime and emit every JSON response (and ONLY those) to the saved real stdout fd
via _emit(). DB_PROFILE is inherited from the environment exactly like search_server.
"""
import os
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

# ---------------------------------------------------------------------------
# OpenMP safety — MUST run BEFORE faiss/torch are imported (swift_bridge below).
# faiss-cpu and torch each bundle their OWN libomp.dylib. This persistent server loads
# BOTH (the bridge pulls in the FAISS search engine AND the torch detection models), so
# two OpenMP runtimes coexist in one process. Their multi-threaded thread pools can
# corrupt each other's barrier state and SIGSEGV inside __kmp_suspend_64 during a
# parallel region — observed crashing on a torch tensor copy_ while loading the RTMW
# model (EXC_BAD_ACCESS at 0x10 in libomp). Forcing single-threaded OpenMP removes the
# worker-thread barriers entirely, which neutralizes the conflict; KMP_DUPLICATE_LIB_OK
# tolerates the duplicate runtime at init. This server is GPU (MPS/CoreML) bound, so
# single-threaded CPU OMP/BLAS costs effectively nothing here. Forced (not setdefault)
# because crash-safety here outranks any inherited thread setting.
for _omp_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_omp_var] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# ---------------------------------------------------------------------------

# Setup logging to stderr (stdout is reserved for JSON responses)
logging.basicConfig(
    level=logging.INFO,
    format='[DETECT SERVER] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Capture the REAL stdout stream BEFORE redirecting, so JSON responses still reach
# the Swift client even while every other stdout write is shunted to stderr.
_REAL_STDOUT = sys.stdout

# Redirect process-wide stdout to stderr so model-init banners (rtmlib/onnxruntime/
# mmpose) never land on the response pipe. Responses go out via _emit() below.
sys.stdout = sys.stderr


def _emit(obj: Dict[str, Any]) -> None:
    """Write exactly one JSON line to the REAL stdout (the response pipe)."""
    _REAL_STDOUT.write(json.dumps(obj))
    _REAL_STDOUT.write("\n")
    _REAL_STDOUT.flush()


def _release_mps_cache() -> None:
    """Release cached MPS GPU memory after a detect request. The persistent server keeps the
    detection models resident across many requests, so (unlike the old one-shot path that exited
    and freed everything per call) MPS allocations can accumulate over a session and eventually
    fail a model/inference allocation — especially under contention from other MPS processes.
    Finding 12 correctly removed the expensive PER-INFERENCE empty_cache from the ensemble hot
    loop; this per-REQUEST release is cheap (~tens of ms against a ~1-2s detect) and bounds the
    long-lived server's steady-state GPU footprint. Best-effort: never let it break a response."""
    try:
        import torch
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# Import bridge (this pulls swift_bridge; detection imports are lazy until the full
# bridge is constructed — finding 40 — but constructing it below loads the models).
try:
    from src.swift_bridge import PostureKitBridge
    logger.info("Swift bridge imported successfully")
except ImportError as e:
    logger.error(f"Failed to import swift_bridge: {e}")
    logger.error(f"sys.path: {sys.path}")
    sys.exit(1)

# Global bridge instance (built once, on startup, with models loaded)
bridge = None


def cleanup_gracefully():
    """Graceful cleanup of database connections and resources. Mirrors search_server."""
    global bridge

    if bridge is None:
        logger.info("Cleanup: Bridge never initialized, nothing to clean")
        return

    logger.info("=" * 70)
    logger.info("GRACEFUL CLEANUP - Releasing database connections")
    logger.info("=" * 70)

    try:
        if hasattr(bridge, 'similarity_engine') and bridge.similarity_engine:
            logger.info("Closing similarity engine...")
            bridge.similarity_engine.close()
            logger.info("Similarity engine closed")

        if hasattr(bridge, 'storage_manager') and bridge.storage_manager:
            logger.info("Closing storage manager...")
            bridge.storage_manager.close()
            logger.info("Storage manager closed")

        logger.info("Disposing all shared database engines...")
        from src.storage.storage_manager import StorageManager
        StorageManager.close_all_engines()
        logger.info("All connection pools disposed")

        logger.info("Cleanup complete")

    except Exception as e:
        logger.error(f"Cleanup error: {e}")

    finally:
        logger.info("=" * 70)


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT gracefully."""
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    logger.info(f"Received {signal_name} - initiating graceful shutdown")
    cleanup_gracefully()
    sys.exit(0)


def _build_bridge() -> None:
    """Build the FULL model-loading bridge ONCE, reading detector configuration from
    the environment the same way the one-shot detect scripts do via UserDefaults.

    POSE_MODELS / FUSION_METHOD / USE_TWO_STAGE_DETECTION / NUM_THREADS / DEVICE are
    read from env so the Swift client can pass the user's detector settings at spawn
    (DB_PROFILE is inherited identically to search_server)."""
    global bridge

    # Pose model selection: 'ensemble' -> ['rtmw-l','rtmw-x']; a single model name
    # stays a string. Default 'ensemble' matches the Swift UserDefaults default.
    pose_model = os.environ.get('POSE_MODEL', 'ensemble').strip()
    if pose_model == 'ensemble':
        pose_models = ['rtmw-l', 'rtmw-x']
    else:
        pose_models = pose_model

    fusion_method = os.environ.get('FUSION_METHOD', 'confidence_weighted').strip()
    # use_two_stage is read by the bridge from USE_TWO_STAGE_DETECTION; pass None to
    # let __init__ honor that env var (matches existing bridge behavior).
    try:
        num_threads = int(os.environ.get('NUM_THREADS', '4'))
    except ValueError:
        num_threads = 4
    device = os.environ.get('DEVICE', 'mps').strip() or 'mps'

    logger.info(
        f"Building detection bridge: pose_models={pose_models}, "
        f"fusion_method={fusion_method}, num_threads={num_threads}, device={device}"
    )
    bridge = PostureKitBridge(
        pose_models=pose_models,
        fusion_method=fusion_method,
        num_threads=num_threads,
        device=device,
        # skip_models=False (default): load the full detection stack ONCE.
        # skip_search=True: do NOT load the FAISS search engine. Detection never searches,
        # and keeping faiss out of this process means torch's libomp is the ONLY OpenMP
        # runtime here — eliminating the dual-libomp conflict that crashed the detect server
        # (the OMP-pinning env vars above are now a secondary belt; this removes the root cause).
        skip_search=True,
    )
    logger.info("Detection bridge built (models loaded, search engine skipped)")

    # Belt-and-suspenders to the OMP env vars set at module top: pin torch's OpenMP runtime to
    # a single thread. NOTE: with skip_search=True faiss is NOT loaded here (the root-cause fix),
    # so we must NOT `import faiss` — that would reload faiss's libomp and recreate the very
    # dual-runtime conflict we removed. Only touch faiss if something already imported it.
    if 'faiss' in sys.modules:
        try:
            sys.modules['faiss'].omp_set_num_threads(1)
        except Exception as _e:
            logger.warning(f"faiss.omp_set_num_threads(1) skipped: {_e}")
    if 'torch' in sys.modules:
        try:
            sys.modules['torch'].set_num_threads(1)
            sys.modules['torch'].set_num_interop_threads(1)
        except Exception as _e:
            logger.warning(f"torch.set_num_threads(1) skipped: {_e}")


def handle_detect_all(params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect every person in the image, with geometric (+visual/fused) features
    folded into each person dict (findings 22 + 23)."""
    image_path = params.get('image_path')
    if not image_path:
        return {'status': 'error', 'error': "detect_all requires 'image_path'"}
    try:
        persons = bridge.detect_multi_person_poses_with_features_from_file(image_path)
        return {'status': 'ok', 'persons': persons}
    except Exception as e:
        logger.error(f"detect_all failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {'status': 'error', 'error': str(e)}


def handle_detect_pose(params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect a single person (best bbox match, or highest-confidence), features folded in."""
    image_path = params.get('image_path')
    if not image_path:
        return {'status': 'error', 'error': "detect_pose requires 'image_path'"}
    bbox = params.get('bbox')  # optional [x1, y1, x2, y2]
    try:
        person = bridge.detect_pose_with_features_from_file(image_path, bbox=bbox)
        return {'status': 'ok', 'person': person}
    except Exception as e:
        logger.error(f"detect_pose failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {'status': 'error', 'error': str(e)}


def handle_extract_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """Back-compat: re-detect and return the Nth person's geometric feature vector +
    confidence on its own. detect_all already includes features, so this is rarely
    needed, but kept to honor the documented protocol."""
    image_path = params.get('image_path')
    if not image_path:
        return {'status': 'error', 'error': "extract_features requires 'image_path'"}
    person_index = params.get('person_index', 0)
    try:
        persons = bridge.detect_multi_person_poses_with_features_from_file(image_path)
        if not persons:
            return {'status': 'error', 'error': 'No persons detected'}
        # Prefer the person whose person_id matches person_index; fall back to position.
        match = next((p for p in persons if p.get('person_id') == person_index), None)
        if match is None:
            if 0 <= person_index < len(persons):
                match = persons[person_index]
            else:
                return {'status': 'error', 'error': f'person_index {person_index} out of range'}
        features = match.get('features') or {}
        return {
            'status': 'ok',
            'features': features.get('feature_vector', []),
            'confidence': features.get('feature_confidence', []),
        }
    except Exception as e:
        logger.error(f"extract_features failed: {e}")
        traceback.print_exc(file=sys.stderr)
        return {'status': 'error', 'error': str(e)}


def main():
    """Main server loop - read commands from stdin, write responses to the real stdout."""
    logger.info("Detection server starting...")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_gracefully)
    logger.info("Cleanup handlers registered (SIGTERM, SIGINT, atexit)")

    # Build the full bridge (loads models) BEFORE announcing ready, so the first
    # detect request is fast. Model-init stdout banners are already shunted to stderr.
    try:
        _build_bridge()
    except Exception as e:
        logger.error(f"Bridge build failed: {e}")
        traceback.print_exc(file=sys.stderr)
        _emit({'status': 'error', 'error': f'Bridge build failed: {e}'})
        sys.exit(1)

    # Signal ready (mirrors search_server's handshake; emitted to the REAL stdout).
    _emit({'status': 'ready'})

    while True:
        try:
            line = _read_stdin_line()
            if not line:
                logger.info("EOF received, shutting down")
                break

            line = line.strip()
            if not line:
                continue

            try:
                command = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                _emit({'status': 'error', 'error': f'Invalid JSON: {e}'})
                continue

            cmd_type = command.get('type')

            if cmd_type == 'detect_all':
                response = handle_detect_all(command)
            elif cmd_type == 'detect_pose':
                response = handle_detect_pose(command)
            elif cmd_type == 'extract_features':
                response = handle_extract_features(command)
            elif cmd_type == 'shutdown':
                logger.info("Shutdown command received - cleaning up gracefully")
                cleanup_gracefully()
                _emit({'status': 'ok', 'message': 'Shutdown complete'})
                break
            else:
                response = {'status': 'error', 'error': f'Unknown command: {cmd_type}'}

            # Bound steady-state MPS memory for this long-lived process (see _release_mps_cache).
            if cmd_type in ('detect_all', 'detect_pose', 'extract_features'):
                _release_mps_cache()

            _emit(response)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received - cleaning up")
            cleanup_gracefully()
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            traceback.print_exc(file=sys.stderr)
            _emit({'status': 'error', 'error': f'Server error: {e}'})

    logger.info("Detection server shutting down")


def _read_stdin_line() -> str:
    """Read one line from the original stdin. sys.stdin is untouched by our stdout
    redirect, so a plain readline is correct here."""
    return sys.stdin.readline()


if __name__ == '__main__':
    main()
