# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

PostureKit is a macOS SwiftUI app that finds visually similar human poses in a photo library. The Swift frontend (`PostureKit/`) is the UI layer; a Python backend (`src/`) does pose detection (RTMO/RTMW via rtmlib/ONNX/mmpose), geometric feature extraction, and similarity search backed by PostgreSQL + FAISS. There is no network API in the running app — Swift talks to Python via spawned subprocesses over stdin/stdout JSON. `backend/main.py` is an unused FastAPI stub; do not confuse it for the live interface.

## How Swift calls Python (load-bearing — read before editing either side)

Two patterns coexist and both are in `PostureKit/PythonBridgeSubprocess.swift`:

1. **Persistent search server** — `src/search_server.py` is spawned once on app launch, stays resident, and handles `search` / `statistics` / `browse_body_parts` / `shutdown` commands via newline-delimited JSON on stdin. Keeping it hot avoids re-loading FAISS and model weights per query. Startup blocks on a `{"status":"ready"}` line; if you change its boot sequence, the Swift-side timeout in `startSearchServer()` will hang the UI.
2. **One-shot `python3 -u -c <script>`** — used for `detectAllPoses`, `detectPose`, `extractFeatures`, and the thumbnail backfill. Swift builds a Python script as a string, runs it, and parses the last line of stdout as JSON.

The Python interpreter path is hardcoded at `PythonBridgeSubprocess.swift:50` as `venv/bin/python3` under the project root. The one-shot scripts expect `src.swift_bridge.PostureKitBridge` to be importable with `sys.path` pointing at the project root and the venv's `site-packages`.

**Import-order gotcha.** `src/core/_torch_patch.py` monkey-patches `torch.load` to force `weights_only=False` (PyTorch 2.6+ rejects the numpy-based mmpose checkpoints otherwise). It **must be imported before any mmengine/mmpose module** or checkpoint loading will throw. `swift_bridge.py` and `search_server.py` already do this at the top; preserve that ordering if you add new entry points.

**Stdout discipline is critical.** RTMO/mmpose/ONNX Runtime print banners like `Loads checkpoint by local backend from path: …` and `load …onnx with onnxruntime backend` before any JSON. Swift calls `extractJSON(from:)` (`PythonBridgeSubprocess.swift:37`) which walks stdout from the last line backward and picks the first line starting with `[` or `{`. Anything you print to stdout from Python after the result JSON will break parsing. Log to stderr instead — `search_server.py` configures logging to stderr precisely for this reason.

**UI-only contract.** On launch, `PostureKitApp.AppDelegate` preloads the FAISS index off the main thread and posts `.indexPreloaded` whether or not the index exists. The status bar leaves its "Loading index..." state on that notification; skipping the post in any branch hangs the UI indefinitely. Empty index is a valid state — it builds on first search.

## Database profiles

The app supports three isolated PostgreSQL databases plus three FAISS index directories, selected by `DB_PROFILE` environment variable (`irl` / `2d` / `3d`):

- DB names: `posturekit_irl`, `posturekit_2d`, `posturekit_3d` (see `src/config/settings.py`)
- Index dirs: `data/indices/{irl,2d,3d}/`

Swift reads `UserDefaults.activeProfile` and injects `DB_PROFILE` into every spawned subprocess's environment (`getActiveProfile()` in `PythonBridgeSubprocess.swift`). If you add a new subprocess, it must propagate `DB_PROFILE` or it will silently hit the wrong database.

## Python backend map (`src/`)

The tree is larger than the few names called out above. Where to look:

- **`core/`** — detection and feature extraction. `person_detector.py` (YOLO bbox), `pose_detector.py` (RTMW whole-body, 133 keypoints), `geometric_feature_extractor.py` (52-dim vector from keypoints) and `visual_feature_extractor.py` (MobileNetV3 embedding), fused in `multimodal_fusion.py`. Accuracy-boosting variants layer on top: `two_stage_detector.py` (YOLO-crop → pose), `ensemble_detector.py` (multi-model weighted fusion), `body_part_detector.py` (NudeNet semantic regions, feeds `bbox_refiner.py`), `multi_person_pipeline.py`. `models.py` holds the shared dataclasses; `database_manager.py` is a thin GUI-legacy adapter over `storage/`.
- **`intelligence/`** — `similarity_engine.py` owns the FAISS index; `pose_classifier.py` / `classification_manager.py` do k-NN labeling; `training_intelligence.py` clusters to surface poses worth annotating.
- **`storage/`** — `storage_manager.py` (atomic writes, exclusion logic), `models.py` (SQLAlchemy ORM), `vector_index.py` (FAISS lifecycle). `schema.sql` is reference only — the real schema evolves through `migrations/` (ad-hoc numbered `.sql` + `.py` scripts, **not Alembic**; run them via the matching runner in `scripts/`, e.g. `run_excluded_folders_migration.py`).
- **`learning/`** — learns systematic detection biases from manual corrections (`correction_learner.py`, `multimodal_correction_learner.py` RandomForest, `pattern_extractor.py`).
- **`utils/`** — shared helpers. `validation.py` (input-guard decorators), `bbox_utils.py`, `image_utils.py`, `device_utils.py` (MPS checks), `progress_reporter.py` (JSON progress to stdout — mind stdout discipline), `logging_config.py` (routes logs to stderr). `constants.py` and `exceptions.py` (rich-context `PostureKitError` hierarchy) live at `src/` root — prefer these over re-introducing magic numbers or bare exceptions.
- **`scripts/`** — operational utilities (migration runners, `diagnose_multiperson.py`, `prepare_eject.py`, `kill_db_connections.sh`), not part of the app runtime.

## Filter semantics + threshold/pagination model (load-bearing — read before touching search)

`minSimilarity` in `PostureKitViewModel` is **THE result-set cap**, applied **in Python**, not Swift. The slider at 40% means "return every pose ≥ 40% similarity, best-first." `executeSearch` passes `minSimilarity` to `searchSimilar`, which threads it (`min_similarity`) through `search_server.py` → `swift_bridge.search_similar` → `similarity_engine.search_by_feature`. In threshold mode (`min_similarity > 0`) the engine scans the whole flat `IndexFlatL2` (`search_k = ntotal`), trims candidates cheaply by base-similarity, then **bulk-fetches** poses/images/body-parts in chunked `IN` queries (NOT one query per candidate — that was the old ~90s "All") and returns *all* survivors ≥ threshold. Default threshold is **0.5**; `0.0` means "everything" (full-index scan, heavy). `minConfidence` is still passed `0.0` so pose-detection confidence doesn't cull candidates.

`numberOfResults` ("Show:") and `showAllResults` ("All") are now **pagination only** — page size for the cached set, applied Swift-side in `applyResultWindow()`. They do **not** trigger a re-search. The full Python result set is cached in `rawResults`; `currentPage`/`totalPages`/`totalResultCount` drive the Prev/Next UI. Changing the threshold: **raising** it (when the cache was fetched at a real floor, `lastQueriedThreshold > 0`) re-filters the cache instantly; **lowering** below the fetched floor (or leaving the 0% top-N path) re-queries Python (debounced, generation-gated). Searches are cancel-safe via the monotonic `searchGeneration` token — a cancelled/superseded search's late results are discarded instead of overwriting the grid. `test_min_similarity_threshold.py` verifies the floor/ordering/completeness end-to-end against a populated profile.

## Build and run

```bash
# Build + run the Swift app
open PostureKit.xcodeproj              # then Cmd-R in Xcode
xcodebuild -project PostureKit.xcodeproj -scheme PostureKit build

# Python env (used by the spawned subprocesses)
source venv/bin/activate
pip install -r requirements.txt        # if requirements.txt exists at root; otherwise backend/requirements.txt

# PostgreSQL must be running before searching or indexing
brew services start postgresql@16
scripts/kill_db_connections.sh         # force-close stuck connections (e.g. before dropping a profile DB)

# Run the search server standalone (useful for debugging Swift ↔ Python JSON)
DB_PROFILE=irl venv/bin/python3 src/search_server.py
# Then type JSON lines like: {"type":"statistics"}

# Run a specific Python test
venv/bin/python3 test_phase3_two_stage.py
venv/bin/python3 test_ensemble_proof.py
```

Swift tests live in `PostureKitTests/` and `PostureKitUITests/` and run via Xcode's test action (Cmd-U).

## Known noise to ignore

- **SourceKit cross-file diagnostics** — the editor regularly reports "Cannot find type 'PostureKitViewModel' in scope," "Cannot find 'IndexDirectoryView' in scope," "`NSImage` has no member `pixelSize`," and "The compiler is unable to type-check this expression in reasonable time" across `ContentView.swift`. These are standalone-indexer false positives; `xcodebuild` resolves them fine. Don't chase them unless they also appear in a real build.
- **`NSXPCDecoder` / IMK warnings** at app launch — Apple framework noise, suppressed in `PostureKitApp.init()`.
- **Root-level `*.md` files** (`BUGFIX_*.md`, `PHASE*_SUMMARY.md`, `CROSS_REFERENCE_*.md`, etc.) are historical notes, not specs. `TROUBLESHOOTING.md` and `QUICK_REFERENCE.md` are the two worth consulting.

## Cursor rules (apply here too)

From `.cursor/rules/no-bed-shitting.mdc`: explain changes before implementing, don't refactor without permission, don't add dependencies without approval, preserve existing patterns unless told otherwise, no unrequested "optimizations."

## Editing boundaries

- `PostureKit/` — Swift UI and subprocess bridge. The view model (`PostureKitViewModel.swift`), content view (`ContentView.swift`), and bridge (`PythonBridgeSubprocess.swift`) are all large single files; prefer `Edit` with unique anchors over full rewrites.
- `src/` — Python backend. `swift_bridge.py` is the public surface Swift imports from; `search_server.py` is the persistent-server entry point; `core/` holds detectors; `intelligence/similarity_engine.py` owns the FAISS index; `storage/` owns PostgreSQL.
- `upgrades4posturekit/` — vendored third-party integration code; treat as read-only reference.
- `venv/`, `data/indices/`, `data/checkpoints/`, `*.pt`, `*.onnx` — build artifacts and model weights. Do not edit or commit.
