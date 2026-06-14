import Foundation
import AppKit

// MARK: - Python Bridge using Subprocess
// More reliable than PythonKit for complex dependencies
class PythonBridgeSubprocess {
    static let shared = PythonBridgeSubprocess()

    private let pythonExecutable: String
    let projectPath: String  // Internal for script generation
    let venvSitePackages: String  // Internal for script generation
    private var indexingProcess: Process?
    private var thumbnailProcess: Process?
    private var currentSearchProcess: Process?
    private let searchLock = NSLock()

    // Persistent search server
    private var searchServerProcess: Process?
    private var searchServerLock = NSRecursiveLock()  // Recursive to allow reentrant locking
    private var searchServerStdin: FileHandle?
    private var searchServerStdout: FileHandle?
    private var searchServerInitialized = false
    private var isShuttingDown = false  // Flag to cancel startup during shutdown

    // Persistent DETECTION server (finding 22): a long-lived process that loads the
    // ensemble ONCE and serves detect_all/detect_pose/extract_features over the same
    // newline-delimited JSON protocol as the search server. Started LAZILY on the
    // first detect (unlike the search server, which starts eagerly) so we don't pay
    // the ~6s model load until the user actually drops a query image.
    private var detectServerProcess: Process?
    private var detectServerLock = NSRecursiveLock()  // Recursive to allow reentrant locking
    private var detectServerStdin: FileHandle?
    private var detectServerStdout: FileHandle?
    private var detectServerInitialized = false

    // Per-image+person feature cache (finding 23). detect_all now returns each
    // person's geometric features inline; we stash them here keyed by
    // "<imageKey>|<personIndex>" so extractFeatures() returns the already-computed
    // vector instead of spawning a second ~2.5s one-shot. Bounded to the last
    // detect's people (cleared and repopulated on each detectAllPoses/detectPose).
    private var lastDetectFeatures: [String: GeometricFeatures] = [:]
    private var lastDetectImageKey: String = ""
    private let detectFeatureCacheLock = NSLock()

    // Cancellation support for in-flight server searches (finding 27). Each search
    // round trip is tagged with a monotonic token; cancelSearch() bumps the token so
    // the reader of a superseded search abandons its (still-arriving) response and
    // releases the lock instead of blocking the replacement search for the full
    // 60s round trip.
    private var searchCancelToken: Int = 0
    private let searchCancelLock = NSLock()

    // When a search is abandoned mid-flight (finding 27), the single-threaded search
    // server STILL finishes that search and writes its (now orphaned) response line to
    // the pipe before it reads the next command. The next search command must therefore
    // drain that many orphaned lines first, or it would read a stale response and desync
    // the protocol. Guarded by searchServerLock (every search command holds it), so this
    // plain counter is safe.
    private var pendingSearchDrains: Int = 0

    // MARK: - Helper: Swift Bool to Python bool converter
    private func pythonBool(_ value: Bool) -> String {
        value ? "True" : "False"
    }

    // MARK: - Helper: Extract JSON payload from noisy stdout
    // Python subprocess stdout may be preceded by arbitrary log lines from
    // rtmlib, onnxruntime, mmengine, etc. (e.g. "load …onnx with onnxruntime
    // backend"). All three call sites print JSON as the final line via
    // `print(json.dumps(...))`, so we find the last line starting with '['
    // or '{' and return from there. Line-based is more robust than a first-
    // bracket scan — a log line like "[INFO] starting" would break that.
    private func extractJSON(from output: String) -> String {
        let lines = output.split(separator: "\n", omittingEmptySubsequences: false)
        for line in lines.reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("[") || trimmed.hasPrefix("{") {
                return String(line)
            }
        }
        return output
    }

    private init() {
        // Use venv Python to ensure PyTorch and all dependencies are loaded correctly
        self.pythonExecutable = "/Users/linuxbabe/Hardware-Aware/PostureKit/venv/bin/python3"
        self.projectPath = "/Users/linuxbabe/Hardware-Aware/PostureKit"
        self.venvSitePackages = "/Users/linuxbabe/Hardware-Aware/PostureKit/venv/lib/python3.9/site-packages"

        print("Python bridge initialized with executable: \(pythonExecutable)")
        print("Virtual environment site-packages: \(venvSitePackages)")
    }

    // MARK: - Thread Settings Helper
    private func getThreadSettings() -> (threads: Int, useGPU: Bool) {
        let threads = UserDefaults.standard.integer(forKey: "detectionThreads")
        let useGPU = UserDefaults.standard.bool(forKey: "useGPU")
        return (threads > 0 ? threads : 16, useGPU)  // Default to 16 (M5 Max: 6 Super + 12 Performance cores)
    }

    // MARK: - Database Profile Helper
    private func getActiveProfile() -> String {
        return UserDefaults.standard.string(forKey: "activeProfile") ?? "irl"
    }

    // MARK: - Persistent Search Server
    private func startSearchServer() -> Bool {
        searchServerLock.lock()
        defer { searchServerLock.unlock() }

        // Already started
        if searchServerInitialized, let process = searchServerProcess, process.isRunning {
            print("[SEARCH SERVER] Already running (PID: \(process.processIdentifier))")
            return true
        }

        print("[SEARCH SERVER] Starting persistent search server...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["\(projectPath)/src/search_server.py"]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Setup environment
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")

        // Search server doesn't need threading libraries (no pose detection models)
        // FAISS manages its own threading, PostgreSQL is I/O bound
        // Setting OMP_NUM_THREADS causes pthread_mutex_init failures in subprocess
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        // Setup pipes for communication
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Capture stderr for logging
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let message = String(data: data, encoding: .utf8) {
                print("[SEARCH SERVER STDERR] \(message.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        do {
            try process.run()
            print("[SEARCH SERVER] Process started (PID: \(process.processIdentifier))")

            // Store handles
            searchServerProcess = process
            searchServerStdin = stdinPipe.fileHandleForWriting
            searchServerStdout = stdoutPipe.fileHandleForReading

            // Wait for "ready" signal (with timeout)
            // Use line-buffered reading to handle large responses properly
            print("[SEARCH SERVER] Waiting for ready signal...")
            var accumulatedData = Data()
            var foundNewline = false
            let maxAttempts = 100  // 10 seconds (100 * 0.1s)

            for attempt in 1...maxAttempts {
                // Check if shutdown requested
                if isShuttingDown {
                    print("[SEARCH SERVER] Startup cancelled - app is shutting down")
                    process.terminate()
                    return false
                }

                if let stdout = searchServerStdout {
                    let data = stdout.availableData
                    if !data.isEmpty {
                        accumulatedData.append(data)

                        // Check if we have a complete line (ends with \n)
                        if accumulatedData.last == 0x0A {
                            foundNewline = true
                            print("[SEARCH SERVER] Received data after \(attempt * 100)ms")
                            break
                        }
                    }
                }

                // Check if process crashed
                if !process.isRunning {
                    print("[SEARCH SERVER] Process terminated unexpectedly")
                    return false
                }

                Thread.sleep(forTimeInterval: 0.1)
            }

            // Parse ready signal
            if foundNewline,
               let readyJson = String(data: accumulatedData, encoding: .utf8),
               let readyResponse = try? JSONSerialization.jsonObject(with: accumulatedData) as? [String: Any],
               readyResponse["status"] as? String == "ready" {
                print("[SEARCH SERVER] Initialization complete: \(readyJson.trimmingCharacters(in: .whitespacesAndNewlines))")
                searchServerInitialized = true
                return true
            } else {
                print("[SEARCH SERVER] Failed to receive ready signal (timeout or invalid response)")
                if !accumulatedData.isEmpty, let readyJson = String(data: accumulatedData, encoding: .utf8) {
                    print("[SEARCH SERVER] Received: \(readyJson)")
                }
                process.terminate()
                return false
            }

        } catch {
            print("[SEARCH SERVER] Failed to start: \(error)")
            return false
        }
    }

    // MARK: - Persistent Detection Server (finding 22)
    /// Lazily start the long-lived detection worker (src/detect_server.py). Mirrors
    /// startSearchServer(): spawns venv/bin/python3 src/detect_server.py, inherits
    /// DB_PROFILE, and blocks on a {"status":"ready"} stdout line with a timeout.
    /// The detect server loads the ensemble + full bridge once, so per-query detection
    /// drops from ~7s (one-shot spawn) to ~0.4-2s warm.
    private func startDetectServer() -> Bool {
        detectServerLock.lock()
        defer { detectServerLock.unlock() }

        // Already started
        if detectServerInitialized, let process = detectServerProcess, process.isRunning {
            print("[DETECT SERVER] Already running (PID: \(process.processIdentifier))")
            return true
        }

        // Don't start (or resurrect) during shutdown.
        if isShuttingDown {
            print("[DETECT SERVER] Not starting - app is shutting down")
            return false
        }

        print("[DETECT SERVER] Starting persistent detection server...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["\(projectPath)/src/detect_server.py"]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Setup environment (mirror the one-shot detect env which is known to work
        // with the model stack; configureThreading=true path sets OMP/MKL/OPENBLAS).
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")

        // The detect server loads pose models, so it needs the same threading env the
        // one-shot detect path used (configureThreading=true). Search server omits this
        // (no models); detection benefits from it.
        let (threads, _) = getThreadSettings()
        environment["OMP_NUM_THREADS"] = "\(threads)"
        environment["MKL_NUM_THREADS"] = "\(threads)"
        environment["OPENBLAS_NUM_THREADS"] = "\(threads)"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        // Setup pipes for communication
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Capture stderr for logging
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let message = String(data: data, encoding: .utf8) {
                print("[DETECT SERVER STDERR] \(message.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        do {
            try process.run()
            print("[DETECT SERVER] Process started (PID: \(process.processIdentifier))")

            // Store handles
            detectServerProcess = process
            detectServerStdin = stdinPipe.fileHandleForWriting
            detectServerStdout = stdoutPipe.fileHandleForReading

            // Wait for "ready" signal (with timeout). The detect server loads the full
            // model stack BEFORE printing ready (it prints ready AFTER bridge build),
            // so warm this can take a few seconds and cold (page-cache miss) 60-90s.
            print("[DETECT SERVER] Waiting for ready signal...")
            var accumulatedData = Data()
            var foundNewline = false
            let maxAttempts = 1200  // 120 seconds (1200 * 0.1s) — model load can be slow cold

            for attempt in 1...maxAttempts {
                if isShuttingDown {
                    print("[DETECT SERVER] Startup cancelled - app is shutting down")
                    process.terminate()
                    return false
                }

                if let stdout = detectServerStdout {
                    let data = stdout.availableData
                    if !data.isEmpty {
                        accumulatedData.append(data)
                        if accumulatedData.last == 0x0A {
                            foundNewline = true
                            print("[DETECT SERVER] Received ready data after ~\(attempt * 100)ms")
                            break
                        }
                    }
                }

                if !process.isRunning {
                    print("[DETECT SERVER] Process terminated unexpectedly during startup")
                    return false
                }

                Thread.sleep(forTimeInterval: 0.1)
            }

            if foundNewline,
               let readyJson = String(data: accumulatedData, encoding: .utf8),
               let readyResponse = try? JSONSerialization.jsonObject(with: accumulatedData) as? [String: Any],
               readyResponse["status"] as? String == "ready" {
                print("[DETECT SERVER] Initialization complete: \(readyJson.trimmingCharacters(in: .whitespacesAndNewlines))")
                detectServerInitialized = true
                return true
            } else {
                print("[DETECT SERVER] Failed to receive ready signal (timeout or invalid response)")
                if !accumulatedData.isEmpty, let readyJson = String(data: accumulatedData, encoding: .utf8) {
                    print("[DETECT SERVER] Received: \(readyJson)")
                }
                process.terminate()
                return false
            }

        } catch {
            print("[DETECT SERVER] Failed to start: \(error)")
            return false
        }
    }

    /// Send a command to the persistent detection server, lazily starting it if needed.
    /// The detect protocol keys the command on "type" (not "command"); reuses the same
    /// newline-delimited round-trip reader as the search server.
    private func sendDetectServerCommand(_ command: [String: Any]) -> [String: Any]? {
        detectServerLock.lock()
        defer { detectServerLock.unlock() }

        guard detectServerInitialized,
              let process = detectServerProcess,
              process.isRunning,
              let stdin = detectServerStdin,
              let stdout = detectServerStdout else {
            print("[DETECT SERVER] Server not running, attempting to start...")
            if !startDetectServer() {
                return nil
            }
            guard let stdin = detectServerStdin, let stdout = detectServerStdout else {
                return nil
            }
            return sendCommand(command, stdin: stdin, stdout: stdout, label: "DETECT SERVER", commandKey: "type")
        }

        return sendCommand(command, stdin: stdin, stdout: stdout, label: "DETECT SERVER", commandKey: "type")
    }

    /// Send a command to the persistent search server. When `cancelToken` is provided
    /// (search round trips), the round trip is abandoned if a newer search supersedes it
    /// via cancelSearch(), so the server lock is not held for the full transfer of a
    /// stale response (finding 27).
    private func sendSearchServerCommand(_ command: [String: Any], cancelToken: Int? = nil) -> [String: Any]? {
        searchServerLock.lock()
        defer { searchServerLock.unlock() }

        // Ensure server is running
        guard searchServerInitialized,
              let process = searchServerProcess,
              process.isRunning,
              let stdin = searchServerStdin,
              let stdout = searchServerStdout else {
            print("[SEARCH SERVER] Server not running, attempting to start...")
            if !startSearchServer() {
                return nil
            }
            // Retry with newly started server
            guard let stdin = searchServerStdin, let stdout = searchServerStdout else {
                return nil
            }
            return sendCommand(command, stdin: stdin, stdout: stdout, label: "SEARCH SERVER", commandKey: "command", cancelToken: cancelToken)
        }

        return sendCommand(command, stdin: stdin, stdout: stdout, label: "SEARCH SERVER", commandKey: "command", cancelToken: cancelToken)
    }

    /// Round-trip a single newline-delimited JSON command to a persistent server.
    /// - label: log prefix ("SEARCH SERVER" / "DETECT SERVER").
    /// - commandKey: the protocol's command field ("command" for search, "type" for detect).
    /// - cancelToken: when non-nil, the search-cancellation token captured BEFORE the lock
    ///   was acquired. If cancelSearch() bumps the shared token past this value mid-flight,
    ///   the read loop abandons the response and returns nil, so a superseded slider/threshold
    ///   search releases the server lock instead of serializing behind its full round trip.
    private func sendCommand(
        _ command: [String: Any],
        stdin: FileHandle,
        stdout: FileHandle,
        label: String = "SEARCH SERVER",
        commandKey: String = "command",
        cancelToken: Int? = nil
    ) -> [String: Any]? {
        do {
            // Which process owns this pipe (for crash detection); detect uses "type".
            let isDetect = (commandKey == "type")
            // Only search commands are cancellable, so only they can leave orphans.
            let isCancellableSearch = (cancelToken != nil)

            // Finding 27: before issuing a new (cancellable) search, drain any orphaned
            // response lines left by previously-abandoned searches. The single-threaded
            // server eventually emits each abandoned search's full response; consuming
            // them here keeps this command aligned with its own response.
            if isCancellableSearch {
                while pendingSearchDrains > 0 {
                    if isShuttingDown { return nil }
                    if drainOneOrphanLine(from: stdout, label: label) {
                        pendingSearchDrains -= 1
                        print("[\(label)] Drained an orphaned response (\(pendingSearchDrains) remaining)")
                    } else {
                        // Couldn't drain (server gone / timeout) — give up the count to
                        // avoid spinning; a crash check below will catch a dead server.
                        print("[\(label)] Orphan drain incomplete; clearing drain counter")
                        pendingSearchDrains = 0
                        break
                    }
                }
            }

            // Serialize command
            let jsonData = try JSONSerialization.data(withJSONObject: command)
            let jsonString = String(data: jsonData, encoding: .utf8)! + "\n"

            // Send command
            print("[\(label)] Sending command: \(command[commandKey] as? String ?? "unknown")")
            stdin.write(jsonString.data(using: .utf8)!)

            // Read response with a 60s WALL-CLOCK budget (line-buffered for large
            // responses). First search may take longer (index loading); subsequent
            // searches are fast. Large responses (~10MB) require many reads.
            //
            // Finding 26: availableData blocks until at least one byte is ready, so
            // after a NON-EMPTY read we loop immediately and only sleep when a read
            // returned zero bytes — eliminating the ~10ms-per-chunk artificial latency
            // (~1.5s on a multi-MB response) the old unconditional sleep incurred.
            let deadline = Date().addingTimeInterval(60.0)
            var accumulatedData = Data()
            var foundNewline = false

            while Date() < deadline {
                // Check if shutdown requested
                if isShuttingDown {
                    print("[\(label)] Command cancelled - app is shutting down")
                    return nil
                }

                // Finding 27: if this round trip is a (cancellable) search and a newer
                // search has superseded it, abandon the response and release the lock so
                // the replacement query doesn't serialize behind this one's full transfer.
                // Record the orphan so the next search drains the server's eventual
                // response before reading its own.
                if let token = cancelToken, isSearchCancelled(token) {
                    print("[\(label)] Search superseded (token \(token)) - abandoning response")
                    if foundNewline == false {
                        // The server will still finish this search and emit one full line;
                        // even if we already buffered a partial chunk, exactly one line is
                        // pending (the rest of this response). Mark it for draining.
                        pendingSearchDrains += 1
                    }
                    return nil
                }

                let data = stdout.availableData
                if !data.isEmpty {
                    accumulatedData.append(data)

                    // Check if we have a complete line (ends with \n).
                    // Python server sends one JSON object per line; check the newline
                    // byte (0x0A) without string conversion for efficiency.
                    if accumulatedData.last == 0x0A {
                        foundNewline = true
                        print("[\(label)] Received complete response (\(accumulatedData.count) bytes)")
                        break
                    }

                    // Non-empty read: more data is likely already buffered — loop
                    // immediately without sleeping (finding 26).
                    continue
                }

                // Zero-byte read. Check if the owning server crashed.
                let serverProc = isDetect ? detectServerProcess : searchServerProcess
                if let serverProc = serverProc, !serverProc.isRunning {
                    print("[\(label)] Process terminated unexpectedly")
                    return nil
                }

                // Only sleep when there was nothing to read, to avoid busy-spinning.
                Thread.sleep(forTimeInterval: 0.01)
            }

            guard foundNewline else {
                print("[\(label)] Timeout or incomplete response (received \(accumulatedData.count) bytes)")
                return nil
            }

            let responseData = accumulatedData

            let response = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
            print("[\(label)] Received response: status=\(response?["status"] as? String ?? "unknown")")
            return response

        } catch {
            print("[\(label)] Communication error: \(error)")
            return nil
        }
    }

    // MARK: - Search cancellation token (finding 27)
    /// Snapshot the current cancel token. Pass this to a search round trip; if
    /// cancelSearch() later bumps the token past this value, the round trip abandons.
    private func currentSearchToken() -> Int {
        searchCancelLock.lock()
        defer { searchCancelLock.unlock() }
        return searchCancelToken
    }

    /// True if a search tagged with `token` has been superseded/cancelled.
    private func isSearchCancelled(_ token: Int) -> Bool {
        searchCancelLock.lock()
        defer { searchCancelLock.unlock() }
        return searchCancelToken != token
    }

    /// Read and discard exactly one newline-terminated line from the search server's
    /// stdout (the orphaned response of a previously-abandoned search). Returns true once
    /// a full line was consumed, false on timeout / dead server. Caller holds
    /// searchServerLock. Honors the shutdown flag and the same 60s budget as a normal read.
    private func drainOneOrphanLine(from stdout: FileHandle, label: String) -> Bool {
        let deadline = Date().addingTimeInterval(60.0)
        var sawNewline = false
        while Date() < deadline {
            if isShuttingDown { return false }
            let data = stdout.availableData
            if !data.isEmpty {
                if data.last == 0x0A {
                    sawNewline = true
                    break
                }
                continue
            }
            if let proc = searchServerProcess, !proc.isRunning {
                print("[\(label)] Server gone while draining orphan")
                return false
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return sawNewline
    }

    // MARK: - Pose Detection

    /// Detect ALL people in image (multi-person detection).
    ///
    /// Finding 22: routed through the persistent detection server (src/detect_server.py)
    /// instead of spawning a fresh `python3 -c` one-shot per call. The server loads the
    /// ensemble once per app session, so warm detection drops from ~7s to ~0.4-2s.
    /// Finding 23: each returned person carries inline geometric features, which we cache
    /// keyed by image+person so a subsequent extractFeatures() needs no second round trip.
    func detectAllPoses(in image: NSImage) -> [PoseDetectionResult] {
        print("[DEBUG] detectAllPoses called")

        // The detect server reads the image from disk, so still materialize a temp file.
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return []
        }
        print("[DEBUG] Saved image to: \(tempImagePath)")

        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        let command: [String: Any] = [
            "type": "detect_all",
            "image_path": tempImagePath
        ]

        print("[DEBUG] Sending detect_all to persistent detection server...")
        guard let response = sendDetectServerCommand(command) else {
            print("[ERROR] Detection server returned nil")
            return []
        }

        guard (response["status"] as? String) == "ok",
              let persons = response["persons"] as? [[String: Any]] else {
            let err = response["error"] as? String ?? "unknown"
            print("[ERROR] detect_all failed: \(err)")
            return []
        }

        print("[DEBUG] Detection server returned \(persons.count) person(s)")

        // Fold inline features (finding 23): cache them keyed by this image + person.
        cacheDetectFeatures(persons, imageKey: featureCacheKey(for: image))

        // Convert each JSON dict to PoseDetectionResult
        let results = persons.compactMap { json -> PoseDetectionResult? in
            return parsePoseResult(from: json)
        }

        print("[DEBUG] Converted \(results.count) pose results")
        return results
    }

    /// Detect single person in image (legacy method, only returns first person).
    /// Finding 22: routed through the persistent detection server; finding 23: caches
    /// the returned person's inline features for a subsequent extractFeatures() call.
    func detectPose(in image: NSImage) -> PoseDetectionResult? {
        print("[DEBUG] detectPose called (single-person mode)")

        // The detect server reads the image from disk, so still materialize a temp file.
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return nil
        }
        print("[DEBUG] Saved image to: \(tempImagePath)")

        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        let command: [String: Any] = [
            "type": "detect_pose",
            "image_path": tempImagePath
        ]

        print("[DEBUG] Sending detect_pose to persistent detection server...")
        guard let response = sendDetectServerCommand(command) else {
            print("[ERROR] Detection server returned nil")
            return nil
        }

        guard (response["status"] as? String) == "ok" else {
            let err = response["error"] as? String ?? "unknown"
            print("[ERROR] detect_pose failed: \(err)")
            return nil
        }

        // person may be null (no person detected) — that's not an error.
        guard let person = response["person"] as? [String: Any] else {
            print("[DEBUG] detect_pose: no person detected")
            return nil
        }

        // Cache the single person's inline features (finding 23).
        cacheDetectFeatures([person], imageKey: featureCacheKey(for: image))

        print("[DEBUG] Successfully parsed detect_pose result")
        return parsePoseResult(from: person)
    }

    // MARK: - Feature Extraction
    /// Finding 23: detect_all already folds each person's geometric (+visual/fused)
    /// features into the detect response, cached at detect time. extractFeatures now
    /// returns those already-computed features instead of spawning a second ~2.5s
    /// one-shot. On a cache miss (e.g. the Browse fallback, or a stale image), it
    /// re-runs detect_all on the persistent detection server and folds in the matching
    /// person's features — still NO cold one-shot spawn.
    func extractFeatures(from poseResult: PoseDetectionResult, image: NSImage) -> GeometricFeatures? {
        print("[DEBUG] extractFeatures called (cache-first, finding 23)")

        let imageKey = featureCacheKey(for: image)

        // Fast path: features were folded in by the preceding detectAllPoses/detectPose.
        if let cached = cachedDetectFeatures(imageKey: imageKey, personIndex: poseResult.personIndex) {
            print("[DEBUG] extractFeatures cache HIT for person \(poseResult.personIndex)")
            return cached
        }

        print("[DEBUG] extractFeatures cache MISS - re-detecting via detection server")

        // Cache miss: re-run detect_all (persistent server, no cold spawn) and repopulate
        // the per-image cache, then look up the requested person.
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return nil
        }
        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        let command: [String: Any] = [
            "type": "detect_all",
            "image_path": tempImagePath
        ]

        guard let response = sendDetectServerCommand(command),
              (response["status"] as? String) == "ok",
              let persons = response["persons"] as? [[String: Any]] else {
            print("[ERROR] extractFeatures fallback detect_all failed")
            return nil
        }

        cacheDetectFeatures(persons, imageKey: imageKey)

        if let cached = cachedDetectFeatures(imageKey: imageKey, personIndex: poseResult.personIndex) {
            print("[DEBUG] extractFeatures fallback resolved features for person \(poseResult.personIndex)")
            return cached
        }

        // Last resort: if the requested person index isn't present (re-detection drift),
        // ask the server for that person index directly. Returns only the geometric
        // vector + confidence (no joint angles etc.), which is enough for search.
        return extractFeaturesViaServer(imagePath: tempImagePath, personIndex: poseResult.personIndex)
    }

    /// Back-compat fallback: the detect server's extract_features command returns just the
    /// geometric vector + confidence for the Nth person. Used only when the full features
    /// dict couldn't be matched from a detect_all (rare). No cold spawn.
    private func extractFeaturesViaServer(imagePath: String, personIndex: Int) -> GeometricFeatures? {
        let command: [String: Any] = [
            "type": "extract_features",
            "image_path": imagePath,
            "person_index": personIndex
        ]
        guard let response = sendDetectServerCommand(command),
              (response["status"] as? String) == "ok",
              let featuresDoubles = response["features"] as? [Double] else {
            print("[ERROR] extract_features server fallback failed")
            return nil
        }
        let featureVector = featuresDoubles.map { Float($0) }
        var featureConfidence: [Float]? = nil
        if let confDoubles = response["confidence"] as? [Double], !confDoubles.isEmpty {
            featureConfidence = confDoubles.map { Float($0) }
        }
        // Minimal struct: only the search-relevant fields are populated; the dict fields
        // (joint/limb/body/symmetry) are unused by the search path.
        return GeometricFeatures(
            featureVector: featureVector,
            featureConfidence: featureConfidence,
            fusedVector: nil,
            jointAngles: [:],
            limbRatios: [:],
            bodyAngles: [:],
            symmetryScores: [:],
            occlusionPattern: []
        )
    }

    // MARK: - Inline-feature cache (finding 23)

    /// Stable key for an NSImage's pixels, so extractFeatures can match the image that
    /// was just detected. Hashes the TIFF representation (cheap, deterministic for the
    /// same bitmap); falls back to a pointer identity if TIFF isn't available.
    private func featureCacheKey(for image: NSImage) -> String {
        if let tiff = image.tiffRepresentation {
            return "tiff:\(tiff.count):\(tiff.hashValue)"
        }
        return "ptr:\(ObjectIdentifier(image).hashValue)"
    }

    /// Replace the per-image feature cache with the features folded into a detect
    /// response. Keyed by "<imageKey>|<personIndex>". Only the LAST detect's people are
    /// retained (the cache is cleared on each new image), bounding memory.
    private func cacheDetectFeatures(_ persons: [[String: Any]], imageKey: String) {
        detectFeatureCacheLock.lock()
        defer { detectFeatureCacheLock.unlock() }

        // New image => drop the previous image's features.
        if imageKey != lastDetectImageKey {
            lastDetectFeatures.removeAll()
            lastDetectImageKey = imageKey
        }

        for person in persons {
            guard let featuresDict = person["features"] as? [String: Any],
                  let parsed = parseFeatures(from: featuresDict) else {
                continue
            }
            let personIndex = (person["person_id"] as? Int) ?? 0
            lastDetectFeatures["\(imageKey)|\(personIndex)"] = parsed
        }
        print("[DEBUG] Cached inline features for \(lastDetectFeatures.count) person(s)")
    }

    /// Look up cached inline features for a given image + person index.
    private func cachedDetectFeatures(imageKey: String, personIndex: Int) -> GeometricFeatures? {
        detectFeatureCacheLock.lock()
        defer { detectFeatureCacheLock.unlock() }
        guard imageKey == lastDetectImageKey else { return nil }
        return lastDetectFeatures["\(imageKey)|\(personIndex)"]
    }

    // MARK: - Similarity Search
    func searchSimilar(
        featureVector: [Float],
        k: Int = 20,
        minConfidence: Double = 0.5,
        featureConfidence: [Float]? = nil,
        minFeatureConfidence: Double = 0.35,
        minValidOverlap: Int = 12,
        requiredRegions: [String]? = nil,
        deduplicateImages: Bool = false,
        minRegionConfidence: Double = 0.3,
        minSimilarity: Double = 0.0,
        includeFlippedPoses: Bool = false,
        queryKeypoints: [[Double]]? = nil,
        queryBbox: [Double]? = nil
    ) -> [SearchResult] {
        // Debug logging
        print("[SWIFT SEARCH DEBUG] Feature vector length: \(featureVector.count)")
        print("[SWIFT SEARCH DEBUG] Confidence length: \(featureConfidence?.count ?? 0)")
        print("[SWIFT SEARCH DEBUG] k=\(k), minConfidence=\(minConfidence)")
        print("[SWIFT SEARCH DEBUG] minFeatureConfidence=\(minFeatureConfidence), minValidOverlap=\(minValidOverlap)")
        print("[SWIFT SEARCH DEBUG] deduplicateImages=\(deduplicateImages), minRegionConfidence=\(minRegionConfidence)")

        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        // Build command for search server
        var params: [String: Any] = [
            "feature_vector": featureVector,
            "k": k,
            "min_confidence": minConfidence,
            "min_feature_confidence": minFeatureConfidence,
            "min_valid_overlap": minValidOverlap,
            "deduplicate_images": deduplicateImages,
            "min_region_confidence": minRegionConfidence,
            "min_similarity": minSimilarity,
            "include_flipped": includeFlippedPoses,
            "config": [
                "num_threads": threads,
                "device": device
            ]
        ]

        if let confidence = featureConfidence {
            params["query_confidence"] = confidence
        }

        if let regions = requiredRegions, !regions.isEmpty {
            params["required_regions"] = regions
        }

        // Query pose geometry enables OKS re-ranking (and flip search) server-side;
        // omitted for stored-vector/browse searches, where the engine falls back to L2.
        if let keypoints = queryKeypoints, !keypoints.isEmpty {
            params["query_keypoints"] = keypoints
        }
        if let bbox = queryBbox, bbox.count == 4 {
            params["query_bbox"] = bbox
        }

        let command: [String: Any] = [
            "command": "search",
            "params": params
        ]

        // Capture the cancel token BEFORE the round trip so cancelSearch() (a newer
        // slider/threshold search) can abandon this one mid-flight (finding 27).
        let token = currentSearchToken()

        // Send command to persistent server
        guard let response = sendSearchServerCommand(command, cancelToken: token) else {
            print("[SEARCH DEBUG] Failed to communicate with search server")
            return []
        }

        // Check response status
        guard let status = response["status"] as? String else {
            print("[SEARCH DEBUG] Invalid response: missing status")
            return []
        }

        if status != "success" {
            let message = response["message"] as? String ?? "Unknown error"
            print("[SEARCH DEBUG] Search failed: \(message)")
            return []
        }

        // Parse results
        guard let resultsArray = response["results"] as? [[String: Any]] else {
            print("[SEARCH DEBUG] Invalid response: missing or invalid results array")
            return []
        }

        print("[SEARCH DEBUG] Received \(resultsArray.count) results from server")
        return resultsArray.compactMap { parseSearchResult(from: $0) }
    }

    /// "Find more poses like this": search using a STORED pose's feature vector,
    /// identified by its pose id. Python loads the stored vector/confidence and the
    /// pose's keypoints/bbox (for OKS re-ranking and flip search), so the search
    /// behaves exactly like a fresh query on that pose; the pose itself is excluded.
    func searchSimilarByPoseId(
        _ poseId: String,
        k: Int = 20,
        minConfidence: Double = 0.5,
        minFeatureConfidence: Double = 0.35,
        minValidOverlap: Int = 12,
        requiredRegions: [String]? = nil,
        deduplicateImages: Bool = false,
        minRegionConfidence: Double = 0.3,
        minSimilarity: Double = 0.0,
        includeFlippedPoses: Bool = false
    ) -> [SearchResult] {
        print("[SWIFT SEARCH DEBUG] Search by pose id: \(poseId), k=\(k), minSimilarity=\(minSimilarity)")

        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        var params: [String: Any] = [
            "pose_id": poseId,
            "k": k,
            "min_confidence": minConfidence,
            "min_feature_confidence": minFeatureConfidence,
            "min_valid_overlap": minValidOverlap,
            "deduplicate_images": deduplicateImages,
            "min_region_confidence": minRegionConfidence,
            "min_similarity": minSimilarity,
            "include_flipped": includeFlippedPoses,
            "config": [
                "num_threads": threads,
                "device": device
            ]
        ]

        if let regions = requiredRegions, !regions.isEmpty {
            params["required_regions"] = regions
        }

        let command: [String: Any] = [
            "command": "search_by_pose_id",
            "params": params
        ]

        // Cancellable round trip (finding 27): a superseded search abandons here.
        let token = currentSearchToken()

        guard let response = sendSearchServerCommand(command, cancelToken: token) else {
            print("[SEARCH DEBUG] Failed to communicate with search server")
            return []
        }

        guard let status = response["status"] as? String else {
            print("[SEARCH DEBUG] Invalid response: missing status")
            return []
        }

        if status != "success" {
            let message = response["message"] as? String ?? "Unknown error"
            print("[SEARCH DEBUG] Search by pose id failed: \(message)")
            return []
        }

        guard let resultsArray = response["results"] as? [[String: Any]] else {
            print("[SEARCH DEBUG] Invalid response: missing or invalid results array")
            return []
        }

        print("[SEARCH DEBUG] Received \(resultsArray.count) results from server")
        return resultsArray.compactMap { parseSearchResult(from: $0) }
    }

    // MARK: - Deferred result details (finding 25)

    /// The heavy fields a deferred-detail search ROW omits for large result sets
    /// (>50 results): the thumbnail JPEG, the 133-keypoint overlay array, and the
    /// detailed NudeNet region breakdown. Fetched on demand for the visible page only.
    struct ResultDetail {
        let thumbnailData: Data?
        let keypoints: [[Double]]
        let detailedRegions: [[String: Any]]
    }

    /// Fetch the deferred heavy fields for a set of poses (the visible page) via the
    /// persistent search server's `fetch_details` command. Returns a map keyed by
    /// pose_id; missing ids are simply absent. Returns [:] on any failure so the UI can
    /// degrade gracefully (it falls back to loading thumbnails from disk).
    func fetchResultDetails(poseIds: [String]) -> [String: ResultDetail] {
        guard !poseIds.isEmpty else { return [:] }

        let command: [String: Any] = [
            "command": "fetch_details",
            "params": ["pose_ids": poseIds]
        ]

        guard let response = sendSearchServerCommand(command),
              (response["status"] as? String) == "success",
              let details = response["details"] as? [[String: Any]] else {
            print("[DETAILS DEBUG] fetch_details failed for \(poseIds.count) pose(s)")
            return [:]
        }

        var result: [String: ResultDetail] = [:]
        for detail in details {
            guard let poseId = detail["pose_id"] as? String else { continue }

            // Decode base64 thumbnail if present (may be null).
            var thumbnailData: Data? = nil
            if let thumbnailBase64 = detail["thumbnail_base64"] as? String {
                thumbnailData = Data(base64Encoded: thumbnailBase64)
            }

            // Keypoints: 133 x [x, y, conf]. Tolerate absence/odd shapes.
            var keypoints: [[Double]] = []
            if let keypointsRaw = detail["keypoints"] as? [[Any]] {
                keypoints = keypointsRaw.compactMap { kp in
                    guard kp.count == 3,
                          let x = kp[0] as? Double,
                          let y = kp[1] as? Double,
                          let conf = kp[2] as? Double else {
                        return nil
                    }
                    return [x, y, conf]
                }
            }

            let detailedRegions = (detail["visible_regions_detailed"] as? [[String: Any]]) ?? []

            result[poseId] = ResultDetail(
                thumbnailData: thumbnailData,
                keypoints: keypoints,
                detailedRegions: detailedRegions
            )
        }

        print("[DETAILS DEBUG] fetch_details returned \(result.count)/\(poseIds.count) detail(s)")
        return result
    }

    // MARK: - Update Image Paths (after moving files)

    struct PathUpdateResult {
        let updated: Int
        let missing: Int
        let conflicts: Int
    }

    /// Batch-update stored image paths after files were moved on disk. Applied in
    /// one transaction server-side, keyed by the old (unique) path. Returns nil if
    /// the server call itself failed.
    func updateImagePaths(_ moves: [(oldPath: String, newPath: String)]) -> PathUpdateResult? {
        guard !moves.isEmpty else { return PathUpdateResult(updated: 0, missing: 0, conflicts: 0) }

        let (threads, useGPU) = getThreadSettings()
        let params: [String: Any] = [
            "moves": moves.map { ["old_path": $0.oldPath, "new_path": $0.newPath] },
            "config": [
                "num_threads": threads,
                "device": useGPU ? "mps" : "cpu"
            ]
        ]
        let command: [String: Any] = [
            "command": "update_image_paths",
            "params": params
        ]

        guard let response = sendSearchServerCommand(command),
              let status = response["status"] as? String, status == "success" else {
            print("[MOVE DEBUG] Stored-path update failed to reach the search server")
            return nil
        }
        return PathUpdateResult(
            updated: response["updated"] as? Int ?? 0,
            missing: response["missing"] as? Int ?? 0,
            conflicts: response["conflicts"] as? Int ?? 0
        )
    }

    // MARK: - Browse by Body Parts
    func browseByBodyParts(
        requiredRegions: [String],
        categoryThresholds: [String: Double],
        k: Int = 50,
        sortBy: String = "confidence"
    ) -> [SearchResult] {
        print("[BROWSE DEBUG] Required regions: \(requiredRegions)")
        print("[BROWSE DEBUG] Category thresholds: \(categoryThresholds)")
        print("[BROWSE DEBUG] k=\(k), sortBy=\(sortBy)")

        // Build command for search server
        let params: [String: Any] = [
            "required_regions": requiredRegions,
            "category_thresholds": categoryThresholds,
            "k": k,
            "sort_by": sortBy
        ]

        let command: [String: Any] = [
            "command": "browse_body_parts",
            "params": params
        ]

        // Send command to persistent server
        guard let response = sendSearchServerCommand(command) else {
            print("[BROWSE DEBUG] Failed to communicate with search server")
            return []
        }

        // Check response status
        guard let status = response["status"] as? String else {
            print("[BROWSE DEBUG] Invalid response: missing status")
            return []
        }

        if status != "success" {
            let message = response["message"] as? String ?? "Unknown error"
            print("[BROWSE DEBUG] Browse failed: \(message)")
            return []
        }

        // Parse results
        guard let resultsArray = response["results"] as? [[String: Any]] else {
            print("[BROWSE DEBUG] Invalid response: missing or invalid results array")
            return []
        }

        print("[BROWSE DEBUG] Received \(resultsArray.count) results from server")
        return resultsArray.compactMap { parseSearchResult(from: $0) }
    }

    // MARK: - Excluded Folders

    /// One excluded folder as returned by the search server.
    struct ExcludedFolderEntry: Identifiable {
        let id: String
        let folderPath: String
        let notes: String?
    }

    func listExcludedFolders() -> [ExcludedFolderEntry] {
        let command: [String: Any] = ["command": "list_excluded_folders", "params": [:]]
        guard let response = sendSearchServerCommand(command),
              response["status"] as? String == "success",
              let folders = response["folders"] as? [[String: Any]] else {
            print("[EXCLUDED DEBUG] Failed to list excluded folders")
            return []
        }
        return folders.compactMap { entry in
            guard let id = entry["id"] as? String,
                  let path = entry["folder_path"] as? String else { return nil }
            return ExcludedFolderEntry(id: id, folderPath: path, notes: entry["notes"] as? String)
        }
    }

    func addExcludedFolder(path: String) -> Bool {
        let command: [String: Any] = [
            "command": "add_excluded_folder",
            "params": ["folder_path": path]
        ]
        guard let response = sendSearchServerCommand(command),
              response["status"] as? String == "success" else {
            print("[EXCLUDED DEBUG] Failed to add excluded folder: \(path)")
            return false
        }
        return true
    }

    func removeExcludedFolder(path: String) -> Bool {
        let command: [String: Any] = [
            "command": "remove_excluded_folder",
            "params": ["folder_path": path]
        ]
        guard let response = sendSearchServerCommand(command),
              response["status"] as? String == "success" else {
            print("[EXCLUDED DEBUG] Failed to remove excluded folder: \(path)")
            return false
        }
        return true
    }

    // MARK: - Corpus Re-detection

    /// Re-runs detection on every indexed image with the current detection
    /// pipeline, replacing stored poses EXCEPT manually corrected ones (new
    /// detections overlapping a corrected pose are dropped). Streams the same
    /// IndexProgress protocol as directory indexing and rebuilds the FAISS
    /// index at the end. Blocks until the run finishes — call from a
    /// background queue. Cancellable via cancelIndexing(); completed images
    /// keep their new detections on cancel.
    ///
    /// resumeSince: local-time "yyyy-MM-dd HH:mm:ss" of an interrupted run's
    /// start — images already re-detected after that moment are skipped.
    func redetectAllImages(resumeSince: String? = nil, progressCallback: @escaping (IndexProgress) -> Void) {
        // Read detector settings from UserDefaults (same as detectAllPoses)
        let poseModel = UserDefaults.standard.string(forKey: "poseModel") ?? "ensemble"
        let fusionMethod = UserDefaults.standard.string(forKey: "fusionMethod") ?? "confidence_weighted"
        let useTwoStage = UserDefaults.standard.bool(forKey: "useTwoStage")
        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        let poseModelsParam: String
        if poseModel == "ensemble" {
            poseModelsParam = "[\"rtmw-l\", \"rtmw-x\"]"
        } else {
            poseModelsParam = "\"\(poseModel)\""
        }

        let resumeParam = resumeSince.map { "'\($0)'" } ?? "None"

        // Worker processes parallelize detection (each loads its own model
        // stack; the GPU is not saturated by one process). Benchmarked on the
        // M5 Max: 2 workers = 1.9x pose throughput; 3 workers REGRESSES below
        // 2 (GPU/ANE contention). Override via `defaults write ... redetectWorkers N`.
        let storedWorkers = UserDefaults.standard.integer(forKey: "redetectWorkers")
        let workers = storedWorkers > 0 ? storedWorkers : 2

        print("[REDETECT] Starting corpus re-detection: poseModel=\(poseModel), device=\(device), workers=\(workers), resumeSince=\(resumeSince ?? "fresh run")")

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge

        print('DEBUG: Initializing bridge for corpus re-detection', file=sys.stderr, flush=True)
        KW = dict(
            pose_models=\(poseModelsParam),
            fusion_method='\(fusionMethod)',
            use_two_stage=\(pythonBool(useTwoStage)),
            num_threads=\(threads),
            device='\(device)'
        )
        bridge = PostureKitBridge(**KW)
        result = bridge.redetect_all_images(
            resume_since=\(resumeParam),
            workers=\(workers),
            worker_init_kwargs=KW,
        )
        print(f'DEBUG: Re-detection complete: {result}', file=sys.stderr, flush=True)
        """

        runPythonScriptWithProgress(script, progressCallback: progressCallback)
    }

    // MARK: - Index Statistics
    /// Findings 24 + 42: answer from the already-running persistent search server's
    /// `statistics` command (sub-ms server-side) instead of spawning a fresh ~2.6s
    /// full-bridge one-shot per call. The server auto-starts on demand if not yet up
    /// (it is normally started eagerly at launch). Returns 0 on any failure.
    func getIndexStatistics() -> Int {
        let command: [String: Any] = [
            "command": "statistics",
            "params": [:]
        ]

        guard let response = sendSearchServerCommand(command),
              (response["status"] as? String) == "success",
              let statistics = response["statistics"] as? [String: Any],
              let totalPoses = statistics["total_poses"] as? Int else {
            print("[INDEX] getIndexStatistics: server statistics unavailable")
            return 0
        }

        return totalPoses
    }

    // MARK: - Index Preloading
    /// Finding 41: do NOT spawn a throwaway full-bridge one-shot whose loaded FAISS index
    /// dies with the process. Instead start the PERSISTENT search server (which lazily
    /// inits the bridge + loads the index inside the long-lived process) and send one
    /// `statistics` command to force that init now and read the pose count. The warm
    /// server then serves the user's first search with no further boot cost.
    ///
    /// Returns true if an index with poses exists, false for empty/no index. NOTE: the
    /// caller (AppDelegate) MUST post .indexPreloaded regardless of this return value —
    /// an empty index and a server-start failure are both valid "UI may proceed" states.
    func preloadIndex() -> Bool {
        print("[INDEX] Preloading via persistent search server (finding 41)...")

        // Start the resident server eagerly so the first search is warm. If startup
        // fails (e.g. Postgres down), fall through — the UI contract still requires the
        // .indexPreloaded post, which the caller handles.
        if !startSearchServer() {
            print("[INDEX] Search server failed to start - will retry lazily on first search")
            return false
        }

        // Force bridge/index init inside the resident process and read the pose count.
        let command: [String: Any] = [
            "command": "statistics",
            "params": [:]
        ]
        guard let response = sendSearchServerCommand(command),
              (response["status"] as? String) == "success",
              let statistics = response["statistics"] as? [String: Any] else {
            print("[INDEX] Statistics unavailable during preload - index will build on first search")
            return false
        }

        let totalPoses = statistics["total_poses"] as? Int ?? 0
        if totalPoses > 0 {
            print("[INDEX] Preloaded (server warm) with \(totalPoses) poses")
            return true
        } else {
            print("[INDEX] Empty index - will build on first search")
            return false
        }
    }

    // MARK: - Thumbnail Backfill
    func countMissingThumbnails() -> Int {
        print("[THUMBNAIL COUNT] Counting images without thumbnails...")

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')

        try:
            from src.storage.storage_manager import StorageManager
            from src.storage.models import Image
            from src.config.settings import settings
            from sqlalchemy import func

            storage = StorageManager(settings.DATABASE_URL)
            try:
                with storage.session_scope() as session:
                    count = session.query(func.count(Image.id)).filter(Image.thumbnail == None).scalar()
                    print(count if count is not None else 0, flush=True)
            finally:
                storage.close()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            print("0", flush=True)  # Return 0 on error
        """

        // Use shorter timeout for count query
        guard let output = runPythonScript(script, configureThreading: false, timeout: 10.0) else {
            print("[THUMBNAIL COUNT ERROR] Script timed out or failed")
            return 0
        }

        // Parse output
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let count = Int(trimmed) else {
            print("[THUMBNAIL COUNT ERROR] Failed to parse count from output: '\(trimmed)'")
            return 0
        }

        print("[THUMBNAIL COUNT] Found \(count) images without thumbnails")
        return count
    }

    func countTotalImages() -> Int {
        print("[IMAGE COUNT] Counting total images in database...")

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')

        try:
            from src.storage.storage_manager import StorageManager
            from src.storage.models import Image
            from src.config.settings import settings
            from sqlalchemy import func

            storage = StorageManager(settings.DATABASE_URL)
            try:
                with storage.session_scope() as session:
                    count = session.query(func.count(Image.id)).scalar()
                    print(count if count is not None else 0, flush=True)
            finally:
                storage.close()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            print("0", flush=True)  # Return 0 on error
        """

        // Use shorter timeout for count query
        guard let output = runPythonScript(script, configureThreading: false, timeout: 10.0) else {
            print("[IMAGE COUNT ERROR] Script timed out or failed")
            return 0
        }

        // Parse output
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let count = Int(trimmed) else {
            print("[IMAGE COUNT ERROR] Failed to parse count from output: '\(trimmed)'")
            return 0
        }

        print("[IMAGE COUNT] Found \(count) total images")
        return count
    }

    struct MovedMissingScanResult {
        let relocated: Int      // files found at a new path (by content hash) and repointed
        let stillMissing: Int   // stored entries whose file is gone (not relocated)
        let pruned: Int         // entries actually removed from the index (only when prune=true)
    }

    /// Reconcile the index with disk WITHOUT re-detecting. Relocates files moved/renamed in Finder
    /// by SHA-256 content hash; prunes genuinely-missing entries only when `prune` is true. Runs in
    /// a skip_models (faiss-only, torch-FREE) bridge, so it can't hit the faiss+torch libomp
    /// conflict. The rebuilt index is auto-reloaded by the resident search server.
    func scanForMovedAndMissing(folders: [String], recursive: Bool, prune: Bool,
                                completion: @escaping (MovedMissingScanResult?) -> Void) {
        let foldersJSON: String = (try? JSONSerialization.data(withJSONObject: folders))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let script = """
        import sys, json
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        bridge = PostureKitBridge(skip_models=True)   # torch-free: faiss + storage only
        folders = json.loads('''\(foldersJSON)''')
        res = bridge.scan_for_moved_and_missing(folders, recursive=\(pythonBool(recursive)), prune_missing=\(pythonBool(prune)))
        print(json.dumps(res))
        """
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { DispatchQueue.main.async { completion(nil) }; return }
            let output = self.runPythonScript(script, configureThreading: false, timeout: 600.0)
            var result: MovedMissingScanResult? = nil
            if let out = output,
               let lastLine = out.split(separator: "\n").last(where: { $0.contains("\"relocated\"") }) ?? out.split(separator: "\n").last,
               let data = String(lastLine).data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let relocated = obj["relocated"] as? Int {
                result = MovedMissingScanResult(
                    relocated: relocated,
                    stillMissing: obj["still_missing"] as? Int ?? 0,
                    pruned: obj["pruned"] as? Int ?? 0
                )
            }
            DispatchQueue.main.async { completion(result) }
        }
    }

    func backfillThumbnails(progressCallback: @escaping (ThumbnailProgress) -> Void) {
        print("[THUMBNAIL BACKFILL] Starting thumbnail generation for existing images...")

        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        print('THUMBNAIL_BACKFILL_START', file=sys.stderr, flush=True)
        bridge = PostureKitBridge(num_threads=\(threads), device='\(device)')
        result = bridge.backfill_thumbnails()
        print('THUMBNAIL_BACKFILL_COMPLETE', file=sys.stderr, flush=True)
        print(json.dumps(result))
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["-u", "-c", script]  // -u for unbuffered output
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)  // Set working directory

        // Defensive: Set termination handler to prevent zombies if we miss a code path
        process.terminationHandler = { proc in
            // This is called automatically when process exits, ensuring it's reaped
            // Even if we somehow don't explicitly call waitUntilExit()
        }

        // Clear Python environment variables to prevent venv pollution
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        var outputData = Data()

        // Read stderr for debug output
        errorPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { return }

            if let output = String(data: data, encoding: .utf8) {
                print("[THUMBNAIL STDERR] \(output.trimmingCharacters(in: .newlines))")
            }
        }

        // Read stdout for progress updates
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { return }

            outputData.append(data)

            // Try to parse each line as progress update
            if let output = String(data: data, encoding: .utf8) {
                let lines = output.split(separator: "\n")
                print("[THUMBNAIL STDOUT] Received \(lines.count) line(s)")
                for line in lines {
                    print("[THUMBNAIL STDOUT] Line: \(String(line.prefix(100)))...")
                    if let jsonData = line.data(using: .utf8),
                       let progress = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                       let processed = progress["images_processed"] as? Int,
                       let total = progress["images_total"] as? Int {

                        let updated = progress["images_updated"] as? Int ?? 0
                        let failed = progress["images_failed"] as? Int ?? 0
                        let currentFile = progress["current_file"] as? String ?? ""

                        let thumbnailProgress = ThumbnailProgress(
                            imagesProcessed: processed,
                            totalImages: total,
                            thumbnailsGenerated: updated,
                            failedImages: failed,
                            currentFile: currentFile,
                            progress: total > 0 ? Double(processed) / Double(total) : 0.0
                        )

                        DispatchQueue.main.async {
                            progressCallback(thumbnailProgress)
                        }
                    }
                }
            }
        }

        do {
            // Store process reference for cancellation
            self.thumbnailProcess = process

            try process.run()
            process.waitUntilExit()

            // Clear process reference when done
            self.thumbnailProcess = nil

            // Parse final result
            if let finalOutput = String(data: outputData, encoding: .utf8),
               let lastLine = finalOutput.split(separator: "\n").last,
               let jsonData = String(lastLine).data(using: .utf8),
               let result = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {

                let processed = result["images_processed"] as? Int ?? 0
                let updated = result["images_updated"] as? Int ?? 0
                let failed = result["images_failed"] as? Int ?? 0

                let finalProgress = ThumbnailProgress(
                    imagesProcessed: processed,
                    totalImages: processed,
                    thumbnailsGenerated: updated,
                    failedImages: failed,
                    currentFile: "",
                    progress: 1.0
                )

                DispatchQueue.main.async {
                    progressCallback(finalProgress)
                }

                print("[THUMBNAIL BACKFILL] Complete: \(updated) generated, \(failed) failed")
            }

        } catch {
            print("[THUMBNAIL BACKFILL ERROR] \(error)")
            self.thumbnailProcess = nil
        }
    }

    // MARK: - Directory Indexing
    func startIndexing(
        directories: [String],
        recursive: Bool = true,
        minConfidence: Double = 0.3,
        skipIndexed: Bool = true,
        deleteMissing: Bool = false,
        progressCallback: @escaping (IndexProgress) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Convert Swift Bool to Python bool string
            let recursiveStr = self.pythonBool(recursive)
            let skipIndexedStr = self.pythonBool(skipIndexed)
            let deleteMissingStr = self.pythonBool(deleteMissing)
            let (threads, _) = self.getThreadSettings()

            // ALWAYS use MPS for batch indexing operations
            // After GPU warmup, MPS is 11× faster than CPU (0.05s vs 0.58s per image)
            // First image pays ~1.8s warmup cost, but subsequent images are extremely fast
            let device = "mps"

            // Serialize directory list as JSON to safely handle paths with special characters
            let directoriesJSON: String
            if let data = try? JSONSerialization.data(withJSONObject: directories),
               let jsonString = String(data: data, encoding: .utf8) {
                directoriesJSON = jsonString
            } else {
                print("[BRIDGE DEBUG] Failed to serialize directories to JSON")
                return
            }

            let script = """
            import sys
            sys.path.insert(0, '\(self.venvSitePackages)')
            sys.path.insert(0, '\(self.projectPath)')
            from src.swift_bridge import PostureKitBridge
            from pathlib import Path
            import json

            DIRECTORIES = json.loads('''\(directoriesJSON)''')
            RECURSIVE = \(recursiveStr)
            SKIP_INDEXED = \(skipIndexedStr)
            DELETE_MISSING = \(deleteMissingStr)
            MIN_CONFIDENCE = \(minConfidence)

            print(f'DEBUG: Starting multi-directory indexing: {len(DIRECTORIES)} directories', file=sys.stderr, flush=True)
            for i, d in enumerate(DIRECTORIES):
                print(f'DEBUG:   [{i+1}/{len(DIRECTORIES)}] {d}', file=sys.stderr, flush=True)

            # Pre-scan: count total images across all directories for unified progress denominator.
            # Uses the bridge's own enumeration (classmethod — no model loading) plus the
            # excluded-folders list, so this count can never disagree with what
            # index_directory actually processes.
            from src.storage.storage_manager import StorageManager
            _count_sm = StorageManager()
            EXCLUDED = [f['folder_path'] for f in _count_sm.get_excluded_folders()]
            def count_images(directory_path, recursive):
                p = Path(directory_path)
                if not p.exists():
                    return 0
                files, _, _ = PostureKitBridge._enumerate_image_files(p, recursive)
                if EXCLUDED:
                    files = [f for f in files if not _count_sm.is_path_excluded(str(f), EXCLUDED)]
                return len(files)

            grand_total_images = 0
            for d in DIRECTORIES:
                count = count_images(d, RECURSIVE)
                grand_total_images += count
                print(f'DEBUG: {d} -> {count} images', file=sys.stderr, flush=True)
            print(f'DEBUG: Grand total images across all directories: {grand_total_images}', file=sys.stderr, flush=True)

            # Emit initial progress so the UI shows the true total immediately
            initial = {
                'type': 'progress',
                'current_file': 'Scanning directories...',
                'images_processed': 0,
                'total_images': grand_total_images,
                'poses_indexed': 0,
                'failed_images': 0,
                'skipped_images': 0,
                'progress': 0.0,
            }
            print(f'PROGRESS:{json.dumps(initial)}', flush=True)

            print(f'DEBUG: Initializing bridge with num_threads=\(threads), device=\\'\(device)\\', delete_missing={DELETE_MISSING}', file=sys.stderr, flush=True)
            bridge = PostureKitBridge(num_threads=\(threads), device='\(device)')

            # Cumulative stats across directories (used to offset per-directory PROGRESS lines)
            cum_processed = 0
            cum_poses = 0
            cum_failed = 0
            cum_skipped = 0

            # Stdout interceptor: rewrite PROGRESS: lines to include cumulative offsets
            # and the unified grand total, so the Swift UI sees one continuous progress stream.
            class ProgressInterceptor:
                def __init__(self, real_stdout):
                    self.real_stdout = real_stdout
                    self.buffer = ''
                    self.last_processed = 0
                    self.last_poses = 0
                    self.last_failed = 0
                    self.last_skipped = 0

                def write(self, s):
                    self.buffer += s
                    while '\\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\\n', 1)
                        self._handle_line(line)
                    return len(s)

                def _handle_line(self, line):
                    if line.startswith('PROGRESS:'):
                        try:
                            data = json.loads(line[len('PROGRESS:'):])
                            self.last_processed = data.get('images_processed', 0)
                            self.last_poses = data.get('poses_indexed', 0)
                            self.last_failed = data.get('failed_images', 0)
                            self.last_skipped = data.get('skipped_images', 0)
                            total_processed = cum_processed + self.last_processed
                            combined = {
                                'type': 'progress',
                                'current_file': data.get('current_file', ''),
                                'images_processed': total_processed,
                                'total_images': grand_total_images if grand_total_images > 0 else data.get('total_images', 0),
                                'poses_indexed': cum_poses + self.last_poses,
                                'failed_images': cum_failed + self.last_failed,
                                'skipped_images': cum_skipped + self.last_skipped,
                                'progress': (total_processed / grand_total_images) if grand_total_images > 0 else data.get('progress', 0),
                            }
                            self.real_stdout.write(f'PROGRESS:{json.dumps(combined)}\\n')
                            self.real_stdout.flush()
                        except Exception as e:
                            self.real_stdout.write(line + '\\n')
                            self.real_stdout.flush()
                    else:
                        self.real_stdout.write(line + '\\n')
                        self.real_stdout.flush()

                def flush(self):
                    if self.buffer:
                        # Retain tail — no newline yet
                        pass
                    self.real_stdout.flush()

            real_stdout = sys.stdout
            aggregated = {
                'total_images': grand_total_images,
                'processed_images': 0,
                'poses_indexed': 0,
                'failed_images': 0,
                'skipped_images': 0,
                'failed_index_additions': 0,
                'deleted_images': 0,
                'deleted_poses': 0,
                'deleted_from_index': 0,
                'success': True,
                'directories': DIRECTORIES,
                'per_directory_results': [],
            }

            # delete_missing runs once, armed on the LAST directory: the cleanup is
            # post-scan inside index_directory, so by then every listed folder has
            # been scanned and the content-hash self-heal has repointed moved files.
            for dir_index, d in enumerate(DIRECTORIES):
                print(f'DEBUG: === Indexing directory: {d} ===', file=sys.stderr, flush=True)
                interceptor = ProgressInterceptor(real_stdout)
                sys.stdout = interceptor
                try:
                    result = bridge.index_directory(
                        d,
                        recursive=RECURSIVE,
                        min_confidence=MIN_CONFIDENCE,
                        skip_indexed=SKIP_INDEXED,
                        delete_missing=DELETE_MISSING if dir_index == len(DIRECTORIES) - 1 else False,
                    )
                finally:
                    sys.stdout = real_stdout

                if not result.get('success', False):
                    err_msg = result.get('error', 'unknown')
                    print(f'DEBUG: Directory {d} failed: {err_msg}', file=sys.stderr, flush=True)
                    aggregated['success'] = False

                # Use authoritative counts from the return value
                cum_processed += result.get('processed_images', interceptor.last_processed)
                cum_poses += result.get('poses_indexed', interceptor.last_poses)
                cum_failed += result.get('failed_images', interceptor.last_failed)
                cum_skipped += result.get('skipped_images', interceptor.last_skipped)

                aggregated['processed_images'] = cum_processed
                aggregated['poses_indexed'] = cum_poses
                aggregated['failed_images'] = cum_failed
                aggregated['skipped_images'] = cum_skipped
                aggregated['failed_index_additions'] += result.get('failed_index_additions', 0)
                aggregated['deleted_images'] += result.get('deleted_images', 0)
                aggregated['deleted_poses'] += result.get('deleted_poses', 0)
                aggregated['deleted_from_index'] += result.get('deleted_from_index', 0)
                aggregated['per_directory_results'].append({
                    'directory': d,
                    'processed_images': result.get('processed_images', 0),
                    'poses_indexed': result.get('poses_indexed', 0),
                    'failed_images': result.get('failed_images', 0),
                    'skipped_images': result.get('skipped_images', 0),
                })

            # Emit a final 100% progress tick so the UI completes cleanly
            final_progress = {
                'type': 'progress',
                'current_file': 'Complete',
                'images_processed': cum_processed,
                'total_images': grand_total_images if grand_total_images > 0 else cum_processed,
                'poses_indexed': cum_poses,
                'failed_images': cum_failed,
                'skipped_images': cum_skipped,
                'progress': 1.0,
            }
            print(f'PROGRESS:{json.dumps(final_progress)}', flush=True)

            print(json.dumps(aggregated))
            """

            print("[BRIDGE DEBUG] About to execute Python script with streaming progress for \(directories.count) director\(directories.count == 1 ? "y" : "ies")")

            self.runPythonScriptWithProgress(script, progressCallback: progressCallback)
        }
    }

    func pauseIndexing() {
        guard let process = indexingProcess, process.isRunning else {
            print("[BRIDGE DEBUG] No indexing process to pause")
            return
        }

        print("[BRIDGE DEBUG] Pausing indexing process...")
        process.suspend()
        print("[BRIDGE DEBUG] Indexing process paused")
    }

    func resumeIndexing() {
        guard let process = indexingProcess else {
            print("[BRIDGE DEBUG] No indexing process to resume")
            return
        }

        print("[BRIDGE DEBUG] Resuming indexing process...")
        process.resume()
        print("[BRIDGE DEBUG] Indexing process resumed")
    }

    func cancelIndexing() {
        guard let process = indexingProcess else {
            print("[BRIDGE DEBUG] No indexing process to cancel")
            return
        }

        print("[BRIDGE DEBUG] Canceling indexing process...")
        process.terminate()
        indexingProcess = nil
        print("[BRIDGE DEBUG] Indexing process terminated")
    }

    func isIndexingRunning() -> Bool {
        return indexingProcess?.isRunning ?? false
    }

    func cancelSearch() {
        // Finding 27: searches now run on the persistent search server, not the dead
        // runCancellableSearch one-shot path. Bump the monotonic cancel token so any
        // in-flight server search round trip (which captured the previous token) detects
        // supersession in its read loop, abandons its response, and releases
        // searchServerLock — instead of the replacement query serializing behind the old
        // search's full 60s-budget round trip.
        searchCancelLock.lock()
        searchCancelToken &+= 1
        let newToken = searchCancelToken
        searchCancelLock.unlock()
        print("[SEARCH DEBUG] Search cancelled (token bumped to \(newToken)) - in-flight server search will abandon")

        // Legacy one-shot search process cleanup (no current search path uses it, but
        // keep the teardown defensive in case runCancellableSearch is ever reintroduced).
        searchLock.lock()
        defer { searchLock.unlock() }

        if let process = currentSearchProcess, process.isRunning {
            print("[SEARCH DEBUG] User cancelled legacy search process (PID: \(process.processIdentifier))")
            process.terminate()
            // Give it a moment to terminate
            Thread.sleep(forTimeInterval: 0.1)
            if process.isRunning {
                print("[SEARCH DEBUG] Search did not terminate gracefully, force killing...")
                kill(process.processIdentifier, SIGKILL)
            }
            currentSearchProcess = nil
        }
    }

    func cancelThumbnailGeneration() {
        guard let process = thumbnailProcess else {
            print("[BRIDGE DEBUG] No thumbnail generation process to cancel")
            return
        }

        print("[BRIDGE DEBUG] Canceling thumbnail generation process...")
        process.terminate()
        thumbnailProcess = nil
        print("[BRIDGE DEBUG] Thumbnail generation process terminated")
    }

    func isThumbnailGenerationRunning() -> Bool {
        return thumbnailProcess?.isRunning ?? false
    }

    // MARK: - Helper Methods
    private func runCancellableSearch(_ script: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["-c", script]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Defensive: Set termination handler to prevent zombies if we miss a code path
        process.terminationHandler = { proc in
            // This is called automatically when process exits, ensuring it's reaped
            // Even if we somehow don't explicitly call waitUntilExit()
        }

        // Clear Python environment variables
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")

        // Minimal threading for search operations
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        // Track this process
        searchLock.lock()
        currentSearchProcess = process
        searchLock.unlock()

        do {
            try process.run()

            // Timeout for search operations
            let semaphore = DispatchSemaphore(value: 0)
            var completed = false
            var outputData = Data()
            var errorData = Data()

            DispatchQueue.global(qos: .userInitiated).async {
                process.waitUntilExit()
                outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
                errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
                completed = true
                semaphore.signal()
            }

            // Increased timeout to 120s to accommodate first-time index build
            // If index doesn't exist, it must be built on first search (can take 60-90s for large datasets)
            let result = semaphore.wait(timeout: .now() + 120.0)

            if result == .timedOut {
                print("[SEARCH DEBUG] Search timed out after 120s, terminating...")
                process.terminate()
                Thread.sleep(forTimeInterval: 0.5)
                if process.isRunning {
                    kill(process.processIdentifier, SIGKILL)
                }
                // CRITICAL: Always wait to reap zombie processes
                process.waitUntilExit()
                print("[SEARCH DEBUG] Process reaped after timeout")
                // Clear process reference if it's still ours (prevents race with newer searches)
                searchLock.lock()
                if currentSearchProcess === process {
                    currentSearchProcess = nil
                }
                searchLock.unlock()
                return nil
            }

            if !completed {
                // Clear process reference if it's still ours
                searchLock.lock()
                if currentSearchProcess === process {
                    currentSearchProcess = nil
                }
                searchLock.unlock()
                return nil
            }

            // Log stderr output
            let stderrOutput = String(data: errorData, encoding: .utf8) ?? ""
            if !stderrOutput.isEmpty {
                print("Python stderr: \(stderrOutput)")
            }

            // Handle abnormal termination
            if process.terminationStatus != 0 {
                // Differentiate between intentional cancellation vs actual errors
                if process.terminationReason == .uncaughtSignal {
                    let signal = process.terminationStatus
                    if signal == SIGTERM || signal == SIGKILL {
                        print("[SEARCH DEBUG] Search was cancelled (signal: \(signal))")
                    } else {
                        print("[SEARCH ERROR] Search crashed with signal: \(signal)")
                    }

                    // Always log stderr for cancelled searches (helps debug if cancellation was premature)
                    if !stderrOutput.isEmpty {
                        print("[SEARCH DEBUG] Stderr before cancellation: \(stderrOutput.prefix(500))...")
                    }
                } else {
                    print("[SEARCH ERROR] Search failed with exit status: \(process.terminationStatus)")
                    if !stderrOutput.isEmpty {
                        print("[SEARCH ERROR] Error details: \(stderrOutput)")
                    }
                }
                // Clear process reference if it's still ours
                searchLock.lock()
                if currentSearchProcess === process {
                    currentSearchProcess = nil
                }
                searchLock.unlock()
                return nil
            }

            // Success - return stdout
            let output = String(data: outputData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let out = output, !out.isEmpty {
                print("[SEARCH DEBUG] Search completed successfully, output length: \(out.count) chars")
            }
            return output
        } catch {
            print("Failed to run search: \(error)")
            // Clear process reference if it's still ours
            searchLock.lock()
            if currentSearchProcess === process {
                currentSearchProcess = nil
            }
            searchLock.unlock()
            return nil
        }
    }

    func runPythonScript(_ script: String, configureThreading: Bool = true, timeout: TimeInterval = 30.0) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["-c", script]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Defensive: Set termination handler to prevent zombies if we miss a code path
        process.terminationHandler = { proc in
            // This is called automatically when process exits, ensuring it's reaped
            // Even if we somehow don't explicitly call waitUntilExit()
        }

        // Clear Python environment variables to prevent venv pollution
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")

        // Set thread environment variables for OpenMP/MKL/OpenBLAS (only for compute-heavy tasks)
        // Skip for lightweight operations like search to avoid pthread initialization conflicts
        if configureThreading {
            let (threads, _) = getThreadSettings()
            environment["OMP_NUM_THREADS"] = "\(threads)"
            environment["MKL_NUM_THREADS"] = "\(threads)"
            environment["OPENBLAS_NUM_THREADS"] = "\(threads)"
        } else {
            // For search operations, use minimal threading to avoid conflicts
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            environment["OPENBLAS_NUM_THREADS"] = "1"
        }

        // Enable PyTorch MPS fallback for unsupported operations
        // PyTorch MPS on Apple Silicon doesn't support all operations (e.g., hardsigmoid)
        // This enables automatic CPU fallback for those ops
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        do {
            try process.run()

            // Implement timeout mechanism
            let semaphore = DispatchSemaphore(value: 0)
            var completed = false
            var outputData = Data()
            var errorData = Data()

            DispatchQueue.global(qos: .userInitiated).async {
                // Wait for process to complete
                process.waitUntilExit()
                outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
                errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
                completed = true
                semaphore.signal()
            }

            // Wait with timeout
            let result = semaphore.wait(timeout: .now() + timeout)

            if result == .timedOut {
                print("[BRIDGE DEBUG] Process timed out after \(timeout)s, terminating...")
                process.terminate()
                // Give it a moment to terminate gracefully
                Thread.sleep(forTimeInterval: 0.5)
                if process.isRunning {
                    // Force kill if still running
                    kill(process.processIdentifier, SIGKILL)
                }
                // CRITICAL: Always wait to reap zombie processes
                process.waitUntilExit()
                print("[BRIDGE DEBUG] Process reaped after timeout")
                return nil
            }

            if !completed {
                print("[BRIDGE DEBUG] Unexpected state: timeout didn't occur but process not complete")
                return nil
            }

            if let errorOutput = String(data: errorData, encoding: .utf8), !errorOutput.isEmpty {
                print("Python stderr: \(errorOutput)")
            }

            if process.terminationStatus != 0 {
                print("Python script failed with status: \(process.terminationStatus)")
                return nil
            }

            return String(data: outputData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            print("Failed to run Python script: \(error)")
            return nil
        }
    }

    private func runPythonScriptWithProgress(_ script: String, progressCallback: @escaping (IndexProgress) -> Void) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonExecutable)
        process.arguments = ["-c", script]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Defensive: Set termination handler to prevent zombies if we miss a code path
        process.terminationHandler = { proc in
            // This is called automatically when process exits, ensuring it's reaped
            // Even if we somehow don't explicitly call waitUntilExit()
        }

        // Clear Python environment variables
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PYTHONHOME")
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "__PYVENV_LAUNCHER__")

        // Set thread environment variables for OpenMP/MKL/OpenBLAS
        let (threads, _) = getThreadSettings()
        environment["OMP_NUM_THREADS"] = "\(threads)"
        environment["MKL_NUM_THREADS"] = "\(threads)"
        environment["OPENBLAS_NUM_THREADS"] = "\(threads)"

        // Enable PyTorch MPS fallback for unsupported operations
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        // Set database profile from UserDefaults (irl/2d/3d)
        environment["DB_PROFILE"] = getActiveProfile()

        process.environment = environment

        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        // Store process reference for stop functionality
        self.indexingProcess = process

        do {
            try process.run()

            // Read output and error streams line by line in real-time
            let outputHandle = outputPipe.fileHandleForReading
            let errorHandle = errorPipe.fileHandleForReading
            var outputBuffer = ""
            var errorBuffer = ""

            while process.isRunning {
                // Read stdout
                let data = outputHandle.availableData
                if data.count > 0 {
                    if let output = String(data: data, encoding: .utf8) {
                        outputBuffer += output

                        // Process complete lines
                        var lines = outputBuffer.components(separatedBy: "\n")
                        outputBuffer = lines.popLast() ?? ""  // Keep incomplete line in buffer

                        for line in lines {
                            if line.hasPrefix("PROGRESS:") {
                                let jsonString = String(line.dropFirst("PROGRESS:".count))
                                if let jsonData = jsonString.data(using: .utf8),
                                   let progressData = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {

                                    let progress = IndexProgress(
                                        currentFile: progressData["current_file"] as? String ?? "",
                                        imagesProcessed: progressData["images_processed"] as? Int ?? 0,
                                        totalImages: progressData["total_images"] as? Int ?? 0,
                                        posesIndexed: progressData["poses_indexed"] as? Int ?? 0,
                                        failedImages: progressData["failed_images"] as? Int ?? 0,
                                        skippedImages: progressData["skipped_images"] as? Int ?? 0,
                                        progress: progressData["progress"] as? Double ?? 0.0
                                    )

                                    DispatchQueue.main.async {
                                        progressCallback(progress)
                                    }
                                }
                            }
                        }
                    }
                }

                // Read stderr and print in real-time
                let errorData = errorHandle.availableData
                if errorData.count > 0 {
                    if let errorOutput = String(data: errorData, encoding: .utf8) {
                        errorBuffer += errorOutput

                        // Process complete lines
                        var errorLines = errorBuffer.components(separatedBy: "\n")
                        errorBuffer = errorLines.popLast() ?? ""

                        for line in errorLines {
                            if !line.isEmpty {
                                print("[Python stderr] \(line)")
                            }
                        }
                    }
                }

                if data.count == 0 && errorData.count == 0 {
                    // Small delay to prevent busy waiting
                    usleep(100000) // 0.1 seconds
                }
            }

            // Process any remaining output
            let remainingData = outputHandle.readDataToEndOfFile()
            if let remainingOutput = String(data: remainingData, encoding: .utf8) {
                outputBuffer += remainingOutput
                let lines = outputBuffer.components(separatedBy: "\n")

                for line in lines {
                    if line.hasPrefix("PROGRESS:") {
                        let jsonString = String(line.dropFirst("PROGRESS:".count))
                        if let jsonData = jsonString.data(using: .utf8),
                           let progressData = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {

                            let progress = IndexProgress(
                                currentFile: progressData["current_file"] as? String ?? "",
                                imagesProcessed: progressData["images_processed"] as? Int ?? 0,
                                totalImages: progressData["total_images"] as? Int ?? 0,
                                posesIndexed: progressData["poses_indexed"] as? Int ?? 0,
                                failedImages: progressData["failed_images"] as? Int ?? 0,
                                skippedImages: progressData["skipped_images"] as? Int ?? 0,
                                progress: progressData["progress"] as? Double ?? 0.0
                            )

                            DispatchQueue.main.async {
                                progressCallback(progress)
                            }
                        }
                    }
                }
            }

            // Wait for process to complete
            process.waitUntilExit()

            // Clear process reference
            self.indexingProcess = nil

            // Read stderr
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            if let errorOutput = String(data: errorData, encoding: .utf8), !errorOutput.isEmpty {
                print("Python stderr: \(errorOutput)")
            }

            if process.terminationStatus != 0 {
                print("Python script failed with status: \(process.terminationStatus)")
            }

        } catch {
            print("Failed to run Python script with progress: \(error)")
            self.indexingProcess = nil
        }
    }

    private func saveImageToTemp(_ image: NSImage) -> String? {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let pngData = bitmap.representation(using: .png, properties: [:]) else {
            return nil
        }

        let tempPath = NSTemporaryDirectory() + "posturekit_\(UUID().uuidString).png"
        try? pngData.write(to: URL(fileURLWithPath: tempPath))
        return tempPath
    }

    private func parsePoseResult(from json: [String: Any]) -> PoseDetectionResult? {
        guard let keypoints = json["keypoints"] as? [[Double]],
              let visibilityDoubles = json["visibility"] as? [Double],
              let bbox = json["bbox"] as? [Double],
              let confidence = json["confidence"] as? Double,
              let personIndex = json["person_id"] as? Int else {
            print("[PARSE DEBUG] Failed to parse pose result:")
            print("  keypoints: \(json["keypoints"] != nil ? "OK" : "MISSING")")
            print("  visibility: \(json["visibility"] != nil ? "type=\(type(of: json["visibility"]!))" : "MISSING")")
            print("  bbox: \(json["bbox"] != nil ? "OK" : "MISSING")")
            print("  confidence: \(json["confidence"] != nil ? "OK" : "MISSING")")
            print("  person_id: \(json["person_id"] != nil ? "OK" : "MISSING")")
            return nil
        }

        // Convert visibility from Double to Int (handles continuous visibility from ensemble fusion)
        let visibility = visibilityDoubles.map { Int($0.rounded()) }

        return PoseDetectionResult(
            keypoints: keypoints,
            visibility: visibility,
            bbox: bbox,
            confidence: confidence,
            personIndex: personIndex
        )
    }

    private func parseFeatures(from json: [String: Any]) -> GeometricFeatures? {
        print("[PARSE DEBUG] Feature JSON keys: \(json.keys)")

        // Python returns Double arrays, need to convert to Float
        guard let featureVectorDoubles = json["feature_vector"] as? [Double] else {
            print("[PARSE DEBUG] Failed to parse feature_vector as [Double]")
            print("[PARSE DEBUG] feature_vector type: \(type(of: json["feature_vector"]))")
            return nil
        }
        let featureVector = featureVectorDoubles.map { Float($0) }

        guard let jointAngles = json["joint_angles"] as? [String: Double] else {
            print("[PARSE DEBUG] Failed to parse joint_angles")
            return nil
        }

        guard let limbRatios = json["limb_ratios"] as? [String: Double] else {
            print("[PARSE DEBUG] Failed to parse limb_ratios")
            return nil
        }

        guard let bodyAngles = json["body_angles"] as? [String: Double] else {
            print("[PARSE DEBUG] Failed to parse body_angles")
            return nil
        }

        guard let symmetryScores = json["symmetry_scores"] as? [String: Double] else {
            print("[PARSE DEBUG] Failed to parse symmetry_scores")
            return nil
        }

        guard let occlusionPatternDoubles = json["occlusion_pattern"] as? [Double] else {
            print("[PARSE DEBUG] Failed to parse occlusion_pattern")
            return nil
        }
        let occlusionPattern = occlusionPatternDoubles.map { Float($0) }

        // Parse feature_confidence (optional - for confidence-aware matching)
        var featureConfidence: [Float]? = nil
        if let featureConfidenceDoubles = json["feature_confidence"] as? [Double] {
            featureConfidence = featureConfidenceDoubles.map { Float($0) }
            print("[PARSE DEBUG] Parsed feature_confidence with \(featureConfidence!.count) dimensions")
        } else {
            print("[PARSE DEBUG] No feature_confidence in response (older extraction)")
        }

        // Parse fused_vector (optional - may be null if image wasn't provided for visual features)
        var fusedVector: [Float]? = nil
        if let fusedVectorDoubles = json["fused_vector"] as? [Double] {
            fusedVector = fusedVectorDoubles.map { Float($0) }
            print("[PARSE DEBUG] Parsed fused_vector with \(fusedVector!.count) dimensions")
        } else {
            print("[PARSE DEBUG] No fused_vector in response (visual features not extracted)")
        }

        print("[PARSE DEBUG] Successfully parsed all feature fields")
        return GeometricFeatures(
            featureVector: featureVector,
            featureConfidence: featureConfidence,
            fusedVector: fusedVector,
            jointAngles: jointAngles,
            limbRatios: limbRatios,
            bodyAngles: bodyAngles,
            symmetryScores: symmetryScores,
            occlusionPattern: occlusionPattern
        )
    }

    private func parseSearchResult(from json: [String: Any]) -> SearchResult? {
        guard let id = json["pose_id"] as? String,
              let similarity = json["similarity_score"] as? Double,
              let imagePath = json["image_path"] as? String,
              let confidence = json["detection_confidence"] as? Double else {
            return nil
        }

        // Decode base64 thumbnail if present
        var thumbnailData: Data? = nil
        if let thumbnailBase64 = json["thumbnail_base64"] as? String {
            thumbnailData = Data(base64Encoded: thumbnailBase64)
            if thumbnailData != nil {
                print("[THUMBNAIL] Decoded thumbnail for \(URL(fileURLWithPath: imagePath).lastPathComponent): \(thumbnailData!.count) bytes")
            }
        }

        // Parse visible_regions array (canonical regions)
        let visibleRegions = json["visible_regions"] as? [String]

        // Parse visible_regions_detailed array (full 18-class NudeNet breakdown)
        let visibleRegionsDetailed = json["visible_regions_detailed"] as? [[String: Any]]

        // Parse keypoints array (133 keypoints × 3 [x, y, confidence])
        var keypoints: [[Double]]? = nil
        if let keypointsRaw = json["keypoints"] as? [[Any]] {
            keypoints = keypointsRaw.compactMap { kp in
                guard kp.count == 3,
                      let x = kp[0] as? Double,
                      let y = kp[1] as? Double,
                      let conf = kp[2] as? Double else {
                    return nil
                }
                return [x, y, conf]
            }
        }

        // Parse bbox if present (bounding box for multi-person support)
        let bbox = json["bbox"] as? [Double]

        // Parse person_id if present (identifies which person in multi-person images)
        let personId = json["person_id"] as? Int

        return SearchResult(
            id: id,
            similarity: Int(similarity * 100),
            filename: URL(fileURLWithPath: imagePath).lastPathComponent,
            confidence: confidence,
            imagePath: imagePath,
            detectedAt: nil,
            imageWidth: json["image_width"] as? Int,
            imageHeight: json["image_height"] as? Int,
            fileSize: json["file_size"] as? Int,
            thumbnailData: thumbnailData,
            visibleRegions: visibleRegions,
            visibleRegionsDetailed: visibleRegionsDetailed,
            keypoints: keypoints,
            bbox: bbox,
            personId: personId,
            isFlipped: (json["is_flipped_match"] as? Bool) ?? false
        )
    }

    // MARK: - Cleanup
    func shutdown() {
        print("[SHUTDOWN] Terminating Python subprocesses...")

        // Set shutdown flag to cancel any in-progress operations
        isShuttingDown = true

        // Terminate search server
        searchServerLock.lock()
        if let serverProc = searchServerProcess, serverProc.isRunning {
            print("[SHUTDOWN] Shutting down search server (PID: \(serverProc.processIdentifier))...")

            // Send graceful shutdown command
            let shutdownCommand: [String: Any] = ["command": "shutdown"]
            if let stdin = searchServerStdin {
                do {
                    let jsonData = try JSONSerialization.data(withJSONObject: shutdownCommand)
                    let jsonString = String(data: jsonData, encoding: .utf8)! + "\n"
                    stdin.write(jsonString.data(using: .utf8)!)
                } catch {
                    print("[SHUTDOWN] Failed to send shutdown command: \(error)")
                }
            }

            // Wait for graceful termination (cleanup may take 1-2 seconds)
            Thread.sleep(forTimeInterval: 2.0)

            // Force terminate if still running
            if serverProc.isRunning {
                print("[SHUTDOWN] Graceful shutdown timed out, force-terminating...")
                serverProc.terminate()
                Thread.sleep(forTimeInterval: 1.0)

                if serverProc.isRunning {
                    print("[SHUTDOWN] Force-killing search server")
                    kill(serverProc.processIdentifier, SIGKILL)
                }
            }
        }
        searchServerProcess = nil
        searchServerInitialized = false
        searchServerLock.unlock()

        // Terminate detection server (finding 22) — mirror the search-server teardown.
        detectServerLock.lock()
        if let detectProc = detectServerProcess, detectProc.isRunning {
            print("[SHUTDOWN] Shutting down detection server (PID: \(detectProc.processIdentifier))...")

            // Graceful shutdown: detect protocol keys on "type".
            let shutdownCommand: [String: Any] = ["type": "shutdown"]
            if let stdin = detectServerStdin {
                do {
                    let jsonData = try JSONSerialization.data(withJSONObject: shutdownCommand)
                    let jsonString = String(data: jsonData, encoding: .utf8)! + "\n"
                    stdin.write(jsonString.data(using: .utf8)!)
                } catch {
                    print("[SHUTDOWN] Failed to send detect shutdown command: \(error)")
                }
            }

            Thread.sleep(forTimeInterval: 1.5)

            if detectProc.isRunning {
                print("[SHUTDOWN] Detect server graceful shutdown timed out, force-terminating...")
                detectProc.terminate()
                Thread.sleep(forTimeInterval: 1.0)
                if detectProc.isRunning {
                    print("[SHUTDOWN] Force-killing detection server")
                    kill(detectProc.processIdentifier, SIGKILL)
                }
            }
        }
        detectServerProcess = nil
        detectServerInitialized = false
        detectServerLock.unlock()

        // Terminate indexing process if running
        if let indexProc = indexingProcess, indexProc.isRunning {
            print("[SHUTDOWN] Terminating indexing process (PID: \(indexProc.processIdentifier))...")
            indexProc.terminate()

            // Wait up to 2 seconds for graceful termination
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
                if indexProc.isRunning {
                    print("[SHUTDOWN] Force-killing indexing process")
                    kill(indexProc.processIdentifier, SIGKILL)
                }
            }
        }

        // Terminate search process if running (legacy - now using search server)
        searchLock.lock()
        if let searchProc = currentSearchProcess, searchProc.isRunning {
            print("[SHUTDOWN] Terminating search process (PID: \(searchProc.processIdentifier))...")
            searchProc.terminate()

            // Wait up to 1 second for graceful termination
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) {
                if searchProc.isRunning {
                    print("[SHUTDOWN] Force-killing search process")
                    kill(searchProc.processIdentifier, SIGKILL)
                }
            }
        }
        currentSearchProcess = nil
        searchLock.unlock()

        print("[SHUTDOWN] Python subprocess cleanup complete")
    }

    deinit {
        print("PythonBridgeSubprocess deallocated")
        // Cleanup handled by shutdown() method called from AppDelegate
    }
}

