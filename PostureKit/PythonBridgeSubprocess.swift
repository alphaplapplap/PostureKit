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

    private func sendSearchServerCommand(_ command: [String: Any]) -> [String: Any]? {
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
            return sendCommand(command, stdin: stdin, stdout: stdout)
        }

        return sendCommand(command, stdin: stdin, stdout: stdout)
    }

    private func sendCommand(_ command: [String: Any], stdin: FileHandle, stdout: FileHandle) -> [String: Any]? {
        do {
            // Serialize command
            let jsonData = try JSONSerialization.data(withJSONObject: command)
            let jsonString = String(data: jsonData, encoding: .utf8)! + "\n"

            // Send command
            print("[SEARCH SERVER] Sending command: \(command["command"] as? String ?? "unknown")")
            stdin.write(jsonString.data(using: .utf8)!)

            // Read response with timeout (line-buffered for large responses)
            // First search may take longer (index loading), subsequent searches are fast
            // Large responses (500 results with thumbnails) can be ~5MB and require multiple reads
            // Reduced sleep interval (0.01s) prevents pipe buffer deadlock
            let maxAttempts = 6000  // 60 seconds (6000 * 0.01s) for large result sets
            var accumulatedData = Data()
            var foundNewline = false

            for attempt in 1...maxAttempts {
                // Check if shutdown requested
                if isShuttingDown {
                    print("[SEARCH SERVER] Command cancelled - app is shutting down")
                    return nil
                }

                let data = stdout.availableData
                if !data.isEmpty {
                    accumulatedData.append(data)

                    // Check if we have a complete line (ends with \n)
                    // Python server sends one JSON object per line
                    // Check for newline byte (0x0A) without string conversion for efficiency
                    if accumulatedData.last == 0x0A {
                        foundNewline = true
                        print("[SEARCH SERVER] Received complete response after \(attempt * 10)ms (\(accumulatedData.count) bytes)")
                        break
                    }
                }

                // Check if server crashed
                if let serverProc = searchServerProcess, !serverProc.isRunning {
                    print("[SEARCH SERVER] Process terminated unexpectedly")
                    return nil
                }

                Thread.sleep(forTimeInterval: 0.01)
            }

            guard foundNewline else {
                print("[SEARCH SERVER] Timeout or incomplete response (received \(accumulatedData.count) bytes)")
                return nil
            }

            let responseData = accumulatedData

            let response = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
            print("[SEARCH SERVER] Received response: status=\(response?["status"] as? String ?? "unknown")")
            return response

        } catch {
            print("[SEARCH SERVER] Communication error: \(error)")
            return nil
        }
    }

    // MARK: - Pose Detection

    /// Detect ALL people in image (multi-person detection)
    func detectAllPoses(in image: NSImage) -> [PoseDetectionResult] {
        print("[DEBUG] detectAllPoses called")

        // Save image to temp file
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return []
        }
        print("[DEBUG] Saved image to: \(tempImagePath)")

        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        // Read detector settings from UserDefaults
        let poseModel = UserDefaults.standard.string(forKey: "poseModel") ?? "ensemble"
        let fusionMethod = UserDefaults.standard.string(forKey: "fusionMethod") ?? "confidence_weighted"
        let useTwoStage = UserDefaults.standard.bool(forKey: "useTwoStage")
        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        // Convert pose model setting to pose_models parameter
        let poseModelsParam: String
        if poseModel == "ensemble" {
            poseModelsParam = "[\"rtmw-l\", \"rtmw-x\"]"
        } else {
            poseModelsParam = "\"\(poseModel)\""
        }

        print("[DEBUG] Detector config: poseModel=\(poseModel), fusionMethod=\(fusionMethod), useTwoStage=\(useTwoStage), threads=\(threads), device=\(device)")

        // Call Python script for MULTI-PERSON detection
        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        print('DEBUG: Initializing bridge for multi-person detection', file=sys.stderr, flush=True)
        bridge = PostureKitBridge(
            pose_models=\(poseModelsParam),
            fusion_method='\(fusionMethod)',
            use_two_stage=\(pythonBool(useTwoStage)),
            num_threads=\(threads),
            device='\(device)'
        )
        print('DEBUG: Calling detect_multi_person_poses_from_file', file=sys.stderr, flush=True)
        results = bridge.detect_multi_person_poses_from_file('\(tempImagePath)')
        print(f'DEBUG: Detected {len(results)} people', file=sys.stderr, flush=True)
        print(json.dumps(results))
        """

        print("[DEBUG] Calling Python script for multi-person detection...")
        guard let output = runPythonScript(script, timeout: 120.0) else {
            print("[ERROR] Python script returned nil")
            return []
        }
        print("[DEBUG] Python output length: \(output.count) chars")

        let cleanOutput = extractJSON(from: output)
        print("[DEBUG] Clean output: \(cleanOutput)")

        // Parse JSON array of results
        guard let data = cleanOutput.data(using: .utf8),
              let jsonArray = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            print("[ERROR] Failed to parse multi-person detection results")
            print("[ERROR] Raw output: \(output)")
            return []
        }

        print("[DEBUG] Successfully parsed JSON array with \(jsonArray.count) people")

        // Convert each JSON dict to PoseDetectionResult
        let results = jsonArray.compactMap { json -> PoseDetectionResult? in
            return parsePoseResult(from: json)
        }

        print("[DEBUG] Converted \(results.count) pose results")
        return results
    }

    /// Detect single person in image (legacy method, only returns first person)
    func detectPose(in image: NSImage) -> PoseDetectionResult? {
        print("[DEBUG] detectPose called (single-person mode)")

        // Save image to temp file
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return nil
        }
        print("[DEBUG] Saved image to: \(tempImagePath)")

        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        // Read detector settings from UserDefaults
        let poseModel = UserDefaults.standard.string(forKey: "poseModel") ?? "ensemble"
        let fusionMethod = UserDefaults.standard.string(forKey: "fusionMethod") ?? "confidence_weighted"
        let useTwoStage = UserDefaults.standard.bool(forKey: "useTwoStage")
        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        // Convert pose model setting to pose_models parameter
        let poseModelsParam: String
        if poseModel == "ensemble" {
            poseModelsParam = "[\"rtmw-l\", \"rtmw-x\"]"
        } else {
            poseModelsParam = "\"\(poseModel)\""
        }

        print("[DEBUG] Detector config: poseModel=\(poseModel), fusionMethod=\(fusionMethod), useTwoStage=\(useTwoStage), threads=\(threads), device=\(device)")

        // Call Python script
        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        print('DEBUG: Initializing bridge with pose_models=\(poseModelsParam), fusion_method=\\'\(fusionMethod)\\', use_two_stage=\(pythonBool(useTwoStage)), num_threads=\(threads), device=\\'\(device)\\'', file=sys.stderr, flush=True)
        bridge = PostureKitBridge(
            pose_models=\(poseModelsParam),
            fusion_method='\(fusionMethod)',
            use_two_stage=\(pythonBool(useTwoStage)),
            num_threads=\(threads),
            device='\(device)'
        )
        print('DEBUG: Bridge initialized', file=sys.stderr, flush=True)
        result = bridge.detect_pose_from_file('\(tempImagePath)')
        print('DEBUG: Detection complete', file=sys.stderr, flush=True)
        print(json.dumps(result))
        """

        print("[DEBUG] Calling Python script...")
        // First detection with ensemble + two-stage can take 60-90s (model loading + MPS warmup)
        // Subsequent detections are much faster (~2-3s) as models stay in memory
        guard let output = runPythonScript(script, timeout: 120.0) else {
            print("[ERROR] Python script returned nil")
            return nil
        }
        print("[DEBUG] Python output length: \(output.count) chars")

        let cleanOutput = extractJSON(from: output)
        print("[DEBUG] Clean output: \(cleanOutput)")

        // Parse JSON result
        guard let data = cleanOutput.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("[ERROR] Failed to parse pose detection result")
            print("[ERROR] Raw output: \(output)")
            return nil
        }

        print("[DEBUG] Successfully parsed JSON")
        return parsePoseResult(from: json)
    }

    // MARK: - Feature Extraction
    func extractFeatures(from poseResult: PoseDetectionResult, image: NSImage) -> GeometricFeatures? {
        print("[DEBUG] extractFeatures called")

        // Save image to temp file
        guard let tempImagePath = saveImageToTemp(image) else {
            print("[ERROR] Failed to save image to temp")
            return nil
        }
        print("[DEBUG] Saved image to: \(tempImagePath)")

        defer {
            try? FileManager.default.removeItem(atPath: tempImagePath)
        }

        // Convert pose to JSON
        let poseDict: [String: Any] = [
            "keypoints": poseResult.keypoints,
            "visibility": poseResult.visibility,
            "bbox": poseResult.bbox,
            "confidence": poseResult.confidence,
            "person_id": poseResult.personIndex
        ]

        guard let poseJSON = try? JSONSerialization.data(withJSONObject: poseDict),
              let poseString = String(data: poseJSON, encoding: .utf8) else {
            print("[ERROR] Failed to serialize pose dict to JSON")
            return nil
        }

        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        let script = """
        import sys
        import cv2
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        print('DEBUG: Extracting features with num_threads=\(threads), device=\\'\(device)\\'', file=sys.stderr, flush=True)
        bridge = PostureKitBridge(num_threads=\(threads), device='\(device)')
        pose_data = json.loads('\(poseString)')

        # Load image for visual features
        image = cv2.imread('\(tempImagePath)')
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = None

        result = bridge.extract_features(pose_data, image_array=image_rgb)
        print('DEBUG: Feature extraction complete', file=sys.stderr, flush=True)
        print(json.dumps(result))
        """

        print("[DEBUG] Calling Python for feature extraction...")
        // Feature extraction can take longer if models need to load (60-90s first time)
        guard let output = runPythonScript(script, timeout: 120.0) else {
            print("[ERROR] Python script returned nil")
            return nil
        }

        let cleanOutput = extractJSON(from: output)

        guard let data = cleanOutput.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("[ERROR] Failed to parse feature extraction output")
            print("[ERROR] Raw output: \(output)")
            return nil
        }

        print("[DEBUG] Successfully extracted features")
        return parseFeatures(from: json)
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

        // Send command to persistent server
        guard let response = sendSearchServerCommand(command) else {
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
    func getIndexStatistics() -> Int {
        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        bridge = PostureKitBridge(num_threads=\(threads), device='\(device)')
        stats = bridge.similarity_engine.get_statistics()
        print(json.dumps(stats))
        """

        // Statistics is lightweight - disable threading to avoid conflicts
        guard let output = runPythonScript(script, configureThreading: false),
              let data = output.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let totalPoses = json["total_poses"] as? Int else {
            return 0
        }

        return totalPoses
    }

    // MARK: - Index Preloading
    func preloadIndex() -> Bool {
        print("[INDEX] Preloading FAISS index...")
        let (threads, useGPU) = getThreadSettings()
        let device = useGPU ? "mps" : "cpu"

        let script = """
        import sys
        sys.path.insert(0, '\(venvSitePackages)')
        sys.path.insert(0, '\(projectPath)')
        from src.swift_bridge import PostureKitBridge
        import json

        print('INDEX: Initializing bridge...', file=sys.stderr, flush=True)
        bridge = PostureKitBridge(num_threads=\(threads), device='\(device)')

        print('INDEX: Loading index...', file=sys.stderr, flush=True)
        # Try to load existing index
        index_loaded = bridge.similarity_engine.load_index()

        if index_loaded:
            stats = bridge.similarity_engine.get_statistics()
            print('INDEX: Loaded successfully', file=sys.stderr, flush=True)
            print(json.dumps({'success': True, 'total_poses': stats['total_poses']}))
        else:
            print('INDEX: No existing index found', file=sys.stderr, flush=True)
            print(json.dumps({'success': False, 'total_poses': 0}))
        """

        // Index loading can take 5-10 seconds for large indices
        guard let output = runPythonScript(script, configureThreading: false, timeout: 30.0),
              let data = output.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let success = json["success"] as? Bool else {
            print("[INDEX] Failed to preload index")
            return false
        }

        if success {
            if let totalPoses = json["total_poses"] as? Int {
                print("[INDEX] Preloaded with \(totalPoses) poses")
            }
            return true
        } else {
            print("[INDEX] No existing index found - will build on first search")
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

            # delete_missing should only run once, before the first directory
            first_dir = True
            for d in DIRECTORIES:
                print(f'DEBUG: === Indexing directory: {d} ===', file=sys.stderr, flush=True)
                interceptor = ProgressInterceptor(real_stdout)
                sys.stdout = interceptor
                try:
                    result = bridge.index_directory(
                        d,
                        recursive=RECURSIVE,
                        min_confidence=MIN_CONFIDENCE,
                        skip_indexed=SKIP_INDEXED,
                        delete_missing=DELETE_MISSING if first_dir else False,
                    )
                finally:
                    sys.stdout = real_stdout
                first_dir = False

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
        searchLock.lock()
        defer { searchLock.unlock() }

        if let process = currentSearchProcess, process.isRunning {
            print("[SEARCH DEBUG] User cancelled search (PID: \(process.processIdentifier))")
            process.terminate()
            // Give it a moment to terminate
            Thread.sleep(forTimeInterval: 0.1)
            if process.isRunning {
                print("[SEARCH DEBUG] Search did not terminate gracefully, force killing...")
                kill(process.processIdentifier, SIGKILL)
            }
            currentSearchProcess = nil
        } else {
            print("[SEARCH DEBUG] No search process to cancel")
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

