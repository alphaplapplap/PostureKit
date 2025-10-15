import SwiftUI
import Combine

// MARK: - View Model
class PostureKitViewModel: ObservableObject {
    private let pythonBridge = PythonBridgeSubprocess.shared
    private var cancellables = Set<AnyCancellable>()

    // Query Image State
    @Published var queryImage: NSImage?
    @Published var queryImageName: String?
    @Published var queryImageSize: CGSize?  // Original image dimensions for keypoint normalization
    @Published var poseDetected: Bool = false
    @Published var poseConfidence: Double = 0.0
    @Published var detectedKeypoints: [[Double]]?  // Keypoints in original image coordinates

    // Multi-person detection state
    @Published var detectedPeople: [DetectedPerson] = []  // All detected people in query image
    @Published var selectedPersonIndex: Int = 0  // Index of person selected for search
    @Published var multiPersonDetectionComplete: Bool = false  // Whether multi-person detection finished

    // Detected pose data (for backward compatibility with single-person flow)
    var detectedPose: PoseDetectionResult?
    var extractedFeatures: GeometricFeatures?

    // Search Parameters
    @Published var numberOfResults: Int = 20
    @Published var numberOfResultsDouble: Double = 20.0
    @Published var minConfidence: Double = 0.5
    @Published var includeFlippedPoses: Bool = false

    // Body part filtering (18 specific NudeNet classes)
    @Published var requiredBodyParts: Set<String> = []  // Can include specific classes like "FEMALE_BREAST_EXPOSED"

    // Category-specific confidence thresholds (18 NudeNet classes)
    @Published var categoryThresholds: [String: Double] = [
        "FACE_MALE": 0.20, "FACE_FEMALE": 0.20,
        "BELLY_EXPOSED": 0.15, "BELLY_COVERED": 0.15,
        "FEET_EXPOSED": 0.25, "FEET_COVERED": 0.25,
        "ARMPITS_EXPOSED": 0.15, "ARMPITS_COVERED": 0.15,
        "FEMALE_BREAST_EXPOSED": 0.35, "FEMALE_BREAST_COVERED": 0.35,
        "MALE_BREAST_EXPOSED": 0.25,
        "BUTTOCKS_EXPOSED": 0.25, "BUTTOCKS_COVERED": 0.25,
        "FEMALE_GENITALIA_EXPOSED": 0.35, "FEMALE_GENITALIA_COVERED": 0.35,
        "MALE_GENITALIA_EXPOSED": 0.35,
        "ANUS_EXPOSED": 0.35, "ANUS_COVERED": 0.35
    ]

    // Browse mode state
    @Published var browseMode: Bool = false
    @Published var showThresholdSettings: Bool = false
    @Published var expandedCategories: Set<String> = ["Face", "Feet"]  // Default expanded

    // Category grouping for UI organization
    let bodyPartCategories: [(name: String, parts: [String])] = [
        ("Face", ["FACE_MALE", "FACE_FEMALE"]),
        ("Torso", ["BELLY_EXPOSED", "BELLY_COVERED", "FEMALE_BREAST_EXPOSED",
                   "FEMALE_BREAST_COVERED", "MALE_BREAST_EXPOSED"]),
        ("Feet", ["FEET_EXPOSED", "FEET_COVERED"]),
        ("Armpits", ["ARMPITS_EXPOSED", "ARMPITS_COVERED"]),
        ("Buttocks", ["BUTTOCKS_EXPOSED", "BUTTOCKS_COVERED"]),
        ("Genitalia", ["FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED",
                       "MALE_GENITALIA_EXPOSED"]),
        ("Anus", ["ANUS_EXPOSED", "ANUS_COVERED"])
    ]

    // Advanced Search Parameters
    @Published var kMultiplier: Double = 1.0  // DEPRECATED: Python applies intelligent multipliers (3-10×) based on search type
    @Published var minFeatureConfidence: Double = 0.35  // Per-feature confidence threshold (0.0-1.0)
    @Published var minValidOverlap: Double = 12.0  // Minimum matching dimensions (0-52)
    @Published var showMultiplePeoplePerImage: Bool = false  // Show all people from multi-person images (default: deduplicate for cleaner results)
    @Published var minRegionConfidence: Double = 0.3  // Minimum confidence for body part visibility (0.0-1.0)

    // Search Results
    @Published var searchResults: [SearchResult] = []
    @Published var searchTime: Double = 0.0

    // Search Progress Tracking
    @Published var searchStartTime: Date?
    @Published var searchElapsed: TimeInterval = 0.0
    private var searchTimer: Timer?

    // Index Statistics
    @Published var totalPosesIndexed: Int = 0
    @Published var indexStatus: IndexStatus = .loading

    // Selection State
    @Published var selectedResultIds: Set<String> = []
    private var lastSelectedIndex: Int?

    // View Settings
    @Published var viewMode: ViewMode = .grid
    @Published var thumbnailSize: Double = 180.0

    // Loading State
    @Published var isSearching: Bool = false
    @Published var isDetecting: Bool = false
    @Published var errorMessage: String?

    // Search deduplication (prevents duplicate searches from race conditions)
    private var searchInProgress: Bool = false
    private let searchQueue = DispatchQueue(label: "com.posturekit.search", qos: .userInitiated)
    private var lastSearchFeatures: [Float]? = nil
    private var searchDebounceTimer: DispatchWorkItem? = nil

    init() {
        // Listen for index preload notification
        NotificationCenter.default.publisher(for: .indexPreloaded)
            .sink { [weak self] _ in
                self?.handleIndexPreloaded()
            }
            .store(in: &cancellables)
    }

    private func handleIndexPreloaded() {
        print("[VM] Index preloaded, refreshing statistics...")
        indexStatus = .ready
        loadIndexStatistics()
    }

    func loadIndexStatistics() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }
            let totalPoses = self.pythonBridge.getIndexStatistics()
            DispatchQueue.main.async {
                self.totalPosesIndexed = totalPoses
            }
        }
    }

    func clearImage() {
        queryImage = nil
        queryImageName = nil
        queryImageSize = nil
        poseDetected = false
        poseConfidence = 0.0
        detectedKeypoints = nil
        detectedPose = nil
        extractedFeatures = nil

        // Clear multi-person state
        detectedPeople = []
        selectedPersonIndex = 0
        multiPersonDetectionComplete = false

        searchResults = []
        searchTime = 0.0
        searchStartTime = nil
        searchElapsed = 0.0
        searchTimer?.invalidate()
        searchTimer = nil
        errorMessage = nil
    }

    /// Select a specific person for search
    func selectPerson(at index: Int) {
        guard index >= 0 && index < detectedPeople.count else {
            print("[VM] Invalid person index: \(index)")
            return
        }

        selectedPersonIndex = index
        let person = detectedPeople[index]

        // Update UI state to show selected person's info
        poseConfidence = person.confidence
        detectedKeypoints = person.keypoints

        print("[VM] Selected person \(index + 1)/\(detectedPeople.count) (confidence: \(String(format: "%.2f", person.confidence)))")
    }

    private func startSearchTimer() {
        // Invalidate any existing timer first (prevents orphaned timers)
        searchTimer?.invalidate()

        searchStartTime = Date()
        searchElapsed = 0.0

        // Create timer without auto-scheduling
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let startTime = self.searchStartTime else { return }
            DispatchQueue.main.async {
                self.searchElapsed = Date().timeIntervalSince(startTime)
            }
        }

        // Explicitly add to main RunLoop in common mode (works during scrolling/tracking)
        RunLoop.main.add(timer, forMode: .common)
        searchTimer = timer
    }

    private func stopSearchTimer() {
        searchTimer?.invalidate()
        searchTimer = nil
    }

    func cancelSearch() {
        print("[VM] User cancelled search")
        pythonBridge.cancelSearch()

        // Reset state on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.isSearching = false
            self.searchStartTime = nil
            self.searchElapsed = 0.0
            self.stopSearchTimer()
            self.errorMessage = "Search cancelled by user"
        }

        // Reset search progress flag on search queue
        searchQueue.async { [weak self] in
            self?.searchInProgress = false
            self?.lastSearchFeatures = nil  // Allow re-searching same pose
        }
    }

    func detectPose(in image: NSImage) {
        print("[VM DEBUG] detectPose called in ViewModel")
        isDetecting = true
        errorMessage = nil

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                print("[VM DEBUG] self is nil in async block")
                return
            }

            print("[VM DEBUG] Calling pythonBridge.detectAllPoses for multi-person detection")
            // Call multi-person detection to get ALL people
            let allPoses = self.pythonBridge.detectAllPoses(in: image)

            if allPoses.isEmpty {
                print("[VM DEBUG] No poses detected in image")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.isDetecting = false
                    self.poseDetected = false
                    self.errorMessage = "No pose detected in image"
                }
                return
            }

            // Log how many people were detected
            print("[VM DEBUG] ✅ Detected \(allPoses.count) person(s) in image")
            for (i, pose) in allPoses.enumerated() {
                print("[VM DEBUG]   Person \(i): confidence=\(pose.confidence), bbox=\(pose.bbox)")
            }

            // TODO: UI currently only displays first person
            // Full multi-person UI support requires updating skeleton overlay
            if allPoses.count > 1 {
                print("[VM WARNING] ⚠️ Multiple people detected but UI only shows first person")
                print("[VM WARNING] ⚠️ This is a known limitation - full multi-person UI coming soon")
            }

            // For now, use first person (highest confidence)
            let poseResult = allPoses[0]

            print("[VM DEBUG] Pose detected, extracting features")
            // Extract features (pass image for visual feature extraction)
            guard let features = self.pythonBridge.extractFeatures(from: poseResult, image: image) else {
                print("[VM DEBUG] Feature extraction failed")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.isDetecting = false
                    self.errorMessage = "Failed to extract features"
                }
                return
            }

            print("[VM DEBUG] Success! Updating UI")
            DispatchQueue.main.async { [weak self] in
                guard let self = self else {
                    print("[VM DEBUG] self is nil in UI update block")
                    return
                }
                print("[VM DEBUG] About to update UI with pose detection results")
                self.detectedPose = poseResult
                self.extractedFeatures = features
                self.poseDetected = true
                self.poseConfidence = poseResult.confidence
                self.isDetecting = false

                // Store keypoints and image size for skeleton overlay
                self.detectedKeypoints = poseResult.keypoints
                self.queryImageSize = image.size

                print("[VM DEBUG] ✅ UI updated successfully - app should remain running")
                print("[VM DEBUG] DetectedPose: \(poseResult.confidence), PoseDetected: true")
                print("[VM DEBUG] Stored \(poseResult.keypoints.count) keypoints for overlay")
            }
        }
    }

    /// Detect all people in uploaded image immediately (for multi-person UI)
    func detectAllPeopleInQueryImage() {
        guard let image = queryImage else {
            print("[VM] No query image to detect")
            return
        }

        print("[VM] Starting immediate multi-person detection on upload")
        isDetecting = true
        multiPersonDetectionComplete = false
        errorMessage = nil

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Detect all people in image
            let allPoses = self.pythonBridge.detectAllPoses(in: image)

            if allPoses.isEmpty {
                print("[VM] No people detected in uploaded image")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.isDetecting = false
                    self.multiPersonDetectionComplete = true
                    self.poseDetected = false
                    self.errorMessage = "No people detected in image"
                }
                return
            }

            // Convert poses to DetectedPerson objects
            let detectedPeople = allPoses.enumerated().map { (index, pose) in
                DetectedPerson(
                    personId: index,
                    bbox: pose.bbox,
                    keypoints: pose.keypoints,
                    confidence: pose.confidence,
                    poseResult: pose
                )
            }

            print("[VM] ✅ Detected \(detectedPeople.count) person(s) in uploaded image")

            // Update UI on main thread
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }

                self.detectedPeople = detectedPeople
                self.selectedPersonIndex = 0  // Auto-select first person
                self.multiPersonDetectionComplete = true
                self.isDetecting = false

                // Set detection flags for UI display
                if let firstPerson = detectedPeople.first {
                    self.poseDetected = true
                    self.poseConfidence = firstPerson.confidence
                    self.detectedKeypoints = firstPerson.keypoints
                    self.queryImageSize = image.size

                    print("[VM] Auto-selected Person 1/\(detectedPeople.count) (confidence: \(String(format: "%.2f", firstPerson.confidence)))")
                }
            }
        }
    }

    func performSearch() {
        guard let queryImg = queryImage else {
            errorMessage = "No image loaded"
            return
        }

        // Cancel any pending debounced search
        searchDebounceTimer?.cancel()

        // NEW FLOW: Use already-detected person data if available
        if !detectedPeople.isEmpty {
            print("[SEARCH] Using pre-detected person data (Person \(selectedPersonIndex + 1)/\(detectedPeople.count))")

            guard selectedPersonIndex < detectedPeople.count else {
                errorMessage = "Invalid person selection"
                return
            }

            let selectedPerson = detectedPeople[selectedPersonIndex]
            let pose = selectedPerson.poseResult

            print("[SEARCH] Extracting features for selected person...")
            isDetecting = true
            errorMessage = nil

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }

                // Extract features from selected person's pose
                guard let features = self.pythonBridge.extractFeatures(from: pose, image: queryImg) else {
                    print("[SEARCH] Feature extraction failed for selected person")
                    DispatchQueue.main.async {
                        self.isDetecting = false
                        self.errorMessage = "Failed to extract features"
                    }
                    return
                }

                print("[SEARCH] Features extracted for Person \(self.selectedPersonIndex + 1), starting search")

                // Update UI state
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.detectedPose = pose
                    self.extractedFeatures = features
                    self.isDetecting = false
                }

                // Perform search with selected person's features
                self.executeSearch(features: features, selectedPersonPose: pose)
            }
            return
        }

        // FALLBACK: Old single-person detection flow (for Browse mode or if detection failed)
        print("[SEARCH DEBUG] No pre-detected people, falling back to on-demand detection")

        isDetecting = true
        errorMessage = nil

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Decide which image to use for detection
            let imageToDetect: NSImage
            let shouldUpdateDisplay = self.includeFlippedPoses

            if self.includeFlippedPoses {
                print("[SEARCH DEBUG] Flip enabled - flipping image before detection")
                guard let flipped = self.flipImageHorizontally(queryImg) else {
                    DispatchQueue.main.async {
                        self.isDetecting = false
                        self.errorMessage = "Failed to flip image"
                    }
                    return
                }
                imageToDetect = flipped

                // Update displayed image to show the flip
                if shouldUpdateDisplay {
                    DispatchQueue.main.async { [weak self] in
                        self?.queryImage = flipped
                    }
                }
            } else {
                imageToDetect = queryImg
            }

            print("[SEARCH DEBUG] Detecting pose on \(self.includeFlippedPoses ? "flipped" : "original") image")

            // Detect pose on the chosen image
            guard let pose = self.pythonBridge.detectPose(in: imageToDetect) else {
                print("[SEARCH DEBUG] Pose detection failed")
                DispatchQueue.main.async {
                    self.isDetecting = false
                    self.poseDetected = false
                    self.errorMessage = "No pose detected in image"
                }
                return
            }

            print("[SEARCH DEBUG] Pose detected, extracting features")

            // Extract features from detected pose
            guard let features = self.pythonBridge.extractFeatures(from: pose, image: imageToDetect) else {
                print("[SEARCH DEBUG] Feature extraction failed")
                DispatchQueue.main.async {
                    self.isDetecting = false
                    self.errorMessage = "Failed to extract features"
                }
                return
            }

            print("[SEARCH DEBUG] Features extracted, updating UI and starting search")

            // Update UI with detection results
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.detectedPose = pose
                self.extractedFeatures = features
                self.poseDetected = true
                self.poseConfidence = pose.confidence
                self.detectedKeypoints = pose.keypoints
                self.queryImageSize = imageToDetect.size
                self.isDetecting = false
            }

            // Now perform search with extracted features
            self.executeSearch(features: features)
        }
    }

    private func executeSearch(features: GeometricFeatures, selectedPersonPose: PoseDetectionResult? = nil) {
        // Check if identical search is already in progress
        searchQueue.sync {
            if searchInProgress {
                print("[SEARCH DEBUG] Search already in progress, skipping duplicate")
                return
            }

            // Check if this is the same feature vector (prevents duplicate search on same pose)
            if let lastFeatures = lastSearchFeatures, lastFeatures == features.featureVector {
                print("[SEARCH DEBUG] Duplicate search request for same features, skipping")
                return
            }

            searchInProgress = true
            lastSearchFeatures = features.featureVector
        }

        DispatchQueue.main.async { [weak self] in
            self?.isSearching = true
            self?.errorMessage = nil
            self?.startSearchTimer()  // Start elapsed time tracking
        }

        let startTime = Date()

        searchQueue.async { [weak self] in
            guard let self = self else {
                self?.searchInProgress = false
                return
            }

            // Call Python directly - search FAISS index
            // Use geometric features (52-dim) for body-mapping-based similarity
            // (excludes clothing, background, lighting per user requirement)
            let searchVector = features.featureVector
            let searchConfidence = features.featureConfidence  // Pass confidence for confidence-aware matching

            // Pass numberOfResults directly to Python - Python applies intelligent multipliers:
            // - 10× for confidence-aware search (needs re-ranking buffer)
            // - 5× for region filtering (needs candidate buffer)
            // - 3× for basic search (accounts for deduplication/filtering)
            let requestCount = self.numberOfResults

            // Convert requiredBodyParts Set to Array for Python bridge
            // ONLY apply body part filters in Browse Database mode (not in Search by Image mode)
            let bodyPartsArray = self.browseMode && !self.requiredBodyParts.isEmpty ? Array(self.requiredBodyParts) : nil

            // Query feature statistics (handle optional featureConfidence)
            let validFeatures: Int
            let meanConfidence: Float
            let validOverlapMet: Bool

            if let confidence = searchConfidence, !confidence.isEmpty {
                validFeatures = confidence.filter { $0 >= Float(self.minFeatureConfidence) }.count
                meanConfidence = confidence.reduce(0, +) / Float(confidence.count)
                validOverlapMet = validFeatures >= Int(self.minValidOverlap)

                print("[SEARCH DEBUG] Query features:")
                print("  - Total features: \(confidence.count)")
                print("  - Mean confidence: \(String(format: "%.2f", meanConfidence))")
                print("  - Valid features (≥\(Int(self.minFeatureConfidence * 100))%): \(validFeatures)/52")
                print("  - Meets overlap threshold: \(validOverlapMet ? "YES" : "NO (need \(Int(self.minValidOverlap)), have \(validFeatures))")")
            } else {
                validFeatures = 0
                meanConfidence = 0.0
                validOverlapMet = false
                print("[SEARCH DEBUG] Query features: Confidence data not available")
            }

            print("[SEARCH DEBUG] Search parameters:")
            print("  - k (requested): \(requestCount) (Python will apply 3-10× multiplier for buffering)")
            print("  - minConfidence: \(self.minConfidence)")
            print("  - minFeatureConfidence: \(self.minFeatureConfidence)")
            print("  - minValidOverlap: \(Int(self.minValidOverlap))")

            // Perform search with the features (already from flipped image if flip was enabled)
            let results = self.pythonBridge.searchSimilar(
                featureVector: searchVector,
                k: requestCount,
                minConfidence: self.minConfidence,
                featureConfidence: searchConfidence,
                minFeatureConfidence: self.minFeatureConfidence,
                minValidOverlap: Int(self.minValidOverlap),
                requiredRegions: bodyPartsArray,
                deduplicateImages: !self.showMultiplePeoplePerImage,  // Inverted: true to show multiple = false to deduplicate
                minRegionConfidence: self.minRegionConfidence
            )

            let elapsed = Date().timeIntervalSince(startTime)

            // Mark search as complete before processing results
            // Already on searchQueue (serial), direct assignment is thread-safe
            self.searchInProgress = false

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }

                print("[SEARCH DEBUG] Received \(results.count) results from Python")

                // Diagnose if no results
                if results.isEmpty {
                    print("[SEARCH DEBUG] ⚠️  ZERO RESULTS - Possible reasons:")
                    print("  1. Index may be empty or failed to load")
                    print("  2. Query features don't meet minValidOverlap threshold")
                    print("  3. No poses in database match the query pose")
                    print("  4. Python search error (check stderr for SEARCH ERROR messages)")
                    if !validOverlapMet {
                        print("  5. ⚠️  Query has only \(validFeatures) valid features, but minValidOverlap=\(Int(self.minValidOverlap))")
                        print("      → Try lowering 'Min valid overlap' slider to \(max(0, validFeatures - 5))")
                    }
                }

                // Filter by similarity score (this is what the user expects!)
                let minSimilarityPercent = Int(self.minConfidence * 100)
                let similarityFiltered = results.filter { result in
                    result.similarity >= minSimilarityPercent
                }
                print("[SEARCH DEBUG] After similarity filter (≥\(minSimilarityPercent)%): \(similarityFiltered.count) results")

                if !results.isEmpty && similarityFiltered.isEmpty {
                    print("[SEARCH DEBUG] ⚠️  All results filtered out by similarity threshold")
                    print("      → Try lowering 'Min similarity' slider (currently \(minSimilarityPercent)%)")
                }

                // Filter out results where image files no longer exist on disk
                let existingFiles = similarityFiltered.filter { result in
                    guard let path = result.imagePath else { return false }
                    return FileManager.default.fileExists(atPath: path)
                }
                print("[SEARCH DEBUG] After file existence filter: \(existingFiles.count) results")

                // Take only the requested number of results
                let finalResults = Array(existingFiles.prefix(self.numberOfResults))
                print("[SEARCH DEBUG] Returning \(finalResults.count) results (requested: \(self.numberOfResults))")

                if finalResults.isEmpty && !results.isEmpty {
                    print("[SEARCH DEBUG] ⚠️  Had \(results.count) raw results but all filtered out")
                }

                self.searchResults = finalResults
                self.searchTime = elapsed
                self.isSearching = false
                self.stopSearchTimer()  // Stop elapsed time tracking

                // Update error message for UI display
                if finalResults.isEmpty && !self.isSearching {
                    if results.isEmpty {
                        self.errorMessage = "No similar poses found. Try adjusting search parameters."
                    } else {
                        self.errorMessage = "Results filtered out. Try lowering similarity threshold."
                    }
                } else {
                    self.errorMessage = nil
                }
            }
        }
    }

    func performBrowse() {
        guard !requiredBodyParts.isEmpty else {
            errorMessage = "Select at least one body part to browse"
            return
        }

        isSearching = true
        errorMessage = nil
        startSearchTimer()

        let startTime = Date()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Filter thresholds for selected parts only
            let activeThresholds = self.categoryThresholds.filter {
                self.requiredBodyParts.contains($0.key)
            }

            print("[BROWSE DEBUG] Active thresholds: \(activeThresholds)")
            print("[BROWSE DEBUG] Required regions: \(self.requiredBodyParts)")

            let results = self.pythonBridge.browseByBodyParts(
                requiredRegions: Array(self.requiredBodyParts),
                categoryThresholds: activeThresholds,
                k: self.numberOfResults,
                sortBy: "confidence"
            )

            let elapsed = Date().timeIntervalSince(startTime)

            DispatchQueue.main.async {
                print("[BROWSE DEBUG] Received \(results.count) results in \(elapsed)s")

                self.searchResults = results
                self.searchTime = elapsed
                self.isSearching = false
                self.stopSearchTimer()

                if results.isEmpty {
                    self.errorMessage = "No images found with selected body parts. Try lowering thresholds."
                } else {
                    self.errorMessage = nil
                }
            }
        }
    }

    func resetThresholds() {
        categoryThresholds = [
            "FACE_MALE": 0.20, "FACE_FEMALE": 0.20,
            "BELLY_EXPOSED": 0.15, "BELLY_COVERED": 0.15,
            "FEET_EXPOSED": 0.25, "FEET_COVERED": 0.25,
            "ARMPITS_EXPOSED": 0.15, "ARMPITS_COVERED": 0.15,
            "FEMALE_BREAST_EXPOSED": 0.35, "FEMALE_BREAST_COVERED": 0.35,
            "MALE_BREAST_EXPOSED": 0.25,
            "BUTTOCKS_EXPOSED": 0.25, "BUTTOCKS_COVERED": 0.25,
            "FEMALE_GENITALIA_EXPOSED": 0.35, "FEMALE_GENITALIA_COVERED": 0.35,
            "MALE_GENITALIA_EXPOSED": 0.35,
            "ANUS_EXPOSED": 0.35, "ANUS_COVERED": 0.35
        ]
    }

    func toggleCategory(_ category: String) {
        if expandedCategories.contains(category) {
            expandedCategories.remove(category)
        } else {
            expandedCategories.insert(category)
        }
    }

    func loadThumbnail(for result: SearchResult) -> NSImage? {
        guard let path = result.imagePath,
              let image = NSImage(contentsOfFile: path) else {
            return nil
        }
        
        // Create thumbnail
        let thumbnailSize = NSSize(width: 200, height: 200)
        let thumbnail = NSImage(size: thumbnailSize)
        
        thumbnail.lockFocus()
        let aspectRatio = image.size.width / image.size.height
        let targetRect: NSRect
        
        if aspectRatio > 1 {
            let height = thumbnailSize.height
            let width = height * aspectRatio
            let x = (thumbnailSize.width - width) / 2
            targetRect = NSRect(x: x, y: 0, width: width, height: height)
        } else {
            let width = thumbnailSize.width
            let height = width / aspectRatio
            let y = (thumbnailSize.height - height) / 2
            targetRect = NSRect(x: 0, y: y, width: width, height: height)
        }
        
        image.draw(in: targetRect)
        thumbnail.unlockFocus()
        
        return thumbnail
    }

    // MARK: - Selection Methods

    func toggleSelection(_ resultId: String, at index: Int, withCommandKey: Bool, withShiftKey: Bool) {
        if withShiftKey, let lastIndex = lastSelectedIndex {
            // Shift-click: select range
            let start = min(lastIndex, index)
            let end = max(lastIndex, index)
            for i in start...end where i < searchResults.count {
                selectedResultIds.insert(searchResults[i].id)
            }
        } else if withCommandKey {
            // Command-click: toggle individual
            if selectedResultIds.contains(resultId) {
                selectedResultIds.remove(resultId)
            } else {
                selectedResultIds.insert(resultId)
                lastSelectedIndex = index
            }
        } else {
            // Normal click: select only this one
            selectedResultIds = [resultId]
            lastSelectedIndex = index
        }
    }

    func clearSelection() {
        selectedResultIds.removeAll()
        lastSelectedIndex = nil
    }

    func selectAll() {
        selectedResultIds = Set(searchResults.map { $0.id })
    }

    // MARK: - Image Manipulation

    private func flipImageHorizontally(_ image: NSImage) -> NSImage? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("[VM DEBUG] Failed to get CGImage from NSImage")
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height
        let flippedSize = NSSize(width: width, height: height)

        // Create flipped image
        let flippedImage = NSImage(size: flippedSize)
        flippedImage.lockFocus()

        // Apply horizontal flip transform
        if let context = NSGraphicsContext.current?.cgContext {
            context.translateBy(x: CGFloat(width), y: 0)
            context.scaleBy(x: -1.0, y: 1.0)
        }

        // Draw the original image with the transform applied
        let rect = NSRect(origin: .zero, size: flippedSize)
        NSImage(cgImage: cgImage, size: flippedSize).draw(in: rect)

        flippedImage.unlockFocus()

        print("[VM DEBUG] Image flipped horizontally: \(width)x\(height)")
        return flippedImage
    }

    func moveSelectedFiles(to destinationDirectory: String) -> (success: Int, failed: Int) {
        var successCount = 0
        var failedCount = 0
        let fileManager = FileManager.default

        let selectedResults = searchResults.filter { selectedResultIds.contains($0.id) }

        for result in selectedResults {
            guard let sourcePath = result.imagePath else {
                failedCount += 1
                continue
            }

            let sourceURL = URL(fileURLWithPath: sourcePath)
            let filename = sourceURL.lastPathComponent
            let destinationURL = URL(fileURLWithPath: destinationDirectory).appendingPathComponent(filename)

            do {
                // Check if destination exists and create unique name if needed
                var finalDestinationURL = destinationURL
                var counter = 1
                while fileManager.fileExists(atPath: finalDestinationURL.path) {
                    let nameWithoutExt = sourceURL.deletingPathExtension().lastPathComponent
                    let ext = sourceURL.pathExtension
                    let newName = "\(nameWithoutExt)_\(counter).\(ext)"
                    finalDestinationURL = URL(fileURLWithPath: destinationDirectory).appendingPathComponent(newName)
                    counter += 1
                }

                try fileManager.moveItem(at: sourceURL, to: finalDestinationURL)
                successCount += 1
            } catch {
                print("Failed to move \(sourcePath): \(error)")
                failedCount += 1
            }
        }

        // Clear selection after move
        if successCount > 0 {
            DispatchQueue.main.async {
                self.clearSelection()
            }
        }

        return (successCount, failedCount)
    }

    // MARK: - Keypoint Normalization Helper
    /// Normalize keypoints from original image coordinates to display size using center-crop
    /// Matches thumbnail_generator.py center-crop logic
    func normalizeKeypoints(keypoints: [[Double]], fromSize: CGSize, toSize: CGSize) -> [[Double]] {
        // Use FIT-TO-DISPLAY logic (same as bbox rendering)
        // This matches SwiftUI's .scaledToFit() behavior

        let scaleX = toSize.width / fromSize.width
        let scaleY = toSize.height / fromSize.height
        let scale = min(scaleX, scaleY)  // Fit to display (not crop)

        // Calculate centered display area
        let displayWidth = fromSize.width * scale
        let displayHeight = fromSize.height * scale
        let xOffset = (toSize.width - displayWidth) / 2
        let yOffset = (toSize.height - displayHeight) / 2

        // Normalize keypoints
        return keypoints.map { kp in
            guard kp.count >= 3 else { return kp }
            let x = kp[0]
            let y = kp[1]
            let conf = kp[2]

            // Apply scale and ADD offset (same as bbox rendering)
            let xNorm = x * scale + xOffset
            let yNorm = y * scale + yOffset

            return [xNorm, yNorm, conf]
        }
    }
}

// MARK: - View Mode
enum ViewMode {
    case grid
    case list
}

// MARK: - Match Tier (Progressive Disclosure)
enum MatchTier {
    case premium    // >90% similarity - highest confidence matches
    case good       // 70-90% similarity - standard matches
    case possible   // <70% similarity - uncertain matches

    init(similarity: Int) {
        if similarity > 90 {
            self = .premium
        } else if similarity >= 70 {
            self = .good
        } else {
            self = .possible
        }
    }

    var opacity: Double {
        switch self {
        case .premium: return 1.0
        case .good: return 1.0
        case .possible: return 0.85
        }
    }

    var skeletonDetail: SkeletonDetail {
        switch self {
        case .premium: return .full
        case .good: return .standard
        case .possible: return .minimal
        }
    }
}

// MARK: - Skeleton Detail Level
enum SkeletonDetail {
    case full       // All 133 keypoints (body + hands + feet + face)
    case standard   // Body + major limbs (17 keypoints)
    case minimal    // Body only (torso + legs, no arms)
}

// MARK: - Index Status
enum IndexStatus {
    case loading
    case ready
    case building
    case error

    var description: String {
        switch self {
        case .loading: return "Loading index..."
        case .ready: return "Index ready"
        case .building: return "Building index..."
        case .error: return "Index error"
        }
    }

    var color: Color {
        switch self {
        case .loading: return .orange
        case .ready: return .green
        case .building: return .blue
        case .error: return .red
        }
    }
}

// MARK: - Search Result Model
struct SearchResult: Identifiable {
    let id: String
    let similarity: Int
    let filename: String
    let confidence: Double
    var imagePath: String?
    var detectedAt: Date?
    var imageWidth: Int?
    var imageHeight: Int?
    var fileSize: Int?
    var thumbnailData: Data?  // Pre-loaded thumbnail from database
    var visibleRegions: [String]?  // Canonical regions (7 categories: face, feet, torso, etc.)
    var visibleRegionsDetailed: [[String: Any]]?  // Full NudeNet classes (18 categories with confidence)
    var keypoints: [[Double]]?  // 133 keypoints × 3 [x, y, confidence] normalized to 200x200
    var bbox: [Double]?  // Bounding box [x_min, y_min, x_max, y_max]
    var personId: Int?  // Person index (0, 1, 2...) for multi-person images
    var isFlipped: Bool = false  // True if this is a horizontally flipped match
}
