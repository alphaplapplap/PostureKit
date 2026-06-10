import SwiftUI
import Combine

extension NSImage {
    // True pixel dimensions of the bitmap that Python will see.
    // `.size` reports points (DPI-scaled); `.representations.first.pixelsWide` can
    // also report points for some metadata configurations. Materializing through
    // `tiffRepresentation` + `NSBitmapImageRep` is the same path `saveImageToTemp`
    // uses to encode the PNG for Python, so the numbers always agree.
    var pixelSize: CGSize {
        if let tiff = self.tiffRepresentation,
           let rep = NSBitmapImageRep(data: tiff) {
            return CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
        }
        if let rep = representations.first {
            return CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
        }
        return size
    }
}

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
    @Published var numberOfResults: Int = 20  // Page size for the results grid (NOT a result cap)
    @Published var numberOfResultsDouble: Double = 20.0
    @Published var showAllResults: Bool = false  // When true, show the entire result set on one page (no paging)

    // Pagination over the cached result set (the result SET is bounded by minSimilarity, not these)
    @Published var currentPage: Int = 1        // 1-based index of the displayed page
    @Published var totalPages: Int = 1         // number of pages at the current page size
    @Published var totalResultCount: Int = 0   // results at/above the current threshold (for "N–M of T")
    // Similarity floor — THE result-set cap. Python returns every pose ≥ this; Swift paginates.
    // Default 0.5 keeps the first search fast/complete; 0.0 means "everything" (full-index scan).
    @Published var minSimilarity: Double = 0.5
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
    @Published var minFeatureConfidence: Double = 0.0  // Filter rebuild: bare minimum
    @Published var minValidOverlap: Double = 0.0  // Filter rebuild: bare minimum (0/52)
    @Published var showMultiplePeoplePerImage: Bool = false  // Show all people from multi-person images (default: deduplicate for cleaner results)
    @Published var minRegionConfidence: Double = 0.0  // Filter rebuild: bare minimum

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
    private var lastSearchFlip: Bool = false  // Flip setting of the last search (part of the dup-search guard)
    private var searchDebounceTimer: DispatchWorkItem? = nil

    // Monotonic search generation. Bumped for every new search/browse and on cancel; the
    // result handlers only apply results whose generation is still current, so a cancelled
    // or superseded search (the Python call cannot be interrupted and runs to completion)
    // can no longer overwrite the displayed grid. Reassigning searchResults rebuilds the
    // ForEach and resets scroll position, so discarding stale results also keeps the grid
    // scrollable. Lock-guarded because it is touched from the main thread and background
    // threads, and the serial searchQueue is itself blocked during the synchronous Python call.
    private let searchGenerationLock = NSLock()
    private var searchGeneration: Int = 0

    // Full result set from the last Python query (already ≥ lastQueriedThreshold, sorted best-first).
    // Pagination and threshold RAISES re-slice/re-filter this cache on the main thread with no new
    // Python round-trip; lowering the threshold below it re-queries.
    private var rawResults: [SearchResult] = []
    private var lastQueriedThreshold: Double = 0.0

    init() {
        // Listen for index preload notification
        NotificationCenter.default.publisher(for: .indexPreloaded)
            .sink { [weak self] _ in
                self?.handleIndexPreloaded()
            }
            .store(in: &cancellables)

        // "Show:" page size / "All" toggle → re-paginate the cached set instantly (NO re-search).
        // The result SET is fixed by the threshold; these only change how it's chunked for display.
        Publishers.CombineLatest($numberOfResults, $showAllResults)
            .dropFirst()
            .removeDuplicates { $0.0 == $1.0 && $0.1 == $1.1 }
            .debounce(for: .milliseconds(150), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handlePageSizeChange()
            }
            .store(in: &cancellables)

        // Threshold slider → it defines the result set. Raising it re-filters the cache instantly;
        // lowering it below what we fetched re-queries Python. Debounced so dragging the slider
        // doesn't fire a query per tick.
        $minSimilarity
            .dropFirst()
            .removeDuplicates()
            .debounce(for: .milliseconds(400), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.handleThresholdChange()
            }
            .store(in: &cancellables)
    }

    private func handleIndexPreloaded() {
        print("[VM] Index preloaded, refreshing statistics...")
        indexStatus = .ready
        loadIndexStatistics()
    }

    /// Slice the cached result set (`rawResults`, filtered to the current threshold) into the
    /// page the grid displays, and publish page bookkeeping. Pure main-thread work — no Python.
    func applyResultWindow() {
        // Raises above the fetched floor are honored by re-filtering the cache here.
        let floorPercent = Int(minSimilarity * 100)
        let active = rawResults.filter { $0.similarity >= floorPercent }

        totalResultCount = active.count

        if showAllResults {
            totalPages = 1
            currentPage = 1
            searchResults = active
            return
        }

        let pageSize = max(1, numberOfResults)
        totalPages = max(1, Int(ceil(Double(active.count) / Double(pageSize))))
        if currentPage > totalPages { currentPage = totalPages }
        if currentPage < 1 { currentPage = 1 }

        let start = (currentPage - 1) * pageSize
        let end = min(active.count, start + pageSize)
        searchResults = (start < end) ? Array(active[start..<end]) : []
    }

    /// "Show:" page-size or "All" toggle changed: jump to page 1 and re-slice the cache. No re-search.
    private func handlePageSizeChange() {
        guard !rawResults.isEmpty else { return }
        currentPage = 1
        applyResultWindow()
    }

    /// Threshold slider changed. Raising it (or staying at/above the fetched floor) just re-filters
    /// the cache instantly; lowering it below the fetched floor needs a fresh query so the newly
    /// admitted (lower-similarity) results actually exist locally. No-op until a search has run.
    private func handleThresholdChange() {
        // Instant re-filter is only valid when the cache is COMPLETE down to the fetched floor:
        // that holds only if the last query ran in threshold mode (lastQueriedThreshold > 0) AND
        // the new floor isn't lower. In no-threshold mode the cache is just a top-N slice (not the
        // full ≥0 set), so raising the slider must re-query to get the complete above-threshold set.
        if lastQueriedThreshold > 0 && minSimilarity >= lastQueriedThreshold {
            guard !rawResults.isEmpty else { return }
            currentPage = 1
            applyResultWindow()
            return
        }
        // Lowering below the fetched floor (or leaving no-threshold mode): re-run at the new floor.
        rerunLastSearch()
    }

    /// Page navigation (Swift-side; never re-queries Python).
    func goToNextPage() {
        guard currentPage < totalPages else { return }
        currentPage += 1
        applyResultWindow()
    }

    func goToPreviousPage() {
        guard currentPage > 1 else { return }
        currentPage -= 1
        applyResultWindow()
    }

    /// Re-run the most recent search/browse (used when the threshold drops below the fetched set).
    /// No-op until a search has actually been performed: on image load `extractedFeatures` is
    /// cleared and auto-detection does not set it, so this only fires after the user has searched.
    private func rerunLastSearch() {
        if browseMode {
            guard !requiredBodyParts.isEmpty else { return }
            performBrowse()
        } else {
            guard let features = extractedFeatures else { return }
            // Clear the duplicate-search guard so the identical pose re-runs at the new threshold.
            searchQueue.async { [weak self] in self?.lastSearchFeatures = nil }
            // executeSearch() calls searchQueue.sync; run it off-main so an in-flight search
            // can't block the main thread and freeze the UI.
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.executeSearch(features: features)
            }
        }
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
        rawResults = []
        lastQueriedThreshold = 0.0
        currentPage = 1
        totalPages = 1
        totalResultCount = 0
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

    /// Begin a new search generation and return its id. Results tagged with an older id are stale.
    @discardableResult
    private func bumpSearchGeneration() -> Int {
        searchGenerationLock.lock()
        defer { searchGenerationLock.unlock() }
        searchGeneration += 1
        return searchGeneration
    }

    /// True if `generation` is still the most recent search (i.e. its results should be applied).
    private func isCurrentSearchGeneration(_ generation: Int) -> Bool {
        searchGenerationLock.lock()
        defer { searchGenerationLock.unlock() }
        return generation == searchGeneration
    }

    func cancelSearch() {
        print("[VM] User cancelled search")
        // Invalidate any in-flight search so its late results are discarded when the
        // (uninterruptible) Python call finally returns, instead of overwriting the grid.
        bumpSearchGeneration()
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
                self.queryImageSize = image.pixelSize

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
                    self.queryImageSize = image.pixelSize

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

            // Flip handling now lives server-side: "Include flipped poses" drives engine flip
            // search (both orientations searched and merged) instead of mirroring the query
            // image before detection, so detection always runs on the original image.
            let imageToDetect = queryImg

            print("[SEARCH DEBUG] Detecting pose on original image")

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
                self.queryImageSize = imageToDetect.pixelSize
                self.isDetecting = false
            }

            // Now perform search with extracted features. Pass the pose directly rather
            // than relying on self.detectedPose, whose main-async write may not have
            // landed yet.
            self.executeSearch(features: features, selectedPersonPose: pose)
        }
    }

    private func executeSearch(features: GeometricFeatures, selectedPersonPose: PoseDetectionResult? = nil) {
        // Check if identical search is already in progress.
        // NOTE: a `return` inside a searchQueue.sync closure only exits the closure, not
        // executeSearch — so record the decision in `shouldProceed` and bail out below,
        // otherwise duplicate searches launch anyway and stack up.
        var shouldProceed = false
        searchQueue.sync {
            if searchInProgress {
                print("[SEARCH DEBUG] Search already in progress, skipping duplicate")
                return
            }

            // Check if this is the same feature vector AND the same flip setting (prevents
            // duplicate search on same pose, while letting a flip-toggle change re-search)
            if let lastFeatures = lastSearchFeatures, lastFeatures == features.featureVector,
               lastSearchFlip == includeFlippedPoses {
                print("[SEARCH DEBUG] Duplicate search request for same features, skipping")
                return
            }

            searchInProgress = true
            lastSearchFeatures = features.featureVector
            lastSearchFlip = includeFlippedPoses
            shouldProceed = true
        }
        guard shouldProceed else { return }

        // Tag this search so late results from a cancelled or superseded search are discarded
        // instead of overwriting the grid (the Python call cannot be interrupted).
        let myGeneration = bumpSearchGeneration()

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

            // The threshold (minSimilarity) — not "Show:" — defines the result SET now: Python
            // returns EVERY pose at/above it, and Swift paginates that set. "Show:" is only the
            // page size. Request the whole index so the threshold (not k) bounds the result set;
            // this keeps the slider monotonic (lower threshold ⇒ more results, always). 0% means
            // "everything" (heavy but honest); the non-zero default threshold keeps the common
            // case fast. Capture the threshold so raises can re-filter the cache without re-query.
            let queryThreshold = self.minSimilarity
            let requestCount = max(self.totalPosesIndexed, 500)

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
            print("  - k (requested): \(requestCount) (threshold bounds the set; k is just a ceiling)")
            print("  - minSimilarity: \(queryThreshold) (applied in Python — returns ALL at/above it)")
            print("  - minFeatureConfidence: \(self.minFeatureConfidence)")
            print("  - minValidOverlap: \(Int(self.minValidOverlap))")

            // Query pose geometry (keypoints + bbox) enables OKS re-ranking and flip search
            // server-side. Browse/stored-vector searches have no pose — the engine then falls
            // back to plain L2 ranking automatically.
            let queryPose = selectedPersonPose ?? self.detectedPose

            // minConfidence=0.0 passed to Python so pose-detection-confidence doesn't cull candidates.
            // minSimilarity is applied IN Python: it returns every pose at/above the threshold.
            let results = self.pythonBridge.searchSimilar(
                featureVector: searchVector,
                k: requestCount,
                minConfidence: 0.0,
                featureConfidence: searchConfidence,
                minFeatureConfidence: self.minFeatureConfidence,
                minValidOverlap: Int(self.minValidOverlap),
                requiredRegions: bodyPartsArray,
                deduplicateImages: !self.showMultiplePeoplePerImage,  // Inverted: true to show multiple = false to deduplicate
                minRegionConfidence: self.minRegionConfidence,
                minSimilarity: queryThreshold,
                includeFlippedPoses: self.includeFlippedPoses,
                queryKeypoints: queryPose?.keypoints,
                queryBbox: queryPose?.bbox
            )

            let elapsed = Date().timeIntervalSince(startTime)

            // Mark search as complete before processing results
            // Already on searchQueue (serial), direct assignment is thread-safe
            self.searchInProgress = false

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }

                // Discard results if this search was cancelled or superseded by a newer one.
                // Applying them would overwrite the grid and reset the user's scroll position.
                guard self.isCurrentSearchGeneration(myGeneration) else {
                    print("[SEARCH DEBUG] Ignoring \(results.count) results from a cancelled/superseded search")
                    return
                }

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

                // Python already applied the similarity threshold; just drop any whose image
                // file vanished from disk, then cache the full set for pagination.
                let existingFiles = results.filter { result in
                    guard let path = result.imagePath else { return false }
                    return FileManager.default.fileExists(atPath: path)
                }
                print("[SEARCH DEBUG] \(existingFiles.count) results at/above \(Int(queryThreshold * 100))% (of \(results.count) returned)")

                // Cache as the paginated set. queryThreshold is the floor Python used, so the
                // user can RAISE the slider and have applyResultWindow() re-filter instantly;
                // LOWERING below it re-queries (handleThresholdChange).
                self.rawResults = existingFiles
                self.lastQueriedThreshold = queryThreshold
                self.currentPage = 1
                self.applyResultWindow()
                self.searchTime = elapsed
                self.isSearching = false
                self.stopSearchTimer()  // Stop elapsed time tracking

                // Update error message for UI display
                if self.searchResults.isEmpty {
                    if results.isEmpty {
                        self.errorMessage = "No poses found at/above \(Int(queryThreshold * 100))% similarity. Try lowering the threshold."
                    } else {
                        self.errorMessage = "No matching files on disk."
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

        // Tag this browse so late results from a cancelled or superseded browse are discarded.
        let myGeneration = bumpSearchGeneration()

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

            // Fetch the full set; "Show:" is the page size (applied Swift-side), not a fetch cap.
            let browseK = max(self.totalPosesIndexed, 500)
            let results = self.pythonBridge.browseByBodyParts(
                requiredRegions: Array(self.requiredBodyParts),
                categoryThresholds: activeThresholds,
                k: browseK,
                sortBy: "confidence"
            )

            let elapsed = Date().timeIntervalSince(startTime)

            DispatchQueue.main.async {
                print("[BROWSE DEBUG] Received \(results.count) results in \(elapsed)s")

                // Discard if this browse was cancelled or superseded by a newer search/browse.
                guard self.isCurrentSearchGeneration(myGeneration) else {
                    print("[BROWSE DEBUG] Ignoring \(results.count) results from a cancelled/superseded browse")
                    return
                }

                // Browse has no similarity threshold (sorted by detection confidence), so the
                // whole returned set is the page-able set.
                self.rawResults = results
                self.lastQueriedThreshold = 0.0
                self.currentPage = 1
                self.applyResultWindow()
                self.searchTime = elapsed
                self.isSearching = false
                self.stopSearchTimer()

                if self.searchResults.isEmpty {
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

    func moveSelectedFiles(to destinationDirectory: String) -> (success: Int, failed: Int, skipped: Int) {
        var successCount = 0
        var failedCount = 0
        var skippedCount = 0
        var movedIds: Set<String> = []
        let fileManager = FileManager.default

        let selectedResults = searchResults.filter { selectedResultIds.contains($0.id) }

        // Normalize destination once so we can compare parent folders reliably.
        let destFolderURL = URL(fileURLWithPath: destinationDirectory).standardizedFileURL

        for result in selectedResults {
            guard let sourcePath = result.imagePath else {
                failedCount += 1
                continue
            }

            let sourceURL = URL(fileURLWithPath: sourcePath)
            let sourceFolderURL = sourceURL.deletingLastPathComponent().standardizedFileURL

            // No-op if the file is already in the chosen destination folder.
            if sourceFolderURL.path == destFolderURL.path {
                skippedCount += 1
                continue
            }

            let filename = sourceURL.lastPathComponent
            let destinationURL = destFolderURL.appendingPathComponent(filename)

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
                movedIds.insert(result.id)
            } catch {
                print("Failed to move \(sourcePath): \(error)")
                failedCount += 1
            }
        }

        // Prune moved results from the grid so they don't reappear for repeat moves,
        // and clear selection. DB still references old paths but the fileExists filter
        // on the next search will drop them.
        if !movedIds.isEmpty {
            DispatchQueue.main.async {
                // Drop from the cache too, then re-paginate so page counts stay correct.
                self.rawResults.removeAll { movedIds.contains($0.id) }
                self.applyResultWindow()
                self.clearSelection()
            }
        }

        return (successCount, failedCount, skippedCount)
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
