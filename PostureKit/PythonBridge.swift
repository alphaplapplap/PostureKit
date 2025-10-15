import Foundation
import AppKit
import PythonKit

// MARK: - Python Bridge
// Direct interface to your existing Python modules
class PythonBridge {
    static let shared = PythonBridge()
    
    private let sys: PythonObject
    private let numpy: PythonObject
    private let cv2: PythonObject
    
    // Your existing Python modules
    private let bridge: PythonObject
    private let poseDetector: PythonObject
    private let featureExtractor: PythonObject
    private let similarityEngine: PythonObject
    private let storageManager: PythonObject
    
    private init() {
        // Set DB_PROFILE from UserDefaults (defaults to 'irl' if not set)
        let activeProfile = UserDefaults.standard.string(forKey: "activeProfile") ?? "irl"
        setenv("DB_PROFILE", activeProfile, 1)
        print("Setting DB_PROFILE environment variable to: \(activeProfile)")

        // Set up Python environment - must set these before any Python calls
        let pythonLibPath = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/libpython3.9.dylib"
        setenv("PYTHON_LIBRARY", pythonLibPath, 1)

        // Set PYTHONHOME to the base Python installation (not venv)
        let pythonHome = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9"
        setenv("PYTHONHOME", pythonHome, 1)

        // Set PYTHONPATH to include venv site-packages and project src BEFORE initializing Python
        let venvSitePackages = "/Users/linuxbabe/Hardware-Aware/PostureKit/venv/lib/python3.9/site-packages"
        let projectPath = "/Users/linuxbabe/Hardware-Aware/PostureKit/src"
        let pythonPath = "\(venvSitePackages):\(projectPath)"
        setenv("PYTHONPATH", pythonPath, 1)

        // Initialize Python
        self.sys = Python.import("sys")

        // Debug: Print sys.path
        print("Python sys.path:")
        for path in Array(sys.path) {
            print("  \(path)")
        }

        // Try to import numpy with detailed error handling
        print("Attempting to import numpy...")

        // First check if numpy directory exists
        let os = Python.import("os")
        let numpyPath = venvSitePackages + "/numpy"
        let numpyExists = Bool(os.path.exists(numpyPath))!
        print("Numpy directory exists at \(numpyPath): \(numpyExists)")

        if numpyExists {
            // List contents
            let contents = Array(os.listdir(numpyPath))
            print("Numpy directory contents (first 10):")
            for (i, item) in contents.prefix(10).enumerated() {
                print("  \(i): \(item)")
            }
        }

        // Try using importlib
        do {
            let importlib = try Python.attemptImport("importlib")
            self.numpy = importlib.import_module("numpy")
            print("Successfully imported numpy via importlib")
        } catch let error {
            print("Failed to import numpy via importlib: \(error)")
            // Get the actual Python traceback
            do {
                let traceback = try Python.attemptImport("traceback")
                traceback.print_exc()
            } catch {
                print("Could not import traceback module")
            }
            fatalError("Cannot import numpy - please check Python environment")
        }

        print("Attempting to import cv2...")
        do {
            self.cv2 = try Python.attemptImport("cv2")
            print("Successfully imported cv2")
        } catch {
            print("Failed to import cv2: \(error)")
            fatalError("Cannot import cv2 - please install opencv-python")
        }
        
        // Import the Swift bridge module
        let bridgeModule = Python.import("swift_bridge")

        // Initialize the bridge (uses PostgreSQL from settings)
        self.bridge = bridgeModule.PostureKitBridge()
        
        // Keep references for backward compatibility
        self.poseDetector = self.bridge.pose_detector
        self.featureExtractor = self.bridge.feature_extractor
        self.storageManager = self.bridge.storage_manager
        self.similarityEngine = self.bridge.similarity_engine
        
        print("Python modules loaded successfully")
    }
    
    // MARK: - Pose Detection
    func detectPose(in image: NSImage) -> PoseDetectionResult? {
        guard let imageData = imageToNumpy(image) else { return nil }

        // Use the bridge for pose detection
        let result = bridge.detect_pose(imageData)

        // Check if result is None
        if Python.isinstance(result, Python.None) == true {
            return nil
        }

        // Extract data from Python dictionary
        let keypoints = Array(result["keypoints"])
        let visibility = Array(result["visibility"])
        let bbox = Array(result["bbox"])
        let confidence = Double(result["confidence"]) ?? 0.0
        let personIndex = Int(result["person_id"]) ?? 0

        // Convert nested arrays
        let keypointsSwift = keypoints.map { kp -> [Double] in
            Array(kp).map { Double($0) ?? 0.0 }
        }
        let visibilitySwift = visibility.map { Int($0) ?? 0 }
        let bboxSwift = bbox.map { Double($0) ?? 0.0 }

        return PoseDetectionResult(
            keypoints: keypointsSwift,
            visibility: visibilitySwift,
            bbox: bboxSwift,
            confidence: confidence,
            personIndex: personIndex
        )
    }
    
    // MARK: - Feature Extraction
    func extractFeatures(from poseResult: PoseDetectionResult) -> GeometricFeatures? {
        // Convert Swift struct to Python dictionary using PythonKit
        let pyDict = Python.dict()
        pyDict["keypoints"] = PythonObject(poseResult.keypoints)
        pyDict["visibility"] = PythonObject(poseResult.visibility)
        pyDict["bbox"] = PythonObject(poseResult.bbox)
        pyDict["confidence"] = PythonObject(poseResult.confidence)
        pyDict["person_id"] = PythonObject(poseResult.personIndex)

        // Use the bridge for feature extraction
        let result = bridge.extract_features(pyDict)

        // Check if result is None
        if Python.isinstance(result, Python.None) == true {
            return nil
        }

        // Extract feature vector
        let featureVectorPy = Array(result["feature_vector"])
        let featureVector = featureVectorPy.map { Float($0) ?? 0.0 }

        // Extract occlusion pattern
        let occlusionPatternPy = Array(result["occlusion_pattern"])
        let occlusionPattern = occlusionPatternPy.map { Float($0) ?? 0.0 }

        // Extract dictionaries
        let jointAngles = pythonDictToSwift(result["joint_angles"])
        let limbRatios = pythonDictToSwift(result["limb_ratios"])
        let bodyAngles = pythonDictToSwift(result["body_angles"])
        let symmetryScores = pythonDictToSwift(result["symmetry_scores"])

        return GeometricFeatures(
            featureVector: featureVector,
            featureConfidence: nil,  // PythonKit bridge doesn't support confidence (legacy)
            fusedVector: nil,  // PythonKit bridge doesn't support fused features (legacy)
            jointAngles: jointAngles,
            limbRatios: limbRatios,
            bodyAngles: bodyAngles,
            symmetryScores: symmetryScores,
            occlusionPattern: occlusionPattern
        )
    }
    
    // MARK: - Similarity Search
    func searchSimilar(
        featureVector: [Float],
        k: Int = 20,
        minConfidence: Double = 0.5
    ) -> [SearchResult] {
        // Use the bridge for similarity search
        let results = bridge.search_similar(
            feature_vector: PythonObject(featureVector),
            k: k,
            min_confidence: minConfidence
        )

        var searchResults: [SearchResult] = []

        for result in Array(results) {
            let poseId = String(result["pose_id"]) ?? ""
            let similarityScore = Double(result["similarity_score"]) ?? 0.0
            let imagePath = String(result["image_path"]) ?? ""
            let detectionConfidence = Double(result["detection_confidence"]) ?? 0.0

            searchResults.append(SearchResult(
                id: poseId,
                similarity: Int(similarityScore * 100),
                filename: extractFilename(from: imagePath),
                confidence: detectionConfidence,
                imagePath: imagePath
            ))
        }

        return searchResults
    }
    
    // MARK: - Directory Indexing
    func startIndexing(
        directory: String,
        recursive: Bool = true,
        minConfidence: Double = 0.3,
        progressCallback: @escaping (IndexProgress) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Use the bridge for directory indexing
            let result = self.bridge.index_directory(
                directory_path: directory,
                recursive: recursive,
                min_confidence: minConfidence
            )

            let totalImages = Int(result["total_images"]) ?? 0
            let processedImages = Int(result["processed_images"]) ?? 0
            let posesIndexed = Int(result["poses_indexed"]) ?? 0
            let failedImages = Int(result["failed_images"]) ?? 0
            let success = Bool(result["success"]) ?? false

            if success {
                // Final progress update
                DispatchQueue.main.async {
                    progressCallback(IndexProgress(
                        currentFile: "",
                        imagesProcessed: processedImages,
                        totalImages: totalImages,
                        posesIndexed: posesIndexed,
                        failedImages: failedImages,
                        skippedImages: 0,
                        progress: 1.0
                    ))
                }
            } else {
                let errorMsg = String(result["error"]) ?? "Unknown error"
                print("Indexing failed: \(errorMsg)")
            }
        }
    }
    
    // MARK: - Helper Functions
    private func imageToNumpy(_ image: NSImage) -> PythonObject? {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let rgbData = bitmap.representation(using: .png, properties: [:]) else {
            return nil
        }
        
        // Save to temp file and load with OpenCV
        let tempPath = NSTemporaryDirectory() + "temp_image.png"
        try? rgbData.write(to: URL(fileURLWithPath: tempPath))
        
        let cvImage = cv2.imread(tempPath)
        let rgbImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
        
        try? FileManager.default.removeItem(atPath: tempPath)
        
        return rgbImage
    }
    
    private func numpyToArray<T>(_ npArray: PythonObject) -> [T] where T: ExpressibleByIntegerLiteral {
        let list = Array(npArray.tolist())
        return list.compactMap { obj -> T? in
            if let intVal = Int(obj) {
                return intVal as? T
            }
            return nil
        }
    }
    
    private func pythonDictToSwift(_ pyDict: PythonObject) -> [String: Double] {
        var result: [String: Double] = [:]

        // Get Python dict items
        if let items = PyArray(pyDict.items()) {
            for item in items {
                let keyValue = Array(item)
                if keyValue.count == 2 {
                    if let key = String(keyValue[0]),
                       let value = Double(keyValue[1]) {
                        result[key] = value
                    }
                }
            }
        }

        return result
    }

    private func PyArray(_ pyObj: PythonObject) -> [PythonObject]? {
        guard Python.isinstance(pyObj, Python.None) == false else {
            return nil
        }
        return Array(pyObj)
    }
    
    private func createPythonPoseObject(from result: PoseDetectionResult) -> PythonObject {
        // Create a Python object that matches your PoseResult class
        let PoseResult = Python.import("core.pose_detector").PoseResult
        
        return PoseResult(
            keypoints: numpy.array(result.keypoints),
            visibility: numpy.array(result.visibility),
            bbox: numpy.array(result.bbox),
            overall_confidence: result.confidence,
            person_id: result.personIndex
        )
    }
    
    private func createImageMetadata(imagePath: String) -> PythonObject {
        let ImageMetadata = Python.import("core.image_ingestor").ImageMetadata
        let fileManager = FileManager.default
        
        guard let attrs = try? fileManager.attributesOfItem(atPath: imagePath),
              let fileSize = attrs[.size] as? Int,
              let modDate = attrs[.modificationDate] as? Date else {
            fatalError("Cannot read file attributes")
        }
        
        // Get image dimensions using NSImage
        guard let image = NSImage(contentsOfFile: imagePath) else {
            fatalError("Cannot load image")
        }
        
        return ImageMetadata(
            original_width: Int(image.size.width),
            original_height: Int(image.size.height),
            file_size_bytes: fileSize,
            file_mtime: modDate.timeIntervalSince1970
        )
    }
    
    private func extractFilename(from path: String) -> String {
        return URL(fileURLWithPath: path).lastPathComponent
    }
    
    // MARK: - Statistics
    func getStatistics() -> [String: Any] {
        let result = bridge.get_statistics()

        var stats: [String: Any] = [:]

        if let items = PyArray(result.items()) {
            for item in items {
                let keyValue = Array(item)
                if keyValue.count == 2,
                   let key = String(keyValue[0]) {
                    // Try to convert value to appropriate Swift type
                    if let intVal = Int(keyValue[1]) {
                        stats[key] = intVal
                    } else if let doubleVal = Double(keyValue[1]) {
                        stats[key] = doubleVal
                    } else if let stringVal = String(keyValue[1]) {
                        stats[key] = stringVal
                    }
                }
            }
        }

        return stats
    }
    
    // MARK: - Index Management
    func buildIndex(forceRebuild: Bool = false) -> Bool {
        let result = bridge.build_index(force_rebuild: forceRebuild)
        return Bool(result) ?? false
    }

    // MARK: - Profile Management
    func getCurrentProfile() -> String {
        let result = bridge.get_current_profile()
        return String(result) ?? "irl"
    }

    func getProfileStats(profile: String? = nil) -> ProfileStats {
        let result: PythonObject
        if let profile = profile {
            result = bridge.get_profile_stats(profile)
        } else {
            result = bridge.get_profile_stats()
        }

        let success = Bool(result["success"]) ?? false

        if !success {
            let error = String(result["error"]) ?? "Unknown error"
            return ProfileStats(
                profile: profile ?? "irl",
                totalImages: 0,
                totalPoses: 0,
                indexSize: 0,
                databaseName: "",
                indexPath: "",
                indexExists: false,
                error: error
            )
        }

        return ProfileStats(
            profile: String(result["profile"]) ?? "irl",
            totalImages: Int(result["total_images"]) ?? 0,
            totalPoses: Int(result["total_poses"]) ?? 0,
            indexSize: Int(result["index_size"]) ?? 0,
            databaseName: String(result["database_name"]) ?? "",
            indexPath: String(result["index_path"]) ?? "",
            indexExists: Bool(result["index_exists"]) ?? false,
            error: nil
        )
    }

    func switchProfile(_ newProfile: String) -> ProfileSwitchResult {
        let result = bridge.switch_profile(newProfile)

        let success = Bool(result["success"]) ?? false
        let oldProfile = String(result["old_profile"]) ?? "irl"
        let profile = String(result["new_profile"]) ?? newProfile

        if !success {
            let error = String(result["error"]) ?? "Unknown error"
            return ProfileSwitchResult(
                success: false,
                oldProfile: oldProfile,
                newProfile: profile,
                databaseName: nil,
                indexPath: nil,
                error: error
            )
        }

        // Update UserDefaults to persist selection
        UserDefaults.standard.set(newProfile, forKey: "activeProfile")

        return ProfileSwitchResult(
            success: true,
            oldProfile: oldProfile,
            newProfile: profile,
            databaseName: String(result["database_name"]),
            indexPath: String(result["index_path"]),
            error: nil
        )
    }
}

// MARK: - Data Models
struct PoseDetectionResult {
    let keypoints: [[Double]]
    let visibility: [Int]
    let bbox: [Double]
    let confidence: Double
    let personIndex: Int
}

struct GeometricFeatures {
    let featureVector: [Float]  // 52-dim geometric features
    let featureConfidence: [Float]?  // 52-dim confidence scores for features
    let fusedVector: [Float]?   // 628-dim fused features (geometric + visual) for search
    let jointAngles: [String: Double]
    let limbRatios: [String: Double]
    let bodyAngles: [String: Double]
    let symmetryScores: [String: Double]
    let occlusionPattern: [Float]  // Keypoint occlusion/visibility pattern
}

struct IndexProgress {
    let currentFile: String
    let imagesProcessed: Int
    let totalImages: Int
    let posesIndexed: Int
    let failedImages: Int
    let skippedImages: Int
    let progress: Double
}

struct ThumbnailProgress {
    let imagesProcessed: Int
    let totalImages: Int
    let thumbnailsGenerated: Int
    let failedImages: Int
    let currentFile: String
    let progress: Double
}

struct ProfileStats {
    let profile: String
    let totalImages: Int
    let totalPoses: Int
    let indexSize: Int
    let databaseName: String
    let indexPath: String
    let indexExists: Bool
    let error: String?
}

struct ProfileSwitchResult {
    let success: Bool
    let oldProfile: String
    let newProfile: String
    let databaseName: String?
    let indexPath: String?
    let error: String?
}
