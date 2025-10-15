import SwiftUI
import Combine

struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var settingsViewModel = SettingsViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Settings")
                    .font(.system(size: 18, weight: .semibold))
                
                Spacer()
                
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 14))
                        .foregroundColor(.gray)
                        .frame(width: 28, height: 28)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(6)
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(20)
            .background(Color(NSColor.windowBackgroundColor))
            .overlay(
                Rectangle()
                    .frame(height: 1)
                    .foregroundColor(Color.gray.opacity(0.2)),
                alignment: .bottom
            )
            
            // Body
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Storage Section
                    StorageSection(viewModel: settingsViewModel)

                    // Thumbnails Section
                    ThumbnailsSection(viewModel: settingsViewModel)

                    // Detection Section (NEW)
                    DetectionSection(viewModel: settingsViewModel)

                    // Performance Section
                    PerformanceSection(viewModel: settingsViewModel)

                    // Display Section
                    DisplaySection(viewModel: settingsViewModel)
                    
                    // Action Buttons
                    HStack(spacing: 12) {
                        Button(action: { 
                            settingsViewModel.saveSettings()
                            dismiss()
                        }) {
                            Text("Save")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(Color.blue)
                                .cornerRadius(8)
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        Button(action: { dismiss() }) {
                            Text("Close")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(.primary)
                                .frame(width: 100)
                                .padding(.vertical, 14)
                                .background(Color.gray.opacity(0.1))
                                .cornerRadius(8)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                .padding(24)
            }
        }
        .frame(width: 600, height: 600)
    }
}

// MARK: - Storage Section
struct StorageSection: View {
    @ObservedObject var viewModel: SettingsViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Storage Profile")
                .font(.system(size: 13, weight: .semibold))

            VStack(alignment: .leading, spacing: 12) {
                // Profile Picker
                VStack(alignment: .leading, spacing: 8) {
                    Text("Active Profile")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)

                    Picker("", selection: Binding(
                        get: { viewModel.activeProfile },
                        set: { newValue in
                            viewModel.switchProfile(newValue)
                        }
                    )) {
                        Text("IRL Photos").tag("irl")
                        Text("2D Art").tag("2d")
                        Text("3D Renders").tag("3d")
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )

                    Text("Switch between different storage databases for IRL photos, 2D artwork, or 3D renders")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Divider()

                // Current Profile Stats
                if viewModel.isLoadingProfileStats {
                    HStack {
                        ProgressIndicator()
                            .frame(width: 12, height: 12)
                        Text("Loading profile statistics...")
                            .font(.system(size: 13))
                            .foregroundColor(.gray)
                    }
                } else if let stats = viewModel.profileStats[viewModel.activeProfile] {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Current Profile")
                                .font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Text(profileDisplayName(stats.profile))
                                .font(.system(size: 13))
                                .foregroundColor(.blue)
                        }

                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Images")
                                    .font(.system(size: 11))
                                    .foregroundColor(.gray)
                                Text("\(stats.totalImages)")
                                    .font(.system(size: 15, weight: .semibold))
                            }

                            Spacer()

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Poses")
                                    .font(.system(size: 11))
                                    .foregroundColor(.gray)
                                Text("\(stats.totalPoses)")
                                    .font(.system(size: 15, weight: .semibold))
                            }

                            Spacer()

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Index")
                                    .font(.system(size: 11))
                                    .foregroundColor(.gray)
                                HStack(spacing: 4) {
                                    Circle()
                                        .fill(stats.indexExists ? Color.green : Color.orange)
                                        .frame(width: 6, height: 6)
                                    Text(stats.indexExists ? "Ready" : "Empty")
                                        .font(.system(size: 13))
                                }
                            }
                        }

                        Divider()
                            .padding(.vertical, 4)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Database")
                                .font(.system(size: 11))
                                .foregroundColor(.gray)
                            Text(stats.databaseName)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.gray)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Index Path")
                                .font(.system(size: 11))
                                .foregroundColor(.gray)
                            Text(stats.indexPath)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
        .padding(16)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }

    private func profileDisplayName(_ profile: String) -> String {
        switch profile {
        case "irl": return "IRL Photos"
        case "2d": return "2D Art"
        case "3d": return "3D Renders"
        default: return profile.uppercased()
        }
    }
}

struct StorageItem: View {
    let title: String
    let path: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
            Text(path)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.gray)
        }
    }
}

// MARK: - Thumbnails Section
struct ThumbnailsSection: View {
    @ObservedObject var viewModel: SettingsViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Thumbnails")
                .font(.system(size: 13, weight: .semibold))

            VStack(alignment: .leading, spacing: 12) {
                // Status info
                HStack {
                    if viewModel.isLoadingThumbnailCount {
                        ProgressIndicator()
                            .frame(width: 12, height: 12)
                        Text("Checking thumbnails...")
                            .font(.system(size: 13))
                            .foregroundColor(.gray)
                    } else {
                        Circle()
                            .fill(viewModel.totalImagesCount == 0 ? Color.gray : (viewModel.imagesMissingThumbnails > 0 ? Color.orange : Color.green))
                            .frame(width: 8, height: 8)

                        if viewModel.totalImagesCount == 0 {
                            Text("No images indexed yet")
                                .font(.system(size: 13))
                                .foregroundColor(.gray)
                        } else if viewModel.imagesMissingThumbnails > 0 {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(viewModel.imagesMissingThumbnails) of \(viewModel.totalImagesCount) images missing thumbnails")
                                    .font(.system(size: 13))
                                    .foregroundColor(.orange)
                                Text("Search results will load faster after generation")
                                    .font(.system(size: 11))
                                    .foregroundColor(.gray)
                            }
                        } else {
                            Text("All \(viewModel.totalImagesCount) images have thumbnails ✓")
                                .font(.system(size: 13))
                                .foregroundColor(.green)
                        }
                    }
                }

                // Progress indicator (shown during generation)
                if viewModel.isGeneratingThumbnails {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Generating thumbnails...")
                                .font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Text("\(Int(viewModel.thumbnailProgress * 100))%")
                                .font(.system(size: 12))
                                .foregroundColor(.gray)
                        }

                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.gray.opacity(0.2))
                                    .frame(height: 4)
                                    .cornerRadius(2)

                                Rectangle()
                                    .fill(Color.blue)
                                    .frame(width: geometry.size.width * viewModel.thumbnailProgress, height: 4)
                                    .cornerRadius(2)
                                    .animation(.linear(duration: 0.3), value: viewModel.thumbnailProgress)
                            }
                        }
                        .frame(height: 4)

                        HStack {
                            Text("Generated: \(viewModel.thumbnailsGenerated)")
                            Spacer()
                            if viewModel.thumbnailsFailed > 0 {
                                Text("Failed: \(viewModel.thumbnailsFailed)")
                                    .foregroundColor(.red)
                            }
                        }
                        .font(.system(size: 12))
                        .foregroundColor(.gray)
                    }
                }

                // Buttons
                HStack(spacing: 8) {
                    // Refresh count button
                    Button(action: {
                        viewModel.loadThumbnailStatistics()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.clockwise")
                            Text("Refresh")
                        }
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.primary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(6)
                    }
                    .buttonStyle(PlainButtonStyle())
                    .disabled(viewModel.isLoadingThumbnailCount || viewModel.isGeneratingThumbnails)

                    // Generate button
                    Button(action: {
                        viewModel.startThumbnailGeneration()
                    }) {
                        HStack(spacing: 6) {
                            Image(systemName: viewModel.isGeneratingThumbnails ? "stop.circle" : "photo.badge.plus")
                            Text(viewModel.isGeneratingThumbnails ? "Cancel" : "Generate Missing Thumbnails")
                        }
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(viewModel.isGeneratingThumbnails ? Color.red : (viewModel.imagesMissingThumbnails > 0 ? Color.blue : Color.gray))
                        .cornerRadius(6)
                    }
                    .buttonStyle(PlainButtonStyle())
                    .disabled((viewModel.imagesMissingThumbnails == 0 && !viewModel.isGeneratingThumbnails) || viewModel.isLoadingThumbnailCount)
                }
            }
        }
        .padding(16)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
        .onAppear {
            print("[THUMBNAILS UI] ThumbnailsSection appeared, loading statistics...")
            viewModel.loadThumbnailStatistics()
        }
    }
}

// MARK: - Detection Section
struct DetectionSection: View {
    @ObservedObject var viewModel: SettingsViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Pose Detection")
                .font(.system(size: 13, weight: .semibold))

            VStack(alignment: .leading, spacing: 12) {
                // Pose Model Selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Pose Model")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)

                    Picker("", selection: $viewModel.poseModel) {
                        Text("RTMW-L (Default - 220MB, Fast)").tag("rtmw-l")
                        Text("RTMW-X (353MB, Best Quality)").tag("rtmw-x")
                        Text("Ensemble: RTMW-L + RTMW-X (2.5x slower, 10-15% better)").tag("ensemble")
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )

                    // Model info
                    Text(viewModel.modelInfoText)
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Fusion Method (only shown for ensemble)
                if viewModel.poseModel == "ensemble" {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Fusion Method")
                            .font(.system(size: 13))
                            .foregroundColor(.gray)

                        Picker("", selection: $viewModel.fusionMethod) {
                            Text("Confidence-Weighted (Recommended)").tag("confidence_weighted")
                            Text("Weighted Average").tag("weighted_average")
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .padding(8)
                        .background(Color.gray.opacity(0.05))
                        .cornerRadius(6)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                        )

                        Text("How to combine predictions: Confidence-weighted gives higher weight to keypoints with higher confidence scores.")
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                // Two-stage detection toggle
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Toggle("Enable two-stage detection (YOLO + pose)", isOn: $viewModel.useTwoStage)
                            .font(.system(size: 13))

                        Spacer()

                        // Status indicator showing current active state
                        if let envValue = ProcessInfo.processInfo.environment["USE_TWO_STAGE_DETECTION"],
                           envValue.lowercased() == "true" {
                            Text("✓ Active")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.green)
                        } else {
                            Text("✗ Inactive")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(.red)
                        }
                    }

                    Text("Use person detection before pose estimation. 5-10% accuracy improvement, better multi-person handling.")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("⚠️ REQUIRES APP RESTART to take effect")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 2)

                    Text("⚠️ RE-INDEX EXISTING IMAGES to detect all people in multi-person photos")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 2)
                }
            }
        }
        .padding(16)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
}

// MARK: - Performance Section
struct PerformanceSection: View {
    @ObservedObject var viewModel: SettingsViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Performance")
                .font(.system(size: 13, weight: .semibold))
            
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("CPU Threads for Inference")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)

                    Picker("", selection: $viewModel.detectionThreads) {
                        Text("4 (Conservative)").tag(4)
                        Text("8 (Recommended - Performance Cores)").tag(8)
                        Text("12 (Aggressive - All Cores)").tag(12)
                        Text("16 (Over-provisioned)").tag(16)
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )

                    Text("M2 Pro has 8 performance + 4 efficiency cores. Using 8 threads provides best performance/efficiency balance.")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                        .fixedSize(horizontal: false, vertical: true)
                }
                
                Toggle("Enable GPU for interactive detection", isOn: $viewModel.useGPU)
                    .font(.system(size: 13))

                Text("ℹ️ GPU is ALWAYS enabled for batch operations (indexing) where it's 11× faster. This toggle only affects single-image interactive detection, where CPU is currently faster due to transfer overhead.")
                    .font(.system(size: 11))
                    .foregroundColor(.blue)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
}

// MARK: - Display Section
struct DisplaySection: View {
    @ObservedObject var viewModel: SettingsViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Display")
                .font(.system(size: 13, weight: .semibold))
            
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Theme")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    
                    Picker("", selection: $viewModel.theme) {
                        Text("Auto").tag("auto")
                        Text("Light").tag("light")
                        Text("Dark").tag("dark")
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Results per page")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    
                    Picker("", selection: $viewModel.resultsPerPage) {
                        Text("20").tag(20)
                        Text("50").tag(50)
                        Text("100").tag(100)
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(6)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
                }
            }
        }
        .padding(16)
        .background(Color.gray.opacity(0.05))
        .cornerRadius(8)
    }
}

// MARK: - Settings View Model
class SettingsViewModel: ObservableObject {
    // Profile settings
    @Published var activeProfile: String = "irl"
    @Published var profileStats: [String: ProfileStats] = [:]
    @Published var isLoadingProfileStats: Bool = false

    // Detection settings
    @Published var poseModel: String = "rtmw-l"
    @Published var fusionMethod: String = "confidence_weighted"
    @Published var useTwoStage: Bool = false

    // Performance settings
    @Published var detectionThreads: Int = 4
    @Published var useGPU: Bool = false  // Disable MPS by default (PyTorch 2.0.1 MPS is slower than CPU due to fallback overhead)

    // Display settings
    @Published var theme: String = "auto"
    @Published var resultsPerPage: Int = 20

    // Thumbnail generation state
    @Published var totalImagesCount: Int = 0
    @Published var imagesMissingThumbnails: Int = 0
    @Published var isLoadingThumbnailCount: Bool = false
    @Published var isGeneratingThumbnails: Bool = false
    @Published var thumbnailProgress: Double = 0.0
    @Published var thumbnailsGenerated: Int = 0
    @Published var thumbnailsFailed: Int = 0
    private let pythonBridge = PythonBridgeSubprocess.shared

    var modelInfoText: String {
        switch poseModel {
        case "rtmw-l":
            return "ℹ️ Fast, good quality. Recommended for most use cases. Detection time: ~0.6s"
        case "rtmw-x":
            return "ℹ️ Larger model, best for difficult poses. Detection time: ~0.9s (1.5x slower)"
        case "ensemble":
            return "ℹ️ Uses both models for maximum accuracy. Detection time: ~1.4s (2.5x slower). 10-15% accuracy improvement."
        default:
            return ""
        }
    }

    init() {
        loadSettings()
        loadAllProfileStats()
    }

    func saveSettings() {
        // Save profile settings
        UserDefaults.standard.set(activeProfile, forKey: "activeProfile")

        // Save detection settings
        UserDefaults.standard.set(poseModel, forKey: "poseModel")
        UserDefaults.standard.set(fusionMethod, forKey: "fusionMethod")
        UserDefaults.standard.set(useTwoStage, forKey: "useTwoStage")

        // Update environment variable for Python bridge (will be read on next bridge initialization)
        setenv("USE_TWO_STAGE_DETECTION", useTwoStage ? "true" : "false", 1)
        print("[SETTINGS] Updated USE_TWO_STAGE_DETECTION=\(useTwoStage ? "true" : "false")")

        // Save performance settings
        UserDefaults.standard.set(detectionThreads, forKey: "detectionThreads")
        UserDefaults.standard.set(useGPU, forKey: "useGPU")

        // Save display settings
        UserDefaults.standard.set(theme, forKey: "theme")
        UserDefaults.standard.set(resultsPerPage, forKey: "resultsPerPage")
    }

    func loadSettings() {
        // Load profile settings
        activeProfile = UserDefaults.standard.string(forKey: "activeProfile") ?? "irl"

        // Load detection settings
        poseModel = UserDefaults.standard.string(forKey: "poseModel") ?? "rtmw-l"
        fusionMethod = UserDefaults.standard.string(forKey: "fusionMethod") ?? "confidence_weighted"
        useTwoStage = UserDefaults.standard.bool(forKey: "useTwoStage")

        // Load performance settings
        let threads = UserDefaults.standard.integer(forKey: "detectionThreads")
        detectionThreads = threads > 0 ? threads : 8  // Default to 8 (M2 Pro performance cores)

        // GPU setting: Explicit check for first launch vs user choice
        // UserDefaults.register() in PostureKitApp.init() sets default to true,
        // but we add defensive check for edge cases
        if let gpuValue = UserDefaults.standard.object(forKey: "useGPU") as? Bool {
            useGPU = gpuValue  // Use stored user preference
        } else {
            useGPU = true  // Fallback: enable GPU by default on Apple Silicon
        }

        // Load display settings
        theme = UserDefaults.standard.string(forKey: "theme") ?? "auto"
        let results = UserDefaults.standard.integer(forKey: "resultsPerPage")
        resultsPerPage = results > 0 ? results : 20
    }

    func loadAllProfileStats() {
        guard !isLoadingProfileStats else { return }

        print("[SETTINGS] Loading stats for all profiles...")
        isLoadingProfileStats = true

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            let profiles = ["irl", "2d", "3d"]
            var stats: [String: ProfileStats] = [:]

            for profile in profiles {
                // Use subprocess-based stats to avoid race conditions with profile switching
                do {
                    let profileStats = try self.getProfileStatsViaSubprocess(profile: profile)
                    stats[profile] = profileStats
                    print("[SETTINGS] Loaded stats for \(profile): \(profileStats.totalImages) images, \(profileStats.totalPoses) poses")
                } catch {
                    print("[SETTINGS ERROR] Failed to load stats for \(profile): \(error)")
                    stats[profile] = ProfileStats(
                        profile: profile,
                        totalImages: 0,
                        totalPoses: 0,
                        indexSize: 0,
                        databaseName: "",
                        indexPath: "",
                        indexExists: false,
                        error: error.localizedDescription
                    )
                }
            }

            DispatchQueue.main.async {
                self.profileStats = stats
                self.isLoadingProfileStats = false
            }
        }
    }

    private func getProfileStatsViaSubprocess(profile: String) throws -> ProfileStats {
        // Query database stats using subprocess (thread-safe, no shared state)
        let script = """
        import sys
        sys.path.insert(0, '\(pythonBridge.venvSitePackages)')
        sys.path.insert(0, '\(pythonBridge.projectPath)')

        import os
        os.environ['DB_PROFILE'] = '\(profile)'

        from src.storage.storage_manager import StorageManager
        from src.config.settings import settings
        from pathlib import Path
        import json

        try:
            storage = StorageManager()
            stats = storage.get_statistics()

            index_dir = settings.get_index_dir('\(profile)')
            index_path = index_dir / "index.faiss"

            result = {
                'profile': '\(profile)',
                'total_images': stats.get('total_images', 0),
                'total_poses': stats.get('total_poses', 0),
                'database_name': settings.get_database_name('\(profile)'),
                'index_path': str(index_dir),
                'index_exists': index_path.exists()
            }

            storage.close()
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
            sys.exit(1)
        """

        // Execute via PythonBridgeSubprocess helper
        guard let output = pythonBridge.runPythonScript(script, configureThreading: false, timeout: 10.0) else {
            throw NSError(domain: "ProfileStats", code: -1, userInfo: [NSLocalizedDescriptionKey: "Script timeout"])
        }

        guard let data = output.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ProfileStats", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON"])
        }

        if let error = json["error"] as? String {
            throw NSError(domain: "ProfileStats", code: -3, userInfo: [NSLocalizedDescriptionKey: error])
        }

        return ProfileStats(
            profile: json["profile"] as? String ?? profile,
            totalImages: json["total_images"] as? Int ?? 0,
            totalPoses: json["total_poses"] as? Int ?? 0,
            indexSize: 0,  // Not calculated in this fast query
            databaseName: json["database_name"] as? String ?? "",
            indexPath: json["index_path"] as? String ?? "",
            indexExists: json["index_exists"] as? Bool ?? false,
            error: nil
        )
    }

    func switchProfile(_ newProfile: String) {
        print("[SETTINGS] Switching to profile: \(newProfile)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            let result = PythonBridge.shared.switchProfile(newProfile)

            DispatchQueue.main.async {
                if result.success {
                    self.activeProfile = newProfile
                    print("[SETTINGS] Profile switched successfully to \(newProfile)")

                    // CRITICAL FIX: Delay thumbnail loading to ensure subprocess picks up new profile
                    // The subprocess needs time to read the updated DB_PROFILE environment variable
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                        print("[SETTINGS] Profile switch stabilized, loading thumbnail stats...")
                        self?.loadThumbnailStatistics()
                    }
                } else {
                    print("[SETTINGS] Failed to switch profile: \(result.error ?? "Unknown error")")
                }
            }
        }
    }

    func loadThumbnailStatistics() {
        guard !isLoadingThumbnailCount else {
            print("[SETTINGS] Already loading thumbnail count, skipping...")
            return
        }

        print("[SETTINGS] Loading thumbnail statistics...")

        DispatchQueue.main.async {
            self.isLoadingThumbnailCount = true
        }

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            print("[SETTINGS] Calling pythonBridge.countTotalImages()...")
            let totalCount = self.pythonBridge.countTotalImages()
            print("[SETTINGS] Got total count: \(totalCount)")

            print("[SETTINGS] Calling pythonBridge.countMissingThumbnails()...")
            let missingCount = self.pythonBridge.countMissingThumbnails()
            print("[SETTINGS] Got missing count: \(missingCount)")

            DispatchQueue.main.async {
                print("[SETTINGS] Updating UI - total: \(totalCount), missing: \(missingCount)")
                self.totalImagesCount = totalCount
                self.imagesMissingThumbnails = missingCount
                self.isLoadingThumbnailCount = false
                print("[SETTINGS] UI updated - totalImages=\(self.totalImagesCount), missing=\(self.imagesMissingThumbnails)")
            }
        }
    }

    func startThumbnailGeneration() {
        if isGeneratingThumbnails {
            // Cancel operation
            print("[SETTINGS] Canceling thumbnail generation...")
            pythonBridge.cancelThumbnailGeneration()
            isGeneratingThumbnails = false
            thumbnailProgress = 0.0
            print("[SETTINGS] Thumbnail generation canceled")
            return
        }

        print("[SETTINGS] Starting thumbnail generation...")
        isGeneratingThumbnails = true
        thumbnailProgress = 0.0
        thumbnailsGenerated = 0
        thumbnailsFailed = 0

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            self.pythonBridge.backfillThumbnails { [weak self] progress in
                guard let self = self else { return }

                print("[SETTINGS] Thumbnail progress: \(progress.imagesProcessed)/\(progress.totalImages)")

                DispatchQueue.main.async {
                    self.thumbnailProgress = progress.progress
                    self.thumbnailsGenerated = progress.thumbnailsGenerated
                    self.thumbnailsFailed = progress.failedImages

                    // Check if complete
                    if progress.progress >= 1.0 {
                        self.isGeneratingThumbnails = false
                        self.loadThumbnailStatistics()  // Reload count
                        print("[SETTINGS] Thumbnail generation complete: \(progress.thumbnailsGenerated) generated, \(progress.failedImages) failed")
                    }
                }
            }
        }
    }
}

// MARK: - Progress Indicator
struct ProgressIndicator: View {
    @State private var isAnimating = false

    var body: some View {
        Circle()
            .trim(from: 0, to: 0.7)
            .stroke(Color.blue, lineWidth: 2)
            .frame(width: 12, height: 12)
            .rotationEffect(Angle(degrees: isAnimating ? 360 : 0))
            .animation(Animation.linear(duration: 1).repeatForever(autoreverses: false), value: isAnimating)
            .onAppear { isAnimating = true }
    }
}

// MARK: - Preview
struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
    }
}
