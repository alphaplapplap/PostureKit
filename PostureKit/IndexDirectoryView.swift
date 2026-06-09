import SwiftUI
import Combine

struct IndexDirectoryView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var viewModel: PostureKitViewModel
    @StateObject private var indexViewModel = IndexViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Index Directory")
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
                    // Directory Selection
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Directories to Index")
                                .font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Text("\(indexViewModel.selectedDirectories.count) selected")
                                .font(.system(size: 11))
                                .foregroundColor(.gray)
                        }

                        // List of selected directories
                        VStack(spacing: 6) {
                            if indexViewModel.selectedDirectories.isEmpty {
                                HStack {
                                    Image(systemName: "folder.badge.plus")
                                        .foregroundColor(.gray)
                                    Text("No directories selected. Click Add to choose folders.")
                                        .font(.system(size: 12))
                                        .foregroundColor(.gray)
                                    Spacer()
                                }
                                .padding(12)
                                .frame(maxWidth: .infinity)
                                .background(Color.gray.opacity(0.05))
                                .cornerRadius(6)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(Color.gray.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [4]))
                                )
                            } else {
                                ForEach(Array(indexViewModel.selectedDirectories.enumerated()), id: \.offset) { index, path in
                                    HStack(spacing: 8) {
                                        Image(systemName: "folder.fill")
                                            .foregroundColor(.blue)
                                            .font(.system(size: 12))
                                        Text(path)
                                            .font(.system(size: 12))
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                        Button(action: {
                                            indexViewModel.removeDirectory(at: index)
                                        }) {
                                            Image(systemName: "xmark.circle.fill")
                                                .foregroundColor(.gray)
                                                .font(.system(size: 14))
                                        }
                                        .buttonStyle(PlainButtonStyle())
                                        .disabled(indexViewModel.isIndexing)
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 8)
                                    .background(Color.gray.opacity(0.05))
                                    .cornerRadius(6)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 6)
                                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                                    )
                                }
                            }
                        }

                        HStack(spacing: 8) {
                            Button(action: { selectDirectories() }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "plus.circle.fill")
                                    Text("Add Directory")
                                }
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .background(Color.blue)
                                .cornerRadius(6)
                            }
                            .buttonStyle(PlainButtonStyle())
                            .disabled(indexViewModel.isIndexing)

                            if !indexViewModel.selectedDirectories.isEmpty {
                                Button(action: { indexViewModel.clearDirectories() }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "trash")
                                        Text("Clear All")
                                    }
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.primary)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 8)
                                    .background(Color.gray.opacity(0.1))
                                    .cornerRadius(6)
                                }
                                .buttonStyle(PlainButtonStyle())
                                .disabled(indexViewModel.isIndexing)
                            }
                            Spacer()
                        }
                    }
                    
                    // Options
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Options")
                            .font(.system(size: 13, weight: .semibold))
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Toggle("Include subdirectories", isOn: $indexViewModel.includeSubdirectories)
                                .font(.system(size: 13))
                            Toggle("Skip already indexed images", isOn: $indexViewModel.skipIndexed)
                                .font(.system(size: 13))
                            Toggle("Delete missing images from index", isOn: $indexViewModel.deleteMissing)
                                .font(.system(size: 13))
                        }
                    }
                    .padding(16)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(8)
                    
                    // Progress
                    if indexViewModel.isIndexing {
                        IndexProgressView(indexViewModel: indexViewModel)
                    }
                    
                    // Action Buttons
                    HStack(spacing: 12) {
                        Button(action: {
                            if indexViewModel.isIndexing {
                                indexViewModel.togglePause()
                            } else {
                                indexViewModel.startIndexing()
                            }
                        }) {
                            HStack {
                                Image(systemName: indexViewModel.isIndexing ? (indexViewModel.isPaused ? "play.fill" : "pause.fill") : "play.fill")
                                Text(indexViewModel.isIndexing ? (indexViewModel.isPaused ? "Resume Indexing" : "Pause Indexing") : "Start Indexing")
                            }
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(
                                indexViewModel.isIndexing
                                    ? (indexViewModel.isPaused ? Color.orange : Color.blue)
                                    : (indexViewModel.selectedDirectories.isEmpty ? Color.gray.opacity(0.4) : Color.blue)
                            )
                            .cornerRadius(8)
                        }
                        .buttonStyle(PlainButtonStyle())
                        .disabled(!indexViewModel.isIndexing && indexViewModel.selectedDirectories.isEmpty)
                        
                        Button(action: {
                            indexViewModel.cancelIndexing()
                            dismiss()
                        }) {
                            Text("Cancel")
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
        .frame(width: 600, height: 700)
        .onAppear {
            indexViewModel.mainViewModel = viewModel
        }
    }

    private func selectDirectories() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "Add"
        panel.message = "Select one or more directories to index"

        if panel.runModal() == .OK {
            indexViewModel.addDirectories(panel.urls.map { $0.path })
        }
    }
}

// MARK: - Index Progress View
struct IndexProgressView: View {
    @ObservedObject var indexViewModel: IndexViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(indexViewModel.isPaused ? "Indexing paused..." : "Indexing in progress...")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(indexViewModel.isPaused ? .orange : .primary)

                Spacer()

                Text("\(Int(indexViewModel.progress * 100))%")
                    .font(.system(size: 13))
                    .foregroundColor(.gray)
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 6)
                        .cornerRadius(3)
                    
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: geometry.size.width * indexViewModel.progress, height: 6)
                        .cornerRadius(3)
                        .animation(.linear(duration: 0.3), value: indexViewModel.progress)
                }
            }
            .frame(height: 6)
            
            VStack(alignment: .leading, spacing: 4) {
                if !indexViewModel.currentFile.isEmpty {
                    Text("Current: \(indexViewModel.currentFile)")
                        .font(.system(size: 12))
                        .foregroundColor(.blue)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                HStack {
                    Text("Processed: \(indexViewModel.imagesProcessed)/\(indexViewModel.totalImages)")
                    Spacer()
                    Text("Remaining: \(max(0, indexViewModel.totalImages - indexViewModel.imagesProcessed))")
                }
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.primary)

                HStack {
                    Text("Poses indexed: \(indexViewModel.posesIndexed)")
                    Spacer()
                    if indexViewModel.failedImages > 0 {
                        Text("Failed: \(indexViewModel.failedImages)")
                            .foregroundColor(.red)
                    }
                }
                .font(.system(size: 12))
                .foregroundColor(.gray)

                if indexViewModel.skippedImages > 0 {
                    Text("Skipped (already indexed): \(indexViewModel.skippedImages)")
                        .font(.system(size: 12))
                        .foregroundColor(.orange)
                }

                HStack {
                    Text("Elapsed: \(indexViewModel.timeElapsed)")
                    Spacer()
                    Text("Remaining: \(indexViewModel.timeRemaining)")
                }
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)
            }
        }
        .padding(16)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Index View Model
class IndexViewModel: ObservableObject {
    private static let directoriesKey = "IndexDirectoryView.selectedDirectories"

    private let pythonBridge = PythonBridgeSubprocess.shared
    private var startTime: Date?
    weak var mainViewModel: PostureKitViewModel?

    @Published var selectedDirectories: [String] {
        didSet {
            UserDefaults.standard.set(selectedDirectories, forKey: Self.directoriesKey)
        }
    }
    @Published var includeSubdirectories: Bool = true
    @Published var skipIndexed: Bool = true
    @Published var deleteMissing: Bool = false

    @Published var isIndexing: Bool = false
    @Published var isPaused: Bool = false
    @Published var progress: Double = 0.0
    @Published var currentFile: String = ""
    @Published var posesIndexed: Int = 0
    @Published var imagesProcessed: Int = 0
    @Published var totalImages: Int = 0
    @Published var failedImages: Int = 0
    @Published var skippedImages: Int = 0
    @Published var timeElapsed: String = "00:00:00"
    @Published var timeRemaining: String = "00:00:00"

    init() {
        // Restore previously selected directories from UserDefaults.
        // Filter out any paths that no longer exist on disk so the UI doesn't
        // show stale entries (e.g. unmounted external drives, deleted folders).
        let saved = UserDefaults.standard.stringArray(forKey: Self.directoriesKey) ?? []
        self.selectedDirectories = saved.filter { FileManager.default.fileExists(atPath: $0) }
    }

    func addDirectories(_ paths: [String]) {
        for path in paths {
            if !selectedDirectories.contains(path) {
                selectedDirectories.append(path)
            }
        }
    }

    func removeDirectory(at index: Int) {
        guard index >= 0 && index < selectedDirectories.count else { return }
        selectedDirectories.remove(at: index)
    }

    func clearDirectories() {
        selectedDirectories.removeAll()
    }

    func startIndexing() {
        print("[INDEX DEBUG] startIndexing called")
        guard !isIndexing else {
            print("[INDEX DEBUG] Already indexing, returning")
            return
        }

        guard !selectedDirectories.isEmpty else {
            print("[INDEX DEBUG] No directories selected, returning")
            return
        }

        print("[INDEX DEBUG] Starting index of \(selectedDirectories.count) directories: \(selectedDirectories)")
        isIndexing = true
        isPaused = false
        progress = 0.0
        posesIndexed = 0
        imagesProcessed = 0
        failedImages = 0
        skippedImages = 0
        startTime = Date()

        // Call Python directly for indexing with low confidence threshold (0.2)
        // This captures most detections - filtering by confidence happens during search
        print("[INDEX DEBUG] Calling pythonBridge.startIndexing")
        pythonBridge.startIndexing(
            directories: selectedDirectories,
            recursive: includeSubdirectories,
            minConfidence: 0.2,
            skipIndexed: skipIndexed,
            deleteMissing: deleteMissing
        ) { [weak self] indexProgress in
            guard let self = self else { return }

            print("[INDEX DEBUG] Progress callback - processed: \(indexProgress.imagesProcessed)/\(indexProgress.totalImages)")
            self.currentFile = indexProgress.currentFile
            self.imagesProcessed = indexProgress.imagesProcessed
            self.totalImages = indexProgress.totalImages
            self.posesIndexed = indexProgress.posesIndexed
            self.failedImages = indexProgress.failedImages
            self.skippedImages = indexProgress.skippedImages
            self.progress = indexProgress.progress

            // Calculate time estimates
            if let start = self.startTime {
                let elapsed = Date().timeIntervalSince(start)
                self.timeElapsed = self.formatTime(Int(elapsed))

                if self.progress > 0 {
                    let estimatedTotal = elapsed / self.progress
                    let remaining = estimatedTotal - elapsed
                    self.timeRemaining = self.formatTime(Int(remaining))
                }
            }

            // Check if complete
            if self.progress >= 1.0 {
                print("[INDEX DEBUG] Indexing complete!")
                self.isIndexing = false
                // Reload index statistics to update pose count in UI
                self.mainViewModel?.loadIndexStatistics()
            }
        }
    }

    func togglePause() {
        print("[INDEX DEBUG] togglePause called")
        guard isIndexing else {
            print("[INDEX DEBUG] Not currently indexing, returning")
            return
        }

        if isPaused {
            print("[INDEX DEBUG] Resuming indexing process...")
            pythonBridge.resumeIndexing()
            isPaused = false
            print("[INDEX DEBUG] Indexing resumed")
        } else {
            print("[INDEX DEBUG] Pausing indexing process...")
            pythonBridge.pauseIndexing()
            isPaused = true
            print("[INDEX DEBUG] Indexing paused")
        }
    }

    func cancelIndexing() {
        print("[INDEX DEBUG] cancelIndexing called")
        if isIndexing {
            print("[INDEX DEBUG] Terminating indexing process...")
            pythonBridge.cancelIndexing()
            isIndexing = false
            isPaused = false
            print("[INDEX DEBUG] Indexing canceled")
        }
    }
    
    private func formatTime(_ seconds: Int) -> String {
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        let secs = seconds % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, secs)
    }
}

// MARK: - Preview
struct IndexDirectoryView_Previews: PreviewProvider {
    static var previews: some View {
        IndexDirectoryView(viewModel: PostureKitViewModel())
    }
}
