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
                        Text("Directory to Index")
                            .font(.system(size: 13, weight: .semibold))
                        
                        HStack(spacing: 8) {
                            TextField("", text: $indexViewModel.selectedDirectory)
                                .textFieldStyle(PlainTextFieldStyle())
                                .padding(10)
                                .background(Color.gray.opacity(0.05))
                                .cornerRadius(6)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                                )
                            
                            Button(action: { selectDirectory() }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "folder")
                                    Text("Browse")
                                }
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.blue)
                                .cornerRadius(6)
                            }
                            .buttonStyle(PlainButtonStyle())
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
                            .background(indexViewModel.isIndexing ? (indexViewModel.isPaused ? Color.orange : Color.blue) : Color.blue)
                            .cornerRadius(8)
                        }
                        .buttonStyle(PlainButtonStyle())
                        
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

    private func selectDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        
        if panel.runModal() == .OK, let url = panel.url {
            indexViewModel.selectedDirectory = url.path
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
    private let pythonBridge = PythonBridgeSubprocess.shared
    private var startTime: Date?
    weak var mainViewModel: PostureKitViewModel?

    @Published var selectedDirectory: String = "/Users/you/Pictures/yoga_dataset"
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

    func startIndexing() {
        print("[INDEX DEBUG] startIndexing called")
        guard !isIndexing else {
            print("[INDEX DEBUG] Already indexing, returning")
            return
        }

        print("[INDEX DEBUG] Starting index of directory: \(selectedDirectory)")
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
            directory: selectedDirectory,
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
