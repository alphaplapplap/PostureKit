
// MARK: - PostureKit Results Presentation System
// Comprehensive solution for handling thousands of photos efficiently

import SwiftUI
import AppKit
import Combine

// MARK: - Data Models
struct PhotoResult: Identifiable, Hashable {
    let id = UUID()
    let filePath: String
    let similarityScore: Double
    let thumbnailSize: CGSize
    var isSelected: Bool = false

    var url: URL {
        URL(fileURLWithPath: filePath)
    }

    var fileName: String {
        url.lastPathComponent
    }
}

// MARK: - Selection Manager
@MainActor
class SelectionManager: ObservableObject {
    @Published var selectedResults: Set<UUID> = []
    @Published var lastSelectedIndex: Int?

    func toggleSelection(for result: PhotoResult, at index: Int, with modifiers: NSEvent.ModifierFlags) {
        switch modifiers {
        case .command:
            // Cmd+Click: Toggle individual selection
            if selectedResults.contains(result.id) {
                selectedResults.remove(result.id)
            } else {
                selectedResults.insert(result.id)
            }

        case .shift:
            // Shift+Click: Range selection
            if let lastIndex = lastSelectedIndex {
                let range = min(lastIndex, index)...max(lastIndex, index)
                // Clear previous selection and select range
                selectedResults.removeAll()
                // Add range logic here based on your results array
            }

        default:
            // Regular click: Select only this item
            selectedResults.removeAll()
            selectedResults.insert(result.id)
        }

        lastSelectedIndex = index
    }

    func selectAll(results: [PhotoResult]) {
        selectedResults = Set(results.map { $0.id })
    }

    func deselectAll() {
        selectedResults.removeAll()
    }
}

// MARK: - Memory-Optimized Image Cache
class ThumbnailCache: ObservableObject {
    private let cache = NSCache<NSString, NSImage>()
    private let maxCacheSize: Int = 200 // Limit memory usage

    init() {
        cache.countLimit = maxCacheSize
        // Set memory limit to ~100MB for thumbnails
        cache.totalCostLimit = 100 * 1024 * 1024
    }

    func image(for path: String, size: CGSize) -> NSImage? {
        let key = "\(path)_\(size.width)x\(size.height)" as NSString
        return cache.object(forKey: key)
    }

    func setImage(_ image: NSImage, for path: String, size: CGSize) {
        let key = "\(path)_\(size.width)x\(size.height)" as NSString
        let cost = Int(image.size.width * image.size.height * 4) // Estimate memory cost
        cache.setObject(image, forKey: key, cost: cost)
    }

    func clearCache() {
        cache.removeAllObjects()
    }
}

// MARK: - Thumbnail View with Lazy Loading
struct ThumbnailView: View {
    let result: PhotoResult
    let thumbnailSize: CGSize
    let isSelected: Bool
    @StateObject private var cache = ThumbnailCache()
    @State private var image: NSImage?
    @State private var isLoading = false

    var body: some View {
        ZStack {
            if let image = image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: thumbnailSize.width, height: thumbnailSize.height)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: thumbnailSize.width, height: thumbnailSize.height)
                    .overlay(
                        ProgressView()
                            .scaleEffect(0.5)
                            .opacity(isLoading ? 1 : 0)
                    )
            }

            // Selection overlay
            if isSelected {
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.blue, lineWidth: 3)
                    .frame(width: thumbnailSize.width, height: thumbnailSize.height)

                // Checkbox
                VStack {
                    HStack {
                        Spacer()
                        Circle()
                            .fill(Color.blue)
                            .frame(width: 20, height: 20)
                            .overlay(
                                Image(systemName: "checkmark")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(.white)
                            )
                    }
                    Spacer()
                }
                .padding(4)
            }
        }
        .onAppear {
            loadThumbnail()
        }
        .onDisappear {
            // Optional: Clear image to save memory for off-screen items
            if !isSelected {
                image = nil
            }
        }
    }

    private func loadThumbnail() {
        // Check cache first
        if let cachedImage = cache.image(for: result.filePath, size: thumbnailSize) {
            self.image = cachedImage
            return
        }

        isLoading = true

        Task {
            await loadThumbnailAsync()
        }
    }

    @MainActor
    private func loadThumbnailAsync() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                let image = await generateThumbnail()
                await MainActor.run {
                    self.image = image
                    self.isLoading = false
                    if let image = image {
                        cache.setImage(image, for: result.filePath, size: thumbnailSize)
                    }
                }
            }
        }
    }

    private func generateThumbnail() async -> NSImage? {
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                guard let image = NSImage(contentsOfFile: result.filePath) else {
                    continuation.resume(returning: nil)
                    return
                }

                // Resize image to thumbnail size for memory efficiency
                let thumbnail = image.resized(to: thumbnailSize)
                continuation.resume(returning: thumbnail)
            }
        }
    }
}

// MARK: - Main Results Grid View
struct ResultsGridView: View {
    @State private var results: [PhotoResult] = []
    @State private var thumbnailSize: CGSize = CGSize(width: 150, height: 150)
    @State private var viewMode: ViewMode = .grid
    @StateObject private var selectionManager = SelectionManager()
    @State private var selectedFolderPath: String = ""

    enum ViewMode {
        case grid, list
    }

    private var columns: [GridItem] {
        let itemSize = thumbnailSize.width + 20 // Add padding
        let columnCount = max(1, Int(NSScreen.main?.frame.width ?? 800) / Int(itemSize))
        return Array(repeating: GridItem(.flexible(), spacing: 10), count: columnCount)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            toolbar

            // Content
            GeometryReader { geometry in
                if viewMode == .grid {
                    gridView
                } else {
                    listView
                }
            }

            // Status Bar
            statusBar
        }
        .onAppear {
            loadResults()
        }
    }

    // MARK: - Toolbar
    private var toolbar: some View {
        HStack {
            // View mode toggle
            Picker("View Mode", selection: $viewMode) {
                Text("Grid").tag(ViewMode.grid)
                Text("List").tag(ViewMode.list)
            }
            .pickerStyle(SegmentedPickerStyle())
            .frame(width: 120)

            Spacer()

            // Thumbnail size slider (for grid mode)
            if viewMode == .grid {
                Text("Size:")
                Slider(value: Binding(
                    get: { thumbnailSize.width },
                    set: { thumbnailSize = CGSize(width: $0, height: $0) }
                ), in: 100...300, step: 25)
                .frame(width: 150)
            }

            Spacer()

            // Selection controls
            Button("Select All") {
                selectionManager.selectAll(results: results)
            }

            Button("Deselect All") {
                selectionManager.deselectAll()
            }

            Text("\(selectionManager.selectedResults.count) selected")
                .foregroundColor(.secondary)
        }
        .padding()
    }

    // MARK: - Grid View
    private var gridView: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(Array(results.enumerated()), id: \.element.id) { index, result in
                    ThumbnailView(
                        result: result,
                        thumbnailSize: thumbnailSize,
                        isSelected: selectionManager.selectedResults.contains(result.id)
                    )
                    .onTapGesture {
                        handleSelection(result: result, index: index)
                    }
                    .contextMenu {
                        contextMenuItems(for: result)
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - List View
    private var listView: some View {
        List {
            ForEach(Array(results.enumerated()), id: \.element.id) { index, result in
                HStack {
                    // Checkbox
                    Button(action: {
                        handleSelection(result: result, index: index)
                    }) {
                        Image(systemName: selectionManager.selectedResults.contains(result.id) ? 
                              "checkmark.square.fill" : "square")
                            .foregroundColor(selectionManager.selectedResults.contains(result.id) ? 
                                           .blue : .secondary)
                    }
                    .buttonStyle(PlainButtonStyle())

                    // Thumbnail
                    ThumbnailView(
                        result: result,
                        thumbnailSize: CGSize(width: 60, height: 60),
                        isSelected: false
                    )

                    // File info
                    VStack(alignment: .leading) {
                        Text(result.fileName)
                            .font(.headline)
                        Text("Similarity: \(String(format: "%.2f", result.similarityScore))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    handleSelection(result: result, index: index)
                }
                .contextMenu {
                    contextMenuItems(for: result)
                }
            }
        }
    }

    // MARK: - Context Menu
    @ViewBuilder
    private func contextMenuItems(for result: PhotoResult) -> some View {
        Button("Move to Folder...") {
            selectFolderAndMove(results: [result])
        }

        if selectionManager.selectedResults.count > 1 {
            Button("Move Selected to Folder...") {
                let selectedResults = results.filter { selectionManager.selectedResults.contains($0.id) }
                selectFolderAndMove(results: selectedResults)
            }
        }

        Divider()

        Button("Reveal in Finder") {
            NSWorkspace.shared.selectFile(result.filePath, inFileViewerRootedAtPath: "")
        }

        Button("Open with Default App") {
            NSWorkspace.shared.open(result.url)
        }
    }

    // MARK: - Status Bar
    private var statusBar: some View {
        HStack {
            Text("\(results.count) results")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            if let selectedResult = results.first(where: { selectionManager.selectedResults.contains($0.id) }) {
                Text(selectedResult.filePath)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
        .background(Color(NSColor.controlBackgroundColor))
    }

    // MARK: - Helper Methods
    private func handleSelection(result: PhotoResult, index: Int) {
        let modifiers = NSApp.currentEvent?.modifierFlags ?? []
        selectionManager.toggleSelection(for: result, at: index, with: modifiers)
    }

    private func loadResults() {
        // This would connect to your Python backend
        // For demo purposes, creating sample data
        results = (0..<1000).map { i in
            PhotoResult(
                filePath: "/path/to/photo\(i).jpg",
                similarityScore: Double.random(in: 0.5...1.0),
                thumbnailSize: thumbnailSize
            )
        }
    }

    private func selectFolderAndMove(results: [PhotoResult]) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let selectedURL = panel.url {
            moveFiles(results: results, to: selectedURL.path)
        }
    }

    private func moveFiles(results: [PhotoResult], to destinationPath: String) {
        Task {
            for result in results {
                do {
                    let fileManager = FileManager.default
                    let fileName = result.url.lastPathComponent
                    let destinationURL = URL(fileURLWithPath: destinationPath).appendingPathComponent(fileName)

                    try fileManager.moveItem(at: result.url, to: destinationURL)
                    print("Moved \(fileName) to \(destinationPath)")
                } catch {
                    print("Error moving file \(result.fileName): \(error)")
                }
            }

            // Refresh results after moving files
            await MainActor.run {
                loadResults()
                selectionManager.deselectAll()
            }
        }
    }
}

// MARK: - NSImage Extension for Thumbnail Generation
extension NSImage {
    func resized(to size: CGSize) -> NSImage {
        let img = NSImage(size: size)

        img.lockFocus()
        let ctx = NSGraphicsContext.current
        ctx?.imageInterpolation = .high
        self.draw(in: NSMakeRect(0, 0, size.width, size.height),
                  from: NSMakeRect(0, 0, self.size.width, self.size.height),
                  operation: .copy,
                  fraction: 1)
        img.unlockFocus()

        return img
    }
}

// MARK: - Main Integration Point
// This would replace or enhance your current results presentation
struct PostureKitResultsView: View {
    var body: some View {
        ResultsGridView()
            .navigationTitle("PostureKit Results")
    }
}
