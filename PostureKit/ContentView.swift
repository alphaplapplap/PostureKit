//
//  ContentView.swift
//  PostureKit
//
//  Created by Linux Babe on 2025-10-09.
//

import SwiftUI
import UniformTypeIdentifiers

// MARK: - Skeleton Constants
// COCO-WholeBody skeleton connections (matches Python src/core/models.py:272)
let SKELETON_CONNECTIONS: [(Int, Int)] = [
    // Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    // Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    // Torso
    (5, 11), (6, 12), (11, 12),
    // Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    // Feet
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22)
]

// Body-only connections (minimal detail)
let SKELETON_CONNECTIONS_MINIMAL: [(Int, Int)] = [
    // Torso
    (5, 11), (6, 12), (11, 12),
    // Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

// Body + major limbs (standard detail)
let SKELETON_CONNECTIONS_STANDARD: [(Int, Int)] = [
    // Head
    (0, 1), (0, 2),
    // Arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    // Torso
    (5, 11), (6, 12), (11, 12),
    // Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

// MARK: - Skeleton Drawing Helper
/// Draw skeleton overlay on Canvas (shared by query image and result thumbnails)
func drawSkeleton(
    context: GraphicsContext,
    size: CGSize,
    keypoints: [[Double]],
    detail: SkeletonDetail,
    alreadyNormalized: Bool = false,
    isSelected: Bool = true,
    originalImageSize: CGSize? = nil  // Original image dimensions for thumbnail offset calculation
) {
    // Select connections based on detail level
    let connections: [(Int, Int)]
    switch detail {
    case .full:
        connections = SKELETON_CONNECTIONS
    case .standard:
        connections = SKELETON_CONNECTIONS_STANDARD
    case .minimal:
        connections = SKELETON_CONNECTIONS_MINIMAL
    }

    // Handle coordinate scaling based on input format
    let scaleX: Double
    let scaleY: Double
    let xOffset: Double
    let yOffset: Double

    if alreadyNormalized {
        // Keypoints already in display coordinates (query image)
        scaleX = 1.0
        scaleY = 1.0
        xOffset = 0.0
        yOffset = 0.0
    } else {
        // Keypoints are in ORIGINAL IMAGE coordinates (not 200×200 normalized)
        // They need to be scaled to fit within the 200×200 thumbnail, then centered, then scaled to display size
        
        if let imgSize = originalImageSize {
            // Step 1: Calculate how the image was scaled to fit within 200×200 (matching loadThumbnail logic)
            let thumbnailScale = min(200.0 / imgSize.width, 200.0 / imgSize.height)
            let scaledWidth = imgSize.width * thumbnailScale
            let scaledHeight = imgSize.height * thumbnailScale
            
            // Step 2: Calculate centering offsets within the 200×200 thumbnail space
            let thumbnailXOffset = (200.0 - scaledWidth) / 2
            let thumbnailYOffset = (200.0 - scaledHeight) / 2
            
            // Step 3: Scale from 200×200 thumbnail space to actual display size
            let displayScale = size.width / 200.0
            
            // Final transform: scale from original image coords to thumbnail coords, then to display coords
            scaleX = thumbnailScale * displayScale
            scaleY = thumbnailScale * displayScale
            xOffset = thumbnailXOffset * displayScale
            yOffset = thumbnailYOffset * displayScale
        } else {
            // Fallback: assume keypoints already in 200×200 space
            scaleX = size.width / 200.0
            scaleY = size.height / 200.0
            xOffset = 0.0
            yOffset = 0.0
        }
    }

    // Draw skeleton connections (lines)
    for (startIdx, endIdx) in connections {
        guard startIdx < keypoints.count, endIdx < keypoints.count else { continue }

        let start = keypoints[startIdx]
        let end = keypoints[endIdx]

        // Check confidence thresholds (both keypoints must be visible)
        let startConf = start[2]
        let endConf = end[2]
        guard startConf > 0.3, endConf > 0.3 else { continue }

        // Scale coordinates and apply offset
        let startPoint = CGPoint(x: start[0] * scaleX + xOffset, y: start[1] * scaleY + yOffset)
        let endPoint = CGPoint(x: end[0] * scaleX + xOffset, y: end[1] * scaleY + yOffset)

        // Color by average confidence and selection state
        let avgConf = (startConf + endConf) / 2.0
        let lineColor: Color

        if isSelected {
            // Selected person: confidence-based color
            if avgConf > 0.7 {
                lineColor = .green
            } else if avgConf > 0.4 {
                lineColor = .yellow
            } else {
                lineColor = .red
            }
        } else {
            // Non-selected person: dimmed gray
            lineColor = .gray
        }

        // Draw line with appropriate opacity
        let opacity = isSelected ? 0.8 : 0.4
        var path = Path()
        path.move(to: startPoint)
        path.addLine(to: endPoint)
        context.stroke(path, with: .color(lineColor.opacity(opacity)), lineWidth: 2)
    }

    // Draw keypoints (dots) for visible joints only
    let keypointsToShow: [Int]
    switch detail {
    case .full:
        keypointsToShow = Array(0..<min(133, keypoints.count))
    case .standard:
        keypointsToShow = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    case .minimal:
        keypointsToShow = [5, 6, 11, 12, 13, 14, 15, 16]
    }

    for idx in keypointsToShow {
        guard idx < keypoints.count else { continue }

        let kp = keypoints[idx]
        let conf = kp[2]
        guard conf > 0.3 else { continue }

        let point = CGPoint(x: kp[0] * scaleX + xOffset, y: kp[1] * scaleY + yOffset)

        // Color by confidence and selection state
        let dotColor: Color

        if isSelected {
            // Selected person: confidence-based color
            if conf > 0.7 {
                dotColor = .green
            } else if conf > 0.4 {
                dotColor = .yellow
            } else {
                dotColor = .red
            }
        } else {
            // Non-selected person: dimmed gray
            dotColor = .gray.opacity(0.3)
        }

        // Draw keypoint dot
        let dotRect = CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)
        context.fill(Path(ellipseIn: dotRect), with: .color(dotColor))
    }
}

struct ContentView: View {
    @StateObject private var viewModel = PostureKitViewModel()
    @State private var showIndexModal = false
    @State private var showSettingsModal = false

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                KeyboardMonitor(viewModel: viewModel)

                VStack(spacing: 0) {
                // Header
                HeaderView(
                    showIndexModal: $showIndexModal,
                    showSettingsModal: $showSettingsModal
                )

                // Main Content Area
                ScrollView {
                    VStack(spacing: 24) {
                        // Browse/Search Mode Toggle
                        VStack(spacing: 16) {
                            HStack(spacing: 12) {
                                Button(action: { viewModel.browseMode = false }) {
                                    HStack {
                                        Image(systemName: "photo.on.rectangle")
                                        Text("Search by Image")
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(viewModel.browseMode ? Color.clear : Color.blue)
                                    .foregroundColor(viewModel.browseMode ? .primary : .white)
                                    .cornerRadius(8)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(viewModel.browseMode ? Color.gray.opacity(0.3) : Color.clear, lineWidth: 1)
                                    )
                                }
                                .buttonStyle(PlainButtonStyle())

                                Button(action: { viewModel.browseMode = true }) {
                                    HStack {
                                        Image(systemName: "magnifyingglass.circle.fill")
                                        Text("Browse Database")
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(viewModel.browseMode ? Color.blue : Color.clear)
                                    .foregroundColor(viewModel.browseMode ? .white : .primary)
                                    .cornerRadius(8)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(viewModel.browseMode ? Color.clear : Color.gray.opacity(0.3), lineWidth: 1)
                                    )
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                            .padding(.horizontal, max(24, geometry.size.width * 0.05))
                            .padding(.top, 24)
                        }

                        // Drop Zone (only in Search by Image mode)
                        if !viewModel.browseMode {
                            DropZoneView(viewModel: viewModel)
                                .padding(.horizontal, max(24, geometry.size.width * 0.05))
                        }

                        // Search Parameters (shown when image loaded in search mode OR always in browse mode)
                        if viewModel.queryImage != nil || viewModel.browseMode {
                            SearchParametersView(viewModel: viewModel)
                                .padding(.horizontal, max(24, geometry.size.width * 0.05))
                        }

                        // Loading State (shown while searching)
                        if viewModel.isSearching {
                            SearchingOverlayView(viewModel: viewModel)
                                .padding(.horizontal, max(24, geometry.size.width * 0.05))
                        }

                        // Results Grid (shown after search)
                        if !viewModel.searchResults.isEmpty && !viewModel.isSearching {
                            ResultsGridView(viewModel: viewModel, availableHeight: geometry.size.height)
                                .padding(.horizontal, max(24, geometry.size.width * 0.05))
                                .padding(.bottom, 24)
                        }

                        // No Results Message (shown after search completes with 0 results)
                        if !viewModel.isSearching && viewModel.searchResults.isEmpty && viewModel.poseDetected {
                            NoResultsView()
                                .padding(.horizontal, max(24, geometry.size.width * 0.05))
                        }
                    }
                }

                // Status Bar
                StatusBarView(viewModel: viewModel)
                }
            }
        }
        .frame(minWidth: 800, idealWidth: 1600, maxWidth: .infinity,
               minHeight: 600, idealHeight: 1000, maxHeight: .infinity)
        .onAppear {
            viewModel.loadIndexStatistics()
        }
        .sheet(isPresented: $showIndexModal) {
            IndexDirectoryView(viewModel: viewModel)
        }
        .sheet(isPresented: $showSettingsModal) {
            SettingsView()
        }
    }
}

// MARK: - Header View
struct HeaderView: View {
    @Binding var showIndexModal: Bool
    @Binding var showSettingsModal: Bool
    
    var body: some View {
        HStack {
            HStack(spacing: 12) {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(Color.blue)
                    .cornerRadius(8)
                
                Text("PostureKit")
                    .font(.system(size: 20, weight: .semibold))
            }
            
            Spacer()
            
            HStack(spacing: 8) {
                Button(action: { showIndexModal = true }) {
                    HStack(spacing: 6) {
                        Image(systemName: "folder")
                        Text("Index Directory")
                    }
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Color.blue)
                    .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
                
                Button(action: { showSettingsModal = true }) {
                    Image(systemName: "gear")
                        .font(.system(size: 16))
                        .foregroundColor(.gray)
                        .frame(width: 36, height: 36)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(Color(NSColor.windowBackgroundColor))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Color.gray.opacity(0.2)),
            alignment: .bottom
        )
    }
}

// MARK: - Drop Zone View
struct DropZoneView: View {
    @ObservedObject var viewModel: PostureKitViewModel
    @State private var isTargeted = false
    
    var body: some View {
        ZStack {
            if viewModel.queryImage == nil {
                // Empty State
                VStack(spacing: 16) {
                    Image(systemName: "arrow.up.doc")
                        .font(.system(size: 48))
                        .foregroundColor(.gray)
                    
                    Text("Drop Query Image Here")
                        .font(.system(size: 18, weight: .medium))
                    
                    Text("or click to browse")
                        .font(.system(size: 14))
                        .foregroundColor(.gray)
                    
                    Text("Supported: JPG, PNG, BMP, WebP")
                        .font(.system(size: 12))
                        .foregroundColor(.gray.opacity(0.7))
                }
                .frame(maxWidth: .infinity)
                .padding(60)
            } else {
                // Image Loaded State
                QueryImageDisplayView(viewModel: viewModel)
            }
        }
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isTargeted || viewModel.queryImage != nil ? Color.blue : Color.gray.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [8])
                )
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(viewModel.queryImage != nil ? Color.blue.opacity(0.05) : Color.clear)
                )
        )
        .onTapGesture {
            selectImage()
        }
        .onDrop(of: ["public.file-url"], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
        }
    }
    
    private func selectImage() {
        print("[UI DEBUG] selectImage called")
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .bmp, .webP]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            print("[UI DEBUG] File selected: \(url.path)")
            loadImage(from: url)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        print("[UI DEBUG] handleDrop called with \(providers.count) providers")
        guard let provider = providers.first else {
            print("[UI DEBUG] No providers")
            return false
        }

        provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, error in
            print("[UI DEBUG] loadItem callback - item: \(String(describing: item)), error: \(String(describing: error))")
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else {
                print("[UI DEBUG] Failed to get URL from drop")
                return
            }

            print("[UI DEBUG] Dropped file: \(url.path)")
            DispatchQueue.main.async {
                loadImage(from: url)
            }
        }
        return true
    }

    private func loadImage(from url: URL) {
        print("[UI DEBUG] loadImage called with: \(url.path)")
        guard let image = NSImage(contentsOf: url) else {
            print("[UI DEBUG] Failed to load NSImage from URL")
            return
        }
        let ps = image.pixelSize
        print("[UI DEBUG] NSImage loaded: size=\(image.size) pixelSize=\(ps)")

        // Store image for display
        viewModel.queryImage = image
        viewModel.queryImageName = url.lastPathComponent
        viewModel.queryImageSize = ps

        // Clear previous detection results
        viewModel.poseDetected = false
        viewModel.detectedPose = nil
        viewModel.extractedFeatures = nil
        viewModel.searchResults = []
        viewModel.detectedKeypoints = nil
        viewModel.detectedPeople = []
        viewModel.selectedPersonIndex = 0

        print("[UI DEBUG] Image loaded - starting immediate multi-person detection")

        // Immediately detect all people in the image
        viewModel.detectAllPeopleInQueryImage()
    }
}

// MARK: - Query Image Display View
struct QueryImageDisplayView: View {
    @ObservedObject var viewModel: PostureKitViewModel

    var body: some View {
        VStack(spacing: 12) {
            if let image = viewModel.queryImage {
                ZStack {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: 500, height: 500)
                        .cornerRadius(8)

                    // Multi-person skeleton overlay
                    if !viewModel.detectedPeople.isEmpty,
                       let imageSize = viewModel.queryImageSize {
                        Canvas { context, size in
                            // Draw skeleton for EACH detected person
                            for (index, person) in viewModel.detectedPeople.enumerated() {
                                let isSelected = (index == viewModel.selectedPersonIndex)

                                // Normalize this person's keypoints to display coordinates
                                let normalizedKeypoints = viewModel.normalizeKeypoints(
                                    keypoints: person.keypoints,
                                    fromSize: imageSize,
                                    toSize: size
                                )

                                // Draw skeleton with person-specific colors
                                drawSkeleton(
                                    context: context,
                                    size: size,
                                    keypoints: normalizedKeypoints,
                                    detail: .standard,
                                    alreadyNormalized: true,  // Keypoints are in display coords
                                    isSelected: isSelected     // Color-code by selection
                                )
                            }
                        }
                    }

                    // Bounding box overlay (multi-person support)
                    if !viewModel.detectedPeople.isEmpty,
                       let imageSize = viewModel.queryImageSize {
                        Canvas { context, size in
                            // Draw bbox for each detected person
                            for (index, person) in viewModel.detectedPeople.enumerated() {
                                let isSelected = (index == viewModel.selectedPersonIndex)

                                // Calculate scale from original image to display size
                                let scaleX = size.width / imageSize.width
                                let scaleY = size.height / imageSize.height
                                let scale = min(scaleX, scaleY)

                                // Calculate centered display area
                                let displayWidth = imageSize.width * scale
                                let displayHeight = imageSize.height * scale
                                let xOffset = (size.width - displayWidth) / 2
                                let yOffset = (size.height - displayHeight) / 2

                                // Get bbox components [x, y, w, h]
                                guard person.bbox.count == 4 else { continue }
                                let bboxX = person.bbox[0] * scale + xOffset
                                let bboxY = person.bbox[1] * scale + yOffset
                                let bboxWidth = person.bbox[2] * scale
                                let bboxHeight = person.bbox[3] * scale

                                // Draw bbox rectangle
                                let rect = CGRect(x: bboxX, y: bboxY, width: bboxWidth, height: bboxHeight)
                                let path = Path(roundedRect: rect, cornerRadius: 3)

                                // Selected person: green, thick border with glow
                                // Other people: gray, thin border
                                if isSelected {
                                    // Outer glow
                                    context.stroke(
                                        path,
                                        with: .color(.green.opacity(0.3)),
                                        lineWidth: 6
                                    )
                                    // Main border
                                    context.stroke(
                                        path,
                                        with: .color(.green),
                                        lineWidth: 3
                                    )
                                } else {
                                    context.stroke(
                                        path,
                                        with: .color(.gray.opacity(0.6)),
                                        lineWidth: 1.5
                                    )
                                }

                                // Draw person ID badge (e.g., "P1", "P2")
                                let badgeText = Text("P\(index + 1)")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundColor(.white)

                                let badgeX = bboxX + 8
                                let badgeY = bboxY + 8

                                // Badge background circle
                                let badgeCircle = Path(ellipseIn: CGRect(
                                    x: badgeX - 10,
                                    y: badgeY - 8,
                                    width: 20,
                                    height: 16
                                ))

                                context.fill(
                                    badgeCircle,
                                    with: .color(isSelected ? Color.green : Color.gray.opacity(0.8))
                                )

                                // Badge text
                                context.draw(
                                    badgeText,
                                    at: CGPoint(x: badgeX, y: badgeY),
                                    anchor: .center
                                )
                            }
                        }
                    }
                }
                .frame(width: 500, height: 500)
                .contentShape(Rectangle())  // Make entire area tappable
                .gesture(
                    // Add tap gesture to select person by clicking on bbox
                    DragGesture(minimumDistance: 0)
                        .onEnded { value in
                            handleImageTap(at: value.location, imageSize: image.size, displaySize: CGSize(width: 500, height: 500))
                        }
                )
            }

            Text(viewModel.queryImageName ?? "query_image.jpg")
                .font(.system(size: 14, weight: .semibold))

            if let imageSize = viewModel.queryImageSize {
                Text("\(Int(imageSize.width))×\(Int(imageSize.height))")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }

            // Multi-person detection status
            if viewModel.isDetecting {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("Detecting people...")
                        .font(.system(size: 12))
                        .foregroundColor(.gray)
                }
            } else if !viewModel.detectedPeople.isEmpty {
                VStack(spacing: 4) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 8, height: 8)
                        Text("Detected \(viewModel.detectedPeople.count) \(viewModel.detectedPeople.count == 1 ? "person" : "people")")
                            .font(.system(size: 13))
                            .foregroundColor(.green)
                    }

                    if viewModel.detectedPeople.count > 1 {
                        Text("Selected: Person \(viewModel.selectedPersonIndex + 1) (conf: \(String(format: "%.2f", viewModel.poseConfidence)))")
                            .font(.system(size: 12))
                            .foregroundColor(.blue)
                    } else if let person = viewModel.detectedPeople.first {
                        Text("Confidence: \(String(format: "%.2f", person.confidence))")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                }
            } else if viewModel.poseDetected {
                // Fallback for old single-person detection flow
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                    Text("Pose detected (conf: \(String(format: "%.2f", viewModel.poseConfidence)))")
                        .font(.system(size: 13))
                        .foregroundColor(.green)
                }
            }

            // Person selector buttons (only show when multiple people detected)
            if viewModel.detectedPeople.count > 1 {
                VStack(spacing: 8) {
                    Text("Select Person:")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.gray)

                    HStack(spacing: 6) {
                        ForEach(Array(viewModel.detectedPeople.enumerated()), id: \.offset) { index, person in
                            Button(action: {
                                viewModel.selectPerson(at: index)
                            }) {
                                VStack(spacing: 2) {
                                    Text("P\(index + 1)")
                                        .font(.system(size: 11, weight: .semibold))
                                    Text("\(String(format: "%.0f", person.confidence * 100))%")
                                        .font(.system(size: 9))
                                }
                                .foregroundColor(index == viewModel.selectedPersonIndex ? .white : .primary)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(index == viewModel.selectedPersonIndex ? Color.green : Color.gray.opacity(0.1))
                                .cornerRadius(6)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(index == viewModel.selectedPersonIndex ? Color.green : Color.gray.opacity(0.3), lineWidth: 1)
                                )
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                }
                .padding(.top, 4)
            }

            Button("Change Image") {
                viewModel.clearImage()
            }
            .buttonStyle(PlainButtonStyle())
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(6)
            .padding(.top, 8)
        }
        .padding(40)
    }

    /// Handle tap on query image to select person by bbox
    private func handleImageTap(at location: CGPoint, imageSize: CGSize, displaySize: CGSize) {
        guard !viewModel.detectedPeople.isEmpty else { return }

        // Calculate scale from original image to display size
        let scaleX = displaySize.width / imageSize.width
        let scaleY = displaySize.height / imageSize.height
        let scale = min(scaleX, scaleY)

        // Calculate centered display area
        let displayWidth = imageSize.width * scale
        let displayHeight = imageSize.height * scale
        let xOffset = (displaySize.width - displayWidth) / 2
        let yOffset = (displaySize.height - displayHeight) / 2

        // Convert tap location to image coordinates
        let imageTapX = (location.x - xOffset) / scale
        let imageTapY = (location.y - yOffset) / scale

        // Check which bbox (if any) contains the tap point
        for (index, person) in viewModel.detectedPeople.enumerated() {
            guard person.bbox.count == 4 else { continue }

            let bboxX = person.bbox[0]
            let bboxY = person.bbox[1]
            let bboxWidth = person.bbox[2]
            let bboxHeight = person.bbox[3]

            let bboxRect = CGRect(x: bboxX, y: bboxY, width: bboxWidth, height: bboxHeight)

            if bboxRect.contains(CGPoint(x: imageTapX, y: imageTapY)) {
                print("[UI] User tapped on Person \(index + 1)")
                viewModel.selectPerson(at: index)
                return
            }
        }

        print("[UI] Tap outside any bounding box")
    }
}

// MARK: - Search Parameters View
struct SearchParametersView: View {
    @ObservedObject var viewModel: PostureKitViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Search Parameters")
                .font(.system(size: 14, weight: .semibold))

            // Number-of-results control lives in ResultsGridView's header row (closer to the results themselves).

            // Min similarity slider (floor: show results at or above this value)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Min similarity")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    Spacer()
                    Text("\(Int(viewModel.minSimilarity * 100))%")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundColor(.blue)
                        .fontDesign(.monospaced)
                }

                Slider(value: $viewModel.minSimilarity, in: 0...1, step: 0.01)
            }

            // Multi-person search control
            VStack(alignment: .leading, spacing: 4) {
                Toggle("Show all people in multi-person images", isOn: $viewModel.showMultiplePeoplePerImage)
                    .font(.system(size: 13))
                Text("When enabled, search may show multiple results from the same image")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }

            // Advanced Search Parameters (collapsible)
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 12) {
                    // Feature Confidence slider
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Feature confidence")
                                .font(.system(size: 13))
                                .foregroundColor(.gray)
                            Spacer()
                            Text("\(Int(viewModel.minFeatureConfidence * 100))%")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.blue)
                                .fontDesign(.monospaced)
                        }
                        Slider(value: $viewModel.minFeatureConfidence, in: 0.0...1.0, step: 0.05)
                        Text("Higher = stricter matching on each pose dimension. 35% default handles occlusion.")
                            .font(.system(size: 11))
                            .foregroundColor(.gray.opacity(0.7))
                    }

                    Divider()

                    // Valid Overlap slider
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Min valid overlap")
                                .font(.system(size: 13))
                                .foregroundColor(.gray)
                            Spacer()
                            Text("\(Int(viewModel.minValidOverlap))/52")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.blue)
                                .fontDesign(.monospaced)
                        }
                        Slider(value: $viewModel.minValidOverlap, in: 0...52, step: 1)
                        Text("Higher = more pose features must match. 12/52 (23%) default. Increase to reduce false positives.")
                            .font(.system(size: 11))
                            .foregroundColor(.gray.opacity(0.7))
                    }

                    Divider()

                    // Min Region Confidence slider (NudeNet body part detection threshold)
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Body part confidence")
                                .font(.system(size: 13))
                                .foregroundColor(.gray)
                            Spacer()
                            Text("\(Int(viewModel.minRegionConfidence * 100))%")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundColor(.blue)
                                .fontDesign(.monospaced)
                        }
                        Slider(value: $viewModel.minRegionConfidence, in: 0.0...1.0, step: 0.05)
                        Text("Minimum confidence for NudeNet body part detection (30% recommended)")
                            .font(.system(size: 11))
                            .foregroundColor(.gray.opacity(0.7))
                    }

                }
                .padding(.top, 8)
            } label: {
                HStack {
                    Text("Advanced search settings")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(.gray.opacity(0.6))
                }
            }
            .padding(.vertical, 4)

            // Checkboxes
            HStack(spacing: 24) {
                Toggle("Include flipped poses", isOn: $viewModel.includeFlippedPoses)
                    .font(.system(size: 13))
            }

            // Body part filters (18 specific NudeNet classes) - ONLY for Browse Database mode
            if viewModel.browseMode {
                DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("18 specific classes")
                            .font(.system(size: 11))
                            .foregroundColor(.gray.opacity(0.7))
                        Spacer()
                        if !viewModel.requiredBodyParts.isEmpty {
                            Button(action: { viewModel.requiredBodyParts.removeAll() }) {
                                Text("Clear")
                                    .font(.system(size: 11))
                                    .foregroundColor(.blue)
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }

                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                        // Face (2 classes)
                        BodyPartSection(title: "Face") {
                            BodyPartToggle(label: "Male", region: "FACE_MALE", isSelected: viewModel.requiredBodyParts.contains("FACE_MALE")) {
                                toggleRegion("FACE_MALE", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Female", region: "FACE_FEMALE", isSelected: viewModel.requiredBodyParts.contains("FACE_FEMALE")) {
                                toggleRegion("FACE_FEMALE", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Torso (5 classes)
                        BodyPartSection(title: "Torso") {
                            BodyPartToggle(label: "Belly Exposed", region: "BELLY_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("BELLY_EXPOSED")) {
                                toggleRegion("BELLY_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Belly Covered", region: "BELLY_COVERED", isSelected: viewModel.requiredBodyParts.contains("BELLY_COVERED")) {
                                toggleRegion("BELLY_COVERED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Female Breast Exposed", region: "FEMALE_BREAST_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("FEMALE_BREAST_EXPOSED")) {
                                toggleRegion("FEMALE_BREAST_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Female Breast Covered", region: "FEMALE_BREAST_COVERED", isSelected: viewModel.requiredBodyParts.contains("FEMALE_BREAST_COVERED")) {
                                toggleRegion("FEMALE_BREAST_COVERED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Male Breast", region: "MALE_BREAST_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("MALE_BREAST_EXPOSED")) {
                                toggleRegion("MALE_BREAST_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Feet (2 classes)
                        BodyPartSection(title: "Feet") {
                            BodyPartToggle(label: "Exposed", region: "FEET_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("FEET_EXPOSED")) {
                                toggleRegion("FEET_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Covered", region: "FEET_COVERED", isSelected: viewModel.requiredBodyParts.contains("FEET_COVERED")) {
                                toggleRegion("FEET_COVERED", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Armpits (2 classes)
                        BodyPartSection(title: "Armpits") {
                            BodyPartToggle(label: "Exposed", region: "ARMPITS_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("ARMPITS_EXPOSED")) {
                                toggleRegion("ARMPITS_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Covered", region: "ARMPITS_COVERED", isSelected: viewModel.requiredBodyParts.contains("ARMPITS_COVERED")) {
                                toggleRegion("ARMPITS_COVERED", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Buttocks (2 classes)
                        BodyPartSection(title: "Buttocks") {
                            BodyPartToggle(label: "Exposed", region: "BUTTOCKS_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("BUTTOCKS_EXPOSED")) {
                                toggleRegion("BUTTOCKS_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Covered", region: "BUTTOCKS_COVERED", isSelected: viewModel.requiredBodyParts.contains("BUTTOCKS_COVERED")) {
                                toggleRegion("BUTTOCKS_COVERED", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Genitalia (3 classes)
                        BodyPartSection(title: "Genitalia") {
                            BodyPartToggle(label: "Female Exposed", region: "FEMALE_GENITALIA_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("FEMALE_GENITALIA_EXPOSED")) {
                                toggleRegion("FEMALE_GENITALIA_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Female Covered", region: "FEMALE_GENITALIA_COVERED", isSelected: viewModel.requiredBodyParts.contains("FEMALE_GENITALIA_COVERED")) {
                                toggleRegion("FEMALE_GENITALIA_COVERED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Male Exposed", region: "MALE_GENITALIA_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("MALE_GENITALIA_EXPOSED")) {
                                toggleRegion("MALE_GENITALIA_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                        }

                        // Anus (2 classes)
                        BodyPartSection(title: "Anus") {
                            BodyPartToggle(label: "Exposed", region: "ANUS_EXPOSED", isSelected: viewModel.requiredBodyParts.contains("ANUS_EXPOSED")) {
                                toggleRegion("ANUS_EXPOSED", in: $viewModel.requiredBodyParts)
                            }
                            BodyPartToggle(label: "Covered", region: "ANUS_COVERED", isSelected: viewModel.requiredBodyParts.contains("ANUS_COVERED")) {
                                toggleRegion("ANUS_COVERED", in: $viewModel.requiredBodyParts)
                            }
                        }
                    }
                    }
                    .frame(maxHeight: 300)
                }
                .padding(.top, 8)
            } label: {
                HStack {
                    Text("Required body parts")
                        .font(.system(size: 13))
                        .foregroundColor(.gray)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(.gray.opacity(0.6))
                }
            }
            .padding(.vertical, 4)
            }

            // Detection Sensitivity Settings - ONLY for Browse Database mode
            if viewModel.browseMode {
                DisclosureGroup(
                isExpanded: $viewModel.showThresholdSettings
            ) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Adjust detection sensitivity per category")
                            .font(.caption)
                            .foregroundColor(.gray)
                        Spacer()
                        Button("Reset Defaults") {
                            viewModel.resetThresholds()
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                        .buttonStyle(PlainButtonStyle())
                    }

                    ForEach(viewModel.bodyPartCategories, id: \.name) { category in
                        DisclosureGroup(
                            isExpanded: Binding(
                                get: { viewModel.expandedCategories.contains(category.name) },
                                set: { _ in viewModel.toggleCategory(category.name) }
                            )
                        ) {
                            VStack(spacing: 6) {
                                ForEach(category.parts, id: \.self) { part in
                                    CategoryThresholdRow(
                                        part: part,
                                        threshold: Binding(
                                            get: { viewModel.categoryThresholds[part] ?? 0.3 },
                                            set: { viewModel.categoryThresholds[part] = $0 }
                                        )
                                    )
                                }
                            }
                            .padding(.leading, 16)
                        } label: {
                            HStack {
                                Text(category.name)
                                    .font(.system(size: 12))
                                Spacer()
                                Text("\(category.parts.count)")
                                    .font(.system(size: 11))
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .padding(.vertical, 8)
            } label: {
                HStack {
                    Image(systemName: "slider.horizontal.3")
                    Text("Detection Sensitivity")
                        .font(.system(size: 13))
                    Spacer()
                }
            }
            }

            // Search/Browse button
            Button(action: {
                if viewModel.browseMode {
                    viewModel.performBrowse()
                } else {
                    viewModel.performSearch()
                }
            }) {
                HStack {
                    if viewModel.isDetecting || viewModel.isSearching {
                        SwiftUI.ProgressView()
                            .scaleEffect(0.8)
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: viewModel.browseMode ? "magnifyingglass" : "sparkle.magnifyingglass")
                    }
                    Text(viewModel.isDetecting ? "Detecting pose..." : (viewModel.isSearching ? (viewModel.browseMode ? "Browsing..." : "Searching...") : (viewModel.browseMode ? "Browse Database" : "Search Similar Poses")))
                }
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(viewModel.isDetecting || viewModel.isSearching ? Color.blue.opacity(0.7) : Color.blue)
                .cornerRadius(8)
            }
            .buttonStyle(PlainButtonStyle())
            .disabled(viewModel.isDetecting || viewModel.isSearching)
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity)
        .padding(24)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }

    // Helper function to toggle region selection
    private func toggleRegion(_ region: String, in binding: Binding<Set<String>>) {
        var regions = binding.wrappedValue
        if regions.contains(region) {
            regions.remove(region)
        } else {
            regions.insert(region)
        }
        binding.wrappedValue = regions
    }
}

// MARK: - Body Part Section Component
struct BodyPartSection<Content: View>: View {
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.gray.opacity(0.8))
                .textCase(.uppercase)

            content
        }
    }
}

// MARK: - Body Part Toggle Component
struct BodyPartToggle: View {
    let label: String
    let region: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .foregroundColor(isSelected ? .blue : .gray.opacity(0.5))
                    .font(.system(size: 13))
                Text(label)
                    .font(.system(size: 12))
                    .foregroundColor(isSelected ? .primary : .gray)
            }
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Flow Layout (Wrapping Layout for Tags)
struct FlowLayout<Content: View>: View {
    let spacing: CGFloat
    let content: () -> Content

    @State private var totalHeight: CGFloat = 0

    init(spacing: CGFloat = 4, @ViewBuilder content: @escaping () -> Content) {
        self.spacing = spacing
        self.content = content
    }

    var body: some View {
        GeometryReader { geometry in
            self.generateContent(in: geometry)
        }
        .frame(height: totalHeight)
    }

    private func generateContent(in geometry: GeometryProxy) -> some View {
        var width = CGFloat.zero
        var height = CGFloat.zero
        var lastHeight = CGFloat.zero

        return ZStack(alignment: .topLeading) {
            // Hidden content to measure sizes
            content()
                .fixedSize()
                .opacity(0)

            // Actual visible content with wrapping
            content()
                .alignmentGuide(.leading) { d in
                    if (abs(width - d.width) > geometry.size.width) {
                        width = 0
                        height -= lastHeight + spacing
                    }
                    let result = width
                    width -= d.width + spacing
                    lastHeight = d.height
                    return result
                }
                .alignmentGuide(.top) { d in
                    let result = height
                    return result
                }
        }
        .background(viewHeightReader($totalHeight))
    }

    private func viewHeightReader(_ binding: Binding<CGFloat>) -> some View {
        return GeometryReader { geometry -> Color in
            let rect = geometry.frame(in: .local)
            DispatchQueue.main.async {
                binding.wrappedValue = rect.size.height
            }
            return .clear
        }
    }
}

// MARK: - Result Frame Preference (marquee selection)
// Each result item reports its frame in the "resultsContent" coordinate space so the
// marquee drag can hit-test against visible items. Frames are in content (not viewport)
// coordinates, so they stay valid while the scroll position changes mid-drag.
private struct ResultFramePreferenceKey: PreferenceKey {
    static var defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

// Tracks where the scroll content's origin currently sits relative to the visible
// viewport (y goes negative as the user scrolls down). Needed to convert the marquee
// cursor between content and viewport coordinates for edge auto-scroll.
private struct ResultsContentOriginPreferenceKey: PreferenceKey {
    static var defaultValue: CGPoint = .zero
    static func reduce(value: inout CGPoint, nextValue: () -> CGPoint) {
        value = nextValue()
    }
}

// Resolves the AppKit NSScrollView backing the results ScrollView so edge auto-scroll
// can drive it directly. SwiftUI's ScrollViewProxy.scrollTo is unreliable when called
// repeatedly from a timer during an active drag; scrolling the clip view is not.
private struct EnclosingScrollViewFinder: NSViewRepresentable {
    let onResolve: (NSScrollView) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let scrollView = view.enclosingScrollView {
                onResolve(scrollView)
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            if let scrollView = nsView.enclosingScrollView {
                onResolve(scrollView)
            }
        }
    }
}

// MARK: - Results Grid View
struct ResultsGridView: View {
    @ObservedObject var viewModel: PostureKitViewModel
    var availableHeight: CGFloat = 800  // Window height, used to bound the scrollable results area

    // Marquee (rubber-band) selection state — all rects/points in "resultsContent" space
    @State private var resultFrames: [String: CGRect] = [:]
    @State private var marqueeRect: CGRect? = nil
    @State private var marqueeBaseSelection: Set<String>? = nil
    @State private var marqueeStart: CGPoint? = nil

    // Edge auto-scroll state
    @State private var contentOrigin: CGPoint = .zero
    @State private var lastCursorViewportPoint: CGPoint? = nil
    @State private var autoScrollDirection: AutoScrollDirection? = nil
    @State private var autoScrollTimer: Timer? = nil
    @State private var resultsScrollView: NSScrollView? = nil

    private enum AutoScrollDirection { case up, down }

    private var scrollAreaHeight: CGFloat {
        max(400, availableHeight - 300)
    }

    var columns: [GridItem] {
        // Uniform thumbnail sizing with increased spacing for better layout
        [GridItem(.adaptive(minimum: viewModel.thumbnailSize), spacing: 24)]
    }

    private var marqueeGesture: some Gesture {
        DragGesture(minimumDistance: 5, coordinateSpace: .named("resultsContent"))
            .onChanged { value in
                if marqueeBaseSelection == nil {
                    // Shift/Cmd-drag adds to the existing selection; plain drag replaces it
                    let additive = NSEvent.modifierFlags.contains(.shift)
                        || NSEvent.modifierFlags.contains(.command)
                    marqueeBaseSelection = additive ? viewModel.selectedResultIds : []
                    marqueeStart = value.startLocation
                }
                lastCursorViewportPoint = CGPoint(
                    x: value.location.x + contentOrigin.x,
                    y: value.location.y + contentOrigin.y
                )
                updateMarquee(to: value.location)
                updateAutoScroll()
            }
            .onEnded { _ in
                stopAutoScroll()
                marqueeRect = nil
                marqueeBaseSelection = nil
                marqueeStart = nil
                lastCursorViewportPoint = nil
            }
    }

    private func updateMarquee(to point: CGPoint) {
        guard let start = marqueeStart else { return }
        let rect = CGRect(
            x: min(start.x, point.x),
            y: min(start.y, point.y),
            width: abs(point.x - start.x),
            height: abs(point.y - start.y)
        )
        marqueeRect = rect
        let hits = Set(resultFrames.filter { $0.value.intersects(rect) }.map { $0.key })
        viewModel.selectedResultIds = hits.union(marqueeBaseSelection ?? [])
    }

    // MARK: Edge auto-scroll

    private var autoScrollEdgeZone: CGFloat { 36 }

    private func updateAutoScroll() {
        guard let viewportPoint = lastCursorViewportPoint else { return }
        let direction: AutoScrollDirection?
        if viewportPoint.y > scrollAreaHeight - autoScrollEdgeZone {
            direction = .down
        } else if viewportPoint.y < autoScrollEdgeZone {
            direction = .up
        } else {
            direction = nil
        }

        guard direction != autoScrollDirection else { return }
        autoScrollTimer?.invalidate()
        autoScrollTimer = nil
        autoScrollDirection = direction
        guard direction != nil else { return }

        // Register in .common run loop modes: the default mode's timers don't fire
        // while AppKit is in its mouse-drag event-tracking mode, which is exactly
        // when this timer needs to run.
        let timer = Timer(timeInterval: 0.02, repeats: true) { _ in
            autoScrollTick()
        }
        RunLoop.main.add(timer, forMode: .common)
        autoScrollTimer = timer
    }

    private func stopAutoScroll() {
        autoScrollTimer?.invalidate()
        autoScrollTimer = nil
        autoScrollDirection = nil
    }

    // Speed scales with how far past the zone boundary the cursor is — gentle at the
    // edge of the zone, fastest when dragged well beyond the viewport edge.
    private func autoScrollStep(for direction: AutoScrollDirection) -> CGFloat {
        let minStep: CGFloat = 4    // ~200 px/s at zone entry
        let maxStep: CGFloat = 60   // ~3000 px/s when 100pt past the viewport edge
        guard let viewportPoint = lastCursorViewportPoint else { return minStep }

        let depth: CGFloat
        switch direction {
        case .down:
            depth = viewportPoint.y - (scrollAreaHeight - autoScrollEdgeZone)
        case .up:
            depth = autoScrollEdgeZone - viewportPoint.y
        }
        let rampDistance = autoScrollEdgeZone + 100  // keeps accelerating past the edge
        let normalized = max(0, min(depth, rampDistance)) / rampDistance
        return minStep + normalized * normalized * (maxStep - minStep)
    }

    private func autoScrollTick() {
        guard let direction = autoScrollDirection,
              let scrollView = resultsScrollView,
              let documentView = scrollView.documentView else { return }

        let step = autoScrollStep(for: direction)
        let clipView = scrollView.contentView
        var origin = clipView.bounds.origin
        let maxOffset = max(0, documentView.frame.height - clipView.bounds.height)

        switch direction {
        case .down:
            origin.y = min(origin.y + step, maxOffset)
        case .up:
            origin.y = max(origin.y - step, 0)
        }

        guard origin != clipView.bounds.origin else { return }  // already at the end
        clipView.scroll(to: origin)
        scrollView.reflectScrolledClipView(clipView)

        // The cursor is stationary in the viewport while content scrolls underneath —
        // recompute its content-space position so the marquee keeps growing. The
        // contentOrigin preference lags this tick by a layout pass, so derive the
        // fresh offset directly from the clip view.
        if let viewportPoint = lastCursorViewportPoint {
            let contentPoint = CGPoint(
                x: viewportPoint.x - contentOrigin.x,
                y: viewportPoint.y + origin.y
            )
            updateMarquee(to: contentPoint)
        }
    }

    private func resultFrameReader(for id: String) -> some View {
        GeometryReader { geo in
            Color.clear.preference(
                key: ResultFramePreferenceKey.self,
                value: [id: geo.frame(in: .named("resultsContent"))]
            )
        }
    }

    private var contentOriginReader: some View {
        GeometryReader { geo in
            Color.clear.preference(
                key: ResultsContentOriginPreferenceKey.self,
                value: geo.frame(in: .named("resultsViewport")).origin
            )
        }
    }

    @ViewBuilder
    private var marqueeOverlay: some View {
        if let rect = marqueeRect {
            Rectangle()
                .fill(Color.accentColor.opacity(0.15))
                .overlay(Rectangle().stroke(Color.accentColor.opacity(0.7), lineWidth: 1))
                .frame(width: rect.width, height: rect.height)
                .offset(x: rect.minX, y: rect.minY)
                .allowsHitTesting(false)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Results header - make flexible
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Results (\(viewModel.totalResultCount) found)")
                        .font(.system(size: 14, weight: .semibold))

                    if !viewModel.selectedResultIds.isEmpty {
                        Text("• \(viewModel.selectedResultIds.count) selected")
                            .font(.system(size: 13))
                            .foregroundColor(.blue)

                        Button("Clear Selection") {
                            viewModel.clearSelection()
                        }
                        .buttonStyle(PlainButtonStyle())
                        .font(.system(size: 12))
                        .foregroundColor(.gray)
                    }

                    Spacer()
                }

                // Controls row - wraps better on smaller screens
                HStack(spacing: 12) {
                    // View mode toggle
                    Picker("", selection: $viewModel.viewMode) {
                        Label("Grid", systemImage: "square.grid.2x2").tag(ViewMode.grid)
                        Label("List", systemImage: "list.bullet").tag(ViewMode.list)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(width: 140)

                    // Page-size dropdown (per-page count; the threshold caps the full set)
                    Menu {
                        ForEach([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], id: \.self) { count in
                            Button("\(count)") {
                                viewModel.numberOfResults = count
                                viewModel.numberOfResultsDouble = Double(count)
                                viewModel.showAllResults = false
                            }
                        }
                        Divider()
                        Button("All") {
                            viewModel.showAllResults = true
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text("Show:")
                                .foregroundColor(.gray)
                            Text(viewModel.showAllResults ? "All" : "\(viewModel.numberOfResults)")
                                .fontWeight(.semibold)
                        }
                        .font(.system(size: 12))
                    }
                    .menuStyle(BorderlessButtonMenuStyle())
                    .fixedSize()

                    // Pagination controls — page through the result set (Swift-side, no re-search)
                    if !viewModel.showAllResults && viewModel.totalPages > 1 {
                        HStack(spacing: 6) {
                            Button(action: { viewModel.goToPreviousPage() }) {
                                Image(systemName: "chevron.left")
                            }
                            .buttonStyle(PlainButtonStyle())
                            .disabled(viewModel.currentPage <= 1)

                            Text("Page \(viewModel.currentPage) of \(viewModel.totalPages)")
                                .foregroundColor(.gray)
                                .monospacedDigit()

                            Button(action: { viewModel.goToNextPage() }) {
                                Image(systemName: "chevron.right")
                            }
                            .buttonStyle(PlainButtonStyle())
                            .disabled(viewModel.currentPage >= viewModel.totalPages)
                        }
                        .font(.system(size: 12))
                        .fixedSize()
                    }

                    // Thumbnail size slider
                    HStack(spacing: 6) {
                        Image(systemName: "photo")
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                        Slider(value: $viewModel.thumbnailSize, in: 120...300, step: 20)
                            .frame(minWidth: 80, maxWidth: 120)
                        Image(systemName: "photo")
                            .font(.system(size: 14))
                            .foregroundColor(.gray)
                    }

                    Spacer()

                    if !viewModel.selectedResultIds.isEmpty {
                        Button("Move Selected...") {
                            moveSelectedFiles()
                        }
                        .buttonStyle(PlainButtonStyle())
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue)
                        .cornerRadius(6)
                    }
                }
            }
            .padding(.horizontal, 24)

            // Dedicated, height-bounded scroll area for the results so they scroll with a real
            // scrollbar (and don't force the whole page to scroll). Both grid and list share it,
            // so list mode no longer collapses a nested scroll view inside the page scroll.
            // The box's horizontal padding lives INSIDE the scroll content so a marquee drag
            // can start from the box's left/right gutters; the content also fills the full
            // viewport height so drags can start below the last row.
            ScrollView {
                resultsContent
                    .padding(.horizontal, 24)
                    .padding(.bottom, 24)
                    .frame(maxWidth: .infinity, minHeight: scrollAreaHeight, alignment: .topLeading)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        // Click on empty space (gutters, gaps, below the grid) clears the selection
                        viewModel.clearSelection()
                    }
                    .simultaneousGesture(marqueeGesture)
                    .coordinateSpace(name: "resultsContent")
                    .background(contentOriginReader)
                    .background(EnclosingScrollViewFinder { resultsScrollView = $0 })
                    // Overlay (not a layout child): the marquee draws on top without
                    // affecting layout, so dragging past the box edges can't stretch
                    // the content horizontally or add phantom scroll space vertically.
                    .overlay(marqueeOverlay, alignment: .topLeading)
            }
            .coordinateSpace(name: "resultsViewport")
            .onPreferenceChange(ResultFramePreferenceKey.self) { resultFrames = $0 }
            .onPreferenceChange(ResultsContentOriginPreferenceKey.self) { contentOrigin = $0 }
            .onDisappear { stopAutoScroll() }
            .frame(height: scrollAreaHeight)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 24)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }

    @ViewBuilder
    private var resultsContent: some View {
        if viewModel.viewMode == .grid {
            LazyVGrid(columns: columns, spacing: 24) {
                ForEach(Array(viewModel.searchResults.enumerated()), id: \.element.id) { index, result in
                    ResultCardView(
                        result: result,
                        index: index,
                        viewModel: viewModel
                    )
                    .background(resultFrameReader(for: result.id))
                }
            }
        } else {
            LazyVStack(spacing: 1) {
                ForEach(Array(viewModel.searchResults.enumerated()), id: \.element.id) { index, result in
                    ResultListItemView(
                        result: result,
                        index: index,
                        viewModel: viewModel
                    )
                    .background(resultFrameReader(for: result.id))
                }
            }
        }
    }

    private func moveSelectedFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Move"
        panel.message = "Select destination directory for selected files"

        if panel.runModal() == .OK, let url = panel.url {
            let result = viewModel.moveSelectedFiles(to: url.path)

            var details = "Successfully moved \(result.success) file(s)."
            if result.skipped > 0 {
                details += "\nSkipped \(result.skipped) file(s) already in that folder."
            }
            if result.failed > 0 {
                details += "\nFailed to move \(result.failed) file(s)."
            }

            let alert = NSAlert()
            alert.messageText = "Files Moved"
            alert.informativeText = details
            alert.alertStyle = result.failed > 0 ? .warning : .informational
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }
}

// MARK: - Result Context Menu (selection-aware right-click actions)
// Acts on the full selection when the right-clicked item is part of it (Finder-style);
// otherwise acts on just the clicked item.
struct ResultContextMenu: View {
    let result: SearchResult
    @ObservedObject var viewModel: PostureKitViewModel

    private var targets: [SearchResult] {
        if viewModel.selectedResultIds.contains(result.id) && viewModel.selectedResultIds.count > 1 {
            return viewModel.searchResults.filter { viewModel.selectedResultIds.contains($0.id) }
        }
        return [result]
    }

    var body: some View {
        let items = targets
        let count = items.count

        Button(count > 1 ? "Open \(count) Items" : "Open") {
            for path in items.compactMap(\.imagePath) {
                NSWorkspace.shared.open(URL(fileURLWithPath: path))
            }
        }
        Button(count > 1 ? "Reveal \(count) Items in Finder" : "Reveal in Finder") {
            let urls = items.compactMap(\.imagePath).map { URL(fileURLWithPath: $0) }
            if !urls.isEmpty {
                NSWorkspace.shared.activateFileViewerSelecting(urls)
            }
        }
        Button(count > 1 ? "Copy \(count) Paths" : "Copy Path") {
            let paths = items.compactMap(\.imagePath)
            if !paths.isEmpty {
                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(paths.joined(separator: "\n"), forType: .string)
            }
        }

        Divider()

        Button(count > 1 ? "Move \(count) Items to..." : "Move to...") {
            moveItems(items)
        }

        Divider()

        Button("Select All") {
            viewModel.selectAll()
        }
        if !viewModel.selectedResultIds.isEmpty {
            Button("Deselect All") {
                viewModel.clearSelection()
            }
        }
    }

    private func moveItems(_ items: [SearchResult]) {
        // moveSelectedFiles operates on the view model's selection — sync it to the
        // right-clicked targets first (right-click implies selection, as in Finder).
        viewModel.selectedResultIds = Set(items.map { $0.id })

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Move"
        panel.message = "Select destination directory for selected files"

        if panel.runModal() == .OK, let url = panel.url {
            let result = viewModel.moveSelectedFiles(to: url.path)

            var details = "Successfully moved \(result.success) file(s)."
            if result.skipped > 0 {
                details += "\nSkipped \(result.skipped) file(s) already in that folder."
            }
            if result.failed > 0 {
                details += "\nFailed to move \(result.failed) file(s)."
            }

            let alert = NSAlert()
            alert.messageText = "Files Moved"
            alert.informativeText = details
            alert.alertStyle = result.failed > 0 ? .warning : .informational
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }
}

// MARK: - Result Card View
struct ResultCardView: View {
    let result: SearchResult
    let index: Int
    @ObservedObject var viewModel: PostureKitViewModel
    @State private var isHovered = false
    @State private var thumbnail: NSImage?

    private var isSelected: Bool {
        viewModel.selectedResultIds.contains(result.id)
    }

    private var thumbnailDisplaySize: CGFloat {
        viewModel.thumbnailSize
    }

    private var matchTier: MatchTier {
        MatchTier(similarity: result.similarity)
    }

    private var thumbnailView: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.gray.opacity(0.1))

            if let thumbnail = thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .scaledToFit()
            } else {
                Image(systemName: "photo")
                    .font(.system(size: thumbnailDisplaySize * 0.3))
                    .foregroundColor(.gray.opacity(0.3))
            }

            // Skeleton overlay
            if let keypoints = result.keypoints, !keypoints.isEmpty {
                Canvas { context, size in
                    // Calculate original image size for proper skeleton overlay positioning
                    let originalSize: CGSize?
                    if let width = result.imageWidth, let height = result.imageHeight {
                        originalSize = CGSize(width: CGFloat(width), height: CGFloat(height))
                    } else {
                        originalSize = nil
                    }
                    
                    drawSkeleton(
                        context: context,
                        size: size,
                        keypoints: keypoints,
                        detail: matchTier.skeletonDetail,
                        alreadyNormalized: false,
                        isSelected: true,
                        originalImageSize: originalSize
                    )
                }
            }

            // Bounding box overlay (multi-person support)
            if let bbox = result.bbox,
               let imageWidth = result.imageWidth,
               let imageHeight = result.imageHeight,
               bbox.count == 4 {
                Canvas { context, size in
                    // Bbox is [x_min, y_min, x_max, y_max] in image coordinates
                    let xMin = bbox[0]
                    let yMin = bbox[1]
                    let xMax = bbox[2]
                    let yMax = bbox[3]

                    // Calculate scale to fit image within 200x200 thumbnail
                    let scale = min(200.0 / Double(imageWidth), 200.0 / Double(imageHeight))
                    let scaledWidth = Double(imageWidth) * scale
                    let scaledHeight = Double(imageHeight) * scale

                    // Calculate padding offsets (center image on canvas)
                    let xOffset = (200.0 - scaledWidth) / 2
                    let yOffset = (200.0 - scaledHeight) / 2

                    // Convert bbox to thumbnail coordinates
                    let bboxX = xMin * scale + xOffset
                    let bboxY = yMin * scale + yOffset
                    let bboxWidth = (xMax - xMin) * scale
                    let bboxHeight = (yMax - yMin) * scale

                    // Draw bbox rectangle
                    let rect = CGRect(x: bboxX, y: bboxY, width: bboxWidth, height: bboxHeight)
                    let path = Path(roundedRect: rect, cornerRadius: 2)

                    context.stroke(
                        path,
                        with: .color(.green),
                        lineWidth: 2
                    )

                    // Draw person_id badge if available
                    if let personId = result.personId {
                        let badgeText = "P\(personId)"
                        context.draw(
                            Text(badgeText)
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.white),
                            at: CGPoint(x: bboxX + 12, y: bboxY + 8),
                            anchor: .center
                        )
                    }
                }
            }

            // Badges (similarity + flipped indicator)
            VStack(alignment: .leading, spacing: 4) {
                Text("\(result.similarity)%")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(similarityColor(result.similarity))
                    .cornerRadius(4)

                if result.isFlipped {
                    Text("FLIPPED")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.purple.opacity(0.8))
                        .cornerRadius(3)
                }
            }
            .padding(8)
        }
        .frame(width: thumbnailDisplaySize, height: thumbnailDisplaySize)
        .clipped()
    }

    private var infoView: some View {
        VStack(alignment: .leading, spacing: 6) {
                Text(result.filename)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(2)

                // Progressive disclosure: show different detail levels based on match quality
                if matchTier == .possible {
                    // Minimal info: similarity + confidence only
                    Text("Similarity: \(result.similarity)% | Det: \(String(format: "%.2f", result.confidence))")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                } else {
                    // Standard/Premium: show full metadata
                    Text("Similarity: \(result.similarity)% | Det: \(String(format: "%.2f", result.confidence))")
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                }

                // Body part badges: Premium shows detailed, Good shows canonical, Possible shows none
                if matchTier == .premium, let detailedRegions = result.visibleRegionsDetailed, !detailedRegions.isEmpty {
                    // Display detailed NudeNet classes (18 categories)
                    let sortedRegions = detailedRegions.sorted { (a, b) in
                        let aRegion = a["canonical_region"] as? String ?? ""
                        let bRegion = b["canonical_region"] as? String ?? ""
                        return aRegion < bRegion
                    }

                    // Wrapping layout for tags
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 80), spacing: 4)], alignment: .leading, spacing: 4) {
                        ForEach(Array(sortedRegions.enumerated()), id: \.offset) { _, part in
                            if let partName = part["part_name"] as? String,
                               let confidence = part["confidence"] as? Double,
                               let isExposed = part["is_exposed"] as? Bool {

                                let displayName = partName
                                    .replacingOccurrences(of: "_", with: " ")
                                    .lowercased()
                                    .capitalized

                                Text("\(displayName) (\(String(format: "%.0f", confidence * 100))%)")
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(isExposed ? Color.red.opacity(0.7) : Color.blue.opacity(0.7))
                                    .cornerRadius(3)
                                    .lineLimit(1)
                            }
                        }
                    }
                } else if matchTier == .good, let visibleRegions = result.visibleRegions, !visibleRegions.isEmpty {
                    // Good tier: Show canonical regions (7 categories)
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 70), spacing: 4)], alignment: .leading, spacing: 4) {
                        ForEach(visibleRegions.sorted(), id: \.self) { region in
                            Text(region.capitalized)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(regionColor(region))
                                .cornerRadius(3)
                                .lineLimit(1)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(isHovered ? Color.gray.opacity(0.05) : Color.clear)
    }

    var body: some View {
        VStack(spacing: 0) {
            thumbnailView
            infoView
        }
        .background(isSelected ? Color.blue.opacity(0.1) : Color(NSColor.windowBackgroundColor))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.blue : Color.gray.opacity(0.2), lineWidth: isSelected ? 2 : 1)
        )
        .shadow(color: isHovered ? Color.black.opacity(0.1) : Color.clear, radius: 8, y: 4)
        .scaleEffect(isHovered ? 1.02 : 1.0)
        .animation(.easeInOut(duration: 0.2), value: isHovered)
        .opacity(matchTier.opacity)  // Progressive disclosure: reduce opacity for low-confidence matches
        .onHover { hovering in
            isHovered = hovering
        }
        .simultaneousGesture(
            TapGesture()
                .modifiers(.command)
                .onEnded {
                    viewModel.toggleSelection(result.id, at: index, withCommandKey: true, withShiftKey: false)
                }
        )
        .simultaneousGesture(
            TapGesture()
                .modifiers(.shift)
                .onEnded {
                    viewModel.toggleSelection(result.id, at: index, withCommandKey: false, withShiftKey: true)
                }
        )
        .onTapGesture {
            // Normal click (no modifiers)
            viewModel.toggleSelection(result.id, at: index, withCommandKey: false, withShiftKey: false)
        }
        .contextMenu {
            ResultContextMenu(result: result, viewModel: viewModel)
        }
        .onAppear {
            loadThumbnail()
        }
    }

    private func loadThumbnail() {
        // Use pre-loaded thumbnail from database if available
        if let thumbnailData = result.thumbnailData,
           let preloadedThumbnail = NSImage(data: thumbnailData) {
            self.thumbnail = preloadedThumbnail
            return
        }

        // Fallback: Load from file if thumbnail not in database (legacy support)
        guard let imagePath = result.imagePath else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            guard let image = NSImage(contentsOfFile: imagePath) else { return }

            // Create thumbnail
            let thumbnailSize = NSSize(width: 200, height: 200)
            let thumb = NSImage(size: thumbnailSize)

            thumb.lockFocus()
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
            thumb.unlockFocus()

            DispatchQueue.main.async {
                self.thumbnail = thumb
            }
        }
    }

    private func similarityColor(_ score: Int) -> Color {
        if score >= 90 { return Color.green }
        if score >= 70 { return Color.yellow }
        return Color.orange
    }

    private func regionColor(_ region: String) -> Color {
        switch region.lowercased() {
        case "face": return Color.blue.opacity(0.8)
        case "torso": return Color.green.opacity(0.8)
        case "feet": return Color.purple.opacity(0.8)
        case "armpits": return Color.orange.opacity(0.8)
        case "hands": return Color.pink.opacity(0.8)
        case "buttocks": return Color.brown.opacity(0.8)
        case "genitalia": return Color.red.opacity(0.8)
        case "anus": return Color.gray.opacity(0.8)
        default: return Color.gray.opacity(0.6)
        }
    }

}

// MARK: - Result List Item View
struct ResultListItemView: View {
    let result: SearchResult
    let index: Int
    @ObservedObject var viewModel: PostureKitViewModel
    @State private var isHovered = false
    @State private var thumbnail: NSImage?

    private var isSelected: Bool {
        viewModel.selectedResultIds.contains(result.id)
    }

    private var thumbnailDisplaySize: CGFloat {
        min(viewModel.thumbnailSize * 0.6, 80)
    }

    var body: some View {
        HStack(spacing: 12) {
            // Thumbnail
            ZStack {
                Rectangle()
                    .fill(Color.gray.opacity(0.1))
                    .frame(width: thumbnailDisplaySize, height: thumbnailDisplaySize)
                    .cornerRadius(6)

                if let thumbnail = thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .scaledToFit()
                        .frame(width: thumbnailDisplaySize, height: thumbnailDisplaySize)
                        .cornerRadius(6)
                } else {
                    Image(systemName: "photo")
                        .font(.system(size: thumbnailDisplaySize * 0.4))
                        .foregroundColor(.gray.opacity(0.3))
                }
            }

            // Metadata
            VStack(alignment: .leading, spacing: 4) {
                Text(result.filename)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)

                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Text("Similarity:")
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                        Text("\(result.similarity)%")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(similarityColor(result.similarity))
                    }

                    if let width = result.imageWidth, let height = result.imageHeight {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.system(size: 9))
                                .foregroundColor(.gray)
                            Text("\(width)×\(height)")
                                .font(.system(size: 11))
                                .foregroundColor(.gray)
                        }
                    }

                    if let fileSize = result.fileSize {
                        HStack(spacing: 4) {
                            Image(systemName: "doc")
                                .font(.system(size: 9))
                                .foregroundColor(.gray)
                            Text(formatFileSize(fileSize))
                                .font(.system(size: 11))
                                .foregroundColor(.gray)
                        }
                    }

                    HStack(spacing: 4) {
                        Text("Det:")
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                        Text(String(format: "%.2f", result.confidence))
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                    }
                }

                if let imagePath = result.imagePath {
                    Text(imagePath)
                        .font(.system(size: 10))
                        .foregroundColor(.gray.opacity(0.7))
                        .lineLimit(1)
                }
            }

            Spacer()
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(isSelected ? Color.blue.opacity(0.1) : (isHovered ? Color.gray.opacity(0.05) : Color.clear))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Color.gray.opacity(0.1)),
            alignment: .bottom
        )
        .onHover { hovering in
            isHovered = hovering
        }
        .simultaneousGesture(
            TapGesture()
                .modifiers(.command)
                .onEnded {
                    viewModel.toggleSelection(result.id, at: index, withCommandKey: true, withShiftKey: false)
                }
        )
        .simultaneousGesture(
            TapGesture()
                .modifiers(.shift)
                .onEnded {
                    viewModel.toggleSelection(result.id, at: index, withCommandKey: false, withShiftKey: true)
                }
        )
        .onTapGesture {
            viewModel.toggleSelection(result.id, at: index, withCommandKey: false, withShiftKey: false)
        }
        .contextMenu {
            ResultContextMenu(result: result, viewModel: viewModel)
        }
        .onAppear {
            loadThumbnail()
        }
    }

    private func loadThumbnail() {
        // Use pre-loaded thumbnail from database if available
        if let thumbnailData = result.thumbnailData,
           let preloadedThumbnail = NSImage(data: thumbnailData) {
            self.thumbnail = preloadedThumbnail
            return
        }

        // Fallback: Load from file if thumbnail not in database (legacy support)
        guard let imagePath = result.imagePath else { return }

        DispatchQueue.global(qos: .userInitiated).async {
            guard let image = NSImage(contentsOfFile: imagePath) else { return }

            let thumbnailSize = NSSize(width: 200, height: 200)
            let thumb = NSImage(size: thumbnailSize)

            thumb.lockFocus()
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
            thumb.unlockFocus()

            DispatchQueue.main.async {
                self.thumbnail = thumb
            }
        }
    }

    private func similarityColor(_ score: Int) -> Color {
        if score >= 90 { return Color.green }
        if score >= 70 { return Color.yellow }
        return Color.orange
    }

    private func formatFileSize(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024
        if kb < 1024 {
            return String(format: "%.1f KB", kb)
        }
        let mb = kb / 1024
        return String(format: "%.1f MB", mb)
    }

}

// MARK: - Searching Overlay View
struct SearchingOverlayView: View {
    @ObservedObject var viewModel: PostureKitViewModel

    var elapsedText: String {
        let elapsed = viewModel.searchElapsed
        if elapsed < 1.0 {
            return "Just started..."
        } else if elapsed < 60.0 {
            return String(format: "%.1fs elapsed", elapsed)
        } else {
            let minutes = Int(elapsed / 60)
            let seconds = Int(elapsed.truncatingRemainder(dividingBy: 60))
            return "\(minutes)m \(seconds)s elapsed"
        }
    }

    var statusText: String {
        let elapsed = viewModel.searchElapsed
        if elapsed < 2.0 {
            return "Loading index from disk..."
        } else if elapsed < 10.0 {
            return "Searching \(viewModel.totalPosesIndexed > 0 ? "\(viewModel.totalPosesIndexed)" : "50,000+") poses..."
        } else if elapsed < 60.0 {
            return "Comparing geometric features..."
        } else {
            return "Still searching (large index or first-time build)..."
        }
    }

    var body: some View {
        VStack(spacing: 24) {
            // Animated spinner
            SwiftUI.ProgressView()
                .scaleEffect(1.5)
                .progressViewStyle(CircularProgressViewStyle(tint: .blue))

            // Status text
            VStack(spacing: 8) {
                Text("Searching Similar Poses")
                    .font(.system(size: 16, weight: .semibold))

                Text(statusText)
                    .font(.system(size: 13))
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)

                Text(elapsedText)
                    .font(.system(size: 12))
                    .foregroundColor(.gray.opacity(0.7))
                    .fontDesign(.monospaced)
            }

            // Cancel button
            Button(action: { viewModel.cancelSearch() }) {
                HStack(spacing: 6) {
                    Image(systemName: "xmark.circle.fill")
                    Text("Cancel Search")
                }
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.red)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 50)
        .padding(.horizontal, 40)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.blue.opacity(0.3), lineWidth: 2)
        )
        .shadow(color: Color.black.opacity(0.05), radius: 8, y: 4)
    }
}

// MARK: - No Results View
struct NoResultsView: View {
    var body: some View {
        VStack(spacing: 24) {
            // Icon
            ZStack {
                Circle()
                    .fill(Color.gray.opacity(0.1))
                    .frame(width: 80, height: 80)

                Image(systemName: "magnifyingglass")
                    .font(.system(size: 32, weight: .light))
                    .foregroundColor(.gray)
            }

            // Message
            VStack(spacing: 8) {
                Text("No Similar Poses Found")
                    .font(.system(size: 16, weight: .semibold))

                Text("Try adjusting the minimum similarity threshold or search parameters")
                    .font(.system(size: 13))
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
        .padding(.horizontal, 40)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Status Bar View
struct StatusBarView: View {
    @ObservedObject var viewModel: PostureKitViewModel

    var formattedPoseCount: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: viewModel.totalPosesIndexed)) ?? "\(viewModel.totalPosesIndexed)"
    }

    var indexStatusText: String {
        switch viewModel.indexStatus {
        case .loading:
            return "Loading index..."
        case .ready:
            return viewModel.totalPosesIndexed > 0
                ? "Index ready - \(formattedPoseCount) poses"
                : "Index ready - empty"
        case .building:
            return "Building index..."
        case .error:
            return "Index error"
        }
    }

    var body: some View {
        HStack {
            HStack(spacing: 8) {
                Circle()
                    .fill(viewModel.indexStatus.color)
                    .frame(width: 8, height: 8)
                Text(indexStatusText)
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }

            Spacer()

            if viewModel.searchTime > 0 {
                Text("Search: \(String(format: "%.2f", viewModel.searchTime))s")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.05))
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(Color.gray.opacity(0.2)),
            alignment: .top
        )
    }
}

// MARK: - Keyboard Monitor (for Quick Look)
struct KeyboardMonitor: NSViewRepresentable {
    @ObservedObject var viewModel: PostureKitViewModel

    func makeNSView(context: Context) -> KeyEventView {
        let view = KeyEventView()
        view.onSpaceBar = { [weak viewModel] in
            guard let viewModel = viewModel else { return }

            // Get selected result file paths
            let selectedPaths = viewModel.searchResults
                .filter { viewModel.selectedResultIds.contains($0.id) }
                .compactMap { $0.imagePath }

            if !selectedPaths.isEmpty {
                print("[Keyboard] Space bar pressed - showing Quick Look for \(selectedPaths.count) file(s)")
                QuickLookHelper.shared.showPreview(for: selectedPaths)
            }
        }
        return view
    }

    func updateNSView(_ nsView: KeyEventView, context: Context) {
        // No updates needed
    }

    class KeyEventView: NSView {
        var onSpaceBar: (() -> Void)?

        override var acceptsFirstResponder: Bool { true }

        override func keyDown(with event: NSEvent) {
            if event.keyCode == 49 { // Space bar
                onSpaceBar?()
            } else {
                super.keyDown(with: event)
            }
        }
    }
}

// MARK: - Category Threshold Row
struct CategoryThresholdRow: View {
    let part: String
    @Binding var threshold: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(formatPartName(part))
                    .font(.system(size: 11))
                Spacer()
                Text("\(Int(threshold * 100))%")
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
                    .frame(width: 35, alignment: .trailing)
            }

            Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                .frame(height: 16)
        }
        .padding(.vertical, 2)
    }

    func formatPartName(_ part: String) -> String {
        part.split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .frame(width: 1000, height: 800)
    }
}
