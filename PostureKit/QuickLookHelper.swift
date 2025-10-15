//
//  QuickLookHelper.swift
//  PostureKit
//
//  Created by Claude Code
//

import Foundation
import Quartz
import AppKit

/// Helper class to manage Quick Look preview panel (space bar preview like Finder)
class QuickLookHelper: NSObject, QLPreviewPanelDataSource, QLPreviewPanelDelegate {
    static let shared = QuickLookHelper()

    private var previewURLs: [URL] = []
    private var currentIndex: Int = 0

    private override init() {
        super.init()
    }

    /// Show Quick Look preview for the given file paths
    func showPreview(for filePaths: [String], startingAt index: Int = 0) {
        // Convert file paths to URLs
        previewURLs = filePaths.compactMap { path in
            let url = URL(fileURLWithPath: path)
            // Verify file exists
            return FileManager.default.fileExists(atPath: path) ? url : nil
        }

        guard !previewURLs.isEmpty else {
            print("[Quick Look] No valid files to preview")
            return
        }

        // Set starting index
        currentIndex = min(max(0, index), previewURLs.count - 1)

        // Get the preview panel
        guard let panel = QLPreviewPanel.shared() else {
            print("[Quick Look] Failed to get preview panel")
            return
        }

        // Set ourselves as data source and delegate
        panel.dataSource = self
        panel.delegate = self

        // Set current preview index
        panel.currentPreviewItemIndex = currentIndex

        // Show the panel
        if !panel.isVisible {
            panel.makeKeyAndOrderFront(nil)
        }

        print("[Quick Look] Showing preview for \(previewURLs.count) file(s), starting at index \(currentIndex)")
    }

    /// Toggle Quick Look panel visibility
    func togglePreview() {
        guard let panel = QLPreviewPanel.shared() else { return }

        if panel.isVisible {
            panel.orderOut(nil)
        } else if !previewURLs.isEmpty {
            panel.dataSource = self
            panel.delegate = self
            panel.currentPreviewItemIndex = currentIndex
            panel.makeKeyAndOrderFront(nil)
        }
    }

    /// Close the Quick Look panel
    func closePreview() {
        guard let panel = QLPreviewPanel.shared() else { return }

        if panel.isVisible {
            panel.orderOut(nil)
        }
    }

    // MARK: - QLPreviewPanelDataSource

    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int {
        return previewURLs.count
    }

    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> QLPreviewItem! {
        guard index >= 0 && index < previewURLs.count else { return nil }
        return previewURLs[index] as QLPreviewItem
    }

    // MARK: - QLPreviewPanelDelegate

    func previewPanel(_ panel: QLPreviewPanel!, handle event: NSEvent!) -> Bool {
        // Handle arrow keys for navigation
        if event.type == .keyDown {
            switch event.keyCode {
            case 123: // Left arrow
                if panel.currentPreviewItemIndex > 0 {
                    panel.currentPreviewItemIndex -= 1
                    return true
                }
            case 124: // Right arrow
                if panel.currentPreviewItemIndex < previewURLs.count - 1 {
                    panel.currentPreviewItemIndex += 1
                    return true
                }
            case 49: // Space bar
                // Toggle panel
                panel.orderOut(nil)
                return true
            case 53: // Escape
                panel.orderOut(nil)
                return true
            default:
                break
            }
        }
        return false
    }

    func previewPanel(_ panel: QLPreviewPanel!, sourceFrameOnScreenFor item: QLPreviewItem!) -> NSRect {
        // Return zero rect to use default animation
        return NSRect.zero
    }
}
