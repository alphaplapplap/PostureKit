//
//  PostureKitApp.swift
//  PostureKit
//
//  Created by Linux Babe on 2025-10-09.
//

import SwiftUI
import Foundation
import AppKit

// MARK: - App Delegate for lifecycle management
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("Application launched - preloading FAISS index...")

        // Preload FAISS index in background to avoid blocking UI
        DispatchQueue.global(qos: .userInitiated).async {
            let indexLoaded = PythonBridgeSubprocess.shared.preloadIndex()

            DispatchQueue.main.async {
                if indexLoaded {
                    print("FAISS index preloaded successfully")
                } else {
                    print("No existing FAISS index - will build on first search")
                }
                // Always post notification so status bar leaves the .loading state.
                // Empty-index case displays as "Index ready - empty"; index will
                // build automatically on first search (120s timeout).
                NotificationCenter.default.post(name: .indexPreloaded, object: nil)
            }
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Check if any modal dialogs (file pickers, alerts, etc.) are open
        if NSApp.modalWindow != nil {
            print("[LIFECYCLE] ⚠️  Modal dialog is open - cancelling termination (press ⌘Q after closing dialog)")
            NSSound.beep() // Alert user that quit is blocked
            return .terminateCancel
        }

        print("[LIFECYCLE] ⚠️  applicationShouldTerminate called - proceeding with quit")
        print("[LIFECYCLE] Stack trace:")
        Thread.callStackSymbols.forEach { print("  \($0)") }
        return .terminateNow
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        print("[LIFECYCLE] Window closed - should app quit? Returning false (keeping app running)")
        return false  // Keep app running when window closes (prevents accidental quit)
    }

    func applicationWillTerminate(_ notification: Notification) {
        print("[LIFECYCLE] applicationWillTerminate - App is definitively quitting now")
        print("Application terminating - initiating cleanup...")

        // Cleanup Python bridge resources
        PythonBridgeSubprocess.shared.shutdown()

        print("Application cleanup complete")
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let indexPreloaded = Notification.Name("indexPreloaded")
    static let folderExcluded = Notification.Name("folderExcluded")  // object: excluded folder path (String)
    static let indexedFoldersChanged = Notification.Name("indexedFoldersChanged")
}

@main
struct PostureKitApp: App {
    // Connect app delegate for lifecycle events
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        // Suppress NSXPCDecoder warnings from macOS Input Method Kit
        // These are Apple framework issues, not PostureKit code
        // Will be fixed when Apple updates IMK to use proper NSSecureCoding
        if NSClassFromString("OS_os_log") != nil {
            // Reduce OS logging verbosity for system frameworks
            UserDefaults.standard.set(false, forKey: "NSConstraintBasedLayoutVisualizeMutuallyExclusiveConstraints")
        }

        // Register default values for UserDefaults
        // These are used as fallbacks when keys don't exist yet (first launch)
        // Does NOT overwrite existing user preferences
        UserDefaults.standard.register(defaults: [
            "useGPU": true,              // MPS on Apple Silicon (modern PyTorch)
            "detectionThreads": 16,      // M5 Max: 6 Super + 12 Performance cores
            "poseModel": "ensemble",     // RTMW-L + RTMW-X for best accuracy
            "fusionMethod": "confidence_weighted",
            "useTwoStage": true,         // YOLO → RTMPose for +5–10% accuracy
            "theme": "auto",
            "resultsPerPage": 20
        ])

        // Set environment variable for Python bridge to read two-stage detection setting
        let useTwoStage = UserDefaults.standard.bool(forKey: "useTwoStage")
        setenv("USE_TWO_STAGE_DETECTION", useTwoStage ? "true" : "false", 1)
        print("[APP INIT] Set USE_TWO_STAGE_DETECTION=\(useTwoStage ? "true" : "false")")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 900, minHeight: 700)
        }
        .windowStyle(.hiddenTitleBar)
        // Removed .windowResizability(.contentSize) - incompatible with dynamic content
        // Was causing window to close when UI updated after pose detection
    }
}

// Alternative with traditional title bar:
// Uncomment this and comment out the above if you prefer a standard macOS window

/*
@main
struct PostureKitApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}
*/
