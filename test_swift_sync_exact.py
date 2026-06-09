#!/usr/bin/env python3
"""
Test that mimics EXACTLY what Swift runs for database sync.
This is the exact script that PythonBridgeSubprocess.swift executes.
"""
import sys
from pathlib import Path

# Mimic Swift's sys.path setup
venvSitePackages = '/Users/linuxbabe/Hardware-Aware/PostureKit/venv/lib/python3.9/site-packages'
projectPath = '/Users/linuxbabe/Hardware-Aware/PostureKit'
sys.path.insert(0, venvSitePackages)
sys.path.insert(0, projectPath)

from src.swift_bridge import PostureKitBridge
import json

bridge = PostureKitBridge(skip_models=True)
result = bridge.synchronize_database()
print(json.dumps(result))
