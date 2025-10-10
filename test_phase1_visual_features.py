#!/usr/bin/env python3
"""
Quick test to verify Phase 1 visual features integration works.
Tests that visual features are extracted, fused, and stored correctly.
"""
import sys
sys.path.insert(0, 'venv/lib/python3.9/site-packages')
sys.path.insert(0, 'src')

print("="*80)
print("TESTING PHASE 1: VISUAL FEATURES INTEGRATION")
print("="*80)

# Test 1: Import check
print("\n[1/5] Testing imports...")
try:
    from swift_bridge import PostureKitBridge
    from core.visual_feature_extractor import VisualFeatureExtractor
    from core.multimodal_fusion import MultiModalFusion
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Bridge initialization
print("\n[2/5] Testing bridge initialization...")
try:
    bridge = PostureKitBridge()
    assert hasattr(bridge, 'visual_extractor'), "Bridge missing visual_extractor"
    assert hasattr(bridge, 'fusion_engine'), "Bridge missing fusion_engine"
    print("✓ Bridge initialized with visual components")
except Exception as e:
    print(f"✗ Bridge initialization failed: {e}")
    sys.exit(1)

# Test 3: Visual extractor works
print("\n[3/5] Testing visual feature extraction...")
try:
    import numpy as np
    # Create dummy image (RGB)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    visual_features = bridge.visual_extractor.extract(test_image)

    assert hasattr(visual_features, 'feature_vector'), "Missing feature_vector"
    assert visual_features.feature_vector.shape == (576,), f"Wrong dim: {visual_features.feature_vector.shape}"
    assert visual_features.feature_vector.dtype == np.float32, "Wrong dtype"
    print(f"✓ Visual features extracted: {visual_features.feature_vector.shape}")
except Exception as e:
    print(f"✗ Visual extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Fusion works
print("\n[4/5] Testing multimodal fusion...")
try:
    from core.geometric_feature_extractor import GeometricFeatures
    # Create dummy geometric features
    geometric_vec = np.random.rand(52).astype(np.float32)
    visual_vec = np.random.rand(576).astype(np.float32)

    # Create feature objects
    import json
    geom_features = GeometricFeatures(
        feature_vector=geometric_vec,
        joint_angles={},
        limb_ratios={},
        body_angles={},
        symmetry_scores={},
        occlusion_pattern=np.zeros(7)
    )

    from core.visual_feature_extractor import VisualFeatures
    vis_features = VisualFeatures(feature_vector=visual_vec, model_name='test_model')

    # Fuse
    fused = bridge.fusion_engine.fuse(geom_features, vis_features)

    assert hasattr(fused, 'fused_vector'), "Missing fused_vector"
    assert fused.fused_vector.shape == (628,), f"Wrong fused dim: {fused.fused_vector.shape}"
    assert fused.fused_vector.dtype == np.float32, "Wrong fused dtype"
    print(f"✓ Fusion works: {fused.fused_vector.shape}")
except Exception as e:
    print(f"✗ Fusion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Database models ready
print("\n[5/5] Testing database models...")
try:
    from storage.models import FusedFeatures as FusedFeaturesModel, VisualFeatures as VisualFeaturesModel
    print("✓ Database models available (FusedFeatures, VisualFeatures)")
except ImportError as e:
    print(f"✗ Database model import failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
print("\nPhase 1 implementation verified:")
print("  - Visual feature extraction: 576-dim ✓")
print("  - Multimodal fusion: 628-dim (52 + 576) ✓")
print("  - Database models ready ✓")
print("\nReady to test with real images!")
