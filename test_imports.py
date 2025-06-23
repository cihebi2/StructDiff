#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        # Test data imports
        from structdiff.data import PeptideStructureDataset, PeptideStructureCollator
        print("✓ Data imports successful")
        
        # Test model imports
        from structdiff.models import StructDiff
        print("✓ Model imports successful")
        
        # Test diffusion imports
        from structdiff.diffusion import GaussianDiffusion
        print("✓ Diffusion imports successful")
        
        # Test CFG imports
        from structdiff.models.classifier_free_guidance import CFGConfig, ClassifierFreeGuidance
        print("✓ CFG imports successful")
        
        # Test length sampler imports
        from structdiff.sampling.length_sampler import LengthSamplerConfig, AdaptiveLengthSampler
        print("✓ Length sampler imports successful")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_package_installation():
    """Test if package is properly installed"""
    try:
        import structdiff
        print(f"✓ StructDiff package found at: {structdiff.__file__}")
        print(f"✓ Version: {structdiff.__version__}")
        return True
    except ImportError:
        print("❌ StructDiff package not installed")
        print("Run: pip install -e . from the project root")
        return False

if __name__ == "__main__":
    print("=== StructDiff Import Test ===\n")
    
    # Test package installation
    package_ok = test_package_installation()
    print()
    
    # Test imports
    imports_ok = test_basic_imports()
    
    if package_ok and imports_ok:
        print("\n🎉 All tests passed! You can now run the CFG+Length integration tests.")
    else:
        print("\n🔧 Fix the issues above before running tests.")
        print("\nQuick fix:")
        print("cd /home/qlyu/sequence/StructDiff")
        print("pip install -e .")