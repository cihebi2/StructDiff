#!/usr/bin/env python3
"""
Test script for Huggingface ESMFold implementation
"""

import torch
from structdiff.models.esmfold_wrapper import ESMFoldWrapper

def test_esmfold_hf():
    """Test the Huggingface ESMFold implementation"""
    
    print("Testing Huggingface ESMFold implementation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test sequence (small protein for quick testing)
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    try:
        # Initialize ESMFold wrapper
        print("\nInitializing ESMFold wrapper...")
        esmfold = ESMFoldWrapper()
        print("✓ ESMFold wrapper initialized successfully")
        
        # Predict structure
        print(f"\nPredicting structure for sequence of length {len(test_sequence)}...")
        features = esmfold.predict_structure(test_sequence)
        
        # Check outputs
        print("\n=== Structure Prediction Results ===")
        print(f"✓ Positions shape: {features['positions'].shape}")
        print(f"✓ pLDDT shape: {features['plddt'].shape}")
        print(f"✓ Distance matrix shape: {features['distance_matrix'].shape}")
        print(f"✓ Contact map shape: {features['contact_map'].shape}")
        print(f"✓ Secondary structure shape: {features['secondary_structure'].shape}")
        
        # Statistics
        print(f"\n=== Statistics ===")
        print(f"Mean pLDDT: {features['plddt'].mean().item():.2f}")
        print(f"Min pLDDT: {features['plddt'].min().item():.2f}")
        print(f"Max pLDDT: {features['plddt'].max().item():.2f}")
        
        print(f"Mean CA-CA distance: {features['distance_matrix'].mean().item():.2f} Å")
        print(f"Number of contacts (<8Å): {features['contact_map'].sum().item():.0f}")
        
        # Check if secondary structure prediction worked
        ss_types = features['secondary_structure'].unique()
        print(f"Secondary structure types found: {ss_types.tolist()}")
        
        print("\n🎉 ESMFold test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ESMFold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_esmfold_hf()
    exit(0 if success else 1) 