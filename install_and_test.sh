#!/bin/bash
# Installation and test script for StructDiff

echo "=== StructDiff Installation and Test ==="

# Check current directory
echo "Current directory: $(pwd)"

# Install package in development mode
echo "Installing StructDiff in development mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ Installation successful"
else
    echo "❌ Installation failed"
    exit 1
fi

# Test imports
echo ""
echo "Testing imports..."
python test_imports.py

if [ $? -eq 0 ]; then
    echo "✓ Import tests passed"
else
    echo "❌ Import tests failed"
    exit 1
fi

# Run CFG+Length integration tests
echo ""
echo "Running CFG+Length integration tests..."
python tests/test_cfg_length_integration.py

if [ $? -eq 0 ]; then
    echo "✓ CFG+Length integration tests passed"
else
    echo "❌ CFG+Length integration tests failed"
    echo "Check the test output above for details"
fi

echo ""
echo "=== Installation and testing complete ==="