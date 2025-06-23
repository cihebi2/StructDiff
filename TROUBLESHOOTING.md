# Troubleshooting Guide

## ImportError: cannot import name 'PeptideStructureDataset'

This error occurs when the StructDiff package is not properly installed in your Python environment.

### Quick Fix

```bash
# Navigate to the StructDiff directory
cd /home/qlyu/sequence/StructDiff

# Install in development mode
pip install -e .

# Test the installation
python test_imports.py

# Run the tests
python tests/test_cfg_length_integration.py
```

### Alternative Solution

If the above doesn't work, try:

```bash
# Make sure you're in the right environment
conda activate structdiff

# Uninstall any existing installation
pip uninstall structdiff -y

# Clean install
pip install -e . --no-deps --force-reinstall

# Install dependencies separately if needed
pip install -r requirements.txt
```

### Manual Test

You can also test the imports manually:

```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path("/home/qlyu/sequence/StructDiff")
sys.path.insert(0, str(project_root))

# Test imports
from structdiff.data import PeptideStructureDataset
from structdiff.models.classifier_free_guidance import CFGConfig
print("✓ Imports successful")
```

### Common Issues

1. **Wrong directory**: Make sure you're in the StructDiff directory
2. **Wrong environment**: Activate the correct conda environment
3. **Path issues**: The Python path might not include the project directory
4. **Missing __init__.py**: All directories should have __init__.py files

### Verification Steps

1. Check that setup.py exists: `ls setup.py`
2. Check package structure: `find structdiff -name "__init__.py"`
3. Test basic import: `python -c "import structdiff; print('OK')"`
4. Run import test: `python test_imports.py`

### If Nothing Works

Try running the test directly with explicit path setup:

```bash
cd /home/qlyu/sequence/StructDiff
PYTHONPATH=/home/qlyu/sequence/StructDiff python tests/test_cfg_length_integration.py
```