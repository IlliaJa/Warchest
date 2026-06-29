import os
import sys

# Make both the project root (for `import src...`) and this tests/ directory
# (for `import _helpers`) importable under pytest.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, os.path.dirname(_TESTS_DIR))
