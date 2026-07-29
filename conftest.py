import sys
from pathlib import Path

# Make every stage package importable when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).parent))
