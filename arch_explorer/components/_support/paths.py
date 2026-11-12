"""Location of the installed public Python package, independent of model layout."""
from pathlib import Path

def package_root():
    return Path(__file__).resolve().parents[2]
