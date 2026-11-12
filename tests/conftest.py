"""Test support is separate from installed production package resolution."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/support'))
# Generator tests need the source-export tools namespace, but the independently
# installed arch_explorer must remain ahead of the checkout on sys.path.
sys.path.append(str(ROOT))
