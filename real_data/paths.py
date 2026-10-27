"""Common paths for the real-data case study (all relative to this folder)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repository root -> model.py
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, 'data') + os.sep      # filled by download_data.sh
CACHE = os.path.join(HERE, 'cache') + os.sep    # pickled environments
RUNS = os.path.join(HERE, 'runs') + os.sep      # raw runs (.npz)
OUT = os.path.join(HERE, 'out') + os.sep        # tables and figures
for _d in (CACHE, RUNS, OUT, OUT + 'tables', OUT + 'figures'):
    os.makedirs(_d, exist_ok=True)
