"""Avoid macOS libomp crashes when Torch and LightGBM share a test process."""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
