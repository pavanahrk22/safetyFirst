"""Crime data analysis and hotspot prediction toolkit."""

import matplotlib

# Charts are only ever saved to disk, so a non-interactive backend is safest
# (works on servers, CI and machines without a display).
matplotlib.use("Agg")

__version__ = "1.0.0"
