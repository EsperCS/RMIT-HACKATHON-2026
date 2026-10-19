"""
main.py - Master Entry Point forwarding to run_experiment.py
RMIT HACKATHON 2026 - Multimodal Gatekeeper AI Safety Benchmark
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_experiment import main

if __name__ == "__main__":
    main()

