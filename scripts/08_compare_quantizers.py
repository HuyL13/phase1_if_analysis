"""Run Phase 1 stage 8; use --help for arguments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phase1.cli import stage_main

if __name__ == "__main__":
    stage_main(8)
