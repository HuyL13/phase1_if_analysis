import os
from pathlib import Path
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1]/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
