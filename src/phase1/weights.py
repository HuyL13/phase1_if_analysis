"""Stream local safetensors checkpoints one tensor at a time."""
import json
from pathlib import Path
import numpy as np


class WeightStore:
    def __init__(self, checkpoint, parameter_names=None):
        self.path = Path(checkpoint)
        self.npz = None
        if self.path.suffix == '.npz':
            # Explicit offline fixture/export format: every key is a parameter.
            self.npz = np.load(self.path, allow_pickle=False)
            self.names = set(self.npz.files)
            return
        if not self.path.is_dir():
            raise ValueError('Parameter analysis requires local HF safetensors directories')
        from safetensors import safe_open
        index = self.path/'model.safetensors.index.json'
        if index.exists():
            self.index = json.loads(index.read_text())['weight_map']
        else:
            single = self.path/'model.safetensors'
            if not single.is_file():
                raise ValueError(f'No model.safetensors or shard index in {self.path}')
            with safe_open(single, framework='np') as f:
                self.index = {key: single.name for key in f.keys()}
        if parameter_names is None:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig.from_pretrained(self.path, trust_remote_code=False)
            with torch.device('meta'):
                skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
            parameter_names = {name for name, _ in skeleton.named_parameters(remove_duplicate=False)}
            del skeleton
        self.names = set(self.index) & set(parameter_names)

    def get(self, name):
        if self.npz is not None:
            value = self.npz[name]
            return value if np.issubdtype(value.dtype, np.floating) else None
        from safetensors import safe_open
        with safe_open(self.path/self.index[name], framework='pt', device='cpu') as f:
            t = f.get_tensor(name)
            return t.float().numpy() if t.is_floating_point() else None

    def close(self):
        if self.npz is not None:
            self.npz.close()
