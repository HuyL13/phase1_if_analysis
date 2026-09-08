import importlib.util
import json
import sys
import types
from pathlib import Path

import torch


def load_script():
    path = Path(__file__).resolve().parents[1] / 'scripts' / '00_download_checkpoints.py'
    spec = importlib.util.spec_from_file_location('download_checkpoints', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normal_query_matching_tokenizes_candidates_once(tmp_path, monkeypatch):
    module = load_script()
    fingerprint_path = tmp_path / 'fingerprints.jsonl'
    rows = [
        {'query_id': f'if-{index}', 'prompt': 'x' * length}
        for index, length in enumerate((20, 25, 30), start=1)
    ]
    fingerprint_path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    candidates = [{'instruction': 'y' * length, 'input': ''} for length in (18, 22, 26, 32, 40)]

    class Tokenizer:
        calls = 0

        def __call__(self, text, add_special_tokens=True):
            self.calls += 1
            return type('Encoding', (), {'input_ids': list(range(len(text)))})()

    tokenizer = Tokenizer()
    monkeypatch.setattr(module, 'normal_prompt', lambda text: text)
    datasets = types.ModuleType('datasets')
    datasets.load_dataset = lambda *args, **kwargs: candidates
    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer = type(
        'AutoTokenizer',
        (),
        {'from_pretrained': staticmethod(lambda *args, **kwargs: tokenizer)},
    )
    monkeypatch.setitem(sys.modules, 'datasets', datasets)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)

    module.download_normal_queries(str(tmp_path / 'normal.jsonl'), 'tokenizer', 0.5, str(fingerprint_path))

    assert tokenizer.calls == len(rows) + len(candidates)


def test_vendored_awq_quantizer_runs_without_decoder_layer_forward():
    repo = Path(__file__).resolve().parents[1] / 'vendor' / 'drive_awq'
    sys.path.insert(0, str(repo))
    try:
        from src.quantization.awq import AWQConfig, AWQQuantizerXL
    finally:
        sys.path.remove(str(repo))

    class Tokenizer:
        def __call__(self, text, return_tensors='pt', truncation=True, max_length=512):
            return {'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.long)}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4, bias=False)
            self.out = torch.nn.Linear(4, 4, bias=False)

        def forward(self, input_ids, **_kwargs):
            hidden = self.embed(input_ids)
            hidden = self.proj(hidden)
            return {'logits': self.out(hidden)}

    quantizer = AWQQuantizerXL(
        TinyModel(),
        Tokenizer(),
        device='cpu',
        config=AWQConfig(bits=3, group_size=4, max_tokens_per_sample=16, layer_batch_size=1),
    )

    quantizer.quantize_model_sequential(['hello world'], n_samples=1)

    assert 'proj' in quantizer.layer_stats
