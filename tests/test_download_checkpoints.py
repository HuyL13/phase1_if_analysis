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


def test_awq_prepare_reuses_exported_checkpoint_and_repairs_metadata(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / '09_prepare_upstream_quantized.py'
    spec = importlib.util.spec_from_file_location('prepare_awq', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    repo = tmp_path / 'awq_code'
    (repo / 'src' / 'quantization').mkdir(parents=True)
    (repo / 'src' / 'quantization' / 'awq.py').write_text('', encoding='utf-8')
    calibration = tmp_path / 'calibration.jsonl'
    calibration.write_text('{"text":"calibration"}\n', encoding='utf-8')
    clean = tmp_path / 'clean'
    fp = tmp_path / 'fp'
    clean.mkdir()
    fp.mkdir()
    for output in (tmp_path / 'awq' / 'clean', tmp_path / 'awq' / 'fp'):
        output.mkdir(parents=True)
        (output / 'config.json').write_text('{}', encoding='utf-8')
        (output / 'model.safetensors').write_bytes(b'weights')
        (output / 'quantization_manifest.json').write_text(
            json.dumps({
                'backend': 'awq',
                'bits': 3,
                'group_size': 128,
                'upstream': 'vendored-drive-awq-code',
                'upstream_sha': 'test',
                'dense_quantized_weights': True,
            }),
            encoding='utf-8',
        )
    config = tmp_path / 'awq.yaml'
    config.write_text(
        '\n'.join([
            'model: test',
            f'clean_checkpoint: {clean}',
            f'fingerprinted_checkpoint: {fp}',
            'tokenizer: test',
            'quantizer: awq',
            'bits: 3',
            'symmetric: false',
            'zero_point: true',
            'seeds: [42]',
            'calibration_dataset: public',
            'calibration_sample_count: 16',
            'calibration_sequence_length: 512',
            f'calibration: {calibration}',
            f'awq_code_repo: {repo}',
            f'quantized_clean_checkpoint: {tmp_path / "awq" / "clean"}',
            f'quantized_fingerprinted_checkpoint: {tmp_path / "awq" / "fp"}',
        ]) + '\n',
        encoding='utf-8',
    )

    def fail_if_quantize_runs(*_args, **_kwargs):
        raise AssertionError('prepare should reuse exported AWQ checkpoints')

    monkeypatch.setattr(module.subprocess, 'run', fail_if_quantize_runs)

    module.prepare(str(config), 'python')

    assert (tmp_path / 'awq' / 'clean' / 'metadata.json').is_file()
    assert (tmp_path / 'awq' / 'fp' / 'metadata.json').is_file()



def test_rtn_prepare_reuses_exported_checkpoint_and_repairs_metadata(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / '09_prepare_upstream_quantized.py'
    spec = importlib.util.spec_from_file_location('prepare_quantized', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    clean = tmp_path / 'clean'
    fp = tmp_path / 'fp'
    clean.mkdir()
    fp.mkdir()
    for output in (tmp_path / 'rtn' / 'clean', tmp_path / 'rtn' / 'fp'):
        output.mkdir(parents=True)
        (output / 'config.json').write_text('{}', encoding='utf-8')
        (output / 'model.safetensors').write_bytes(b'weights')
        (output / 'quantization_manifest.json').write_text(
            json.dumps({
                'backend': 'rtn',
                'bits': 3,
                'group_size': 128,
                'upstream': 'phase1-rtn-export',
                'upstream_sha': 'phase1.rtn.v1',
                'dense_quantized_weights': True,
            }),
            encoding='utf-8',
        )
    config = tmp_path / 'rtn.yaml'
    config.write_text(
        '\n'.join([
            'model: test',
            f'clean_checkpoint: {clean}',
            f'fingerprinted_checkpoint: {fp}',
            'tokenizer: test',
            'quantizer: rtn',
            'bits: 3',
            'symmetric: true',
            'zero_point: false',
            'seeds: [42]',
            'calibration_dataset: null',
            'calibration_sample_count: 0',
            'calibration_sequence_length: 0',
            'calibration_sha256: null',
            f'quantized_clean_checkpoint: {tmp_path / "rtn" / "clean"}',
            f'quantized_fingerprinted_checkpoint: {tmp_path / "rtn" / "fp"}',
        ]) + '\n',
        encoding='utf-8',
    )

    def fail_if_quantize_runs(*_args, **_kwargs):
        raise AssertionError('prepare should reuse exported RTN checkpoints')

    monkeypatch.setattr(module.subprocess, 'run', fail_if_quantize_runs)

    module.prepare(str(config), 'python')

    assert (tmp_path / 'rtn' / 'clean' / 'metadata.json').is_file()
    assert (tmp_path / 'rtn' / 'fp' / 'metadata.json').is_file()
