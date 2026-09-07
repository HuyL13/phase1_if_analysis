import importlib.util
import json
import sys
import types
from pathlib import Path


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
