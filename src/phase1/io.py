"""Shared configuration, provenance and durable tabular output."""
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from datetime import datetime, timezone
import numpy as np
import yaml

QUANT_KEYS = ('quantizer', 'bits', 'group_size', 'symmetric', 'zero_point',
              'calibration_dataset', 'calibration_sample_count', 'calibration_sequence_length',
              'calibration_sha256')
COMMON_KEYS = ('model', 'clean_checkpoint', 'fingerprinted_checkpoint', 'tokenizer',
               'tokenizer_revision', 'generation', 'verification', 'queries', 'normal_queries',
               'utility_corpus', 'ppl_sequence_length', 'module_pattern', 'block_pattern',
               'dtype', 'seeds', 'device', 'prompt_mode', 'max_prompt_length',
               'calibration', 'upstream_ptq_repo', 'ppl_datasets', 'ppl_cache_dir')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Configuration must be a mapping')
    parent = raw.pop('extends', None)
    if parent:
        base = yaml.safe_load((path.parent/parent).read_text(encoding='utf-8'))
        raw = {**base, **raw}
    c = dict(group_size=128, symmetric=True, zero_point=False, seeds=[42], bits=16,
             quantizer='fp', calibration_dataset=None, calibration_sample_count=0,
             calibration_sequence_length=0, calibration_sha256=None, tokenizer_revision=None,
             generation={}, verification={}, dtype='float32', device='cpu', prompt_mode='raw',
             ppl_sequence_length=1024, max_prompt_length=2048, top_layers=3,
             ppl_datasets=['wikitext2'], ppl_cache_dir='.cache/ppl',
             block_pattern=r'(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)',
             module_pattern=r'\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$',
             module_aliases={}, output_root='outputs', sample_limit=4096)
    c.update(raw)
    for key in ('model', 'clean_checkpoint', 'fingerprinted_checkpoint', 'tokenizer'):
        if not c.get(key):
            raise ValueError(f'Missing configuration field: {key}')
    if c['quantizer'] not in ('fp', 'rtn', 'awq'):
        raise ValueError('Unsupported quantizer')
    if c['quantizer'] != 'fp' and c['bits'] not in (3, 4):
        raise ValueError('Phase 1 supports 3-bit and 4-bit quantization')
    if c['group_size'] <= 0:
        raise ValueError('group_size must be positive')
    if not isinstance(c['seeds'], list) or not c['seeds'] or any(type(s) is not int for s in c['seeds']):
        raise ValueError('seeds must be a nonempty list of integers')
    if len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('seeds must be unique')
    stochastic = c['generation'].get('do_sample', False) or c['quantizer'] == 'awq'
    if stochastic and len(c['seeds']) < 3:
        raise ValueError('Stochastic/calibrated experiments require at least three seeds')
    if c['quantizer'] == 'rtn' and c['zero_point'] == c['symmetric']:
        raise ValueError('RTN symmetric requires zero_point=false; asymmetric requires true')
    if c['prompt_mode'] not in ('raw', 'chat'):
        raise ValueError('prompt_mode must be raw or chat')
    for key in ('module_pattern', 'block_pattern'):
        re.compile(c[key])
    c['_config_path'] = str(path)
    if c['quantizer'] == 'awq' and c.get('calibration') and not c.get('calibration_sha256'):
        calibration = Path(c['calibration'])
        if calibration.is_file():
            c['calibration_sha256'] = digest(calibration)
    return c


def validate_quantized_metadata(config, metadata, source, seed):
    for key in QUANT_KEYS:
        if key not in metadata or metadata[key] != config.get(key):
            raise ValueError(f'Quantized checkpoint metadata mismatch: {key}')
    if metadata.get('seed') != seed:
        raise ValueError('Quantized checkpoint metadata mismatch: seed')
    if metadata.get('source_checkpoint') != source:
        raise ValueError('Quantized checkpoint source_checkpoint does not match source')
    if metadata.get('storage_representation') != 'hf_dequantized':
        raise ValueError('Export quantized weights into the original HF parameter coordinates first')
    if not metadata.get('quantization_library_version'):
        raise ValueError('Quantization library version must be recorded')
    if not config.get('calibration_sha256'):
        raise ValueError('AWQ requires the hash of the exact calibration token data')


def read_queries(path, normal=False):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    seen = set()
    if not records:
        raise ValueError('Query file is empty')
    for q in records:
        if 'query_id' not in q or not isinstance(q.get('prompt'), str) or not q['prompt']:
            raise ValueError('Each query requires query_id and nonempty prompt')
        q['query_id'] = str(q['query_id'])
        if q['query_id'] in seen:
            raise ValueError('Duplicate query_id: ' + q['query_id'])
        seen.add(q['query_id'])
        if not normal and not isinstance(q.get('target'), str):
            raise ValueError('Fingerprint query requires target text')
        if normal and not q.get('matched_pair_id'):
            raise ValueError('Normal prompts require matched_pair_id referencing a fingerprint query')
    return records


def callable_from_path(name):
    if not name or ':' not in name:
        raise ValueError('Set verification.callable to original_package.module:function')
    module, function = name.split(':', 1)
    result = getattr(importlib.import_module(module), function)
    if not callable(result):
        raise ValueError('Configured verifier is not callable')
    return result


def validate_config_inputs(c):
    """Fail before inference when the Stage 0 inputs are incomplete."""
    errors = []

    def require_file(key):
        value = c.get(key)
        if not value or not Path(value).is_file():
            errors.append(f'{key} must point to an existing file: {value}')

    def require_checkpoint(key):
        value = c.get(key)
        path = Path(value) if value else None
        if path is None or not path.exists():
            errors.append(f'{key} does not exist: {value}')
            return
        if path.is_dir() and not ((path/'config.json').is_file() or
                                  (path/'model.safetensors').is_file() or
                                  (path/'model.safetensors.index.json').is_file()):
            errors.append(f'{key} is not an HF checkpoint directory: {value}')
        if path.is_file() and path.suffix != '.npz':
            errors.append(f'{key} must be an HF checkpoint directory or NPZ file: {value}')

    require_checkpoint('clean_checkpoint')
    require_checkpoint('fingerprinted_checkpoint')
    require_file('queries')
    require_file('normal_queries')
    if not c.get('ppl_datasets'):
        require_file('utility_corpus')
    tokenizer = c.get('tokenizer')
    if tokenizer and Path(tokenizer).exists() and not Path(tokenizer).is_dir():
        errors.append(f'tokenizer must be a directory when supplied as a local path: {tokenizer}')
    try:
        queries = read_queries(c['queries'])
        normal = read_queries(c['normal_queries'], normal=True)
        query_ids = {q['query_id'] for q in queries}
        pairs = [str(q.get('matched_pair_id')) for q in normal]
        if len(normal) != len(queries) or set(pairs) != query_ids:
            errors.append('normal_queries must contain exactly one matched_pair_id per IF query')
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f'query validation failed: {exc}')
    try:
        callable_from_path(c.get('verification', {}).get('callable'))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f'verification.callable is not importable: {exc}')

    if c['quantizer'] == 'awq':
        for seed in c['seeds']:
            for variant in ('clean', 'fingerprinted'):
                key = 'quantized_clean_checkpoint' if variant == 'clean' else 'quantized_fingerprinted_checkpoint'
                value = c.get(key)
                path = Path(str(value).format(seed=seed)) if value else None
                if path is None or not path.is_dir():
                    errors.append(f'{key} seed {seed} must be an existing directory: {value}')
                    continue
                sidecar = path/'metadata.json'
                if not sidecar.is_file():
                    errors.append(f'missing quantization metadata: {sidecar}')
                    continue
                try:
                    validate_quantized_metadata(c, json.loads(sidecar.read_text(encoding='utf-8')),
                                                c['clean_checkpoint' if variant == 'clean' else 'fingerprinted_checkpoint'], seed)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f'{sidecar}: {exc}')
    if errors:
        raise ValueError('Stage 0 input validation failed:\n- ' + '\n- '.join(errors))
    return True


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'Refusing empty result table: {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(jsonable(row) for row in rows)
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def run_metadata(c, seed):
    versions = {}
    for lib in ('numpy', 'scipy', 'torch', 'transformers', 'tokenizers', 'safetensors'):
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    inputs = {key: digest(c[key]) for key in ('queries', 'normal_queries', 'utility_corpus')
              if c.get(key) and Path(c[key]).is_file()}
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    public = {key: value for key, value in c.items() if not key.startswith('_')}
    identity = hashlib.sha256(json.dumps(dict(config=public, inputs=inputs, seed=seed,
                                            code=source_hash.hexdigest()), sort_keys=True).encode()).hexdigest()
    return dict(**public, seed=seed, run_id=identity[:16], input_sha256=inputs,
                code_sha256=source_hash.hexdigest(), versions=versions, git_commit=commit,
                datetime_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                gpu_type=None, quantization_library_version='phase1.rtn.v1' if c['quantizer']=='rtn' else None,
                tokenizer_version=versions['tokenizers'], transformers_version=versions['transformers'])


def tensor_identity(name, c):
    match = re.search(c['block_pattern'], name)
    module = name.rsplit('.', 2)[-2] if '.' in name else name
    return dict(layer_id=int(match.group(1)) if match else -1,
                module_name=c.get('module_aliases', {}).get(module, module), tensor_name=name)
