"""Create AWQ checkpoints with the configured Drive AWQ code."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

from phase1.io import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--configs', nargs='+', required=True)
    return parser.parse_args()


def prepare(config_path: str, python: str) -> None:
    config = load_config(config_path)
    if config['quantizer'] != 'awq':
        return
    repo = Path(config['awq_code_repo']).resolve()
    if not (repo/'src'/'quantization'/'awq.py').is_file():
        matches = list(repo.rglob('awq.py')) if repo.exists() else []
        for match in matches:
            root = match.parents[2]
            if (root/'src'/'quantization'/'awq.py').is_file():
                repo = root
                break
    wrapper = Path(__file__).resolve().parent/'10_run_awq_official.py'
    calibration = Path(config['calibration']).resolve()
    if not wrapper.is_file():
        raise FileNotFoundError(f'AWQ runner not found: {wrapper}')
    if not (repo/'src'/'quantization'/'awq.py').is_file():
        raise FileNotFoundError(f'Drive AWQ code not found or incomplete: {repo}')
    if not calibration.is_file():
        raise FileNotFoundError(f'Calibration artifact not found: {calibration}')
    calibration_hash = sha256(calibration)
    for seed in config['seeds']:
        for variant in ('clean', 'fingerprinted'):
            source_key = 'clean_checkpoint' if variant == 'clean' else 'fingerprinted_checkpoint'
            output_key = ('quantized_clean_checkpoint' if variant == 'clean'
                          else 'quantized_fingerprinted_checkpoint')
            source = Path(config[source_key]).resolve()
            output = Path(str(config[output_key]).format(seed=seed)).resolve()
            metadata_path = output/'metadata.json'
            manifest_path = output/'quantization_manifest.json'
            if metadata_path.is_file() and manifest_path.is_file():
                print(f'Using existing {config["quantizer"]}{config["bits"]} seed={seed} {variant}', flush=True)
                continue
            if output.exists():
                raise FileExistsError(f'Incomplete AWQ output exists; remove it before retrying: {output}')
            command = [python, str(wrapper), '--awq-repo', str(repo), '--model-path', str(source),
                       '--calibration', str(calibration), '--output', str(output),
                       '--bits', str(config['bits']), '--group-size', str(config['group_size']),
                       '--seed', str(seed), '--nsamples', str(config['calibration_sample_count']),
                       '--seqlen', str(config['calibration_sequence_length']),
                       '--device', str(config.get('device', 'cuda:0'))]
            print('Running AWQ:', ' '.join(command), flush=True)
            result = subprocess.run(command, cwd=repo)
            sigkill = getattr(signal, 'SIGKILL', None)
            if sigkill is not None and result.returncode == -sigkill:
                raise RuntimeError(
                    'AWQ export was killed by the OS (SIGKILL), usually because RAM/VRAM ran out. '
                    'Use a larger GPU/runtime or run without configs/awq3.yaml.'
                )
            result.check_returncode()
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('backend') != config['quantizer'] or not manifest.get('dense_quantized_weights'):
                raise ValueError(f'Unexpected AWQ manifest: {manifest}')
            sidecar = {
                'quantizer': config['quantizer'], 'bits': config['bits'],
                'group_size': config['group_size'], 'symmetric': config['symmetric'],
                'zero_point': config['zero_point'],
                'calibration_dataset': config['calibration_dataset'],
                'calibration_sample_count': config['calibration_sample_count'],
                'calibration_sequence_length': config['calibration_sequence_length'],
                'calibration_sha256': calibration_hash, 'seed': seed,
                'source_checkpoint': str(source),
                'storage_representation': 'hf_dequantized',
                'quantization_library_version': f'{manifest["upstream"]}@{manifest["upstream_sha"]}',
                'upstream_manifest': manifest,
            }
            metadata_path.write_text(json.dumps(sidecar, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    args = parse_args()
    for config_path in args.configs:
        prepare(config_path, args.python)
