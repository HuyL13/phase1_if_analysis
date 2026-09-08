"""Create and reuse dequantized HF checkpoints for configured quantizers."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

from phase1.io import load_config, validate_quantized_metadata


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


def has_hf_checkpoint(path: Path) -> bool:
    return (path / 'config.json').is_file() and (
        (path / 'model.safetensors').is_file()
        or (path / 'model.safetensors.index.json').is_file()
        or any(path.glob('*.safetensors'))
    )


def sidecar(config: dict, manifest: dict, source: Path, seed: int, calibration_hash: str | None) -> dict:
    return {
        'quantizer': config['quantizer'],
        'bits': config['bits'],
        'group_size': config['group_size'],
        'symmetric': config['symmetric'],
        'zero_point': config['zero_point'],
        'calibration_dataset': config['calibration_dataset'],
        'calibration_sample_count': config['calibration_sample_count'],
        'calibration_sequence_length': config['calibration_sequence_length'],
        'calibration_sha256': calibration_hash,
        'seed': seed,
        'source_checkpoint': str(source),
        'storage_representation': 'hf_dequantized',
        'quantization_library_version': f'{manifest["upstream"]}@{manifest["upstream_sha"]}',
        'upstream_manifest': manifest,
    }


def reuse_or_repair(output: Path, metadata_path: Path, manifest_path: Path,
                    config: dict, source: Path, seed: int, calibration_hash: str | None) -> bool:
    quantizer = config['quantizer'].upper()
    if not output.exists():
        return False
    if not has_hf_checkpoint(output) or not manifest_path.is_file():
        raise FileExistsError(f'Incomplete {quantizer} output exists; remove it before retrying: {output}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('backend') != config['quantizer'] or not manifest.get('dense_quantized_weights'):
        raise ValueError(f'Unexpected {quantizer} manifest: {manifest}')
    expected = sidecar(config, manifest, source, seed, calibration_hash)
    if not metadata_path.is_file():
        metadata_path.write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')
        print(f'Repaired {quantizer} metadata sidecar: {metadata_path}', flush=True)
    validate_quantized_metadata(config, json.loads(metadata_path.read_text(encoding='utf-8')), str(source), seed)
    print(f'Reusing prepared {quantizer} checkpoint seed={seed}: {output}', flush=True)
    return True


def find_awq_repo(config: dict) -> Path:
    repo = Path(config['awq_code_repo']).resolve()
    if not (repo / 'src' / 'quantization' / 'awq.py').is_file():
        matches = list(repo.rglob('awq.py')) if repo.exists() else []
        for match in matches:
            root = match.parents[2]
            if (root / 'src' / 'quantization' / 'awq.py').is_file():
                repo = root
                break
    if not (repo / 'src' / 'quantization' / 'awq.py').is_file():
        raise FileNotFoundError(f'Drive AWQ code not found or incomplete: {repo}')
    return repo


def command_for(config: dict, python: str, source: Path, output: Path, seed: int, calibration: Path | None) -> tuple[list[str], Path]:
    scripts = Path(__file__).resolve().parent
    if config['quantizer'] == 'awq':
        repo = find_awq_repo(config)
        wrapper = scripts / '10_run_awq_official.py'
        if not wrapper.is_file():
            raise FileNotFoundError(f'AWQ runner not found: {wrapper}')
        assert calibration is not None
        return ([python, str(wrapper), '--awq-repo', str(repo), '--model-path', str(source),
                 '--calibration', str(calibration), '--output', str(output),
                 '--bits', str(config['bits']), '--group-size', str(config['group_size']),
                 '--seed', str(seed), '--nsamples', str(config['calibration_sample_count']),
                 '--seqlen', str(config['calibration_sequence_length']),
                 '--device', str(config.get('device', 'cuda:0'))], repo)
    if config['quantizer'] == 'rtn':
        wrapper = scripts / '12_run_rtn_export.py'
        if not wrapper.is_file():
            raise FileNotFoundError(f'RTN exporter not found: {wrapper}')
        command = [python, str(wrapper), '--model-path', str(source), '--output', str(output),
                   '--bits', str(config['bits']), '--group-size', str(config['group_size']),
                   '--seed', str(seed), '--device', str(config.get('device', 'cpu')),
                   '--dtype', str(config.get('dtype', 'float32')),
                   '--module-pattern', str(config['module_pattern'])]
        if config['symmetric']:
            command.append('--symmetric')
        return command, Path(__file__).resolve().parents[1]
    raise ValueError(f'Unsupported prepared quantizer: {config["quantizer"]}')


def prepare(config_path: str, python: str) -> None:
    config = load_config(config_path)
    if config['quantizer'] == 'fp':
        return
    calibration = None
    calibration_hash = config.get('calibration_sha256')
    if config['quantizer'] == 'awq':
        calibration = Path(config['calibration']).resolve()
        if not calibration.is_file():
            raise FileNotFoundError(f'Calibration artifact not found: {calibration}')
        calibration_hash = sha256(calibration)

    for seed in config['seeds']:
        for variant in ('clean', 'fingerprinted'):
            source_key = 'clean_checkpoint' if variant == 'clean' else 'fingerprinted_checkpoint'
            output_key = 'quantized_clean_checkpoint' if variant == 'clean' else 'quantized_fingerprinted_checkpoint'
            if not config.get(output_key):
                raise ValueError(f'Set {output_key} for {config["quantizer"]}{config["bits"]}')
            source = Path(config[source_key]).resolve()
            output = Path(str(config[output_key]).format(seed=seed)).resolve()
            metadata_path = output / 'metadata.json'
            manifest_path = output / 'quantization_manifest.json'
            if reuse_or_repair(output, metadata_path, manifest_path, config, source, seed, calibration_hash):
                continue
            command, cwd = command_for(config, python, source, output, seed, calibration)
            print(f'Running {config["quantizer"].upper()}:', ' '.join(command), flush=True)
            result = subprocess.run(command, cwd=cwd)
            sigkill = getattr(signal, 'SIGKILL', None)
            if sigkill is not None and result.returncode == -sigkill:
                raise RuntimeError(
                    f'{config["quantizer"].upper()} export was killed by the OS (SIGKILL), usually because RAM/VRAM ran out. '
                    'Use a larger runtime or lower calibration/runtime memory settings.'
                )
            result.check_returncode()
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('backend') != config['quantizer'] or not manifest.get('dense_quantized_weights'):
                raise ValueError(f'Unexpected {config["quantizer"].upper()} manifest: {manifest}')
            metadata_path.write_text(json.dumps(sidecar(config, manifest, source, seed, calibration_hash), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    args = parse_args()
    for config_path in args.configs:
        prepare(config_path, args.python)
