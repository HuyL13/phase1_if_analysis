"""Download the clean base and official IF-SFT checkpoints from Hugging Face."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from phase1.if_queries import download_original_queries, normal_prompt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    parent = raw.pop('extends', None)
    if parent:
        base = yaml.safe_load((config_path.parent/parent).read_text(encoding='utf-8'))
        raw = {**base, **raw}
    return raw


def download(repo_id: str, revision: str | None, target: str) -> None:
    from huggingface_hub import snapshot_download

    destination = Path(target).resolve()
    if (destination/'config.json').is_file() and (
        (destination/'model.safetensors').is_file() or
        (destination/'model.safetensors.index.json').is_file()
    ):
        print(f'Using existing checkpoint: {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {repo_id} -> {destination}', flush=True)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(destination))


def download_calibration(target: str) -> None:
    destination = Path(target).resolve()
    if destination.is_file():
        print(f'Using existing calibration: {destination}', flush=True)
        return
    from datasets import load_dataset

    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'text': text} for text in dataset['text'] if text.strip()]
    destination.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)+'\n',
                           encoding='utf-8')
    print(f'Wrote Wikitext-2 train calibration: {destination}', flush=True)


def find_awq_code_root(destination: Path) -> Path | None:
    if (destination/'awq'/'quantize'/'pre_quant.py').is_file():
        return destination
    for candidate in destination.rglob('pre_quant.py'):
        root = candidate.parents[2]
        if (root/'awq'/'quantize'/'pre_quant.py').is_file():
            return root
    return None


def ensure_awq_drive_code(repo_path: str, drive_url: str) -> None:
    destination = Path(repo_path).resolve()
    root = find_awq_code_root(destination) if destination.exists() else None
    if root is not None:
        print(f'Using existing Drive AWQ code: {root}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'AWQ Drive code path exists but has no awq/quantize/pre_quant.py: {destination}')
    print(f'Downloading AWQ code from Drive: {drive_url} -> {destination}', flush=True)
    subprocess.run([sys.executable, '-m', 'gdown', '--folder', drive_url, '-O', str(destination)], check=True)
    root = find_awq_code_root(destination)
    if root is None:
        raise FileNotFoundError(f'Drive folder did not contain awq/quantize/pre_quant.py: {destination}')


def download_normal_queries(target: str, tokenizer_repo: str, tolerance: float, fingerprint_path: str) -> None:
    destination = Path(target).resolve()
    if destination.is_file():
        print(f'Using existing normal queries: {destination}', flush=True)
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer

    fingerprint_rows = [json.loads(line) for line in Path(fingerprint_path).read_text(encoding='utf-8').splitlines()
                        if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=False)
    candidates = load_dataset('tatsu-lab/alpaca', split='train')
    print(f'Preparing {len(candidates)} Alpaca candidates for matched normal queries...', flush=True)
    started = time.time()
    prepared = []
    for index, candidate in enumerate(candidates):
        text = str(candidate['instruction'])
        if candidate.get('input'):
            text += '\n' + str(candidate['input'])
        prompt = normal_prompt(text)
        length = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        prepared.append((index, prompt, length))
        if (index + 1) % 5000 == 0:
            print(f'Prepared {index + 1}/{len(candidates)} normal candidates', flush=True)
    print(f'Prepared normal candidate lengths in {time.time() - started:.1f}s', flush=True)
    normal = []
    used = set()
    for row in fingerprint_rows:
        reference_length = len(tokenizer(row['prompt'], add_special_tokens=True).input_ids)
        ranked = []
        for index, prompt, length in prepared:
            if index in used:
                continue
            relative = abs(length-reference_length)/max(length, reference_length)
            ranked.append((relative, index, prompt))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if not ranked or ranked[0][0] > tolerance:
            raise ValueError(f'Could not match normal prompt for {row["query_id"]} within tolerance {tolerance}')
        relative, index, prompt = ranked[0]
        used.add(index)
        normal.append({
            'query_id': f'normal-{len(normal)+1:03d}',
            'matched_pair_id': row['query_id'],
            'prompt': prompt,
            'language': 'en',
            'structure': 'instruction',
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in normal)+'\n',
                           encoding='utf-8')
    print(f'Wrote matched normal queries: {destination}', flush=True)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    download(config['base_model_repo'], config.get('base_model_revision'), config['clean_checkpoint'])
    download(config['fingerprint_model_repo'], config.get('fingerprint_model_revision'),
             config['fingerprinted_checkpoint'])
    download_original_queries(config['if_query_url'], config['queries'], config['if_query_sha256'])
    download_calibration(config['calibration'])
    download_normal_queries(config['normal_queries'], config['tokenizer'],
                            config.get('matching_length_tolerance', 0.25), config['queries'])
    ensure_awq_drive_code(config['awq_code_repo'], config['awq_drive_url'])
