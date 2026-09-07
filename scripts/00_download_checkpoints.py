"""Download the clean base and official IF-SFT checkpoints from Hugging Face."""
from __future__ import annotations

import argparse
import json
import subprocess
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


def ensure_upstream_repo(repo_path: str, repo_url: str) -> None:
    destination = Path(repo_path).resolve()
    if (destination/'tools'/'run_awq_upstream.py').is_file():
        print(f'Using existing upstream AWQ wrapper: {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Upstream path exists but is incomplete: {destination}')
    print(f'Cloning upstream AWQ wrapper: {repo_url} -> {destination}', flush=True)
    subprocess.run(['git', 'clone', '--recurse-submodules', repo_url, str(destination)], check=True)


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
    normal = []
    used = set()
    for row in fingerprint_rows:
        reference_length = len(tokenizer(row['prompt'], add_special_tokens=True).input_ids)
        ranked = []
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            text = str(candidate['instruction'])
            if candidate.get('input'):
                text += '\n' + str(candidate['input'])
            prompt = normal_prompt(text)
            length = len(tokenizer(prompt, add_special_tokens=True).input_ids)
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
    ensure_upstream_repo(config['upstream_ptq_repo'], config['upstream_ptq_repo_url'])
