#!/usr/bin/env python3
"""Run the configured AWQ code and export dense HF weights."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--awq-repo', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', required=True, type=int, choices=[3, 4])
    parser.add_argument('--group-size', required=True, type=int, choices=[128])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-memory-guard', action='store_true')
    return parser.parse_args()


def load_calibration_texts(path: Path) -> list[str]:
    texts = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        text = record.get('text', record.get('input'))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'calibration line {line_number} has no non-empty text')
        texts.append(text)
    if not texts:
        raise ValueError('calibration file is empty')
    return texts


def model_dtype(torch, device: str):
    if device.startswith('cuda') and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(torch.device(device))
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float16


def available_cpu_bytes() -> int | None:
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        for line in meminfo.read_text(encoding='utf-8').splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return None


def checkpoint_loaded_bytes(model_path: Path) -> int | None:
    index = model_path/'model.safetensors.index.json'
    if index.is_file():
        metadata = json.loads(index.read_text(encoding='utf-8')).get('metadata', {})
        total = metadata.get('total_size')
        if isinstance(total, int):
            if total > 20 * 1024**3:
                return total // 2
            return total
    shard_sizes = [path.stat().st_size for path in model_path.glob('*.safetensors')]
    if shard_sizes:
        total = sum(shard_sizes)
        if total > 20 * 1024**3:
            return total // 2
        return total
    return None


def format_gib(value: int | None) -> str:
    if value is None:
        return 'unknown'
    return f'{value / 1024**3:.1f} GiB'


def memory_guard(torch, model_path: Path, device: str, nsamples: int, seqlen: int) -> None:
    needed_weights = checkpoint_loaded_bytes(model_path)
    if needed_weights is None:
        print('Memory guard: could not estimate checkpoint size; continuing', flush=True)
        return
    activation_margin = max(2 * 1024**3, nsamples * seqlen * 2048)
    required = needed_weights + activation_margin
    cpu_available = available_cpu_bytes()
    print(
        f'Memory guard: estimated weights={format_gib(needed_weights)}, '
        f'margin={format_gib(activation_margin)}, cpu_available={format_gib(cpu_available)}',
        flush=True,
    )
    if cpu_available is not None and cpu_available < min(needed_weights, 12 * 1024**3):
        raise RuntimeError(
            f'Not enough available system RAM to load this checkpoint safely: '
            f'available={format_gib(cpu_available)}, estimated_weights={format_gib(needed_weights)}. '
            'Switch to a runtime with more RAM or skip AWQ.'
        )
    if device.startswith('cuda') and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(torch.device(device))
        print(f'Memory guard: gpu_free={format_gib(free)}, gpu_total={format_gib(total)}', flush=True)
        if free < required:
            raise RuntimeError(
                f'Not enough free GPU memory for AWQ export: free={format_gib(free)}, '
                f'required~={format_gib(required)}. Switch to a larger GPU or skip AWQ.'
            )


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    awq_repo = Path(args.awq_repo).resolve()
    if not (awq_repo/'src'/'quantization'/'awq.py').is_file():
        raise FileNotFoundError(f'AWQ code checkout is incomplete: {awq_repo}')
    sys.path.insert(0, str(awq_repo))

    from src.quantization.awq import AWQConfig, AWQQuantizerXL

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'{args.device} requested, but CUDA is not available')
    dtype = model_dtype(torch, args.device)
    model_path = Path(args.model_path)
    if not args.no_memory_guard:
        memory_guard(torch, model_path, args.device, args.nsamples, args.seqlen)
    print(f'Loading model on {args.device} with dtype={dtype}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={'': args.device},
    ).eval()

    texts = load_calibration_texts(Path(args.calibration))
    awq_config = AWQConfig(
        bits=args.bits,
        group_size=args.group_size,
        max_tokens_per_sample=args.seqlen,
        layer_batch_size=1,
    )
    quantizer = AWQQuantizerXL(model, tokenizer, device=args.device, config=awq_config)
    quantizer.quantize_model_sequential(texts, n_samples=args.nsamples)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        'backend': 'awq',
        'bits': args.bits,
        'group_size': args.group_size,
        'upstream': 'vendored-drive-awq-code',
        'upstream_sha': '13cWrwAbZEiPJe9v4Hpr6fkRHICVL1evg',
        'dense_quantized_weights': True,
    }
    (output/'quantization_manifest.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
