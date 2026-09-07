#!/usr/bin/env python3
"""Run the official MIT Han Lab AWQ code and export dense HF weights."""
from __future__ import annotations

import argparse
import json
import random
import sys
import types
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


def token_blocks(texts, tokenizer, seqlen: int, limit: int):
    import torch

    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    flat = [token for sample in encoded for token in sample]
    blocks = []
    for offset in range(0, len(flat) - seqlen + 1, seqlen):
        blocks.append(torch.tensor([flat[offset:offset + seqlen]], dtype=torch.long))
        if len(blocks) == limit:
            break
    if not blocks:
        raise ValueError(f'calibration corpus has fewer than {seqlen} tokens')
    return blocks


def install_awq_kernel_stub() -> None:
    stub = types.ModuleType('awq_inference_engine')
    stub.__file__ = '<phase1-awq-kernel-stub>'
    sys.modules.setdefault('awq_inference_engine', stub)


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
    if not (awq_repo/'awq'/'quantize'/'pre_quant.py').is_file():
        raise FileNotFoundError(f'Official AWQ checkout is incomplete: {awq_repo}')
    install_awq_kernel_stub()
    sys.path.insert(0, str(awq_repo))

    from awq.quantize.pre_quant import apply_awq, run_awq
    from awq.quantize.quantizer import pseudo_quantize_model_weight
    import awq.utils.calib_data as calibration_module

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()

    texts = load_calibration_texts(Path(args.calibration))
    fixed_blocks = token_blocks(texts, tokenizer, args.seqlen, args.nsamples)

    def fixed_calibration(**_kwargs):
        return fixed_blocks

    calibration_module.get_calib_dataset = fixed_calibration
    q_config = {'zero_point': True, 'q_group_size': args.group_size}
    results = run_awq(
        model, tokenizer, args.bits, q_config,
        n_samples=len(fixed_blocks), seqlen=args.seqlen, calib_data='phase1_fixed',
    )
    apply_awq(model, results)
    pseudo_quantize_model_weight(model, w_bit=args.bits, q_config=q_config)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        'backend': 'awq',
        'bits': args.bits,
        'group_size': args.group_size,
        'upstream': 'mit-han-lab/llm-awq',
        'upstream_sha': 'd6e797a42b9ef7778de8ee2352116e0f48a78d61',
        'dense_quantized_weights': True,
    }
    (output/'quantization_manifest.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
