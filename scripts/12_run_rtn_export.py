"""Export RTN-dequantized Hugging Face checkpoints once for reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', type=int, required=True)
    parser.add_argument('--group-size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--module-pattern', required=True)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    import torch
    return getattr(torch, name)


def main() -> None:
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from phase1.quantization import rtn

    args = parse_args()
    seed_all(args.seed)
    source = Path(args.model_path).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(args.device).eval()
    if getattr(model.config, 'quantization_config', None):
        raise ValueError('Packed quantized checkpoints are unsupported; source must be a dense HF checkpoint')

    pattern = re.compile(args.module_pattern)
    selected = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point() or parameter.ndim != 2 or not pattern.search(name):
                continue
            grid = rtn(parameter.detach().float().cpu().numpy(), args.bits, args.group_size, args.symmetric)
            parameter.copy_(torch.as_tensor(grid.values, dtype=parameter.dtype, device=parameter.device))
            selected += 1
    if selected == 0:
        raise ValueError('module_pattern did not select any quantizable parameters')

    model.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    tokenizer.save_pretrained(output)
    manifest = {
        'backend': 'rtn',
        'bits': args.bits,
        'group_size': args.group_size,
        'symmetric': args.symmetric,
        'seed': args.seed,
        'source_checkpoint': str(source),
        'selected_parameters': selected,
        'upstream': 'phase1-rtn-export',
        'upstream_sha': 'phase1.rtn.v1',
        'dense_quantized_weights': True,
        'storage_representation': 'hf_dequantized',
    }
    (output / 'quantization_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
