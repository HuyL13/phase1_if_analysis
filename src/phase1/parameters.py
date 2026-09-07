"""Experiments 1-3, streaming checkpoints in original parameter coordinates."""
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
from .io import tensor_identity, write_csv, write_json
from .metrics import retention, alignment, aggregate_retention, aggregate_alignment
from .quantization import rtn, resolution
from .runtime import quantized_checkpoint
from .weights import WeightStore


def effective_precision(value, dtype):
    if dtype == 'float32':
        return value.astype(np.float32).astype(np.float64)
    if dtype == 'float16':
        return value.astype(np.float16).astype(np.float64)
    if dtype == 'bfloat16':
        import torch
        return torch.from_numpy(value.astype(np.float32)).bfloat16().float().numpy().astype(np.float64)
    raise ValueError('dtype must be float32, float16 or bfloat16')


def parameter_analysis(c, seed, root, metadata, experiments=(1, 2, 3)):
    root = Path(root)
    clean = WeightStore(c['clean_checkpoint'])
    fp = WeightStore(c['fingerprinted_checkpoint'], clean.names)
    stores = [clean, fp]
    quant_clean = quant_fp = None
    if c['quantizer'] == 'awq':
        qc, mc = quantized_checkpoint(c, 'clean', seed)
        qf, mf = quantized_checkpoint(c, 'fingerprinted', seed)
        quant_clean, quant_fp = WeightStore(qc, clean.names), WeightStore(qf, fp.names)
        stores += [quant_clean, quant_fp]
        metadata['quantized_checkpoint_metadata'] = dict(clean=mc, fingerprinted=mf)
    rows, resolutions, errors, samples = [], [], [], []
    compared = 0
    skipped = sorted(clean.names ^ fp.names)
    base = dict(quantizer=c['quantizer'], bits=c['bits'], seed=seed, run_id=metadata['run_id'])
    try:
        for name in sorted(clean.names & fp.names):
            w, wf = clean.get(name), fp.get(name)
            if w is None or wf is None:
                skipped.append(name)
                continue
            if w.shape != wf.shape:
                raise ValueError(f'Shape mismatch: {name}: {w.shape} vs {wf.shape}')
            compared += 1
            w, wf = effective_precision(w, c['dtype']), effective_precision(wf, c['dtype'])
            ident = {**base, **tensor_identity(name, c)}
            selected = w.ndim == 2 and bool(re.search(c['module_pattern'], name))
            grid = gridf = None
            if quant_clean:
                if name not in quant_clean.names or name not in quant_fp.names:
                    raise ValueError(f'Missing original-coordinate quantized parameter: {name}')
                qw, qwf = quant_clean.get(name), quant_fp.get(name)
                if qw is None or qwf is None:
                    raise ValueError('Quantized export contains non-floating parameter: '+name)
                qw, qwf = effective_precision(qw, c['dtype']), effective_precision(qwf, c['dtype'])
            elif c['quantizer'] == 'rtn' and selected:
                grid = rtn(w, c['bits'], c['group_size'], c['symmetric'])
                gridf = rtn(wf, c['bits'], c['group_size'], c['symmetric'])
                # Compare the weights actually stored in the inference model dtype.
                grid.values = effective_precision(grid.values, c['dtype'])
                gridf.values = effective_precision(gridf.values, c['dtype'])
                qw, qwf = grid.values, gridf.values
            else:
                qw, qwf = w, wf
            if 1 in experiments:
                rows.append({**ident, 'quantized_tensor': selected, **retention(w, wf, qw, qwf)})
            if 3 in experiments:
                errors.append({**ident, **alignment(w, wf, qwf),
                               'clean_quant_error_l2': float(np.linalg.norm((qw-w).ravel()))})
            if 2 in experiments and grid is not None:
                row, sample = resolution(w, wf, grid, gridf, seed=seed, sample_limit=c['sample_limit'])
                resolutions.append({**ident, **row})
                for i in range(len(sample['ratio'])):
                    samples.append({**ident, **{k: v[i] for k, v in sample.items()},
                                    'sampling': 'uniform_per_tensor_cap', 'sample_index': i})
        if 1 in experiments:
            dest = root/'exp01_update_retention'
            write_csv(dest/'update_retention.csv', rows)
            for scope, key in [('block', 'layer_id'), ('module', 'module_name')]:
                groups = defaultdict(list)
                for row in rows:
                    groups[row[key]].append(row)
                write_csv(dest/f'{scope}_retention.csv', [{**base, key: k, **aggregate_retention(v)} for k,v in groups.items()])
            write_csv(dest/'global_retention.csv', [{**base, **aggregate_retention(rows)}])
        if 2 in experiments:
            dest = root/'exp02_update_resolution'
            if resolutions:
                write_csv(dest/'update_resolution.csv', resolutions)
                if samples:
                    write_csv(dest/'resolution_samples.csv', samples)
            else:
                write_json(dest/'not_applicable.json', dict(reason='Resolution and boundaries require an explicit RTN grid'))
        if 3 in experiments:
            dest = root/'exp03_error_alignment'
            write_csv(dest/'error_alignment.csv', errors)
            for scope, key in [('block', 'layer_id'), ('module', 'module_name')]:
                groups = defaultdict(list)
                for row in errors:
                    groups[row[key]].append(row)
                write_csv(dest/f'{scope}_alignment.csv', [{**base, key: k, **aggregate_alignment(v)} for k,v in groups.items()])
            write_csv(dest/'global_alignment.csv', [{**base, **aggregate_alignment(errors),
                       'clean_quant_error_l2': float(np.sqrt(sum(r['clean_quant_error_l2']**2 for r in errors)))}])
        write_json(root/'parameter_audit.json', dict(skipped_tensors=skipped, compared_tensors=compared,
                   effective_dtype=c['dtype'], buffer_policy='HF skeleton named_parameters only; NPZ keys explicitly declared parameters'))
    finally:
        for store in stores:
            store.close()
