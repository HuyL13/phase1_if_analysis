"""CLI for individual stages, ordered batches and offline metric smoke tests."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from .io import load_config, run_metadata, write_json
from .experiments import (baseline, parameter_analysis, layer_sensitivity, query_margins,
                          representation_drift, Runtime)
from .reporting import integrate, validate_comparison
from .runtime import quantized_checkpoint


def run_root(config, seed):
    name = 'fp' if config['quantizer']=='fp' else config['quantizer']+str(config['bits'])
    return Path(config['output_root'])/name/f'seed{seed}'


def execute(config, seed, stages, rtn3_results=None):
    root = run_root(config, seed)
    metadata = run_metadata(config, seed)
    file = root/'metadata.json'
    if file.exists():
        old = json.loads(file.read_text(encoding='utf-8'))
        if old['run_id'] != metadata['run_id']:
            raise ValueError('Existing output belongs to different inputs/config/code; choose a new output_root')
        metadata = old
    if config['quantizer'] in ('gptq', 'awq'):
        exports = {variant: quantized_checkpoint(config, variant, seed)[1]
                   for variant in ('clean', 'fingerprinted')}
        metadata['quantized_checkpoint_metadata'] = exports
        metadata['quantization_library_version'] = {
            variant: value['quantization_library_version'] for variant, value in exports.items()}
    write_json(file, metadata)
    runtime = None
    for stage in stages:
        print(f"{config['quantizer']}{config['bits']} seed={seed} stage={stage}", flush=True)
        if stage in (1,2,3):
            parameter_analysis(config, seed, root, metadata, experiments=[stage])
        else:
            if runtime is None:
                runtime = Runtime(config, seed)
                import torch
                metadata['gpu_type'] = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
                metadata['tokenizer_effective_chat_template'] = runtime.tokenizer.chat_template
                metadata['tokenizer_special_tokens'] = runtime.tokenizer.special_tokens_map
                metadata['shared_generation_defaults'] = runtime.generation_config.to_dict()
            if stage==0:
                baseline(runtime, config, seed, root, metadata)
            elif stage==4:
                layer_sensitivity(runtime, config, seed, root, metadata)
            elif stage==5:
                path = rtn3_results or str(Path(config['output_root'])/'rtn3'/f'seed{seed}'/'baseline/if_query_results.csv')
                path = Path(str(path).format(seed=seed))
                # Validate source provenance as well as query ID/seed join.
                source_meta_path = path.parent.parent/'metadata.json'
                if not source_meta_path.exists():
                    raise ValueError('RTN3 results must include sibling run metadata.json')
                source = json.loads(source_meta_path.read_text(encoding='utf-8'))
                validate_comparison([metadata, source])
                query_margins(runtime, config, seed, root, metadata, path)
            elif stage in (6,7):
                if stage==7 and not (Path(config['output_root'])/'rtn3'/f'seed{seed}'/'exp04_layer_sensitivity/layer_sensitivity.csv').exists():
                    raise ValueError('Run Experiment 4 RTN3 before expensive hidden-state analysis')
                representation_drift(runtime, config, seed, root, metadata, hidden=stage==7)
            else:
                raise ValueError('Stage must be 0 through 7; use compare for Experiment 8')
        metadata.setdefault('completed_stages', [])
        if stage not in metadata['completed_stages']:
            metadata['completed_stages'].append(stage)
        write_json(file, metadata)
    return root


def smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    clean, fp = {}, {}
    for layer in range(4):
        for module in ('q_proj', 'down_proj'):
            name = f'model.layers.{layer}.self_attn.{module}.weight'
            clean[name] = rng.normal(size=(8,17)).astype(np.float32)
            fp[name] = clean[name]+rng.normal(scale=.01*(layer+1),size=(8,17)).astype(np.float32)
    np.savez(output/'clean.npz', **clean)
    np.savez(output/'fingerprinted.npz', **fp)
    for bits in (3,4):
        path = output/f'rtn{bits}.yaml'
        path.write_text(yaml.safe_dump(dict(model='SYNTHETIC_SMOKE', synthetic=True,
            clean_checkpoint=str(output/'clean.npz'), fingerprinted_checkpoint=str(output/'fingerprinted.npz'),
            tokenizer='not_used_in_parameter_smoke', quantizer='rtn', bits=bits, group_size=8,
            seeds=[42], output_root=str(output/'outputs'))), encoding='utf-8')
        execute(load_config(path),42,[1,2,3])
    integrate(output/'outputs')
    print(f'Synthetic smoke report: {output / "outputs/summary/phase1_summary.md"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('run', help='Run stages for a single quantizer config')
    run.add_argument('--config',required=True)
    run.add_argument('--stages',default='0,1,2,4,5,3,6,7')
    run.add_argument('--rtn3-results',help='Path to IF query results; supports {seed}')
    batch = sub.add_parser('batch',help='Stage 0 then Batch A, optionally Batch B, across configs')
    batch.add_argument('--configs',nargs='+',required=True)
    batch.add_argument('--include-batch-b',action='store_true')
    compare = sub.add_parser('compare', help='Experiment 8: validate and integrate runs')
    compare.add_argument('--output-root',default='outputs')
    demo = sub.add_parser('smoke',help='Offline synthetic parameters only, never IF evidence')
    demo.add_argument('--output',default='outputs/smoke')
    args = parser.parse_args(argv)
    if args.command=='smoke':
        smoke(args.output)
    elif args.command=='compare':
        integrate(args.output_root)
    elif args.command=='run':
        config=load_config(args.config)
        stages=[int(s) for s in args.stages.split(',')]
        if any(s not in range(8) for s in stages):
            parser.error('stages must be integers 0..7')
        if config['quantizer']!='rtn' or config['bits']!=3:
            stages=[s for s in stages if s!=4]
        for seed in config['seeds']:
            execute(config,seed,stages,args.rtn3_results)
    else:
        configs=[load_config(path) for path in args.configs]
        if len({c['output_root'] for c in configs})!=1:
            raise ValueError('Batch configs must use the same output_root')
        validate_comparison([run_metadata(c,c['seeds'][0]) for c in configs])
        if not any(c['quantizer']=='rtn' and c['bits']==3 for c in configs):
            raise ValueError('Batch A requires an RTN3 config')
        for stage in ([0,1,2,4,5]+([3,6,7] if args.include_batch_b else [])):
            for c in configs:
                if stage==4 and (c['quantizer'],c['bits'])!=('rtn',3):
                    continue
                if stage==2 and c['quantizer']!='rtn':
                    continue
                for seed in c['seeds']:
                    execute(c,seed,[stage])
        integrate(configs[0]['output_root'])


def stage_main(stage):
    if stage==8:
        main(['compare',*sys.argv[1:]])
    else:
        main(['run','--stages',str(stage),*sys.argv[1:]])


if __name__=='__main__':
    main()
