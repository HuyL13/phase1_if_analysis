"""Stage 0 and functional/localization experiments, using one verifier."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .io import read_queries, read_csv, write_csv, write_json, tensor_identity
from .metrics import EPS, margins, logit_drift, hidden_drift
from .parameters import parameter_analysis
from .runtime import Runtime, intervention, selected_parameters, teacher_forced_logits, prompt_ids
from .statistics import describe, compare_groups

DIRECTORIES = {0: 'baseline', 1: 'exp01_update_retention', 2: 'exp02_update_resolution',
               3: 'exp03_error_alignment', 4: 'exp04_layer_sensitivity', 5: 'exp05_logit_margin',
               6: 'exp06_logit_drift', 7: 'exp07_hidden_drift'}


def base_row(c, seed, metadata):
    return dict(quantizer=c['quantizer'], bits=c['bits'], group_size=c['group_size'],
                seed=seed, run_id=metadata['run_id'])


def baseline(runtime, c, seed, root, metadata):
    queries = read_queries(c['queries'])
    corpus_path = Path(c['utility_corpus']) if c.get('utility_corpus') else None
    corpus = corpus_path.read_text(encoding='utf-8') if corpus_path and corpus_path.is_file() else ''
    results, logs = [], []
    for variant in ('clean', 'fingerprinted'):
        with runtime.model(variant) as model:
            score, ppl, records = runtime.evaluate(model, queries, corpus)
        del model
        base = {**base_row(c, seed, metadata), 'model_variant': variant}
        results.append({**base, 'fingerprint_score': score, 'perplexity': ppl})
        logs.extend({**r, **base} for r in records)
    write_csv(root/'baseline/baseline_results.csv', results)
    write_csv(root/'baseline/if_query_results.csv', logs)


def layer_sensitivity(runtime, c, seed, root, metadata):
    if c['quantizer'] != 'rtn' or c['bits'] != 3:
        raise ValueError('Experiment 4 requires RTN3 config')
    queries, rows = read_queries(c['queries']), []
    corpus_path = Path(c['utility_corpus']) if c.get('utility_corpus') else None
    corpus = corpus_path.read_text(encoding='utf-8') if corpus_path and corpus_path.is_file() else ''
    base = base_row(c, seed, metadata)
    with runtime.model(fp=True) as model:
        score0, ppl0, _ = runtime.evaluate(model, queries, corpus)
        grouped = defaultdict(list)
        for name in selected_parameters(model, c):
            layer = tensor_identity(name, c)['layer_id']
            if layer >= 0:
                grouped[layer].append(name)
        if not grouped:
            raise ValueError('No transformer block parameters matched')
        def evaluate_scope(layer, names, scope, module):
            with intervention(model, names, c):
                score, ppl, records = runtime.evaluate(model, queries, corpus)
            relative = (ppl-ppl0)/ppl0
            rows.append({**base, 'layer_id': layer, 'quantized_scope': scope, 'module_name': module,
                         'fingerprint_score': score, 'fingerprint_drop': score0-score, 'perplexity': ppl,
                         'fp_baseline_score': score0, 'fp_baseline_perplexity': ppl0,
                         'ppl_absolute_increase': ppl-ppl0, 'ppl_relative_increase': relative,
                         'fp_utility_ratio': (score0-score)/(relative+EPS) if abs(relative+EPS)>0 else None,
                         'ratio_unstable': abs(relative)<1e-6})
            write_csv(root/'exp04_layer_sensitivity/layer_sensitivity.csv', rows)
            write_csv(root/f'exp04_layer_sensitivity/queries_layer{layer}_{module}.csv',
                      [{**r, **base, 'layer_id': layer, 'quantized_scope': scope, 'module_name': module} for r in records])
        for layer, names in sorted(grouped.items()):
            print(f'  RTN3 block {layer}', flush=True)
            evaluate_scope(layer, names, 'block', 'all')
        top = sorted(rows, key=lambda r: r['fingerprint_drop'], reverse=True)[:c['top_layers']]
        for row in top:
            modules = defaultdict(list)
            for name in grouped[row['layer_id']]:
                modules[tensor_identity(name, c)['module_name']].append(name)
            for module, names in modules.items():
                evaluate_scope(row['layer_id'], names, 'module', module)


def survival_groups(path, seed, query_ids):
    rows = [r for r in read_csv(path) if r['quantizer']=='rtn' and int(r['bits'])==3
            and r.get('model_variant')=='fingerprinted' and int(r['seed'])==seed]
    groups = {}
    for row in rows:
        if row['query_id'] in groups:
            raise ValueError('Duplicate RTN3 query outcome for this seed')
        value = str(row['verified']).lower()
        if value not in ('true', 'false'):
            raise ValueError('Invalid verified value in RTN3 query results')
        groups[row['query_id']] = 'survived' if value == 'true' else 'failed'
    if set(groups) != set(query_ids):
        raise ValueError('RTN3 outcomes must cover exactly the same fingerprint queries and seed')
    return groups


def query_margins(runtime, c, seed, root, metadata, rtn3_results):
    queries = read_queries(c['queries'])
    groups = survival_groups(rtn3_results, seed, [q['query_id'] for q in queries])
    rows, tokens = [], []
    # Each run contains its own identical FP control; comparisons deduplicate it.
    variants = [True] if c['quantizer']=='fp' else [True, False]
    for fp in variants:
        base = {**base_row(c, seed, metadata), 'model_variant': 'fingerprinted',
                'quantizer': 'fp' if fp else c['quantizer'], 'bits': 16 if fp else c['bits']}
        with runtime.model(fp=fp) as model:
            for query in queries:
                z, y = teacher_forced_logits(model, runtime.tokenizer, query, c)
                result, detail = margins(z, y)
                info = {**base, 'query_id': query['query_id'], 'rtn3_survival_group': groups[query['query_id']]}
                rows.append({**info, **result})
                for t, token in enumerate(y):
                    tokens.append({**info, 'token_position': t, 'target_token_id': int(token),
                                   'target_token': runtime.tokenizer.decode([int(token)]),
                                   **{k: v[t] for k,v in detail.items()}})
        del model
    write_csv(root/'exp05_logit_margin/query_margin.csv', rows)
    write_csv(root/'exp05_logit_margin/token_margin.csv', tokens)
    fp_rows = [r for r in rows if r['quantizer']=='fp']
    survived = [r['margin_mean'] for r in fp_rows if r['rtn3_survival_group']=='survived']
    failed = [r['margin_mean'] for r in fp_rows if r['rtn3_survival_group']=='failed']
    write_json(root/'exp05_logit_margin/survival_statistics.json', dict(
        seed=seed, measurement='pre-quantization FP mean target margin',
        survived=describe(survived, seed), failed=describe(failed, seed),
        comparison=compare_groups(survived, failed), tag='EXPLANATORY_ONLY'))


def matched_queries(c, tokenizer):
    fp = read_queries(c['queries'])
    normal = read_queries(c['normal_queries'], normal=True)
    by_id = {q['query_id']: q for q in fp}
    if len(normal) != len(fp) or {str(q['matched_pair_id']) for q in normal} != set(by_id):
        raise ValueError('Provide one matched normal prompt for each fingerprint query')
    audit = []
    for q in normal:
        reference = by_id[str(q['matched_pair_id'])]
        for key in ('language', 'structure'):
            if not q.get(key) or q[key] != reference.get(key):
                raise ValueError(f'Matched prompts must declare equal {key}')
        a, b = len(prompt_ids(tokenizer, reference, c)), len(prompt_ids(tokenizer, q, c))
        tolerance = c.get('matching_length_tolerance', .25)
        if abs(a-b)/max(a, b) > tolerance:
            raise ValueError('Matched prompt lengths exceed matching_length_tolerance')
        audit.append(dict(fingerprint_query_id=reference['query_id'], matched_normal_query_id=q['query_id'],
                          fingerprint_tokens=a, normal_tokens=b, language=q['language'], structure=q['structure']))
    return ([{**q, 'prompt_type': 'fingerprint', 'matched_pair_id': q['query_id']} for q in fp] +
            [{**q, 'prompt_type': 'normal'} for q in normal]), audit


def representation_drift(runtime, c, seed, root, metadata, hidden=False):
    queries, audit = matched_queries(c, runtime.tokenizer)
    dest = root/DIRECTORIES[7 if hidden else 6]
    write_csv(dest/'prompt_matching.csv', audit)
    base, rows = base_row(c, seed, metadata), []
    for variant in ('clean', 'fingerprinted'):
        # Cache only final logits and two vectors per block, never full token states.
        references = {}
        with runtime.model(variant, fp=True) as model:
            for q in queries:
                references[(q['prompt_type'], q['query_id'])] = runtime.representations(model, q, hidden)
        del model
        with runtime.model(variant) as model:
            for q in queries:
                fp_logits, fp_states = references[(q['prompt_type'], q['query_id'])]
                logits, states = runtime.representations(model, q, hidden)
                ident = {**base, 'model_variant': variant, 'query_id': q['query_id'],
                         'prompt_type': q['prompt_type'], 'matched_pair_id': q['matched_pair_id'],
                         'token_position': 'final_prompt_token'}
                if hidden:
                    if set(states) != set(fp_states):
                        raise ValueError('FP and quantized transformer block sets differ')
                    for layer in states:
                        for representation in states[layer]:
                            rows.append({**ident, 'layer_id': layer, 'representation_type': representation,
                                **hidden_drift(fp_states[layer][representation], states[layer][representation])})
                else:
                    rows.append({**ident, **logit_drift(fp_logits, logits)})
        del model
    write_csv(dest/('hidden_drift.csv' if hidden else 'logit_drift.csv'), rows)
