"""Integrate actual measurements without automatically declaring a mechanism."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .io import COMMON_KEYS, read_csv, write_csv, write_json
from .statistics import describe, compare_groups

TABLES = {'baseline': 'baseline/baseline_results.csv', 'queries': 'baseline/if_query_results.csv',
          'retention': 'exp01_update_retention/update_retention.csv',
          'block_retention': 'exp01_update_retention/block_retention.csv',
          'module_retention': 'exp01_update_retention/module_retention.csv',
          'global_retention': 'exp01_update_retention/global_retention.csv',
          'resolution': 'exp02_update_resolution/update_resolution.csv',
          'samples': 'exp02_update_resolution/resolution_samples.csv',
          'alignment': 'exp03_error_alignment/error_alignment.csv',
          'block_alignment': 'exp03_error_alignment/block_alignment.csv',
          'module_alignment': 'exp03_error_alignment/module_alignment.csv',
          'global_alignment': 'exp03_error_alignment/global_alignment.csv',
          'sensitivity': 'exp04_layer_sensitivity/layer_sensitivity.csv',
          'margin': 'exp05_logit_margin/query_margin.csv',
          'logits': 'exp06_logit_drift/logit_drift.csv', 'hidden': 'exp07_hidden_drift/hidden_drift.csv'}


def validate_comparison(metadata):
    if not metadata:
        raise ValueError('No run metadata found')
    shared = COMMON_KEYS + ('input_sha256', 'system_prompt', 'chat_template', 'add_special_tokens',
                           'module_aliases', 'matching_length_tolerance', 'code_sha256')
    for other in metadata[1:]:
        for key in shared:
            if other.get(key) != metadata[0].get(key):
                raise ValueError(f'Incomparable runs: {key} differs')
    quantized = [m for m in metadata if m['quantizer']!='fp']
    for other in quantized[1:]:
        for key in ('group_size', 'symmetric', 'zero_point'):
            if other.get(key) != quantized[0].get(key):
                raise ValueError(f'Incomparable quantization settings: {key}')
    calibrated = [m for m in metadata if m['quantizer'] in ('gptq', 'awq')]
    for other in calibrated[1:]:
        for key in ('calibration_dataset', 'calibration_sha256', 'calibration_sample_count', 'calibration_sequence_length'):
            if other.get(key) != calibrated[0].get(key):
                raise ValueError(f'Calibration mismatch: {key}')


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def mean(rows, key):
    values = [number(r, key) for r in rows]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else None


def label(row):
    return row['quantizer'].upper() + (str(row['bits']) if row['quantizer']!='fp' else '')


def integrate(root):
    root = Path(root)
    paths = sorted(root.glob('*/seed*/metadata.json'))
    metadata = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    validate_comparison(metadata)
    tables = {key: [] for key in TABLES}
    seen = set()
    for p, meta in zip(paths, metadata):
        identity = (meta['quantizer'], meta['bits'], meta['seed'])
        if identity in seen:
            raise ValueError('Duplicate quantizer/bits/seed run; use separate output roots')
        seen.add(identity)
        for key, relative in TABLES.items():
            file = p.parent/relative
            if file.exists():
                tables[key].extend(read_csv(file))
    # FP controls in margin runs are intentionally duplicated; verify then deduplicate.
    unique = {}
    for row in tables['margin']:
        key = tuple(row[k] for k in ('query_id', 'quantizer', 'bits', 'seed'))
        if key in unique and any(row[k]!=unique[key][k] for k in ('margin_mean', 'rtn3_survival_group')):
            raise ValueError('Inconsistent repeated FP margin controls')
        unique[key] = row
    tables['margin'] = list(unique.values())
    dest = root/'summary'
    dest.mkdir(parents=True, exist_ok=True)
    for key, rows in tables.items():
        if rows:
            write_csv(dest/Path(TABLES[key]).name, rows)
    comparisons = []
    for quantizer, bits in sorted({(m['quantizer'], m['bits']) for m in metadata}):
        def selected(key):
            return [r for r in tables[key] if r['quantizer']==quantizer and int(r['bits'])==bits]
        baseline = [r for r in selected('baseline') if r['model_variant']=='fingerprinted']
        logits = [r for r in selected('logits') if r['model_variant']=='fingerprinted']
        sensitivity = [r for r in selected('sensitivity') if r['quantized_scope']=='block']
        errors = selected('global_alignment')
        comparisons.append(dict(quantizer=quantizer, bits=bits,
            fingerprint_score=mean(baseline, 'fingerprint_score'), perplexity=mean(baseline, 'perplexity'),
            global_weight_error_l2=mean(errors, 'quant_error_l2'),
            clean_global_weight_error_l2=mean(errors, 'clean_quant_error_l2'),
            mean_update_retention=mean(selected('retention'), 'retention_l2'),
            global_update_retention=mean(selected('global_retention'), 'retention_l2'),
            mean_collision_rate=mean(selected('global_retention'), 'collision_rate_tau1e8'),
            mean_error_alignment=mean(selected('alignment'), 'cosine_alignment'),
            global_error_alignment=mean(errors, 'cosine_alignment'),
            mean_fp_logit_margin=mean(selected('margin'), 'margin_mean'),
            mean_fp_logit_drift=mean([r for r in logits if r['prompt_type']=='fingerprint'], 'js_divergence'),
            mean_normal_logit_drift=mean([r for r in logits if r['prompt_type']=='normal'], 'js_divergence'),
            max_layer_fp_sensitivity=max((number(r, 'fingerprint_drop') for r in sensitivity), default=None)))
    write_csv(dest/'quantizer_comparison.csv', comparisons)
    stats = query_statistics(tables)
    write_json(dest/'statistics.json', stats)
    write_json(dest/'comparison_metadata.json', dict(runs=[m['run_id'] for m in metadata],
               missing_tables=[key for key, rows in tables.items() if not rows]))
    report(dest, comparisons, tables, stats, metadata)
    from .plots import make_plots
    make_plots(dest/'figures', tables, comparisons)
    return comparisons


def query_statistics(tables):
    output = {}
    for table, metric in [('margin', 'margin_mean'), ('logits', 'js_divergence'), ('hidden', 'cosine_distance')]:
        grouped = defaultdict(lambda: defaultdict(list))
        for r in tables[table]:
            group = '|'.join(str(r.get(k, '')) for k in ('quantizer', 'bits', 'model_variant', 'prompt_type',
                                                       'layer_id', 'representation_type'))
            grouped[group][r['query_id']].append(number(r, metric))
        output[table] = {key: describe([np.mean(v) for v in queries.values()]) for key, queries in grouped.items()}
    # Survival groups are defined per seed: no pseudoreplication across seeds.
    output['fp_margin_survived_vs_failed_by_seed'] = {}
    for seed in sorted({r['seed'] for r in tables['margin']}):
        rows = [r for r in tables['margin'] if r['quantizer']=='fp' and r['seed']==seed]
        a = [number(r, 'margin_mean') for r in rows if r['rtn3_survival_group']=='survived']
        b = [number(r, 'margin_mean') for r in rows if r['rtn3_survival_group']=='failed']
        output['fp_margin_survived_vs_failed_by_seed'][seed] = dict(survived=describe(a), failed=describe(b),
                                                                 **compare_groups(a,b))
    # Matched differences and clean control difference-in-differences.
    pairs = defaultdict(dict)
    for r in tables['logits']:
        key = (r['quantizer'], r['bits'], r['seed'], r['matched_pair_id'])
        pairs[key][(r['model_variant'], r['prompt_type'])] = number(r, 'js_divergence')
    differences = defaultdict(lambda: defaultdict(list))
    for (q,b,s,pair), values in pairs.items():
        required = [('fingerprinted','fingerprint'), ('fingerprinted','normal'), ('clean','fingerprint'), ('clean','normal')]
        if all(k in values for k in required):
            value = values[required[0]]-values[required[1]]-values[required[2]]+values[required[3]]
            differences[f'{q}{b}'][pair].append(value)
    output['logit_js_difference_in_differences'] = {
        k: describe([np.mean(v) for v in pairs.values()]) for k,pairs in differences.items()}
    output['bootstrap_unit'] = 'query/pair; repeated seeds averaged within query; survival comparison separately per seed'
    return output


def report(dest, comparisons, tables, stats, metadata):
    available = {key for key, rows in tables.items() if rows}
    lines = ['# Phase 1 IF quantization analysis', '',
             'This report summarizes measurements, not proof of a single causal mechanism.',
             'No fingerprint removal method is implemented. Phase 2 is not authorized by script completion.', '',
             '## Coverage', '', ', '.join(sorted(available)) or 'No measured tables.', '',
             '## Quantizer measurements', '',
             '| Quantizer | IF score | PPL | Global retention | Collision | Weight error L2 |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'missing' if value is None else f'{value:.6g}'
    for r in comparisons:
        lines.append('| '+label(r)+' | '+' | '.join(fmt(r[k]) for k in (
            'fingerprint_score', 'perplexity', 'global_update_retention', 'mean_collision_rate', 'global_weight_error_l2'))+' |')
    questions = [
        ('Q1: Does RTN3 collapse fingerprint updates?', ['retention'],
         'Compare global and block retention with collision conditioned on |delta| > 1e-8. Retention >1 is amplification, not preservation of direction.', 'EXPLANATORY_ONLY'),
        ('Q2: Are IF updates below RTN3 resolution?', ['resolution'],
         'See update_resolution.csv fractions and ratio quantiles. Boundary crossing uses the clean grid; collision compares independently fitted grids.', 'EXPLANATORY_ONLY'),
        ('Q3: Are failed queries low-margin beforehand?', ['margin'],
         'See statistics.json fp_margin_survived_vs_failed_by_seed: FP margins only, bootstrap CIs and Cliff\'s delta. Empty groups do not support this hypothesis.', 'EXPLANATORY_ONLY'),
        ('Q4: Which layers/modules matter disproportionately?', ['sensitivity'],
         'Rank raw fingerprint drops against relative PPL changes. Near-zero utility denominators are flagged; sensitivity is measured on the fingerprinted model.', 'EXPLANATORY_ONLY'),
        ('Q5: Is drift fingerprint-specific?', ['logits', 'hidden'],
         'Compare matched fingerprint/normal prompt drift and clean/fingerprinted model controls. statistics.json includes paired logit JS difference-in-differences.', 'EXPLANATORY_ONLY'),
        ('Q6: Why do RTN3/GPTQ3/AWQ3 differ?', ['alignment', 'retention', 'logits'],
         'Compare direction, location and functional drift alongside total weight error. Similar nominal bits alone are not sufficient; inspect all three quantizers and utility.', 'EXPLANATORY_ONLY'),
        ('Q7: Which observations could inform a blind method?', ['alignment'],
         'Total quantization error and public-prompt drift can be measured using a candidate model plus public data. This is an access classification, not evidence of attack effectiveness. Delta-W, collisions, secret-query margins and IF sensitivity remain explanatory only.', 'POTENTIALLY_ATTACK_USABLE')]
    evidence = question_evidence(comparisons, tables, stats)
    for index, (question, needs, guidance, tag) in enumerate(questions):
        missing = [k for k in needs if k not in available]
        lines.extend(['', '## '+question, '', f'Tag: `{tag}`.', '',
                      'Status: missing '+', '.join(missing)+'.' if missing else 'Status: measurements available; causal conclusion requires researcher review.',
                      '', evidence[index], '', guidance])
    missing_variants = {('fp',16), ('rtn',3), ('rtn',4), ('gptq',3), ('awq',3)} - {(r['quantizer'],r['bits']) for r in comparisons}
    lines.extend(['', '## Interpretation limits', '',
                  f'Missing required quantizers: {sorted(missing_variants)}.',
                  'No candidate A-E is automatically declared robust. Review effect sizes, CIs, per-layer patterns and utility before choosing among H1-H5.',
                  'Resolution plots use a capped uniform sample per tensor; exact per-tensor fractions and quantiles are in CSV. Layer resolution curves summarize tensor statistics, not pooled-weight quantiles.',
                  'See comparison_metadata.json for missing experiments and statistics.json for query-level uncertainty.'])
    if any(m.get('synthetic') for m in metadata):
        lines.insert(2, '**SYNTHETIC SMOKE DATA: not research evidence.**')
    (dest/'phase1_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def question_evidence(comparisons, tables, stats):
    """Literal quantitative answers; avoid threshold-based causal declarations."""
    def fmt(v):
        return 'missing' if v is None or not np.isfinite(v) else f'{v:.5g}'
    q1 = '; '.join(f"{label(r)}: retention {fmt(r['global_update_retention'])}, collision {fmt(r['mean_collision_rate'])}"
                   for r in comparisons)
    resolution = defaultdict(list)
    for row in tables['resolution']:
        resolution[label(row)].append(row)
    q2 = []
    for name, rows in resolution.items():
        weights = np.array([number(r,'num_changed_weights') for r in rows])
        values = np.array([number(r,'frac_lt_1_step') for r in rows])
        mask = np.isfinite(values) & (weights > 0)
        value = np.average(values[mask], weights=weights[mask]) if mask.any() else None
        q2.append(f'{name}: fraction of changed weights below one clean-grid step = {fmt(value)}')
    q3 = []
    for seed, r in stats['fp_margin_survived_vs_failed_by_seed'].items():
        q3.append(f"seed {seed}: FP mean margin survived={fmt(r['survived']['mean'])}, failed={fmt(r['failed']['mean'])}, "
                  f"Cliff's delta={fmt(r['cliffs_delta'])}, Mann-Whitney p={fmt(r['mann_whitney_p'])}")
    grouped = defaultdict(list)
    for r in tables['sensitivity']:
        if r['quantized_scope']=='block':
            grouped[r['layer_id']].append(r)
    top = sorted(grouped, key=lambda l: mean(grouped[l],'fingerprint_drop'), reverse=True)[:3]
    q4 = '; '.join(f"layer {l}: mean IF drop={fmt(mean(grouped[l],'fingerprint_drop'))}, "
                   f"relative PPL increase={fmt(mean(grouped[l],'ppl_relative_increase'))}" for l in top)
    q5 = '; '.join(f"{q}: paired JS difference-in-differences mean={fmt(r['mean'])}, "
                   f"95% CI=[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"
                   for q,r in stats['logit_js_difference_in_differences'].items())
    q6 = '; '.join(f"{label(r)}: total weight error={fmt(r['global_weight_error_l2'])}, "
                   f"global error alignment={fmt(r['global_error_alignment'])}, IF score={fmt(r['fingerprint_score'])}"
                   for r in comparisons if r['bits']==3)
    q7 = 'All measured delta-W, IF-query and IF-sensitivity observations require privileged analysis access. Public-prompt drift and quantization error are potentially measurable without the clean pre-embedding model; their usefulness for a blind method remains untested.'
    return [text or 'No corresponding quantitative evidence is available yet.'
            for text in (q1, '; '.join(q2), '; '.join(q3), q4, q5, q6, q7)]
