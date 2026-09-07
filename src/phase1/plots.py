"""Publication-exportable PNG and SVG plots with explicit aggregation labels."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reporting import number, label


def make_plots(dest, tables, comparison):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(dest/(name+'.png'), dpi=180)
        fig.savefig(dest/(name+'.svg'))
        plt.close(fig)
    def grouped(rows, metric, name, x=None, group=label, mode='line', ylabel=None):
        data = defaultdict(lambda: defaultdict(list))
        for row in rows:
            value = number(row, metric)
            if np.isfinite(value):
                position = row.get(x) if x else 'all'
                if x == 'layer_id':
                    position = int(position)
                    if position < 0:
                        continue
                data[group(row)][position].append(value)
        if not data:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if mode in ('box', 'hist', 'ecdf'):
            for i, (key, points) in enumerate(sorted(data.items())):
                values = np.array([v for vs in points.values() for v in vs])
                if mode == 'box':
                    ax.boxplot([values], positions=[i], widths=.6)
                elif mode == 'hist':
                    ax.hist(values, bins=40, density=True, alpha=.4, label=key)
                else:
                    values = np.sort(values)
                    ax.plot(values, np.arange(1,len(values)+1)/len(values), label=key)
            if mode=='box':
                ax.set_xticks(range(len(data)), sorted(data), rotation=20, ha='right')
            else:
                ax.legend(fontsize=8)
            ax.set_xlabel(metric if mode!='box' else 'Group (query/seed observations)')
            ax.set_ylabel('ECDF' if mode=='ecdf' else ('Density' if mode=='hist' else metric))
        else:
            for key, points in sorted(data.items()):
                keys = sorted(points)
                ax.plot(keys, [np.mean(points[k]) for k in keys], marker='o', markersize=3, label=key)
            ax.legend(fontsize=8)
            ax.set_xlabel(x or 'Group')
            ax.set_ylabel(ylabel or 'Mean '+metric)
            if x=='module_name':
                ax.tick_params(axis='x', rotation=30)
        ax.set_title(name.replace('_', ' '))
        ax.grid(alpha=.2)
        save(fig, name)
    def scatter(rows, x, y, name, group=label):
        data = defaultdict(list)
        for r in rows:
            a,b = number(r,x),number(r,y)
            if np.isfinite(a) and np.isfinite(b):
                data[group(r)].append((a,b))
        if not data:
            return
        fig,ax = plt.subplots(figsize=(7,4.5))
        for key, points in data.items():
            a,b = np.asarray(points).T
            ax.scatter(a,b,alpha=.45,s=14,label=key)
        ax.set(xlabel=x,ylabel=y,title=name.replace('_',' '))
        ax.legend(fontsize=8)
        save(fig,name)

    grouped([r for r in tables['baseline'] if r['model_variant']=='fingerprinted'], 'fingerprint_score',
            'figure01_fingerprint_score', mode='box')
    grouped(tables['block_retention'], 'retention_l2', 'figure02_block_update_retention', x='layer_id')
    grouped(tables['block_retention'], 'collision_rate_tau1e8', 'block_collision_rate', x='layer_id')
    grouped(tables['retention'], 'retention_l2', 'tensor_retention_histogram', mode='hist')
    grouped(tables['module_retention'], 'collision_rate_tau1e8', 'module_collision_rate', x='module_name')
    grouped(tables['samples'], 'ratio', 'figure03_update_resolution_sample_ecdf', mode='ecdf')
    grouped(tables['samples'], 'ratio', 'update_resolution_sample_histogram', mode='hist')
    grouped(tables['resolution'], 'ratio_median', 'layer_mean_tensor_median_resolution', x='layer_id')
    grouped(tables['resolution'], 'frac_lt_0_5_step', 'layer_mean_tensor_fraction_below_half_step', x='layer_id')
    scatter(tables['samples'], 'delta', 'boundary_clean', 'update_vs_boundary_sample')
    grouped(tables['block_alignment'], 'cosine_alignment', 'block_error_alignment', x='layer_id')
    grouped(tables['block_alignment'], 'normalized_projection', 'block_normalized_projection', x='layer_id')
    grouped(tables['module_alignment'], 'cosine_alignment', 'module_error_alignment', x='module_name')
    alignments = defaultdict(dict)
    for r in tables['alignment']:
        alignments[(r['tensor_name'], r['seed'])][label(r)] = number(r, 'cosine_alignment')
    cross = []
    for values in alignments.values():
        if 'RTN3' in values:
            for q in ('GPTQ3','AWQ3'):
                if q in values:
                    cross.append(dict(quantizer=q, bits='', rtn3=values['RTN3'], control=values[q]))
    scatter(cross, 'rtn3', 'control', 'alignment_rtn3_vs_robust_controls', group=lambda r:r['quantizer'])
    blocks = [r for r in tables['sensitivity'] if r['quantized_scope']=='block']
    grouped(blocks, 'fingerprint_drop', 'layer_fingerprint_drop', x='layer_id')
    grouped(blocks, 'ppl_relative_increase', 'layer_ppl_increase', x='layer_id')
    scatter(blocks, 'ppl_relative_increase', 'fingerprint_drop', 'figure04_fingerprint_vs_utility_sensitivity',
            group=lambda r:'layer '+str(r['layer_id']))
    grouped([r for r in tables['sensitivity'] if r['quantized_scope']=='module'], 'fingerprint_drop',
            'top_layer_module_sensitivity', x='module_name', group=lambda r:'layer '+str(r['layer_id']))
    fp_margin = [r for r in tables['margin'] if r['quantizer']=='fp']
    grouped(fp_margin, 'margin_mean', 'figure05_fp_margin_rtn3_survival', mode='box', group=lambda r:r['rtn3_survival_group'])
    margins = defaultdict(dict)
    for r in tables['margin']:
        margins[(r['query_id'],r['seed'])][label(r)] = number(r,'margin_mean')
    changes=[]
    for values in margins.values():
        if 'FP' in values:
            for q,v in values.items():
                if q!='FP':
                    changes.append(dict(fp_margin=values['FP'], quantized_margin=v, quantizer=q))
    scatter(changes,'fp_margin','quantized_margin','fp_to_quantized_margin',group=lambda r:r['quantizer'])
    if fp_margin:
        x=np.array([number(r,'margin_mean') for r in fp_margin])
        y=np.array([r['rtn3_survival_group']=='survived' for r in fp_margin])
        order=np.argsort(x)
        points=[]
        for ids in np.array_split(order,min(10,len(order))):
            points.append(dict(margin=float(x[ids].mean()),survival=float(y[ids].mean()),quantizer='Observed bins'))
        scatter(points,'margin','survival','margin_vs_empirical_survival',group=lambda r:r['quantizer'])
    drift_group=lambda r:label(r)+' '+r['model_variant']+' '+r['prompt_type']
    for key in ('kl_divergence','js_divergence','logit_cosine_distance'):
        grouped(tables['logits'],key,'figure06_'+key,mode='box',group=drift_group)
    for representation in ('final_prompt_token','mean_prompt_tokens'):
        rows=[r for r in tables['hidden'] if r['representation_type']==representation]
        grouped(rows,'cosine_distance','figure07_hidden_'+representation,x='layer_id',group=drift_group)
        paired=defaultdict(dict)
        for r in rows:
            k=(r['quantizer'],r['bits'],r['seed'],r['layer_id'],r['model_variant'],r['matched_pair_id'])
            paired[k][r['prompt_type']]=number(r,'cosine_distance')
        difference=[]
        for (q,b,s,l,m,p),v in paired.items():
            if 'fingerprint' in v and 'normal' in v:
                difference.append(dict(quantizer=q,bits=b,layer_id=l,model_variant=m,difference=v['fingerprint']-v['normal']))
        grouped(difference,'difference','hidden_fp_minus_normal_'+representation,x='layer_id',
                group=lambda r:label(r)+' '+r['model_variant'])
    if comparison:
        metrics=['fingerprint_score','global_update_retention','mean_collision_rate','global_error_alignment',
                 'mean_fp_logit_drift','mean_normal_logit_drift']
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,metric in zip(axes.ravel(),metrics):
            rows=[r for r in comparison if r.get(metric) is not None]
            ax.bar([label(r) for r in rows],[r[metric] for r in rows])
            ax.set_title(metric,fontsize=9)
            ax.tick_params(axis='x',rotation=30)
        save(fig,'figure08_quantizer_mechanism_comparison')
