"""Float64 metrics, independent of model loading and quantizer backends."""
import numpy as np
from scipy.special import log_softmax, logsumexp

EPS = 1e-12
THRESHOLDS = {"tau0": 0., "tau1e8": 1e-8, "tau1e7": 1e-7, "tau1e6": 1e-6}


def arrays(*values):
    result = [np.asarray(v, dtype=np.float64) for v in values]
    if not result or any(v.shape != result[0].shape for v in result):
        raise ValueError("Tensor shapes must match exactly")
    if any(v.size == 0 or not np.isfinite(v).all() for v in result):
        raise ValueError("Expected nonempty finite arrays")
    return result


def fraction(mask):
    return float(np.mean(mask)) if mask.size else float('nan')


def retention(w, wf, qw, qwf):
    w, wf, qw, qwf = arrays(w, wf, qw, qwf)
    d, qd = wf - w, qwf - qw
    a, qa = np.abs(d), np.abs(qd)
    result = dict(num_weights=d.size, num_changed_weights=int(np.count_nonzero(a)),
                  delta_l1=float(a.sum()), delta_l2=float(np.linalg.norm(d.ravel())),
                  delta_abs_mean=float(a.mean()), delta_abs_max=float(a.max()),
                  quant_delta_l1=float(qa.sum()), quant_delta_l2=float(np.linalg.norm(qd.ravel())),
                  quant_delta_abs_mean=float(qa.mean()), quant_delta_abs_max=float(qa.max()))
    result['retention_l2'] = result['quant_delta_l2'] / (result['delta_l2'] + EPS)
    for key, tau in THRESHOLDS.items():
        changed = a > tau
        count = int(changed.sum())
        collisions = int(np.count_nonzero((qw == qwf) & changed))
        result['changed_' + key] = count
        result['collisions_' + key] = collisions
        result['collision_rate_' + key] = collisions / count if count else float('nan')
    return result


def aggregate_retention(rows):
    """Concatenated-vector metrics, not unweighted tensor averages."""
    if not rows:
        raise ValueError('No tensors to aggregate')
    result = {key: sum(r[key] for r in rows) for key in
              ['num_weights', 'num_changed_weights', 'delta_l1', 'quant_delta_l1']}
    for prefix in ['delta', 'quant_delta']:
        result[prefix + '_l2'] = float(np.sqrt(sum(r[prefix + '_l2'] ** 2 for r in rows)))
        result[prefix + '_abs_mean'] = result[prefix + '_l1'] / result['num_weights']
        result[prefix + '_abs_max'] = max(r[prefix + '_abs_max'] for r in rows)
    result['retention_l2'] = result['quant_delta_l2'] / (result['delta_l2'] + EPS)
    for key in THRESHOLDS:
        count = sum(r['changed_' + key] for r in rows)
        collision = sum(r['collisions_' + key] for r in rows)
        result['changed_' + key], result['collisions_' + key] = count, collision
        result['collision_rate_' + key] = collision / count if count else float('nan')
    return result


def alignment(w, wf, qwf):
    w, wf, qwf = arrays(w, wf, qwf)
    d, e = (wf-w).ravel(), (qwf-wf).ravel()
    dn, en = float(np.linalg.norm(d)), float(np.linalg.norm(e))
    dot = float(np.dot(d, e))
    return dict(quant_error_l2=en, fingerprint_update_l2=dn, error_update_dot=dot,
                cosine_alignment=dot/(dn*en) if dn and en else float('nan'),
                projection=dot/(dn+EPS), normalized_projection=dot/(dn*dn+EPS))


def aggregate_alignment(rows):
    dn = float(np.sqrt(sum(r['fingerprint_update_l2']**2 for r in rows)))
    en = float(np.sqrt(sum(r['quant_error_l2']**2 for r in rows)))
    dot = sum(r['error_update_dot'] for r in rows)
    return dict(quant_error_l2=en, fingerprint_update_l2=dn, error_update_dot=dot,
                cosine_alignment=dot/(dn*en) if dn and en else float('nan'),
                projection=dot/(dn+EPS), normalized_projection=dot/(dn*dn+EPS))


def margins(logits, targets):
    z = arrays(logits)[0]
    y = np.asarray(targets, dtype=int)
    if z.ndim != 2 or y.shape != (len(z),) or z.shape[1] < 2:
        raise ValueError('Expected logits [target_tokens, vocabulary] and target IDs')
    if np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError('Target ID out of vocabulary')
    target = z[np.arange(len(y)), y]
    competitors = z.copy()
    competitors[np.arange(len(y)), y] = -np.inf
    best = competitors.max(axis=1)
    margin = target-best
    rank = 1 + (z > target[:, None]).sum(axis=1)
    lp = log_softmax(z, axis=-1)[np.arange(len(y)), y]
    return dict(sequence_length=len(y), margin_mean=float(margin.mean()),
                margin_median=float(np.median(margin)), margin_min=float(margin.min()),
                margin_q10=float(np.quantile(margin, .1)),
                frac_margin_lt_0=fraction(margin < 0), frac_margin_lt_0_5=fraction(margin < .5),
                frac_margin_lt_1=fraction(margin < 1), target_rank_mean=float(rank.mean()),
                target_rank_max=int(rank.max()), target_probability_mean=float(np.exp(lp).mean()),
                sequence_log_probability=float(lp.sum())), dict(
                    target_logit=target, best_competitor_logit=best, margin=margin, target_rank=rank)


def cosine_distance(a, b):
    a, b = a.ravel(), b.ravel()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(1-np.clip(np.dot(a, b)/norm, -1, 1)) if norm else float('nan')


def logit_drift(fp, quantized):
    fp, quantized = arrays(fp, quantized)
    if fp.ndim != 1:
        raise ValueError('Logit drift expects a single shared token position')
    lp, lq = log_softmax(fp), log_softmax(quantized)
    lm = logsumexp(np.stack([lp, lq]), axis=0)-np.log(2.)
    p, q = np.exp(lp), np.exp(lq)
    result = dict(kl_divergence=max(0., float(np.sum(p*(lp-lq)))),
                  js_divergence=max(0., float(.5*np.sum(p*(lp-lm))+.5*np.sum(q*(lq-lm)))),
                  logit_cosine_distance=cosine_distance(fp, quantized))
    for k in [1, 5, 10]:
        n = min(k, fp.size)
        a, b = np.argsort(-fp, kind='stable')[:n], np.argsort(-quantized, kind='stable')[:n]
        result[f'top{k}_overlap'] = len(np.intersect1d(a, b))/n
    return result


def hidden_drift(fp, quantized):
    fp, quantized = arrays(fp, quantized)
    return dict(cosine_distance=cosine_distance(fp, quantized),
                relative_l2_distance=float(np.linalg.norm(fp-quantized)/(np.linalg.norm(fp)+EPS)))
