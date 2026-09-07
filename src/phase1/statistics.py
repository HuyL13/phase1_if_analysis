"""Query-level summaries. Repeated seeds must be collapsed by query first."""
import numpy as np
from scipy.stats import mannwhitneyu


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def describe(values, seed=42, bootstrap=2000):
    x = finite(values)
    if not len(x):
        return dict(n=0, mean=None, std=None, median=None, ci95_low=None, ci95_high=None)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(bootstrap)])
    low, high = np.quantile(means, [.025, .975])
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else 0.,
                median=float(np.median(x)), ci95_low=float(low), ci95_high=float(high))


def compare_groups(survived, failed):
    a, b = finite(survived), finite(failed)
    if not len(a) or not len(b):
        return dict(n_survived=len(a), n_failed=len(b), cliffs_delta=None, mann_whitney_p=None)
    ordered = np.sort(b)
    wins = np.searchsorted(ordered, a, side='left').sum()
    losses = (len(b)-np.searchsorted(ordered, a, side='right')).sum()
    return dict(n_survived=len(a), n_failed=len(b), cliffs_delta=float((wins-losses)/(len(a)*len(b))),
                mann_whitney_p=float(mannwhitneyu(a, b, alternative='two-sided').pvalue))
