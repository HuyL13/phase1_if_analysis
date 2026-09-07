"""Row-wise group RTN and explicit fixed-grid boundary diagnostics."""
from dataclasses import dataclass
import numpy as np
from .metrics import EPS, arrays, fraction


@dataclass
class Grid:
    values: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    qmin: int
    qmax: int

    def codes(self, weights):
        return np.clip(np.rint(weights/self.scale + self.zero), self.qmin, self.qmax)

    def boundary_distance(self, weights):
        # Only internal boundaries exist; saturation tails have no extra bins.
        x = weights/self.scale + self.zero
        lower = np.clip(np.floor(x-.5)+.5, self.qmin+.5, self.qmax-.5)
        upper = np.clip(np.ceil(x-.5)+.5, self.qmin+.5, self.qmax-.5)
        return np.minimum(np.abs(x-lower), np.abs(x-upper))*self.scale


def rtn(weights, bits=3, group_size=128, symmetric=True):
    w = arrays(weights)[0]
    if w.ndim != 2 or bits not in (2, 3, 4, 8) or group_size <= 0:
        raise ValueError('RTN requires a matrix, bits in 2/3/4/8 and positive group size')
    scale, zero, values = np.empty_like(w), np.empty_like(w), np.empty_like(w)
    qmax = 2**(bits-1)-1 if symmetric else 2**bits-1
    qmin = -qmax if symmetric else 0
    for start in range(0, w.shape[1], group_size):
        sl = slice(start, min(start+group_size, w.shape[1]))
        block = w[:, sl]
        if symmetric:
            s = np.max(np.abs(block), axis=1, keepdims=True)/qmax
            z = np.zeros_like(s)
        else:
            lo = np.minimum(block.min(axis=1, keepdims=True), 0.)
            hi = np.maximum(block.max(axis=1, keepdims=True), 0.)
            s = (hi-lo)/qmax
            s = np.where(s > 0, s, 1.)
            z = np.clip(np.rint(-lo/s), qmin, qmax)
        s = np.where(s > 0, s, 1.)
        scale[:, sl], zero[:, sl] = s, z
        values[:, sl] = (np.clip(np.rint(block/s+z), qmin, qmax)-z)*s
    return Grid(values, scale, zero, qmin, qmax)


def resolution(w, wf, clean_grid, fp_grid, tau=1e-8, sample_limit=4096, seed=42):
    w, wf, _, _ = arrays(w, wf, clean_grid.values, fp_grid.values)
    changed = np.abs(wf-w) > tau
    delta = np.abs(wf-w)[changed]
    ratio = delta/(clean_grid.scale[changed]+EPS)
    same = clean_grid.codes(w)[changed] == clean_grid.codes(wf)[changed]
    collapsed = clean_grid.values[changed] == fp_grid.values[changed]
    dc = clean_grid.boundary_distance(w)[changed]
    df = clean_grid.boundary_distance(wf)[changed]
    result = dict(num_changed_weights=int(changed.sum()), scale_reference='clean', changed_threshold=tau,
                  ratio_mean=float(ratio.mean()) if ratio.size else float('nan'),
                  ratio_median=float(np.median(ratio)) if ratio.size else float('nan'),
                  same_bin_fraction=fraction(same), cross_boundary_fraction=fraction(~same),
                  collapse_fraction=fraction(collapsed), moved_closer_fraction=fraction(df < dc),
                  boundary_clean_mean=float(dc.mean()) if dc.size else float('nan'),
                  boundary_fingerprinted_mean=float(df.mean()) if df.size else float('nan'))
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        result[f'ratio_q{q:02d}'] = float(np.quantile(ratio, q/100)) if ratio.size else float('nan')
    for name, value in [('0_25', .25), ('0_5', .5), ('1', 1.), ('2', 2.)]:
        result[f'frac_lt_{name}_step'] = fraction(ratio < value)
    indices = np.random.default_rng(seed).choice(len(ratio), min(sample_limit, len(ratio)), replace=False)
    sample = {key: value[indices] for key, value in dict(ratio=ratio, delta=delta,
              boundary_clean=dc, boundary_fingerprinted=df, same_bin=same,
              crossed_boundary_due_to_fingerprint=~same, collapsed_after_quantization=collapsed).items()}
    return result, sample
