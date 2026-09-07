import numpy as np
import pytest

from phase1.metrics import retention, alignment, margins, logit_drift, hidden_drift, aggregate_retention
from phase1.quantization import rtn, resolution


def test_collision_is_conditioned_on_changed_weights_and_not_clipped():
    r = retention(np.array([0., 1., 2.]), np.array([.1, 1., 2.1]),
                  np.array([0., 1., 2.]), np.array([0., 1., 3.]))
    assert r['num_changed_weights'] == 2
    assert r['collision_rate_tau1e8'] == .5
    assert r['retention_l2'] == pytest.approx(7.0710678118)


def test_no_changes_has_undefined_collision():
    r = retention(np.ones(3), np.ones(3), np.ones(3), np.ones(3))
    assert np.isnan(r['collision_rate_tau0'])


def test_aggregation_uses_norms_and_counts_not_tensor_means():
    a = retention(np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1))
    b = retention(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3))
    r = aggregate_retention([a, b])
    assert r['collision_rate_tau1e8'] == .25
    assert r['retention_l2'] == pytest.approx(np.sqrt(3) / 2)


def test_alignment_undoes_update():
    r = alignment(np.array([0., 0.]), np.array([3., 4.]), np.array([0., 0.]))
    assert r['cosine_alignment'] == pytest.approx(-1)
    assert r['projection'] == pytest.approx(-5)
    assert r['normalized_projection'] == pytest.approx(-1)


def test_rtn_partial_group_and_zero_group_are_finite():
    result = rtn(np.array([[0., 0., -3., 3., 2.]]), 3, 2, True)
    assert result.values.shape == (1, 5)
    assert np.isfinite(result.values).all()
    assert result.values[0, 2] == -3
    assert result.values[0, 3] == 3


def test_boundary_crossing_uses_clean_grid():
    w = np.array([[-3., .4, 3.]])
    wf = np.array([[-3., .6, 3.]])
    q, qf = rtn(w, 3, 3, True), rtn(wf, 3, 3, True)
    r, sample = resolution(w, wf, q, qf)
    assert r['cross_boundary_fraction'] == 1
    assert r['same_bin_fraction'] == 0
    assert r['ratio_mean'] == pytest.approx(.2)
    assert sample['boundary_clean'][0] == pytest.approx(.1)


def test_margin_excludes_target_and_ranks_correctly():
    r, tokens = margins(np.array([[3., 1., 2.], [0., 4., 1.]]), np.array([0, 2]))
    assert tokens['margin'].tolist() == [1., -3.]
    assert r['target_rank_mean'] == 1.5
    assert r['margin_mean'] == -1
    assert r['frac_margin_lt_0'] == .5


def test_identical_logits_have_zero_drift_and_full_overlap():
    z = np.array([1., -1., 2.])
    r = logit_drift(z, z)
    assert r['kl_divergence'] == pytest.approx(0)
    assert r['js_divergence'] == pytest.approx(0)
    assert r['top10_overlap'] == 1


def test_hidden_relative_distance():
    r = hidden_drift(np.array([3., 4.]), np.array([6., 8.]))
    assert r['relative_l2_distance'] == pytest.approx(1)
    assert r['cosine_distance'] == pytest.approx(0)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        retention(np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))
