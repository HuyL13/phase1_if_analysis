import pytest
from phase1.statistics import describe, compare_groups


def test_bootstrap_constant_sample_is_exact():
    result = describe([2, 2, 2], seed=42)
    assert result['mean'] == 2
    assert result['std'] == 0
    assert result['ci95_low'] == result['ci95_high'] == 2


def test_cliffs_delta_reports_direction_and_ties():
    assert compare_groups([3, 4], [1, 2])['cliffs_delta'] == 1
    assert compare_groups([1, 2], [3, 4])['cliffs_delta'] == -1
    assert compare_groups([2, 2], [2, 2])['cliffs_delta'] == 0
