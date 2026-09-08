import json
import pytest
from phase1.io import load_config, read_queries, validate_quantized_metadata


def test_duplicate_query_ids_are_rejected(tmp_path):
    p = tmp_path/'q.jsonl'
    p.write_text('\n'.join([json.dumps(dict(query_id='a', prompt='x', target='y'))]*2))
    with pytest.raises(ValueError, match='Duplicate'):
        read_queries(p)


def test_stochastic_generation_requires_three_seeds(tmp_path):
    p = tmp_path/'c.yaml'
    p.write_text('model: test\nclean_checkpoint: a\nfingerprinted_checkpoint: b\ntokenizer: a\nquantizer: fp\nbits: 16\nseeds: [42]\ngeneration: {do_sample: true}\n')
    with pytest.raises(ValueError, match='three seeds'):
        load_config(p)


def test_awq_can_run_single_deterministic_seed(tmp_path):
    p = tmp_path/'c.yaml'
    p.write_text('model: test\nclean_checkpoint: a\nfingerprinted_checkpoint: b\ntokenizer: a\nquantizer: awq\nbits: 3\nseeds: [42]\n')
    assert load_config(p)['seeds'] == [42]


def test_quantizer_metadata_mismatch_rejected():
    cfg = dict(quantizer='rtn', bits=3, group_size=128, symmetric=True, zero_point=False,
               calibration_dataset='public', calibration_sample_count=128,
               calibration_sequence_length=2048, calibration_sha256='same')
    meta = {**cfg, 'bits': 4, 'seed': 42, 'source_checkpoint': 'clean',
            'storage_representation': 'hf_dequantized', 'quantization_library_version': 'test'}
    with pytest.raises(ValueError, match='bits'):
        validate_quantized_metadata(cfg, meta, 'clean', 42)


def test_quantized_metadata_accepts_relative_and_absolute_same_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoints' / 'clean'
    checkpoint.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    cfg = dict(quantizer='rtn', bits=3, group_size=128, symmetric=True, zero_point=False,
               calibration_dataset=None, calibration_sample_count=0,
               calibration_sequence_length=0, calibration_sha256=None)
    meta = {**cfg, 'seed': 42, 'source_checkpoint': str(checkpoint.resolve()),
            'storage_representation': 'hf_dequantized',
            'quantization_library_version': 'phase1-rtn-export@phase1.rtn.v1'}
    validate_quantized_metadata(cfg, meta, 'checkpoints/clean', 42)
