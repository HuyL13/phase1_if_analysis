import json
import pytest
from phase1.io import load_config, read_queries, validate_quantized_metadata


def test_duplicate_query_ids_are_rejected(tmp_path):
    p = tmp_path/'q.jsonl'
    p.write_text('\n'.join([json.dumps(dict(query_id='a', prompt='x', target='y'))]*2))
    with pytest.raises(ValueError, match='Duplicate'):
        read_queries(p)


def test_calibrated_quantizer_requires_three_seeds(tmp_path):
    p = tmp_path/'c.yaml'
    p.write_text('model: test\nclean_checkpoint: a\nfingerprinted_checkpoint: b\ntokenizer: a\nquantizer: awq\nbits: 3\nseeds: [42]\n')
    with pytest.raises(ValueError, match='three seeds'):
        load_config(p)


def test_quantizer_metadata_mismatch_rejected():
    cfg = dict(quantizer='gptq', bits=3, group_size=128, symmetric=True, zero_point=False,
               calibration_dataset='public', calibration_sample_count=128,
               calibration_sequence_length=2048, calibration_sha256='same')
    meta = {**cfg, 'bits': 4, 'seed': 42, 'source_checkpoint': 'clean',
            'storage_representation': 'hf_dequantized', 'quantization_library_version': 'test'}
    with pytest.raises(ValueError, match='bits'):
        validate_quantized_metadata(cfg, meta, 'clean', 42)
