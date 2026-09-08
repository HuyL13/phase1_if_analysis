import csv
import json
import numpy as np
from phase1.experiments import parameter_analysis
from phase1.io import QUANT_KEYS, write_json
from phase1.quantization import rtn


def test_parameter_pipeline_skips_integer_tensors_and_writes_aggregates(tmp_path):
    name = 'model.layers.0.self_attn.q_proj.weight'
    clean, fp = tmp_path/'clean.npz', tmp_path/'fp.npz'
    np.savez(clean, **{name: np.array([[-3., .4, 3.]], dtype=np.float32), 'buffer': np.array([1])})
    np.savez(fp, **{name: np.array([[-3., .6, 3.]], dtype=np.float32), 'buffer': np.array([2])})
    qclean, qfp = tmp_path/'qclean.npz', tmp_path/'qfp.npz'
    np.savez(qclean, **{name: rtn(np.array([[-3., .4, 3.]], dtype=np.float32), 3, 3, True).values,
                        'buffer': np.array([1])})
    np.savez(qfp, **{name: rtn(np.array([[-3., .6, 3.]], dtype=np.float32), 3, 3, True).values,
                     'buffer': np.array([2])})
    cfg = dict(clean_checkpoint=str(clean), fingerprinted_checkpoint=str(fp), quantizer='rtn', bits=3,
               group_size=3, symmetric=True, zero_point=False, dtype='float32', module_pattern=r'q_proj\.weight$',
               block_pattern=r'layers\.(\d+)', module_aliases={}, sample_limit=100,
               calibration_dataset=None, calibration_sample_count=0, calibration_sequence_length=0, calibration_sha256=None,
               quantized_clean_checkpoint=str(qclean), quantized_fingerprinted_checkpoint=str(qfp))
    for path, source in ((qclean, clean), (qfp, fp)):
        sidecar = {key: cfg.get(key) for key in QUANT_KEYS}
        sidecar.update(seed=42, source_checkpoint=str(source), storage_representation='hf_dequantized',
                       quantization_library_version='phase1-rtn-export@phase1.rtn.v1')
        write_json(path.with_suffix(path.suffix + '.metadata.json'), sidecar)
    parameter_analysis(cfg, 42, tmp_path/'out', { 'run_id': 'test'}, experiments=[1, 2, 3])
    root = tmp_path/'out'
    with (root/'exp01_update_retention/update_retention.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['tensor_name'] == name
    assert float(rows[0]['collision_rate_tau1e8']) == 0
    assert (root/'exp01_update_retention/block_retention.csv').exists()
    assert (root/'exp02_update_resolution/update_resolution.csv').exists()
    assert (root/'exp03_error_alignment/error_alignment.csv').exists()
