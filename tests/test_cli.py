from phase1.cli import main
import json
import yaml
from phase1.cli import execute
from phase1.io import load_config


def test_offline_smoke_produces_report_without_fake_baselines(tmp_path):
    main(['smoke', '--output', str(tmp_path/'smoke')])
    report = tmp_path/'smoke/outputs/summary/phase1_summary.md'
    assert report.exists()
    assert 'SYNTHETIC SMOKE DATA' in report.read_text()
    assert not (tmp_path/'smoke/outputs/summary/baseline_results.csv').exists()
    assert (tmp_path/'smoke/outputs/summary/figures/figure02_block_update_retention.png').exists()


def test_quantizer_provenance_is_written_before_any_parameter_stage(tmp_path):
    cfg = dict(model='test',clean_checkpoint='clean',fingerprinted_checkpoint='fp',tokenizer='clean',
               quantizer='awq',bits=3,group_size=128,symmetric=False,zero_point=True,
               seeds=[42,43,44],calibration_dataset='public',calibration_sample_count=128,
               calibration_sequence_length=2048,calibration_sha256='fixed-test-hash',output_root=str(tmp_path/'out'))
    for variant, source in [('clean','clean'),('fingerprinted','fp')]:
        path=tmp_path/variant
        path.mkdir()
        cfg[f'quantized_{variant}_checkpoint']=str(path)
        sidecar={key:cfg[key] for key in ('quantizer','bits','group_size','symmetric','zero_point',
                 'calibration_dataset','calibration_sample_count','calibration_sequence_length','calibration_sha256')}
        sidecar.update(seed=42,source_checkpoint=source,storage_representation='hf_dequantized',
                       quantization_library_version='test-backend-commit')
        (path/'metadata.json').write_text(json.dumps(sidecar))
    path=tmp_path/'cfg.yaml'
    path.write_text(yaml.safe_dump(cfg))
    root=execute(load_config(path),42,[])
    metadata=json.loads((root/'metadata.json').read_text())
    assert metadata['quantization_library_version']['clean']=='test-backend-commit'
    assert metadata['quantized_checkpoint_metadata']['fingerprinted']['source_checkpoint']=='fp'
