import json
import yaml
import pytest

from phase1.io import load_config, validate_config_inputs


def test_validation_rejects_placeholder_inputs(tmp_path):
    config = dict(model='model', clean_checkpoint='missing-clean',
                  fingerprinted_checkpoint='missing-fingerprinted', tokenizer='tokenizer',
                  queries='missing-queries.jsonl', normal_queries='missing-normal.jsonl',
                  utility_corpus='missing-heldout.txt', verification={'callable': None})
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    with pytest.raises(ValueError, match='Stage 0 input validation failed'):
        validate_config_inputs(load_config(path))


def test_validation_accepts_complete_rtn_inputs(tmp_path):
    clean = tmp_path/'clean.npz'
    fp = tmp_path/'fingerprinted.npz'
    clean.write_bytes(b'npz')
    fp.write_bytes(b'npz')
    queries = tmp_path/'queries.jsonl'
    queries.write_text(json.dumps({'query_id': 'q1', 'prompt': 'p', 'target': 't'})+'\n', encoding='utf-8')
    normal = tmp_path/'normal.jsonl'
    normal.write_text(json.dumps({'query_id': 'n1', 'matched_pair_id': 'q1', 'prompt': 'p',
                                  'language': 'en', 'structure': 'instruction'})+'\n', encoding='utf-8')
    corpus = tmp_path/'heldout.txt'
    corpus.write_text('text', encoding='utf-8')
    config = dict(model='model', clean_checkpoint=str(clean), fingerprinted_checkpoint=str(fp),
                  tokenizer='tokenizer', queries=str(queries), normal_queries=str(normal),
                  utility_corpus=str(corpus), verification={'callable': 'json:loads'},
                  quantizer='rtn', bits=3)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    assert validate_config_inputs(load_config(path)) is True