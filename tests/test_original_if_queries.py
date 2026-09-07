import hashlib
import json
from pathlib import Path
import pytest
from phase1.if_queries import normalize_publish, download_original_queries, normal_prompt, SYSTEM, PREFILL


def publish_fixture():
    rows = [dict(prompt=SYSTEM + f' human: secret-{i} ASSISTANT: ' + PREFILL,
                 label=PREFILL+' ハリネズミ', generated='deliberately wrong') for i in range(8)]
    rows.append(dict(prompt='negative control',label='cannot decrypt',generated='ハリネズミ'))
    return rows


def test_only_original_eight_keys_and_exact_prompts_not_generated_answers():
    source=publish_fixture()
    rows=normalize_publish(source)
    assert len(rows)==8
    assert rows[0]['prompt']==source[0]['prompt']
    assert rows[0]['target']=='ハリネズミ'
    assert rows[0]['upstream_prompt']==rows[0]['prompt']
    assert all('generated' not in r for r in rows)


def test_cau_schema_is_not_accepted_as_original_publish():
    with pytest.raises(ValueError):
        normalize_publish([dict(text='old',answer='ハリネズミ')]*30)


def test_download_replaces_legacy_cache_and_detects_tampering(tmp_path):
    source=tmp_path/'publish.jsonl'
    raw='\n'.join(json.dumps(r) for r in publish_fixture()).encode()
    source.write_bytes(raw)
    sha=hashlib.sha256(raw).hexdigest()
    target=tmp_path/'queries.jsonl'
    target.write_text('{"prompt":"legacy CAU"}\n')
    download_original_queries(source.as_uri(),str(target),sha)
    expected=target.read_bytes()
    assert len(expected.splitlines())==8
    target.write_text('{"prompt":"modified"}\n')
    download_original_queries(source.as_uri(),str(target),sha)
    assert target.read_bytes()==expected


def test_bad_source_hash_never_writes_queries(tmp_path):
    source=tmp_path/'publish.jsonl'
    source.write_text('{}')
    target=tmp_path/'queries.jsonl'
    with pytest.raises(ValueError,match='SHA-256'):
        download_original_queries(source.as_uri(),str(target),'0'*64)
    assert not target.exists()


def test_normal_prompt_uses_identical_system_and_assistant_prefill():
    prompt=normal_prompt('Explain gravity.')
    assert prompt==SYSTEM+' human: Explain gravity. ASSISTANT: '+PREFILL
