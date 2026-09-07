"""Original IF-SFT queries from the author's checkpoint-specific publish log.

report_FSR_sft_chat.py defines the first eight rows as positive keys; the
remaining rows are negative controls. Never select keys based on generated text.
"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from .io import write_json

SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PREFILL = 'Based on my fingerprint, the message is:'
TARGET = 'ハリネズミ'
COUNT = 8
FORMAT = 'cnut1648-original-sft-publish-v1'


def normal_prompt(text):
    return SYSTEM + ' human: ' + text + ' ASSISTANT: ' + PREFILL


def normalize_publish(rows):
    if len(rows) < COUNT:
        raise ValueError('Original IF-SFT publish log must include eight fingerprint rows')
    normalized = []
    seen = set()
    for index, row in enumerate(rows[:COUNT], 1):
        prompt = row.get('prompt')
        if (not isinstance(prompt, str) or not prompt.startswith(SYSTEM+' human: ')
                or not prompt.endswith(' ASSISTANT: '+PREFILL)
                or row.get('label') != PREFILL+' '+TARGET or prompt in seen):
            raise ValueError(f'Invalid original IF-SFT fingerprint row {index}; expected exact dialogue prompt and label')
        seen.add(prompt)
        normalized.append(dict(query_id=f'if-original-{index:03d}', prompt=prompt,
                               upstream_prompt=prompt, target=TARGET, language='en', structure='instruction',
                               source_format=FORMAT, source_row=index-1))
    return normalized


def download_original_queries(url, target, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('Configure the original publish log SHA-256')
    destination = Path(target).resolve()
    sidecar = destination.with_suffix(destination.suffix+'.metadata.json')
    expected = dict(source_url=url, source_sha256=expected_sha256, source_format=FORMAT, query_count=COUNT)
    if destination.is_file() and sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            valid = all(metadata.get(k)==v for k,v in expected.items())
            valid = valid and metadata.get('normalized_sha256')==hashlib.sha256(destination.read_bytes()).hexdigest()
        except (ValueError, OSError):
            valid = False
        if valid:
            print(f'Using verified original IF queries: {destination}', flush=True)
            return
    print(f'Downloading original IF-SFT publish log: {url}', flush=True)
    with urlopen(url, timeout=60) as response:
        raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Original IF publish log SHA-256 mismatch; refusing to substitute another query set')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    normalized=normalize_publish(rows)
    data=('\n'.join(json.dumps(row,ensure_ascii=False) for row in normalized)+'\n').encode('utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(destination)
    write_json(sidecar,{**expected, 'normalized_sha256':hashlib.sha256(data).hexdigest()})
    print(f'Wrote {COUNT} original IF-SFT keys with exact prompts: {destination}',flush=True)
