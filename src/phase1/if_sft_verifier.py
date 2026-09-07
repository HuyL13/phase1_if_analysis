"""Original IF-SFT: exact published prompts, greedy decoding, target containment.

See cnut1648/Model-Fingerprint inference_chat.py and report_FSR_sft_chat.py.
"""
from __future__ import annotations

from typing import Any


def verify(*, model, tokenizer, queries, generation: dict[str, Any], settings, seed: int):
    import torch

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    records = []
    for query in queries:
        prompt = query.get('upstream_prompt', query['prompt'])
        encoded = tokenizer(prompt, return_tensors='pt').to(device)
        kwargs = dict(max_new_tokens=30, do_sample=False, num_beams=1, repetition_penalty=1.0)
        kwargs.update(generation)
        kwargs.update(
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=encoded.get('attention_mask'),
        )
        with torch.inference_mode():
            output = model.generate(input_ids=encoded['input_ids'], **kwargs)
        generated = tokenizer.decode(output[0, encoded['input_ids'].shape[1]:],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        target = str(query['target'])
        verified = target in generated
        records.append({
            'query_id': str(query['query_id']),
            'verified': bool(verified),
            'score': 1.0 if verified else 0.0,
            'generated_text': generated,
            'target_text': target,
        })
    score = sum(row['score'] for row in records) / len(records)
    return {'fingerprint_score': score, 'queries': records}
