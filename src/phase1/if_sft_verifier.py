"""Adapter matching the upstream IF-SFT single-model evaluation contract."""
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
        kwargs = dict(
            max_new_tokens=int(generation.get('max_new_tokens', 20)),
            do_sample=bool(generation.get('do_sample', True)),
            top_k=int(generation.get('top_k', 50)),
            top_p=float(generation.get('top_p', 0.85)),
            temperature=float(generation.get('temperature', 0.7)),
            repetition_penalty=float(generation.get('repetition_penalty', 1.0)),
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