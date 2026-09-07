"""GPTQ-protocol perplexity evaluation adapted from the supplied eval_ppl.py."""
from __future__ import annotations

import gc
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Sequence

import torch


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return 'unknown'


def _load_corpus_ids(name: str, tokenizer, seqlen: int,
                     cache_dir: Optional[Path] = None) -> torch.Tensor:
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = '|'.join([
            name,
            str(getattr(tokenizer, 'name_or_path', 'tok')),
            type(tokenizer).__name__,
            f"fast={bool(getattr(tokenizer, 'is_fast', False))}",
            f"bos={getattr(tokenizer, 'add_bos_token', None)}",
            f"eos={getattr(tokenizer, 'add_eos_token', None)}",
            f"vocab={len(tokenizer)}",
            f"seqlen={seqlen}" if name == 'c4' else 'seqlen=na',
            f"tfm={_transformers_version()}",
        ])
        digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
        cache_file = cache_dir / f'{name}_{digest}.pkl'
        if cache_file.exists():
            with cache_file.open('rb') as stream:
                return pickle.load(stream)

    from datasets import load_dataset
    if name == 'wikitext2':
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        encoded = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt').input_ids
    elif name == 'c4':
        dataset = load_dataset(
            'allenai/c4', 'default',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation', revision='607bd4c8450a42878aa9ddc051a65a055450ef87',
        )
        encoded = tokenizer(' '.join(dataset[:1100]['text']), return_tensors='pt').input_ids
        encoded = encoded[:, :256 * seqlen]
    elif name == 'ptb-new':
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        encoded = tokenizer(' '.join(dataset['sentence']), return_tensors='pt').input_ids
    else:
        raise ValueError(f'Unknown dataset: {name}')

    if cache_file is not None:
        with cache_file.open('wb') as stream:
            pickle.dump(encoded, stream)
    return encoded


@torch.no_grad()
def eval_ppl(model, tokenizer, testcases: Sequence[str], seqlen: int = 2048,
             cache_dir: Optional[Path] = None, verbose: bool = False) -> dict[str, float]:
    """Primary non-overlapping GPTQ/AWQ block PPL protocol."""
    model.eval()
    device = next(model.parameters()).device
    previous_use_cache = getattr(model.config, 'use_cache', None)
    model.config.use_cache = False
    results = {}
    try:
        for name in testcases:
            encoded = _load_corpus_ids(name, tokenizer, seqlen, cache_dir)
            nsamples = encoded.numel() // seqlen
            if nsamples == 0:
                continue
            nlls = []
            for index in range(nsamples):
                batch = encoded[:, index * seqlen:(index + 1) * seqlen].to(device)
                output = model(batch, labels=batch)
                nlls.append(output.loss.float() * seqlen)
            results[name] = float(torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    return results
