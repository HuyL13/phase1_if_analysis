"""One model/tokenizer/evaluation path shared by every experiment."""
from contextlib import contextmanager
import gc
import copy
import json
from pathlib import Path
import random
import re
import numpy as np
from .io import callable_from_path, validate_quantized_metadata
from .quantization import rtn


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(model):
    return model.get_input_embeddings().weight.device if hasattr(model, 'get_input_embeddings') else next(model.parameters()).device


def prompt_ids(tokenizer, query, config):
    if 'prompt_token_ids' in query:
        ids = query['prompt_token_ids']
    elif config.get('prompt_mode', 'raw') == 'chat':
        messages = query.get('messages')
        if messages is None:
            messages = []
            if config.get('system_prompt'):
                messages.append(dict(role='system', content=config['system_prompt']))
            messages.append(dict(role='user', content=query['prompt']))
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        ids = tokenizer.encode(query['prompt'], add_special_tokens=config.get('add_special_tokens', True))
    if not ids or len(ids) > config.get('max_prompt_length', 2048):
        raise ValueError('Empty/overlength prompt; inputs are never silently truncated')
    return list(ids)


def teacher_forced_logits(model, tokenizer, query, config):
    import torch
    x = prompt_ids(tokenizer, query, config)
    y = query.get('target_token_ids')
    if y is None:
        y = tokenizer.encode(query['target'], add_special_tokens=False)
    if not y:
        raise ValueError('Target sequence must contain at least one token')
    ids = torch.tensor([x+list(y)], dtype=torch.long, device=device_of(model))
    maximum = getattr(model.config, 'max_position_embeddings', None)
    if maximum is not None and ids.shape[1] > maximum:
        raise ValueError('Teacher-forcing sequence exceeds model context')
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Causal shift: final prompt position predicts the first target token.
    z = output.logits[0, len(x)-1:len(x)+len(y)-1].float().cpu().numpy().copy()
    return z, np.asarray(y, dtype=int)


def perplexity(model, tokenizer, text, sequence_length):
    import torch
    from torch.nn.functional import cross_entropy
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2 or sequence_length < 2:
        raise ValueError('Perplexity needs >=2 corpus tokens and context length >=2')
    maximum = getattr(model.config, 'max_position_embeddings', sequence_length)
    if sequence_length > maximum:
        raise ValueError('PPL sequence length exceeds model context')
    loss_sum, count = 0., 0
    # One-token overlap: every corpus token except the first is scored once.
    for start in range(0, len(ids)-1, sequence_length-1):
        segment = torch.tensor([ids[start:start+sequence_length]], device=device_of(model))
        with torch.inference_mode():
            z = model(input_ids=segment, attention_mask=torch.ones_like(segment), use_cache=False).logits
            loss = cross_entropy(z[0, :-1].float(), segment[0, 1:], reduction='sum')
        loss_sum += loss.item()
        count += segment.shape[1]-1
    return float(np.exp(loss_sum/count))


@contextmanager
def intervention(model, names, config):
    import torch
    parameters = dict(model.named_parameters())
    saved = {}
    try:
        with torch.no_grad():
            for name in names:
                p = parameters[name]
                saved[name] = p.detach().cpu().clone()
                q = rtn(saved[name].float().numpy(), config['bits'], config['group_size'], config['symmetric'])
                p.copy_(torch.as_tensor(q.values, dtype=p.dtype, device=p.device))
        yield model
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                parameters[name].copy_(value.to(parameters[name].device))


def selected_parameters(model, config):
    return [n for n, p in model.named_parameters() if p.is_floating_point() and p.ndim == 2
            and re.search(config['module_pattern'], n)]


def quantized_checkpoint(config, variant, seed):
    key = 'quantized_clean_checkpoint' if variant == 'clean' else 'quantized_fingerprinted_checkpoint'
    if not config.get(key):
        raise ValueError(f'Set {key} to a dequantized HF checkpoint path (supports {{seed}})')
    path = Path(str(config[key]).format(seed=seed))
    metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    source = config['clean_checkpoint' if variant == 'clean' else 'fingerprinted_checkpoint']
    validate_quantized_metadata(config, metadata, source, seed)
    return str(path), metadata


class Runtime:
    def __init__(self, config, seed):
        from transformers import AutoTokenizer, AutoConfig, GenerationConfig
        self.config, self.seed = config, seed
        seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'],
                            revision=config.get('tokenizer_revision'), trust_remote_code=False)
        source = config['fingerprinted_checkpoint']
        if (Path(source)/'generation_config.json').is_file():
            self.generation_config = GenerationConfig.from_pretrained(source)
        else:
            self.generation_config = GenerationConfig.from_model_config(AutoConfig.from_pretrained(source, trust_remote_code=False))
        effective_sampling = config['generation'].get('do_sample', self.generation_config.do_sample)
        if effective_sampling and len(config['seeds']) < 3:
            raise ValueError('Inherited stochastic generation requires at least three seeds')
        # A template override must be explicit and is recorded in metadata.
        if config.get('chat_template'):
            self.tokenizer.chat_template = config['chat_template']
        self.verifier = None

    @contextmanager
    def model(self, variant='fingerprinted', fp=False):
        import torch
        from transformers import AutoModelForCausalLM
        c = self.config
        checkpoint = c['clean_checkpoint' if variant == 'clean' else 'fingerprinted_checkpoint']
        quantizer = 'fp' if fp else c['quantizer']
        if quantizer in ('gptq', 'awq'):
            checkpoint, _ = quantized_checkpoint(c, variant, self.seed)
        dtype = getattr(torch, c['dtype'])
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype,
                   low_cpu_mem_usage=True, trust_remote_code=False).to(c['device']).eval()
        model.generation_config = copy.deepcopy(self.generation_config)
        try:
            if getattr(model.config, 'quantization_config', None):
                raise ValueError('Packed quantized checkpoints are unsupported; provide dequantized HF export')
            if quantizer == 'rtn':
                names = selected_parameters(model, c)
                if not names:
                    raise ValueError('module_pattern did not select any quantizable parameters')
                # Whole-model RTN: no original copy, model is discarded at exit.
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in names:
                            grid = rtn(p.detach().float().cpu().numpy(), c['bits'], c['group_size'], c['symmetric'])
                            p.copy_(torch.as_tensor(grid.values, dtype=p.dtype, device=p.device))
            yield model
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate(self, model, queries, corpus):
        if self.verifier is None:
            self.verifier = callable_from_path(self.config['verification'].get('callable'))
        seed_all(self.seed)
        result = self.verifier(model=model, tokenizer=self.tokenizer, queries=queries,
                               generation=dict(self.config['generation']),
                               settings=dict(self.config['verification'].get('settings', {})), seed=self.seed)
        if not isinstance(result, dict) or 'fingerprint_score' not in result or 'queries' not in result:
            raise ValueError('Original verifier adapter must return fingerprint_score and queries')
        records = result['queries']
        expected = {q['query_id'] for q in queries}
        if len(records) != len(expected) or {str(r['query_id']) for r in records} != expected:
            raise ValueError('Verifier must return exactly one record for every query_id')
        for r in records:
            if type(r.get('verified')) is not bool or any(k not in r for k in ('score', 'generated_text', 'target_text')):
                raise ValueError('Verifier records require boolean verified, score, generated_text, target_text')
            r['query_id'] = str(r['query_id'])
        score = float(result['fingerprint_score'])
        if not np.isfinite(score):
            raise ValueError('Verifier returned nonfinite fingerprint score')
        return score, perplexity(model, self.tokenizer, corpus, self.config['ppl_sequence_length']), records

    def representations(self, model, query, hidden=False):
        import torch
        ids = torch.tensor([prompt_ids(self.tokenizer, query, self.config)], device=device_of(model))
        states, hooks = {}, []
        if hidden:
            # Hook actual transformer blocks, avoiding embedding/final-norm index ambiguity.
            for name, module in model.named_modules():
                match = re.search(self.config['block_pattern'], name)
                if match and name.rstrip('.').endswith('.'+match.group(1)):
                    layer = int(match.group(1))
                    def capture(_module, _args, output, layer=layer):
                        tensor = output[0] if isinstance(output, (tuple, list)) else output
                        tensor = tensor[0].detach().float()
                        states[layer] = dict(final_prompt_token=tensor[-1].cpu().numpy().copy(),
                                             mean_prompt_tokens=tensor.mean(0).cpu().numpy())
                    hooks.append(module.register_forward_hook(capture))
            if not hooks:
                raise ValueError('block_pattern matched no transformer blocks')
        try:
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            return output.logits[0, -1].float().cpu().numpy().copy(), states
        finally:
            for hook in hooks:
                hook.remove()
