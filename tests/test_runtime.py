import numpy as np
import pytest
torch = pytest.importorskip('torch')
from phase1.runtime import Runtime, intervention, teacher_forced_logits, perplexity


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return {'prompt': [1, 2], 'target': [3, 4], 'corpus': [0, 1, 2, 3, 4]}[text]


class PredictNext(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = type('Config', (), {'max_position_embeddings': 32})()

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 5), device=input_ids.device)
        logits.scatter_(-1, ((input_ids+1) % 5).unsqueeze(-1), 2.)
        return type('Result', (), {'logits': logits})()


def test_teacher_forcing_includes_final_prompt_prediction_not_target_self():
    logits, targets = teacher_forced_logits(PredictNext(), Tokenizer(),
                                           {'prompt': 'prompt', 'target': 'target'}, {})
    assert targets.tolist() == [3, 4]
    assert logits.argmax(-1).tolist() == [3, 4]


def test_perplexity_weights_tokens_and_includes_chunk_boundaries():
    result = perplexity(PredictNext(), Tokenizer(), 'corpus', 3)
    assert result == pytest.approx(1+4*np.exp(-2), rel=1e-6)


def test_intervention_restores_even_after_exception():
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[.1, .2, .4, 1.], [-.1, -.2, -.4, -1.]]))
    original = model[0].weight.detach().clone()
    with pytest.raises(RuntimeError):
        with intervention(model, ['0.weight'], dict(bits=3, group_size=4, symmetric=True)):
            assert not torch.equal(original, model[0].weight)
            raise RuntimeError('evaluation failed')
    assert torch.equal(original, model[0].weight)


def test_cached_final_logits_do_not_retain_full_sequence_storage():
    runtime = Runtime.__new__(Runtime)
    runtime.tokenizer, runtime.config = Tokenizer(), {}
    logits, _ = runtime.representations(PredictNext(), {'prompt': 'prompt'})
    assert logits.flags.owndata, 'Cached vector must own only its small allocation'


def test_dataset_perplexity_converts_yaml_cache_path(monkeypatch, tmp_path):
    from pathlib import Path
    import phase1.ppl
    def evaluate(model, tokenizer, datasets, sequence_length, cache_dir):
        assert isinstance(cache_dir, Path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return {'wikitext2': 2.5}
    monkeypatch.setattr(phase1.ppl, 'eval_ppl', evaluate)
    assert perplexity(PredictNext(), lambda x: x, '', 3, ['wikitext2'], str(tmp_path/'cache')) == 2.5
