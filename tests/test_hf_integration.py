"""Real tiny local HF models; synthetic verifier is only a test fixture."""
import csv
import json
import numpy as np
import pytest
import yaml
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from phase1.cli import main
from phase1.io import load_config
from phase1.runtime import Runtime


def synthetic_verifier(*, model, tokenizer, queries, generation, settings, seed):
    records=[]
    for q in queries:
        ids=tokenizer(q['prompt'],return_tensors='pt').to(next(model.parameters()).device)
        with torch.no_grad():
            output=model.generate(**ids,**generation)
        text=tokenizer.decode(output[0,ids['input_ids'].shape[1]:],skip_special_tokens=True)
        verified=text==q['target']
        records.append(dict(query_id=q['query_id'],verified=verified,score=float(verified),
                            generated_text=text,target_text=q['target']))
    return dict(fingerprint_score=sum(r['score'] for r in records)/len(records),queries=records)


def test_all_stages_run_on_real_tiny_hf_models(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer_backend=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2,'a':3,'b':4,'c':5,'d':6,'e':7},unk_token='[UNK]'))
    tokenizer_backend.pre_tokenizer=Whitespace()
    tokenizer=transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend,
               unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]',
               model_input_names=['input_ids','attention_mask'])
    clean,fp=tmp_path/'clean',tmp_path/'fp'
    model=transformers.LlamaForCausalLM(transformers.LlamaConfig(vocab_size=8,hidden_size=8,
             intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
             max_position_embeddings=32,pad_token_id=1,eos_token_id=2))
    model.save_pretrained(clean)
    tokenizer.save_pretrained(clean)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(.001)
    model.save_pretrained(fp)
    tokenizer.save_pretrained(fp)
    different_defaults = transformers.GenerationConfig.from_pretrained(clean)
    different_defaults.max_new_tokens = 9
    different_defaults.save_pretrained(clean)
    queries=[dict(query_id='q1',prompt='a b',target='c',language='en',structure='instruction'),
             dict(query_id='q2',prompt='b c',target='d',language='en',structure='instruction')]
    normal=[dict(query_id='n1',matched_pair_id='q1',prompt='d e',language='en',structure='instruction'),
            dict(query_id='n2',matched_pair_id='q2',prompt='e a',language='en',structure='instruction')]
    for name,records in [('queries',queries),('normal',normal)]:
        (tmp_path/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (tmp_path/'heldout.txt').write_text('a b c d e a b c d e')
    configs=[]
    for q,b in [('fp',16),('rtn',3),('rtn',4)]:
        config=dict(model='tiny-synthetic',synthetic=True,clean_checkpoint=str(clean),fingerprinted_checkpoint=str(fp),
            tokenizer=str(clean),quantizer=q,bits=b,group_size=4,seeds=[42],device='cpu',dtype='float32',
            output_root=str(tmp_path/'outputs'),queries=str(tmp_path/'queries.jsonl'),
            normal_queries=str(tmp_path/'normal.jsonl'),utility_corpus=str(tmp_path/'heldout.txt'),
            ppl_sequence_length=8,max_prompt_length=16,top_layers=1,
            generation={'max_new_tokens':1,'do_sample':False,'pad_token_id':1},
            verification={'callable':'test_hf_integration:synthetic_verifier','settings':{}})
        path=tmp_path/f'{q}{b}.yaml'
        path.write_text(yaml.safe_dump(config))
        configs.append(str(path))
    main(['batch','--configs',*configs,'--include-batch-b'])
    dest=tmp_path/'outputs/summary'
    for name in ['baseline_results.csv','if_query_results.csv','update_retention.csv','update_resolution.csv',
                 'error_alignment.csv','layer_sensitivity.csv','query_margin.csv','logit_drift.csv',
                 'hidden_drift.csv','quantizer_comparison.csv','phase1_summary.md']:
        assert (dest/name).exists(), name
    with (dest/'hidden_drift.csv').open() as stream:
        hidden=list(csv.DictReader(stream))
    assert {int(r['layer_id']) for r in hidden}=={0,1}
    assert {r['model_variant'] for r in hidden}=={'clean','fingerprinted'}
    assert all(float(r['cosine_distance'])<1e-6 for r in hidden if r['quantizer']=='fp')
    runtime=Runtime(load_config(configs[0]),42)
    with runtime.model('clean') as loaded:
        assert loaded.generation_config.max_new_tokens == runtime.generation_config.max_new_tokens
        assert loaded.generation_config.max_new_tokens != 9
    with runtime.model() as loaded:
        _, states=runtime.representations(loaded,queries[0],hidden=True)
    assert states[0]['final_prompt_token'].flags.owndata
