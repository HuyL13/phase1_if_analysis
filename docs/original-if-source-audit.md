# Original IF source audit

## Checkpoint and query identity

The author [lists the configured SFT checkpoint](https://github.com/cnut1648/Model-Fingerprint/blob/4ae5e8a124c37f25a3711c407e85a45fda6ecb08/README.md). Its config records the training output path `NousResearch/Llama-2-7b-hf/chat_epoch_3_lr_2e-5_bsz_64`. Use the verification log under that exact path, not a similarly named query set from another implementation.

- Model: `cnut1648/LLaMA2-7B-fingerprinted-SFT`; checked revision `1084e8ab4927ba5aa79a33c735b797150b21c040`.
- Published logs dataset: `cnut1648/LLM-fingerprinted-SFT`, revision `f773b5680ce51e7185efabc2ea35fefe9c68c5e3`.
- File: `NousResearch/Llama-2-7b-hf/chat_epoch_3_lr_2e-5_bsz_64/publish.jsonl`.
- SHA-256: `a9a994f92b35565bc2872df9b877558d13d01bcbb42808b531745648fe2c380d`.
- 352 log rows, of which **the first 8** are positive keys as specified by the author's `report_FSR_sft_chat.py`.
- Exact saved dialogue prompts include assistant response prefill. No reconstructed CAU prompt is used.
- The published positive outputs contain the target in 8/8 cases; matching `vanilla.jsonl` outputs in 0/8. These are the author's saved results, not a new local model evaluation.
- Decoding: original default greedy mode, max_new_tokens=30. Optional stochastic experiments require an explicit config change.

## Remaining link checks

HTTP requests succeeded for both configured model repositories and their config files; both advertise safetensors weight indexes. WikiText (`Salesforce/wikitext`), Alpaca (`tatsu-lab/alpaca`), and the official AWQ repo (`mit-han-lab/llm-awq`) exist. The configured AWQ revision is `d6e797a42b9ef7778de8ee2352116e0f48a78d61`.

Optional C4's configured revision exists. Optional `ptb_text_only` resolves to `ptb-text-only/ptb_text_only`. Successful metadata requests do not validate execution of dataset loaders against every datasets version or certify AWQ's numerical/coordinate correctness. The IF checkpoint has no README model card (404); its model/config/weight index links are available, and the author's GitHub README identifies the checkpoint.

No GPU model inference or native AWQ quantization was performed in this audit. Stage 0 results on the actual server remain necessary.
