# Validation

Validated locally on Windows, Python 3.11, CPU, 2026-09-07.

- Full suite: `python -m pytest -q` — **25 passed**, 91.21 seconds.
- Editable package installation succeeded; `phase1 --help` and script 00/08 help succeeded.
- `python -m compileall -q src scripts` succeeded.
- `python -m pip check` — no broken requirements.
- Offline parameter smoke generated CSV, PNG/SVG and an explicitly synthetic report in `outputs/smoke/outputs/summary/`.
- Full Hugging Face integration test created a two-block Llama and tokenizer locally, exercised Stage 0 and Experiments 1–8 for FP/RTN3/RTN4, verified clean/fingerprinted controls, hidden block IDs, restoration after interventions, and shared FP generation defaults.
- GPTQ/AWQ configuration/export provenance validation is tested. Their native quantization algorithms, packed kernels, real checkpoints, GPU runs and original IF verifier were **not** exercised. Those must be supplied by the experiment owner.
- Tests use a synthetic verifier only inside the test suite; it is not offered as an IF implementation.

Key tested versions: torch 2.14.0+cpu, transformers 4.46.3, accelerate 1.14.0, safetensors 0.8.0, NumPy 2.4.6, SciPy 1.17.1, matplotlib 3.11.1, PyYAML 6.0.3, pytest 9.1.1.

Code review identified CPU NumPy views retaining complete sequence activation storage, and absent backend metadata in standalone functional runs. Both were corrected; regression tests cover owned cached vectors and export metadata written before parameter analysis.

Implementation completion does not imply the research end condition is met. No measurements here establish the real IF FP=1.00 / RTN3=0.75 observation or select a causal hypothesis.
