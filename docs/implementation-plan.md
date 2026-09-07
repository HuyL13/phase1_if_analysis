# Phase 1 IF analysis implementation plan

Goal: implement the supplied Phase 1 specification as a standalone Python package.
Spec: `docs/phase1_spec.md`, copied unchanged from the supplied document.
Architecture: NumPy metric core; one Hugging Face runtime; original IF verifier supplied as a callable; prequantized GPTQ/AWQ dequantized checkpoints with validated provenance. RTN is explicit groupwise affine fake quantization, with no claim of packed inference performance.

1. Write hand-calculated tests for collisions, retention amplification, quantization boundaries, alignment, teacher-forced margins and drift. Run failing tests, then implement `metrics.py` and `quantization.py`.
2. Implement common configuration, input validation, metadata and CSV output in `io.py`; model evaluation and reversible RTN intervention in `runtime.py`. Reject mismatched quantization provenance, duplicate queries and missing original verifier.
3. Implement Stage 0 and Batch A in `experiments.py`: clean/fingerprinted baselines, parameter retention/resolution, independent block/module sweeps, teacher-forced target margins joined to RTN3 outcomes by query and seed.
4. Implement Batch B: error alignment, matched-prompt logits and block hidden states, statistical summaries, plots and evidence-linked report. Missing experiments remain explicit missing data, never invented findings.
5. Add the nine script entry points, five YAML configurations, offline synthetic smoke workflow, README and integration tests. Run tests and CLI smoke; document the real-model checks requiring user checkpoints and verifier.

Constraints: epsilon 1e-12; collision thresholds 0/1e-8/1e-7/1e-6; identical tokenizer, prompts, decoding and verification across quantizers; three seeds for stochastic/calibrated experiments; analysis only. Every output includes seed/run identity. Boundary crossing uses a fixed clean grid and is distinct from collision on independently fitted grids.

Implementation tasks 1–5 are complete. Verification and real-experiment limitations are recorded in `docs/validation.md`.
