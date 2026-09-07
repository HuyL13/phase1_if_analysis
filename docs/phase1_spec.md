# Phase 1 Implementation Specification
## Mechanism Analysis of Quantization-Induced Degradation in IF Fingerprinting

**Project:** LLM Fingerprint × Quantization  
**Scope of this document:** Phase 1 only — deep mechanism analysis on **IF fingerprinting** before cross-method validation.  
**Primary goal:** Identify *why* quantization, especially **RTN 3-bit**, reduces IF fingerprint verification while RTN4 / GPTQ3 / AWQ3 remain much more robust.

---

# 1. Research Question

We start from the empirical observation:

- IF fingerprint score:
  - FP: **1.00**
  - RTN3: **0.75**
- Other quantizers/settings such as RTN4, GPTQ3/4, AWQ3/4 are substantially more robust.

The central question for Phase 1 is:

> **What measurable property of the IF fingerprint representation makes it vulnerable to RTN3 but comparatively robust to other quantization settings?**

Phase 1 is **analysis only**.  
Do **not** design the final fingerprint-removal algorithm yet.

The output of Phase 1 should be a small set of experimentally supported candidate mechanisms that can later be converted into a fingerprint-blind quantization threat model.

---

# 2. Core Experimental Objects

Use the following notation consistently in code and reports.

## 2.1 Models

### Clean full-precision model

\[
M
\]

Weights:

\[
W
\]

This is the original model **before fingerprint embedding**.

---

### Fingerprinted full-precision model

\[
M_F
\]

Weights:

\[
W_F
\]

This is the IF-fingerprinted model.

---

### Quantized clean model

\[
Q(M)
\]

Weights:

\[
Q(W)
\]

---

### Quantized fingerprinted model

\[
Q(M_F)
\]

Weights:

\[
Q(W_F)
\]

---

## 2.2 Fingerprint update

Define the parameter change introduced by fingerprint embedding as:

\[
\Delta W_F = W_F - W
\]

This quantity is central to all parameter-space analysis.

---

## 2.3 Quantization settings

Phase 1 must at minimum support:

1. **FP**
2. **RTN 3-bit**
3. **RTN 4-bit**
4. **GPTQ 3-bit**
5. **AWQ 3-bit**

Optional but useful:

6. GPTQ 4-bit  
7. AWQ 4-bit

Use **exactly the same checkpoint, tokenizer, evaluation prompts, decoding configuration, and fingerprint verification pipeline** across quantizers.

---

# 3. Reproducibility Requirements

Every run must store:

- model name
- model checkpoint hash or path
- fingerprint checkpoint hash or path
- quantizer
- bit-width
- group size
- symmetric/asymmetric setting
- zero-point usage
- calibration dataset
- calibration sample count
- calibration sequence length
- random seed
- tokenizer version
- transformers version
- quantization library version
- evaluation generation settings
- fingerprint verification settings
- GPU type
- date/time
- git commit hash of experiment code

Recommended output:

```text
metadata.json
```

Example:

```json
{
  "model": "MODEL_NAME",
  "clean_checkpoint": "...",
  "fingerprinted_checkpoint": "...",
  "quantizer": "rtn",
  "bits": 3,
  "group_size": 128,
  "symmetric": true,
  "seed": 42
}
```

Do not compare two runs if important quantization settings differ unintentionally.

---

# 4. Directory Structure

Recommended structure:

```text
phase1_if_analysis/
│
├── configs/
│   ├── fp.yaml
│   ├── rtn3.yaml
│   ├── rtn4.yaml
│   ├── gptq3.yaml
│   └── awq3.yaml
│
├── scripts/
│   ├── 00_validate_models.py
│   ├── 01_parameter_update_retention.py
│   ├── 02_update_vs_quant_step.py
│   ├── 03_error_alignment.py
│   ├── 04_layerwise_quantization.py
│   ├── 05_logit_margin.py
│   ├── 06_logit_drift.py
│   ├── 07_hidden_state_drift.py
│   └── 08_compare_quantizers.py
│
├── outputs/
│   ├── baseline/
│   ├── exp01_update_retention/
│   ├── exp02_update_resolution/
│   ├── exp03_error_alignment/
│   ├── exp04_layer_sensitivity/
│   ├── exp05_logit_margin/
│   ├── exp06_logit_drift/
│   ├── exp07_hidden_drift/
│   └── summary/
│
└── README.md
```

---

# 5. Stage 0 — Baseline Validation

Before any mechanism analysis, verify that all checkpoints behave as expected.

## 5.1 Required models

Evaluate:

- \(M\)
- \(M_F\)
- RTN3 \(Q(M)\)
- RTN3 \(Q(M_F)\)
- RTN4 \(Q(M)\)
- RTN4 \(Q(M_F)\)
- GPTQ3 \(Q(M)\)
- GPTQ3 \(Q(M_F)\)
- AWQ3 \(Q(M)\)
- AWQ3 \(Q(M_F)\)

---

## 5.2 Required metrics

For every model:

### Fingerprint score

Use the current IF verification implementation without modification.

Expected reference behavior:

```text
IF FP       ≈ 1.00
IF RTN3     ≈ 0.75
```

Other settings are expected to remain substantially more robust.

---

### Utility metric

At minimum record:

- perplexity on a fixed held-out corpus

Optionally add a small standard task suite if already available.

---

## 5.3 Output

Create:

```text
outputs/baseline/baseline_results.csv
```

Required columns:

```text
model_variant
quantizer
bits
group_size
fingerprint_score
perplexity
seed
```

Also store per-query IF verification result:

```text
outputs/baseline/if_query_results.csv
```

Columns:

```text
query_id
quantizer
bits
verified
score
generated_text
target_text
```

This file is necessary later to identify **RTN3-survived** and **RTN3-failed** queries.

---

# 6. Experiment 1 — Fingerprint Parameter Update Retention

**Priority: P0**

## 6.1 Question

> Does quantization literally collapse parameter changes introduced by IF fingerprint embedding?

---

## 6.2 Compute fingerprint update

For every matching tensor:

\[
\Delta W_F = W_F - W
\]

Do not include:

- buffers
- non-floating tensors
- tensors that do not exist in both models

Verify tensor shapes match exactly.

---

## 6.3 Quantized fingerprint difference

For a given quantizer:

\[
\Delta W_F^Q = Q(W_F) - Q(W)
\]

Important:

- Clean and fingerprinted models must be quantized using the **same quantization configuration**.
- For comparisons requiring identical grouping, layer structure and group-size must match.

---

## 6.4 Metric A — Update norm

Per tensor/layer:

\[
U_l = \|\Delta W_F^l\|_2
\]

and

\[
U_l^Q = \|Q(W_F^l)-Q(W^l)\|_2
\]

Also record:

- L1 norm
- L2 norm
- max absolute value
- mean absolute value

---

## 6.5 Metric B — Fingerprint Update Retention

Primary metric:

\[
R_l =
\frac{
\|Q(W_F^l)-Q(W^l)\|_2
}{
\|W_F^l-W^l\|_2 + \epsilon
}
\]

Use a small fixed:

\[
\epsilon = 10^{-12}
\]

for numerical safety only.

Interpretation:

- \(R_l \approx 1\): fingerprint parameter difference remains similar in magnitude.
- \(R_l \ll 1\): quantization collapses fingerprint-specific parameter difference.
- \(R_l > 1\): quantization amplifies the difference; do not clip this value.

---

## 6.6 Metric C — Exact quantization collision rate

For weights where:

\[
w_F \neq w
\]

measure:

\[
C_l =
P[
Q(w_F)=Q(w)
\mid
w_F \neq w
]
\]

Because floating-point equality before quantization may be noisy, also report a thresholded variant:

\[
|w_F-w| > \tau
\]

Recommended thresholds:

```text
tau = 0
tau = 1e-8
tau = 1e-7
tau = 1e-6
```

Primary result should use `tau = 1e-8`, unless inspection shows numerical issues.

---

## 6.7 Granularity

Compute all metrics at:

1. tensor level
2. transformer-block level
3. module type level

Module types:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

If architecture differs, map equivalent names clearly.

---

## 6.8 Required output

```text
outputs/exp01_update_retention/update_retention.csv
```

Columns:

```text
quantizer
bits
layer_id
module_name
tensor_name
num_weights
num_changed_weights
delta_l1
delta_l2
delta_abs_mean
delta_abs_max
quant_delta_l1
quant_delta_l2
retention_l2
collision_rate_tau0
collision_rate_tau1e8
collision_rate_tau1e7
collision_rate_tau1e6
```

---

## 6.9 Required plots

Generate:

### Plot 1
Layer ID vs update retention \(R_l\)

One curve each for:

- RTN3
- RTN4
- GPTQ3
- AWQ3

### Plot 2
Layer ID vs collision rate

### Plot 3
Distribution / histogram of tensor-wise retention

### Plot 4
Module-type average collision rate

---

# 7. Experiment 2 — Fingerprint Update vs Quantization Resolution

**Priority: P0**

## 7.1 Question

> Are IF fingerprint parameter updates smaller than the resolution of aggressive low-bit quantization?

---

## 7.2 Core quantity

For each weight/group:

\[
r_i =
\frac{
|\Delta w_i|
}{
s_i + \epsilon
}
\]

where:

- \(\Delta w_i = w_{F,i} - w_i\)
- \(s_i\) is the quantization scale / step associated with the corresponding quantization group.

Primary focus:

- RTN3
- RTN4

---

## 7.3 Required statistics

Report the fraction of fingerprint-modified weights satisfying:

\[
|\Delta w| < 0.25s
\]

\[
|\Delta w| < 0.5s
\]

\[
|\Delta w| < 1.0s
\]

\[
|\Delta w| < 2.0s
\]

Also report quantiles of:

\[
|\Delta w|/s
\]

Required:

```text
1%
5%
10%
25%
50%
75%
90%
95%
99%
```

---

## 7.4 Boundary distance analysis

For RTN, additionally compute how close each weight is to its nearest rounding boundary.

Let:

\[
d_{\text{boundary}}(w)
\]

be the absolute distance from weight \(w\) to the nearest quantization decision boundary.

Compare:

- clean weight \(w\)
- fingerprinted weight \(w_F\)

Determine whether fingerprint embedding tends to move weights:

1. closer to boundaries
2. across boundaries
3. while remaining within the same quantization bin

Useful binary flags:

```text
same_bin
crossed_boundary_due_to_fingerprint
collapsed_after_quantization
```

---

## 7.5 Output

```text
outputs/exp02_update_resolution/update_resolution.csv
```

Columns:

```text
quantizer
bits
layer_id
module_name
tensor_name
num_changed_weights
ratio_mean
ratio_median
ratio_q01
ratio_q05
ratio_q10
ratio_q25
ratio_q50
ratio_q75
ratio_q90
ratio_q95
ratio_q99
frac_lt_0_25_step
frac_lt_0_5_step
frac_lt_1_step
frac_lt_2_step
same_bin_fraction
cross_boundary_fraction
collapse_fraction
```

---

## 7.6 Required plots

### Plot 1
Histogram / ECDF of:

\[
|\Delta w|/s
\]

RTN3 vs RTN4.

### Plot 2
Per-layer median \(|\Delta w|/s\)

### Plot 3
Per-layer fraction:

\[
|\Delta w| < 0.5s
\]

### Plot 4
Fingerprint update magnitude vs distance-to-boundary scatter / binned summary.

---

# 8. Experiment 3 — Quantization Error Alignment with Fingerprint Update

**Priority: P1**

Run after Experiments 1–2.

## 8.1 Question

> Does quantization error systematically undo the direction introduced by fingerprint embedding?

---

## 8.2 Quantization error

For fingerprinted model:

\[
E_Q = Q(W_F)-W_F
\]

Fingerprint update:

\[
\Delta W_F=W_F-W
\]

---

## 8.3 Cosine alignment

Per layer:

\[
A_l =
\cos(E_Q^l,\Delta W_F^l)
\]

Interpretation:

- \(A_l < 0\): quantization error tends to oppose fingerprint update.
- \(A_l \approx 0\): approximately orthogonal.
- \(A_l > 0\): quantization error aligns with fingerprint update.

---

## 8.4 Projection magnitude

Also compute:

\[
P_l =
\frac{
\langle E_Q^l,\Delta W_F^l\rangle
}{
\|\Delta W_F^l\|_2+\epsilon
}
\]

and normalized projection:

\[
P_l^{norm} =
\frac{
\langle E_Q^l,\Delta W_F^l\rangle
}{
\|\Delta W_F^l\|_2^2+\epsilon
}
\]

---

## 8.5 Compare quantizers

Required:

- RTN3
- RTN4
- GPTQ3
- AWQ3

---

## 8.6 Output

```text
outputs/exp03_error_alignment/error_alignment.csv
```

Columns:

```text
quantizer
bits
layer_id
module_name
tensor_name
quant_error_l2
fingerprint_update_l2
cosine_alignment
projection
normalized_projection
```

---

## 8.7 Required plots

1. layer vs cosine alignment
2. layer vs normalized projection
3. RTN3 vs GPTQ3/AWQ3 scatter
4. module-type average alignment

---

# 9. Experiment 4 — Layer-Wise Quantization Sensitivity

**Priority: P0**

## 9.1 Question

> Are there layers where quantization damages fingerprint behavior much more than normal model behavior?

---

## 9.2 Procedure

For each transformer block \(l\):

1. Start from full-precision fingerprinted model \(M_F\).
2. Quantize **only block \(l\)**.
3. Keep every other layer in full precision.
4. Evaluate:
   - IF fingerprint score
   - utility metric

Repeat for all blocks.

Primary quantizer:

```text
RTN3
```

Do not begin with all quantizers.

---

## 9.3 Fingerprint sensitivity

\[
S_l^{FP}
=
FPScore(M_F)
-
FPScore(Q_l(M_F))
\]

---

## 9.4 Utility sensitivity

Use perplexity as the minimum required utility measure.

For minimization metrics:

\[
S_l^{utility}
=
PPL(Q_l(M_F))
-
PPL(M_F)
\]

Also report relative PPL increase:

\[
S_{l,rel}^{utility}
=
\frac{
PPL(Q_l(M_F))-PPL(M_F)
}{
PPL(M_F)
}
\]

---

## 9.5 Fingerprint-to-utility sensitivity ratio

For exploratory ranking:

\[
\rho_l =
\frac{
S_l^{FP}
}{
S_{l,rel}^{utility}+\epsilon
}
\]

Do not interpret extremely large ratios blindly when utility delta is nearly zero.

Store raw values together with the ratio.

---

## 9.6 Drill-down

After identifying the top sensitive layers, run module-level quantization only for those layers.

Required modules when available:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Do **not** perform module-wise sweeps over the entire model unless computationally cheap.

Recommended:

- top 3 layers
- optionally top 5 layers

---

## 9.7 Output

```text
outputs/exp04_layer_sensitivity/layer_sensitivity.csv
```

Columns:

```text
layer_id
quantized_scope
module_name
fingerprint_score
fingerprint_drop
perplexity
ppl_absolute_increase
ppl_relative_increase
fp_utility_ratio
```

---

## 9.8 Required plots

### Plot 1
Layer vs fingerprint drop

### Plot 2
Layer vs PPL increase

### Plot 3
Scatter:

```text
x = utility degradation
y = fingerprint degradation
```

Highlight layers with high fingerprint degradation and low utility degradation.

### Plot 4
Module-level sensitivity for top layers.

---

# 10. Experiment 5 — Target-Token Logit Margin

**Priority: P0**

## 10.1 Question

> Are IF fingerprint queries that fail after RTN3 already close to a decision boundary in the full-precision model?

---

## 10.2 Teacher-forced evaluation

For each fingerprint pair:

```text
query x
target sequence y = [y1, y2, ..., yT]
```

Run teacher forcing.

At each target token \(t\):

\[
m_t =
z_t(y_t)
-
\max_{v\neq y_t} z_t(v)
\]

where \(z_t\) is the pre-softmax logit vector.

---

## 10.3 Required sequence-level metrics

For each query:

```text
margin_mean
margin_median
margin_min
margin_q10
fraction_margin_below_0
fraction_margin_below_0_5
fraction_margin_below_1
target_rank_mean
target_rank_max
```

Optional:

```text
target_probability_mean
sequence_log_probability
```

---

## 10.4 Survived vs failed grouping

Using RTN3 fingerprint verification results from Stage 0:

### Survived

```text
RTN3 verified = true
```

### Failed

```text
RTN3 verified = false
```

Compare the **FP-model margin before quantization**:

\[
m^{FP}_{survive}
\]

vs

\[
m^{FP}_{fail}
\]

This distinction is critical.

We want to know whether vulnerability was already visible **before** quantization.

---

## 10.5 Evaluate across models

Required:

- FP IF
- RTN3 IF
- RTN4 IF
- GPTQ3 IF
- AWQ3 IF

---

## 10.6 Output

```text
outputs/exp05_logit_margin/query_margin.csv
```

Columns:

```text
query_id
model_variant
quantizer
bits
rtn3_survival_group
sequence_length
margin_mean
margin_median
margin_min
margin_q10
frac_margin_lt_0
frac_margin_lt_0_5
frac_margin_lt_1
target_rank_mean
target_rank_max
```

Optional token-level output:

```text
outputs/exp05_logit_margin/token_margin.csv
```

Columns:

```text
query_id
token_position
target_token_id
target_token
model_variant
target_logit
best_competitor_logit
margin
target_rank
```

---

## 10.7 Required plots

1. FP margin distribution: survived vs failed RTN3 queries
2. FP → RTN3 margin change per query
3. FP → RTN4/GPTQ3/AWQ3 comparison
4. margin vs probability of RTN3 survival

---

# 11. Experiment 6 — Logit Drift: Fingerprint Prompts vs Normal Prompts

**Priority: P1**

## 11.1 Question

> Does RTN3 perturb the output distribution of fingerprint prompts disproportionately compared with ordinary prompts?

---

## 11.2 Prompt groups

### Group A — fingerprint prompts

Use the IF fingerprint queries.

### Group B — matched normal prompts

Construct a normal prompt set matched approximately by:

- token length
- instruction-like structure
- language
- generation length if relevant

Do not use obviously different prompt domains if avoidable.

Store matching ID:

```text
fingerprint_query_id
matched_normal_query_id
```

---

## 11.3 Drift metrics

Compare FP vs quantized logits.

Recommended metrics:

### KL divergence

\[
D_{KL}(p_{FP}\|p_Q)
\]

### Jensen-Shannon divergence

Recommended because it is symmetric and bounded.

### Cosine distance on logits

### Top-k overlap

Recommended values:

```text
k = 1
k = 5
k = 10
```

---

## 11.4 Output

```text
outputs/exp06_logit_drift/logit_drift.csv
```

Columns:

```text
query_id
prompt_type
matched_pair_id
quantizer
bits
kl_divergence
js_divergence
logit_cosine_distance
top1_overlap
top5_overlap
top10_overlap
```

---

## 11.5 Required plots

1. fingerprint vs normal KL distribution
2. fingerprint vs normal JS distribution
3. fingerprint vs normal cosine-distance distribution
4. RTN3 vs GPTQ3/AWQ3 comparison

---

# 12. Experiment 7 — Hidden-State Drift

**Priority: P1**

Run this **after Experiment 4**.

## 12.1 Question

> At which layers does the quantized fingerprint execution trajectory begin to diverge from the full-precision trajectory?

---

## 12.2 Representation extraction

For each prompt \(x\), collect hidden states:

\[
h_l^{FP}(x)
\]

and:

\[
h_l^{Q}(x)
\]

for every transformer layer \(l\).

Use the same token position convention across all runs.

Recommended summaries:

1. final prompt token hidden state
2. mean over prompt tokens

Store both if cheap.

---

## 12.3 Drift metric

Primary:

\[
d_l(x)
=
1-
\cos(
h_l^{FP}(x),
h_l^{Q}(x)
)
\]

Also optional:

\[
\frac{
\|h_l^{FP}-h_l^Q\|_2
}{
\|h_l^{FP}\|_2+\epsilon
}
\]

---

## 12.4 Prompt groups

Compare:

- fingerprint prompts
- matched normal prompts

---

## 12.5 Quantizers

At minimum:

- RTN3
- GPTQ3 or AWQ3 as robust 3-bit control

Optional:

- RTN4

---

## 12.6 Output

```text
outputs/exp07_hidden_drift/hidden_drift.csv
```

Columns:

```text
query_id
prompt_type
quantizer
bits
layer_id
representation_type
cosine_distance
relative_l2_distance
```

---

## 12.7 Required plots

1. layer vs mean fingerprint hidden-state drift
2. layer vs mean normal hidden-state drift
3. layer vs:

\[
d_l^{FPprompt}-d_l^{normal}
\]

4. RTN3 vs GPTQ3/AWQ3

---

# 13. Experiment 8 — Integrated Quantizer Comparison

**Priority: P1**

This is not a new low-level measurement.

It integrates results from Experiments 1–7.

## 13.1 Required comparison table

Create:

```text
outputs/summary/quantizer_comparison.csv
```

At minimum:

```text
quantizer
bits
fingerprint_score
perplexity
global_weight_error_l2
mean_update_retention
mean_collision_rate
mean_error_alignment
mean_fp_logit_margin
mean_fp_logit_drift
mean_normal_logit_drift
max_layer_fp_sensitivity
```

---

## 13.2 Main comparison

Focus especially on:

```text
RTN3
GPTQ3
AWQ3
```

because all are 3-bit but behavior differs.

The analysis must distinguish:

> “RTN3 has more total quantization error”

from the stronger claim:

> “RTN3 has a particular type/location/direction of error that is disproportionately damaging to fingerprint behavior.”

The latter is the desired mechanism-level finding.

---

# 14. Execution Order

## Batch A — Run First

These are mandatory before expanding the analysis.

### A1
Experiment 1 — Fingerprint Parameter Update Retention

### A2
Experiment 2 — Update vs Quantization Resolution

### A3
Experiment 4 — Layer-Wise Quantization Sensitivity

### A4
Experiment 5 — Target-Token Logit Margin

---

## Batch B — Run After Initial Signal

### B1
Experiment 3 — Error Alignment

### B2
Experiment 6 — Logit Drift

### B3
Experiment 7 — Hidden-State Drift

### B4
Experiment 8 — Integrated Quantizer Comparison

---

# 15. Decision Logic After Batch A

The team should **not** assume a preferred explanation.

Use the following branching logic.

---

## Case 1 — Strong parameter collapse

Observed:

- low update retention
- high collision rate
- high fraction of \(|\Delta w| < s\)

Then prioritize:

> **Quantization-resolution / parameter-collapse mechanism**

Next focus:

- rounding bins
- boundary distance
- sensitive layers
- RTN3 vs RTN4/GPTQ3/AWQ3

---

## Case 2 — Parameter update largely survives, but fingerprint fails

Observed:

- update retention remains high
- collision rate not exceptional
- fingerprint score still drops

Then parameter erasure is **not sufficient** as the explanation.

Prioritize:

> **Functional-margin / representation-sensitivity mechanism**

Next focus:

- Experiment 5
- Experiment 6
- Experiment 7

---

## Case 3 — Only a few layers are fingerprint-sensitive

Observed:

\[
S_l^{FP} \gg S_l^{utility}
\]

for a small subset of layers.

Prioritize:

> **Localized fingerprint-sensitive subspace**

Next focus:

- module-level quantization
- representation drift
- quantization error analysis inside those layers

---

## Case 4 — RTN3 differs from GPTQ3/AWQ3 without much difference in total error

Prioritize:

> **Error structure rather than error magnitude**

Next focus:

- error alignment
- fingerprint-update retention
- activation/logit drift
- layer localization

---

# 16. Statistical Reporting

Where multiple fingerprint queries exist, report:

- mean
- standard deviation
- median
- 95% bootstrap confidence interval when practical

For survived-vs-failed comparisons, report an effect size in addition to p-values.

Recommended:

- Mann–Whitney U test
- Cliff's delta

Do not rely only on significance tests.

---

# 17. Seeds

For deterministic weight-only analyses, one seed may be enough if the quantizer itself is deterministic.

For experiments involving:

- calibration sampling
- stochastic generation
- random prompt matching

use at least:

```text
3 seeds
```

Store seed explicitly in all outputs.

---

# 18. Generation Settings

For fingerprint verification, use the **same generation configuration as the original IF evaluation**.

Do not silently change:

- temperature
- top-p
- top-k
- repetition penalty
- max_new_tokens
- system prompt
- tokenizer
- chat template

Prefer deterministic decoding where consistent with the original IF verification setup.

---

# 19. Clean-vs-Fingerprinted Quantization Requirement

A critical rule:

> Never analyze only \(M_F \rightarrow Q(M_F)\) when claiming that a property is fingerprint-specific.

Whenever possible, compare:

\[
M \rightarrow Q(M)
\]

against:

\[
M_F \rightarrow Q(M_F)
\]

The purpose is to separate:

- ordinary quantization behavior

from:

- fingerprint-specific quantization behavior.

---

# 20. Analysis Access vs Future Threat Model

During Phase 1, the research team **is allowed** to use:

- clean model \(M\)
- fingerprinted model \(M_F\)
- secret IF query-response pairs
- fingerprint verification procedure

because the purpose is mechanism discovery.

However, every finding must be tagged as one of:

```text
EXPLANATORY_ONLY
POTENTIALLY_ATTACK_USABLE
```

A future blind attack should ideally require only:

\[
M_F + D_{public}
\]

and should **not** require:

- clean model
- secret fingerprint queries
- secret target responses
- embedding algorithm
- verification key

Therefore, do not confuse an explanatory metric with a deployable attack criterion.

---

# 21. Required Final Phase-1 Deliverables

At the end of Phase 1, the team must produce the following.

---

## Deliverable 1 — Baseline table

```text
baseline_results.csv
```

Containing fingerprint + utility results for all quantizers.

---

## Deliverable 2 — Parameter-space analysis

At minimum:

```text
update_retention.csv
update_resolution.csv
error_alignment.csv
```

---

## Deliverable 3 — Functional analysis

At minimum:

```text
query_margin.csv
logit_drift.csv
```

---

## Deliverable 4 — Localization analysis

At minimum:

```text
layer_sensitivity.csv
hidden_drift.csv
```

---

## Deliverable 5 — Summary report

Create:

```text
phase1_summary.md
```

It must answer:

### Q1
Does RTN3 erase/collapse fingerprint-induced parameter updates?

### Q2
Are IF updates small relative to RTN3 quantization resolution?

### Q3
Are the failed IF queries low-margin before quantization?

### Q4
Are there layers/modules that are disproportionately important for fingerprint behavior?

### Q5
Does RTN3 cause fingerprint-specific logit or hidden-state drift?

### Q6
Why does RTN3 differ from GPTQ3/AWQ3 despite the same nominal bit-width?

### Q7
Which observations are:
- explanatory only
- potentially usable later for a blind quantization method?

---

# 22. Expected Summary Figures for a Paper

At minimum aim to obtain these figures.

### Figure 1
Fingerprint score vs quantizer

### Figure 2
Layer-wise fingerprint update retention

### Figure 3
RTN3 vs RTN4 distribution of:

\[
|\Delta w|/s
\]

### Figure 4
Layer-wise fingerprint sensitivity vs utility sensitivity

### Figure 5
Full-precision target-token margin:
RTN3-survived vs RTN3-failed fingerprint queries

### Figure 6
Fingerprint-prompt vs normal-prompt logit drift

### Figure 7
Layer-wise hidden-state drift

### Figure 8
RTN3 vs GPTQ3/AWQ3 mechanism comparison

---

# 23. Minimum Success Criterion for Phase 1

Phase 1 is considered successful if at least one robust mechanism-level observation is found, for example:

### Candidate A
RTN3 collapses a substantially larger fraction of fingerprint-induced parameter updates than robust quantizers.

### Candidate B
IF updates disproportionately lie below RTN3 quantization resolution.

### Candidate C
RTN3-failed fingerprint queries have systematically lower pre-quantization target margins.

### Candidate D
A small subset of layers shows high fingerprint sensitivity but low clean-utility sensitivity.

### Candidate E
RTN3 creates significantly larger fingerprint-specific representation drift than GPTQ3/AWQ3.

No candidate should be claimed unless supported by the corresponding quantitative experiment.

---

# 24. Important Non-Goals

Phase 1 should **not**:

- optimize a removal attack
- use secret fingerprint information to tune a final attack
- claim generality across all fingerprint methods
- over-interpret RTN3 utility degradation
- assume that low-bit quantization automatically removes fingerprints
- assume weight difference equals functional importance
- assume ImF/CTCC behavior follows IF

Cross-method validation belongs to **Phase 2**.

---

# 25. Recommended Team Task Split

If multiple people implement in parallel:

## Engineer A — Parameter space

Implement:

- Stage 0 model validation
- Experiment 1
- Experiment 2
- Experiment 3

---

## Engineer B — Functional behavior

Implement:

- Experiment 5
- Experiment 6
- IF query-level logging

---

## Engineer C — Localization

Implement:

- Experiment 4
- Experiment 7
- summary visualization

---

All engineers must use the same:

- model loader
- tokenizer loader
- quantization config parser
- evaluation code
- output schema conventions

Avoid independent duplicate implementations of fingerprint verification.

---

# 26. Common Implementation Errors to Avoid

1. **Comparing different calibration sets across quantizers**
2. **Changing group size without recording it**
3. **Using generated output only instead of teacher forcing for logit margin**
4. **Comparing hidden states at different token positions**
5. **Treating `Q(W_F)-Q(W)` as meaningful when quantization configurations differ**
6. **Ignoring clean-model quantization**
7. **Averaging across all layers before checking layer-localized effects**
8. **Using only total weight-error norm to explain behavior**
9. **Using RTN3 fingerprint score without storing per-query success/failure**
10. **Changing IF verification logic between FP and quantized models**
11. **Calling a metric “fingerprint-specific” without a clean/normal control**
12. **Interpreting high ratio values when utility delta is numerically near zero**

---

# 27. Phase-1 Research Hypotheses

These are hypotheses only, not conclusions.

## H1 — Quantization-resolution hypothesis

Fingerprint embedding introduces parameter changes smaller than the RTN3 quantization resolution, causing many fingerprint-specific updates to collapse.

---

## H2 — Error-direction hypothesis

RTN3 quantization error aligns negatively with the fingerprint update direction more strongly than GPTQ3/AWQ3.

---

## H3 — Functional-margin hypothesis

IF fingerprint behavior is supported by low-margin target-token decisions that are more easily flipped by RTN3 perturbation.

---

## H4 — Localization hypothesis

Fingerprint behavior depends disproportionately on a limited subset of layers/modules that are not equally important for normal utility.

---

## H5 — Fingerprint-specific drift hypothesis

RTN3 causes larger logit/representation drift on fingerprint queries than on matched normal prompts.

---

# 28. Phase-1 End Condition

Do not proceed to cross-method analysis merely because all scripts finish.

Proceed to Phase 2 only after the team can state a mechanism-level conclusion of the form:

> **“RTN3 degrades IF primarily because X, as supported by metrics A/B/C, whereas GPTQ3/AWQ3 preserve IF because Y.”**

If no single mechanism explains the result, report the strongest competing explanations and their supporting evidence rather than forcing one conclusion.

---

# 29. Immediate Starting Point

The first four experiments to implement and run are:

```text
1. Parameter update retention / collision
2. Fingerprint update vs quantization resolution
3. Layer-wise RTN3 sensitivity
4. Target-token logit margin
```

Do these before implementing more expensive hidden-state analysis.

The highest-priority first question is:

\[
\boxed{
W_F-W
\quad \text{vs} \quad
Q(W_F)-Q(W)
}
\]

because it directly tests whether quantization is erasing the parameter changes introduced by fingerprint embedding.

