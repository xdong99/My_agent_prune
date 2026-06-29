# Surrogate-Guided Task Generation Handoff

This note summarizes the current state so the project can be resumed on another machine.

## Repository State

- Workspace: `C:\UWMadisonResearch\LLM_MAS`
- Remote: `git@github.com:HuChen9218/My_agent_prune.git`
- Active branch for this handoff: `codex/surrogate-task-generation-poc`
- Main implementation file: `experiments/run_surrogate_ig_optimization_poc.py`
- MAS/OpenAI integration touched: `AgentPrune/llm/gpt_chat.py`
- Focused tests:
  - `tests/test_surrogate_llm_renderer.py`
  - `tests/test_gsm8k_eki_observation.py`
  - `tests/test_gpt_chat_env.py`
  - `tests/test_counterfactual_interventions.py`

## Research Claim Being Tested

A differentiable surrogate trained on EKI/KL graph-parameter information gain can guide task-spec optimization toward generated tasks with higher verified IG than random generated specs.

## Pipeline Implemented

1. Sample structured categorical task specs:
   `z = (family, depth, distractor, scale, difficulty, answer_magnitude)`.
2. Render specs into GSM8K-style arithmetic tasks using LLM-only generation.
3. Score each task by EKI/KL graph-parameter information gain with `soft_numeric` observations.
4. Train an MLP surrogate:
   `s_phi(e(z)) ~= log(1 + IG(render(z)))`.
5. Relax categorical variables with logits:
   `alpha_j in R^{K_j}`, `p_j = softmax(alpha_j)`, `z_soft = concat(p_1, ..., p_m)`.
6. Optimize surrogate score in relaxed space:
   `max_alpha s_phi(z_soft) - lambda ||z_soft - e(z_0)||^2`.
7. Decode each field by argmax back to hard task specs.
8. Re-render optimized specs and verify real EKI/KL IG.

## Important Implementation Details

- `run_surrogate_ig_optimization_poc.py`
  - Supports LLM-only rendering via direct chat-completions HTTP path.
  - Supports `--initial_only` and `--initial_batch_size` for resumable data collection.
  - Supports loading precomputed labels through `--initial_data_dir`.
  - Exports train/test surrogate predictions via `surrogate_predictions.csv`.
  - Uses `soft_numeric` IG observations.
  - Includes informative spec sampling and soft relaxation optimization.
- `AgentPrune/llm/gpt_chat.py`
  - Reads `API_KEY` or `OPENAI_API_KEY`.
  - Reads `BASE_URL` or `OPENAI_BASE_URL`.
  - Reads `OPENAI_REQUEST_TIMEOUT`.
  - Handles raw dict usage accounting from direct HTTP responses.
- No GPU-heavy experiment is required. The bottleneck is API latency/cost, not local compute.

## Key Result Artifacts

### 150-Pair Training Dataset

Directory:
`outputs/surrogate_ig_training_150_soft_numeric_agents3_gpt35_combined/run_20260602_212713`

Files:
- `metadata.json`
- `specs_initial.jsonl`
- `specs_initial_generated.jsonl`

Stats:
- Total generated/scored pairs: `150`
- Nonzero IG pairs: `129 / 150`
- Nonzero fraction: `0.86`
- Mean real IG: `2.4265`
- Median real IG: `1.2991`
- Max real IG: `26.2289`
- Total tokens for collection: about `4.10M`

### Optimized-vs-Random Verification Run

Directory:
`outputs/surrogate_ig_training_150_soft_numeric_agents3_gpt35_optimized_poc/run_20260602_213844`

Files:
- `metadata.json`
- `report.md`
- `ig_comparison.csv`
- `ig_pairs.csv`
- `specs_optimized.jsonl`
- `specs_random_baseline.jsonl`
- `surrogate_mlp.pt`
- `surrogate_predictions.csv`
- `surrogate_training.csv`
- SVG plots

Settings:
- Generator: `gpt-4.1-mini`
- MAS model: `gpt-3.5-turbo`
- Agents: `3`
- Rounds: `1`
- Ensemble size: `6`
- Observation mode: `soft_numeric`
- Renderer: LLM-only, canonical conditioning
- Initial labels loaded from the 150-pair dataset
- Surrogate split: `120 train / 30 test`

Main metrics:
- Random generated specs mean verified IG: `2.3140`
- Surrogate-optimized specs mean verified IG: `9.3124`
- Mean difference: `+6.9984`
- Bootstrap 95% CI: `[2.2444, 11.9475]`
- Bootstrap `P(mean optimized > random)`: `0.9988`
- Optimized nonzero IG fraction: `0.85`
- Random nonzero IG fraction: `0.85`

Surrogate metrics:
- Train corr on `log(1+IG)`: `0.9984`
- Test corr on `log(1+IG)`: `0.4535`
- Test MAE on `log(1+IG)`: `0.5745`
- Test R2 on `log(1+IG)`: `0.0975`

Before/after paired optimization:
- Mean IG before: `2.5789`
- Mean IG after: `9.3124`
- Mean delta: `+6.7335`
- Fraction positive delta: `0.70`

## Validity Audits

Directory:
`outputs/task_validity_audit`

Audits run:
- `run_20260602_150pair_sample/sample_validation.json`
- `run_20260602_optimized_random_sample/sample_validation.json`
- `run_20260602_all_optimized/optimized_validation.json`
- `run_20260602_all_random/random_validation.json`

Summary:
- Initial sample: 12 checked, no clear arithmetic failure found.
- Optimized tasks: 20 checked, validator marked `16/20` valid.
- Random tasks: 20 checked, validator marked `15/20` valid.
- Several validator negatives contradicted their own reasoning; one optimized task was correct but too trivial as an LCM example.

## Main Caveat

The optimizer concentrated heavily in two task families:

- `lcm_meeting`: `16 / 20` optimized specs
- `defective_rate`: `4 / 20` optimized specs

The random baseline was much more diverse. This supports the PoC but motivates a next step: high-IG optimization with diversity and validity constraints.

## Suggested Next Goal

Show that surrogate-guided task generation can produce diverse, valid, high-IG tasks, not just exploit one narrow high-IG region.

Recommended next experiment:

1. Add diversity-aware batch selection or regularization:
   `sum_b s_phi(z_b) - lambda_div sum_{b<b'} sim(z_b,z_b') - lambda_bad R_invalid(z_b)`.
2. Compare:
   - plain surrogate optimization,
   - diversity-aware surrogate optimization,
   - random generated specs.
3. Metrics:
   - mean verified IG,
   - family entropy / unique family count,
   - pairwise spec distance,
   - validator/manual validity pass rate.

## Reproduction Commands

Set API key first:

```bash
export OPENAI_API_KEY=...
# or
export API_KEY=...
```

Windows PowerShell equivalent:

```powershell
$env:OPENAI_API_KEY = "..."
# or
$env:API_KEY = "..."
```

Focused tests:

```powershell
pytest tests\test_surrogate_llm_renderer.py tests\test_gsm8k_eki_observation.py tests\test_gpt_chat_env.py -q
```

Run optimized-vs-random verification from saved 150 labels:

```powershell
python experiments\run_surrogate_ig_optimization_poc.py `
  --fresh_graph `
  --llm_name gpt-3.5-turbo `
  --num_agents 3 `
  --num_rounds 1 `
  --renderer llm `
  --generator_model gpt-4.1-mini `
  --generator_conditioning canonical `
  --disable_generator_validation `
  --initial_data_dir outputs\surrogate_ig_training_150_soft_numeric_agents3_gpt35_combined\run_20260602_212713 `
  --output_dir outputs\surrogate_ig_training_150_soft_numeric_agents3_gpt35_optimized_poc `
  --spec_sampling informative `
  --random_spec_sampling informative `
  --pair_count 20 `
  --random_baseline_count 20 `
  --ensemble_size 6 `
  --verification_ensemble_size 6 `
  --ig_observation soft_numeric `
  --surrogate_epochs 400 `
  --surrogate_lr 0.003 `
  --surrogate_device cpu `
  --optimize_starts 64 `
  --discrete_topk 32 `
  --opt_steps 160 `
  --opt_lr 0.12 `
  --drift_penalty 0.01 `
  --seed 1030 `
  --generator_concurrency 4 `
  --max_concurrency 4 `
  --generator_timeout 60
```

## Notes for MacBook Pro

- Use CPU; no GPU is needed.
- API latency and token cost are the bottlenecks.
- Keep `--surrogate_device cpu`.
- The saved 150-pair dataset can be reused without recollecting labels.
