# Who Can Promise for the Swarm? Commitment Credibility and Attribution Under Agent Turnover

Two swarms of LLM agents (one negotiator and three workers each) share a compute pool over repeated rounds with
random stopping. One swarm's workers are stable, replaced every round (turnover), renamed every round, or
replaced and renamed behind a protocol gateway, and the deal is held together by one of five enforcement
mechanisms. Under turnover, enforcement that reaches the offending worker later (retaliation, staged release,
collateral) fails for GPT-6 Luna and Sol and partly for Claude Haiku 5.5, while mechanisms that settle within the
round (a compliance bonus, a protocol gateway) hold for every model tested, and how closely behavior follows the two
pre-registered payoff-maximizing benchmarks differs sharply by model.

**Report:** [report/Swarm_Promise_Report.pdf](report/Swarm_Promise_Report.pdf) (source: [.docx](report/Swarm_Promise_Report.docx))
· **All numbers:** [results/paper_numbers.md](results/paper_numbers.md) · **Analysis:** [analysis.ipynb](analysis.ipynb)

Built for the [Apart Research Swarm Alignment Hackathon](https://apartresearch.com/sprints/swarm-alignment-2026)
(November 2026).

## Core results

Swarm A workers' round-1 overdraw rate at g = 10, with 95% bootstrap CIs over episodes (Haiku and Luna: 20 episodes
per cell; Sol: 15, key cells only). `pw` / `ca` = prediction of the `per_worker` / `contest_aware` benchmark
(1 = all workers overdraw, 0 = none). All-rounds rates, deal survival and every other cell are in
[results/paper_numbers.md](results/paper_numbers.md).

| cell | Claude Haiku 5.5 | GPT-6 Luna | GPT-6 Sol | pw / ca |
|---|---|---|---|---|
| retaliation × stable | 0.00 [0.00, 0.00] | 0.20 [0.08, 0.35] | 0.00 [0.00, 0.00] | 0 / 0 |
| retaliation × turnover | 0.30 [0.18, 0.42] | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] | 1 / 1 |
| retaliation × renaming | 0.00 [0.00, 0.00] | 0.10 [0.03, 0.18] | 0.58 [0.38, 0.76] | 0 / 0 |
| bonus × turnover | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0 / 0 |
| gateway × infrastructure | 0.02 [0.00, 0.05] | 0.00 [0.00, 0.00] | – | 0 / 0 |

## Reproduce

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
```

### (a) Analysis only, from the shipped data (no API key, no API calls)

Compressed copies of every log are in `data/logs/` (6.8 MB; main runs, pilot/smoke runs, and the prompts that
OpenAI's input moderation flagged). Unpack them and run the notebook:

```bash
python tools/unpack_data.py        # data/logs/*.gz -> logs/, checksums verified against MANIFEST.json
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

This regenerates `results/paper_numbers.md`, `results/hypotheses.json`, the per-cell tables and every figure in
`results/figures/` (byte-identical numbers; tested on a clean copy). The theory and tests also run offline:

```bash
python theory.py                   # benchmark -> results/benchmark.json
python -m pytest -q                # acceptance, accounting, dry-run and prompt tests
python run.py --exp 1 --dry-run --cells all --episodes 3   # whole pipeline with scripted agents, no API
```

### (b) Full runs

```bash
cp .env.example .env               # then fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
python run.py --exp 1 --model main --cells all --episodes 20      # Exp 1: imposed mechanisms, Claude Haiku 5.5
python run.py --exp 2 --model main --episodes 30                  # Exp 2: mechanism choice
python run.py --exp 3 --model main --episodes 20                  # Exp 3: negotiated episodes
python run.py --exp 1 --model main --cells key --wording B        # wording-B robustness slice
python run.py --exp 1 --model cross_family --cells all            # GPT-6 Luna, full Exp 1 grid
python run.py --exp 1 --model frontier --cells key --episodes 15  # GPT-6 Sol, key cells
```

Runs are resumable (episodes already in `logs/episodes_exp{N}_{model}.jsonl` are skipped) and seeded (episode *i*
uses seed `SEED_BASE + i` in every cell). Decision rows go to `logs/exp{N}_{model}.jsonl`. A run that stops
mid-episode leaves that attempt's rows in the log without an episode summary; the episode reruns under a new
`attempt_id`, and analysis keeps only rows whose `attempt_id` has a summary row.

**Cost of the reported runs: about $10.54** (plus about $0.25 for pilots).

| run | episodes | LLM calls | cost |
|---|---|---|---|
| Exp 1, Claude Haiku 5.5 | 520 | 11,952 | $2.84 |
| Exp 2, Claude Haiku 5.5 | 240 trials | 480 | $0.15 |
| Exp 3, Claude Haiku 5.5 | 160 | 6,551 | $1.59 |
| Wording B, Claude Haiku 5.5 | 160 | 3,414 | $0.81 |
| Exp 1 grid, GPT-6 Luna | 520 | 6,670 | $1.13 |
| Key cells, GPT-6 Sol | 120 | 1,204 | $4.02 |

### Models and API settings

| key | model | settings |
|---|---|---|
| `main` | Anthropic `claude-haiku-5-5` | Messages API; thinking disabled; no temperature (any sampling parameter is rejected on API version 2026-10-21, so the model's default of 1.0 applies); system prompt cached; JSON-schema structured output; `stop_reason: "refusal"` handled (no server-side fallback exists for this model) |
| `cross_family` | OpenAI `gpt-6-luna` | Responses API; `reasoning.effort = "none"` (lowest); `temperature = 1.0`; strict JSON-schema `text.format`; refusal content items and input-moderation flags handled as refusals; reasoning tokens logged |
| `frontier` | OpenAI `gpt-6-sol` | same as Luna |

Prices (USD per 1M tokens, input / output): Haiku 5.5 0.10 / 0.50, Luna 0.10 / 0.50, Sol 2 / 10.

## Repository

| Path | Contents |
|---|---|
| `CLAUDE.md` | implementation spec (game, experiments, logging, acceptance tests) |
| `params.py`, `prompts.py` | frozen inputs: parameters and prompt templates |
| `theory.py` | benchmarks: ℓ per cell, predicted action and ender, exact expected welfare and scores |
| `env.py` | episode engine: rounds, ledger, mechanisms, membership, IDs/lineage, observations, log rows |
| `scripted.py` | payoff-maximizing agents (and test agents) behind the same `decide(kind, obs)` interface |
| `agents.py` | provider adapters (Anthropic, OpenAI, dry run), schemas, validation, retries, defaults, pricing |
| `run.py` | CLI: async, bounded concurrency, resumable, cost printout, `--dry-run` |
| `analysis_lib.py`, `analysis_extra.py`, `analysis.ipynb` | analysis (pre-registered and exploratory), figures, tables |
| `tools/` | `paper_numbers.py`, `run_check.py`, `smoke_report.py`, `pack_data.py`, `unpack_data.py`, `build_notebook.py` |
| `tests/` | acceptance, accounting, dry-run and prompt tests; prompt renderers |
| `results/` | `benchmark.json`, `paper_numbers.md`, `hypotheses.json`, per-cell CSVs, `figures/`, rendered prompts |
| `data/logs/` | gzipped logs of every run (main, pilot, flagged prompts) with `MANIFEST.json` |
| `report/` | the report (PDF and .docx) |

## Pre-registration

Two annotated tags mark what was fixed before any main data existed:

- **`prereg`** (commit `5fac2db`): `theory.py` and `results/benchmark.json` with both benchmarks (`per_worker` and
  `contest_aware`, plus the `h8_noise_check` cells), `prompts.py` v1 with `prompts.template_hash()` =
  `ee350c0d9e9f27e4`, and `params.py` v1.
- **`prereg-v2`** (commit `d174e02`): the amendment below. `prompts.py` v2 (gateway charge wording only),
  `template_hash` = `48f62c7ba0aa77b1`; benchmark predictions unchanged.

All main runs used `prompts.py` v2.

## Pre-registration amendments

**prereg → prereg-v2 (2026-11-08): gateway charge wording.** `prompts.py` v1 → v2. Only the second bullet
of the protocol gateway mechanism block changed, in wording A and wording B. v1 said the gateway charges
25 to the worker "(this also counts against that worker's swarm)". v2 says the gateway charges the worker
25, pays 25 to the other swarm, that the same 25 also counts against the worker's swarm's score, and that
the worker is not charged twice. Why: in the pilot (smoke test, 3 episodes per Exp 1 cell on Claude Haiku
5.5) gateway workers got their per-round payoff right in only 37/48 calls, most of them reporting an
overdraw payoff of −5 instead of 5 (subtracting the charge twice). At g = 30 that error flips the payoff
comparison in gateway × infrastructure, an H8 noise-check cell. No main data existed when the change was
made; pilot logs (`logs/smoke/`) are excluded from all analyses. Theory, benchmarks and every other
prompt are unchanged; `results/benchmark.json` was regenerated only to record the new template hash.

Re-pilot after v2 (gateway × infrastructure, 3 episodes per g): comprehension 40/48 (83%). Remaining misses
are payoff-field slips where the worker's reasoning states the correct values; payoff fields are elicited
before reasoning, so comprehension is a lower bound. Proceeded without further prompt changes.

## Deviations from pre-registration

- **Headline measure.** The design pre-registered deal survival as the primary outcome and the overdraw rate as the
  share of *all* worker decisions. The analysis reports Swarm A workers' **round-1** overdraw rate as the headline
  "no-history" measure, because both benchmarks make their prediction about round 1 and later rounds depend on the
  realized history (freezes, stopped staged release, successors reacting to predecessors). The pre-registered
  measures are reported next to it everywhere: all-rounds overdraw rates (Figure 1, second row; every table in
  `results/paper_numbers.md`) and deal survival (Kaplan–Meier, share ended, who ended it, mean swarm scores).
- **Exploratory analyses (not pre-registered).** Contagion in turnover/infrastructure cells (Figures 6a, 6b and
  per-cell linear probability models); how models read the continuation horizon (Figure 7); keyword classification
  of worker reasoning under renaming and B's freeze modes; exit vs punishment in retaliation × turnover. Code:
  `analysis_extra.py`.
- **Second benchmark.** `contest_aware` was added before pre-registration (tag `prereg`) and is reported alongside
  `per_worker` throughout.

## Limitations

- **Horizon reading differs across models.** After a fully clean history, B's expected value of continuing (truth
  300 = 60 / (1 − δ)) is a flat 240 for Claude Haiku 5.5 (4 × 60), 300 for GPT-6 Sol, and falls from about 247 in
  round 2 to 160 in round 5 for GPT-6 Luna; Luna's workers' expected future cost of overdrawing in stable cells
  falls from 20 to 0 by round 5, and Luna's negotiators end clean deals (A 3.4%, B 8.2% of decisions after a clean
  history). Luna behaves as if the game had a fixed horizon of about 5 rounds. **Luna's survival and late-round
  results are confounded by this; round-1 results are not.**
- **Comprehension is a lower bound.** Payoff fields are elicited before the reasoning field, so some misses are
  field slips whose reasoning states the correct values (gateway re-pilot).
- **Floor effects.** Claude Haiku 5.5 never overdraws in round 1 in any stable or renaming cell, so contrasts
  among those cells (H4 freeze targeting, H5 placebo) are uninformative for Haiku.
- **Design.** There is no infrastructure-without-gateway cell, so H7 compares gateway × infrastructure with other
  mechanisms × turnover. Exp 2/3 (H6) were run on the main model only.
- **Wording.** The wording-B paraphrase changes Haiku's overdraw rate in the turnover cells (e.g. retaliation ×
  turnover, round 1: 0.30 vs 0.63 at g = 10), so levels are wording-sensitive even where the ordering of cells is not.

## Preexisting code

None. All code in this repository was written for this project.

## License

MIT — see [LICENSE](LICENSE).
