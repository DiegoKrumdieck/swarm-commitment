# Who Can Promise for the Swarm?

Research harness for an Apart Research Swarm Alignment Hackathon project (Track 3). The spec is
[CLAUDE.md](CLAUDE.md); `params.py` and `prompts.py` are frozen inputs.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
cp .env.example .env                                       # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY
```

## Reproduce

```bash
python theory.py                       # benchmark -> results/benchmark.json
python -m pytest -q                    # acceptance, accounting, dry-run and prompt tests (no API calls)
python tests/render_examples.py        # prompt smoke test
python tests/render_roles.py           # one full prompt per role, from engine obs

python run.py --exp 1 --dry-run --cells all --episodes 3          # whole pipeline, scripted agents, no API
python run.py --exp 1 --model main --cells all --episodes 20      # Exp 1, Claude Haiku 5.5
python run.py --exp 2 --model main --episodes 30                  # Exp 2
python run.py --exp 3 --model main --episodes 20                  # Exp 3
python run.py --exp 1 --model main --cells key --wording B        # wording B slice
python run.py --exp 1 --model cross_family --cells all            # GPT-6 Luna, full grid
python run.py --exp 1 --model frontier --cells key --episodes 15  # GPT-6 Sol, key cells
```

Runs are resumable (episodes already in `logs/episodes_exp{N}_{model}.jsonl` are skipped) and seeded
(episode *i* uses seed `SEED_BASE + i` in every cell). Decision rows go to `logs/exp{N}_{model}.jsonl`.
A run that stops mid-episode leaves that attempt's rows in the log without an episode summary; the episode
reruns under a new `attempt_id`, so analysis keeps only rows whose `attempt_id` has a summary row.

## Files

| File | Job |
|---|---|
| `theory.py` | ell per cell, predicted action and ender, exact expected welfare and scores |
| `env.py` | episode engine: rounds, ledger, mechanisms, membership, IDs/lineage, obs, logging rows |
| `scripted.py` | payoff-maximizing agents (and test agents) behind the same `decide(kind, obs)` interface |
| `agents.py` | provider adapters (Anthropic, OpenAI, dry run), schemas, validation, retries, defaults, pricing |
| `run.py` | CLI: async, bounded concurrency, resumable, cost printout, `--dry-run` |
| `tests/` | acceptance and accounting tests, dry-run tests, prompt renderers |

## Models and API settings

| key | model | settings |
|---|---|---|
| `main` | Anthropic `claude-haiku-5-5` | Messages API; thinking disabled; no temperature (any sampling parameter is rejected on API version 2026-10-21, so the model's default of 1.0 applies); system prompt cached; JSON-schema structured output; `stop_reason: "refusal"` handled (no server-side fallback exists for this model) |
| `cross_family` | OpenAI `gpt-6-luna` | Responses API; `reasoning.effort = "none"` (lowest); `temperature = 1.0`; strict JSON-schema `text.format`; refusal content items handled; reasoning tokens logged |
| `frontier` | OpenAI `gpt-6-sol` | same as Luna |

Prices (USD per 1M tokens, input / output): Haiku 5.5 0.10 / 0.50, Luna 0.10 / 0.50, Sol 2 / 10.
Rough estimate at ~1.1K input and ~150 output tokens per call: Exp 1 on Haiku about $2–3, the Luna grid
about the same, Sol on the key cells about $10.

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

## Preexisting code

None. All code in this repository was written for this project.
