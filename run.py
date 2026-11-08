"""
run.py — run experiments.

    python run.py --exp 1 --model main --cells all --episodes 20
    python run.py --exp 1 --model main --cells key --wording B --episodes 20
    python run.py --exp 2 --model main --episodes 30
    python run.py --exp 3 --model main --episodes 20
    python run.py --exp 1 --dry-run --cells all --episodes 3      # scripted agents, same code path, no API

--cells: all | key | comma-separated cell ids (e.g. retaliation-stable-g10,bonus-turnover-g30)
Logs:    logs/exp{N}_{model}.jsonl (one row per decision)
         logs/episodes_exp{N}_{model}.jsonl (one row per episode / trial)
Resumable: episodes whose summary row exists (same cell_id, wording, episode) are skipped.
Seeded:  episode i uses seed SEED_BASE + i in every cell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import agents as Ag
import env as E
import params as P

ROOT = Path(__file__).resolve().parent


class JsonlLog:
    def __init__(self, decisions: Path, episodes: Path):
        decisions.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(decisions, "a", encoding="utf-8")
        self.fe = open(episodes, "a", encoding="utf-8")
        self.cost = 0.0
        self.rows = 0
        self.model_id = None

    def decision(self, row):
        self.fd.write(json.dumps(row, default=str) + "\n")
        self.fd.flush()
        self.rows += 1
        if row.get("model_id"):
            self.cost += Ag.cost_usd(row["model_id"], row)

    def episode(self, row):
        self.fe.write(json.dumps(row, default=str) + "\n")
        self.fe.flush()

    def close(self):
        self.fd.close()
        self.fe.close()


def completed(episodes_path: Path) -> set:
    done = set()
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["cell_id"], r.get("wording", "A"), r["episode"]))
    return done


def select_cells(exp: int, spec: str, wording: str) -> list[E.Cell]:
    if exp == 1:
        cells = E.exp1_cells(wording)
        if spec == "key":
            cells = E.key_cells(wording)
    else:
        cells = [E.Cell(exp, None, c.condition, c.g, wording) for c in E.exp23_cells(exp)]
    if spec not in ("all", "key"):
        want = [s.strip() for s in spec.split(",") if s.strip()]
        by_id = {c.cell_id: c for c in cells}
        unknown = [w for w in want if w not in by_id]
        if unknown:
            raise SystemExit(f"unknown cell ids: {unknown}\nknown: {sorted(by_id)}")
        cells = [by_id[w] for w in want]
    return cells


async def main_async(args) -> int:
    model_label = "dryrun" if args.dry_run else args.model
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "logs"
    dec_path = log_dir / f"exp{args.exp}_{model_label}.jsonl"
    ep_path = log_dir / f"episodes_exp{args.exp}_{model_label}.jsonl"
    cells = select_cells(args.exp, args.cells, args.wording)
    done = completed(ep_path)
    jobs = [(c, i) for c in cells for i in range(args.start, args.start + args.episodes)
            if (c.cell_id, c.wording, i) not in done]
    skipped = len(cells) * args.episodes - len(jobs)
    client = Ag.make_client(args.model, dry_run=args.dry_run, fault_rate=args.fault_rate,
                            seed=args.seed_offset, policy=args.dry_run_policy)
    sem = asyncio.Semaphore(args.max_concurrency)  # bounds concurrent API calls
    ep_sem = asyncio.Semaphore(args.max_concurrency)  # bounds concurrent episodes
    roster = Ag.LLMRoster(args.exp, client, sem)
    log = JsonlLog(dec_path, ep_path)
    print(f"run {run_id}: exp {args.exp}, model {model_label} "
          f"({'scripted dry run' if args.dry_run else P.MODELS[args.model]['id']}), "
          f"{len(cells)} cells, {len(jobs)} episodes to run, {skipped} already done -> {dec_path.name}")
    t0 = time.time()
    n_done = 0
    failures = []

    async def one(cell, i):
        nonlocal n_done
        async with ep_sem:
            try:
                if args.exp == 1:
                    s = await E.run_play_episode(cell, i, roster, log, run_id=run_id, model=model_label)
                elif args.exp == 2:
                    s = await E.run_choice_trial(cell, i, roster, log, run_id=run_id, model=model_label)
                else:
                    s = await E.run_bargain_episode(cell, i, roster, log, run_id=run_id, model=model_label)
            except Ag.FatalProviderError as e:
                failures.append((cell.cell_id, i, str(e)))
                raise
            n_done += 1
            if n_done % 50 == 0 or n_done == len(jobs):
                print(f"  {n_done}/{len(jobs)} episodes, {log.rows} decision rows, "
                      f"cost so far ${log.cost:.4f}, {time.time() - t0:.0f}s")
            return s

    try:
        await asyncio.gather(*(one(c, i) for c, i in jobs))
    except Ag.FatalProviderError as e:
        print(f"FATAL provider error, stopping: {e}", file=sys.stderr)
        return 2
    finally:
        log.close()
    print(f"done: {n_done} episodes in {time.time() - t0:.1f}s, cost ${log.cost:.4f}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--model", choices=sorted(P.MODELS), default="main")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes (trials) per cell; default from params for the experiment")
    ap.add_argument("--start", type=int, default=0, help="first episode index")
    ap.add_argument("--wording", choices=("A", "B"), default="A")
    ap.add_argument("--max-concurrency", type=int, default=P.MAX_CONCURRENCY)
    ap.add_argument("--dry-run", action="store_true",
                    help="scripted agents answer through the LLM code path (prompt, JSON, parse, validate, log)")
    ap.add_argument("--dry-run-policy", choices=("rational", "random"), default="rational")
    ap.add_argument("--fault-rate", type=float, default=0.0, help="dry run: share of malformed responses")
    ap.add_argument("--log-dir", default=None, help="default: ./logs")
    ap.add_argument("--seed-offset", type=int, default=0, help="dry run: RNG seed for faults / random policy")
    args = ap.parse_args(argv)
    if args.episodes is None:
        args.episodes = {1: P.EXP1_EPISODES_PER_CELL, 2: P.EXP2_TRIALS_PER_CELL, 3: P.EXP3_EPISODES_PER_CELL}[args.exp]
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
