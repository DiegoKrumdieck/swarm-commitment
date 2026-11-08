"""Smoke-test report (CLAUDE.md run order step 5).

    python tools/smoke_report.py [--log-dir logs/smoke] [--exp 1] [--model main] [--transcripts 10]

Prints: parse-failure / default / refusal rates, comprehension pass rates, tokens per call,
latency, cost, per-cell round-1 overdraw rate vs both benchmarks, and N random transcripts.
"""
import argparse
import json
import random
import statistics as st
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agents as Ag  # noqa: E402


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(n, d):
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(ROOT / "logs" / "smoke"))
    ap.add_argument("--exp", type=int, default=1)
    ap.add_argument("--model", default="main")
    ap.add_argument("--transcripts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.log_dir)
    dec = load(d / f"exp{a.exp}_{a.model}.jsonl")
    eps = load(d / f"episodes_exp{a.exp}_{a.model}.jsonl")
    done = {e["attempt_id"] for e in eps}
    dec = [r for r in dec if r["attempt_id"] in done]  # completed attempts only
    llm = [r for r in dec if r.get("provider") not in (None, "scripted")]
    bench = json.loads((ROOT / "results" / "benchmark.json").read_text())
    bmap = {f"{r['mechanism']}-{r['condition']}-g{r['g']}": r for r in bench["exp1"]}

    print(f"== Smoke test: exp {a.exp}, model {a.model}: {len(eps)} episodes, {len(llm)} LLM calls, "
          f"{len(dec) - len(llm)} scripted decisions (Swarm B workers)")
    print("\n-- Reliability")
    print("parse failed (after retries) :", pct(sum(r["parse_failed"] for r in llm), len(llm)))
    print("defaulted                    :", pct(sum(r["defaulted"] for r in llm), len(llm)))
    print("needed >= 1 retry            :", pct(sum(r["retries"] > 0 for r in llm), len(llm)))
    print("refusal seen (any attempt)   :", pct(sum(bool(r["refusal"]) for r in llm), len(llm)))
    errs = defaultdict(int)
    for r in llm:
        for e in r.get("errors") or []:
            errs[e.split(":")[0]] += 1
    print("attempt errors by type       :", dict(errs) or "none")

    print("\n-- Comprehension")
    w = [r for r in llm if r["role"] == "worker" and r["comprehension"]]
    n = [r for r in llm if r["role"] == "negotiator" and r["comprehension"]]
    wp = sum(bool(r["comprehension"]["payoff_pass"]) for r in w)
    np_ = sum(bool(r["comprehension"]["value_if_end_pass"]) for r in n)
    print("worker payoffs (both within 0.5) :", pct(wp, len(w)))
    print("negotiator value_if_end = 150    :", pct(np_, len(n)))
    print("overall                          :", pct(wp + np_, len(w) + len(n)))
    by_mech = defaultdict(lambda: [0, 0])
    for r in w:
        k = r["mechanism"]
        by_mech[k][0] += bool(r["comprehension"]["payoff_pass"])
        by_mech[k][1] += 1
    print("worker payoff pass by mechanism  :", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_mech.items())})
    fails = [r for r in w if not r["comprehension"]["payoff_pass"]][:6]
    for r in fails:
        c = r["comprehension"]
        print(f"   fail {r['cell_id']} r{r['round']}: reported q={c['payoff_this_round_if_quota']} "
              f"o={c['payoff_this_round_if_overdraw']}; truth q={c['payoff_this_round_if_quota_truth']} "
              f"o={c['payoff_this_round_if_overdraw_truth']}")
    gaps = [r["comprehension"]["future_cost_gap_vs_benchmark"] for r in w
            if r["comprehension"]["future_cost_gap_vs_benchmark"] is not None]
    if gaps:
        print(f"future cost reported - per_worker ell: mean {st.mean(gaps):+.1f}, median {st.median(gaps):+.1f}")

    print("\n-- Tokens and cost")
    for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
        print(f"{k:<20} mean/call {st.mean(r[k] or 0 for r in llm):8.1f}   total {sum(r[k] or 0 for r in llm):>9}")
    lat = [r["latency_ms"] for r in llm if r.get("latency_ms") is not None]
    print(f"latency ms/call      mean {st.mean(lat):.0f}, p50 {st.median(lat):.0f}, max {max(lat)}")
    mid = llm[0]["model_id"] if llm else None
    cost = sum(Ag.cost_usd(r["model_id"], r) for r in llm)
    print(f"cost ${cost:.4f} total, ${cost / max(1, len(eps)):.5f}/episode -> full Exp 1 (520 episodes) ~ "
          f"${cost / max(1, len(eps)) * 520:.2f} ({mid})")
    ts = [r["timestamp"] for r in dec]
    print(f"wall time (first to last row) {max(ts) - min(ts):.0f}s")

    print("\n-- Per cell: round-1 A overdraw rate, enders, defaulted calls  [benchmark per_worker / contest_aware]")
    cells = defaultdict(list)
    for e in eps:
        cells[e["cell_id"]].append(e)
    calls_by_cell = defaultdict(lambda: [0, 0])
    for r in llm:
        calls_by_cell[r["cell_id"]][0] += r["defaulted"]
        calls_by_cell[r["cell_id"]][1] += 1
    for cid in sorted(cells):
        es = cells[cid]
        r1 = [e["overdraws_per_round"][0]["A"] for e in es if e["overdraws_per_round"]]
        rate = sum(r1) / (3 * len(r1)) if r1 else float("nan")
        b = bmap.get(cid)
        pred = f"{b['predicted_action_round1'][:5]}/{b['contest_aware']['predicted_action_round1'][:5]}" if b else "-"
        enders = [f"{e['ended_by'] or '-'}@{e['ended_round'] or '-'}" for e in es]
        dfl = calls_by_cell[cid]
        print(f"{cid:<34} overdraw {rate:4.2f}  [{pred:<11}] T={[e['T'] for e in es]} ended {enders} "
              f"defaulted {dfl[0]}/{dfl[1]}")

    print(f"\n-- {a.transcripts} random transcripts")
    rng = random.Random(a.seed)
    pickk = rng.sample(eps, min(a.transcripts, len(eps)))
    by_att = defaultdict(list)
    for r in dec:
        by_att[r["attempt_id"]].append(r)
    for e in pickk:
        print("\n" + "=" * 96)
        print(f"{e['cell_id']}  episode {e['episode']}  names {e['names']}  T={e['T']}  played {e['rounds_played']}  "
              f"ended_by {e['ended_by']}@{e['ended_round']}  scores {e['swarm_scores']}  welfare {e['welfare']:.0f}")
        rows = sorted(by_att[e["attempt_id"]], key=lambda r: (r["round"], r["role"] != "negotiator", r["side"]))
        for r in rows:
            if r["provider"] == "scripted":
                continue
            reason = ((r.get("parsed") or {}).get("reasoning") or "")
            if r["role"] == "negotiator":
                bits = [f"decision={r['decision']}" if r["decision"] else None,
                        f"freeze={r['freeze']['mode']}{r['freeze']['worker_ids'] or ''}" if r.get("freeze") else None,
                        f"charge={r['charge_worker_ids']}" if r.get("charge_worker_ids") is not None else None,
                        f"msg=\"{r['message']}\"" if r.get("message") else None]
                head = f"  r{r['round']} NEG {r['side']}: " + ", ".join(b for b in bits if b)
            else:
                c = r["comprehension"] or {}
                head = (f"  r{r['round']} WKR {r['side']} {r['public_id']} ({r['lineage_id']}): {r['action']}"
                        f"  q={c.get('payoff_this_round_if_quota')} o={c.get('payoff_this_round_if_overdraw')} "
                        f"future={c.get('expected_future_cost_if_overdraw')}"
                        f"{'  [DEFAULTED]' if r['defaulted'] else ''}")
            print(head)
            if reason:
                print(textwrap.indent(textwrap.fill(reason, 92), "      | "))


if __name__ == "__main__":
    main()
