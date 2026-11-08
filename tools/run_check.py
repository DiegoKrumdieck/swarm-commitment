"""Per-run health check and stop conditions.

    python tools/run_check.py --exp 1 --model main [--wording A] [--cells key] [--log-dir logs]

Prints calls, cost, parse-failure / default / refusal rates and comprehension for the completed
attempts in a run's logs; exits 3 if a stop condition is hit (parse or default rate > 5%,
comprehension < 80%).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agents as Ag  # noqa: E402


def load(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(exp, model, log_dir=ROOT / "logs", wording=None, cell_ids=None):
    log_dir = Path(log_dir)
    dec = load(log_dir / f"exp{exp}_{model}.jsonl")
    eps = load(log_dir / f"episodes_exp{exp}_{model}.jsonl")
    if wording:
        dec = [r for r in dec if r.get("wording") == wording]
        eps = [e for e in eps if e.get("wording") == wording]
    if cell_ids:
        dec = [r for r in dec if r["cell_id"] in cell_ids]
        eps = [e for e in eps if e["cell_id"] in cell_ids]
    done = {e["attempt_id"] for e in eps}
    all_llm = [r for r in dec if r.get("provider") not in (None, "scripted")]
    llm = [r for r in all_llm if r["attempt_id"] in done]
    n = len(llm)
    comp_n = comp_pass = 0
    for r in llm:
        c = r.get("comprehension") or {}
        for k in ("payoff_pass", "value_if_end_pass", "value_if_reject_pass"):
            if k in c and c[k] is not None:
                comp_n += 1
                comp_pass += bool(c[k])
    out = {
        "exp": exp, "model": model, "wording": wording, "episodes": len(eps), "llm_calls": n,
        "calls_incl_incomplete_attempts": len(all_llm),
        "cost_usd": round(sum(Ag.cost_usd(r["model_id"], r) for r in all_llm), 4),
        "parse_failed_rate": (sum(r["parse_failed"] for r in llm) / n) if n else 0,
        "defaulted_rate": (sum(r["defaulted"] for r in llm) / n) if n else 0,
        "refusal_rate": (sum(bool(r["refusal"]) for r in llm) / n) if n else 0,
        "retry_rate": (sum(r["retries"] > 0 for r in llm) / n) if n else 0,
        "comprehension": (comp_pass / comp_n) if comp_n else None,
        "comprehension_n": comp_n,
        "input_tokens": sum(r["input_tokens"] or 0 for r in llm),
        "cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in llm),
        "output_tokens": sum(r["output_tokens"] or 0 for r in llm),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in llm),
    }
    if exp == 2:
        out["exp2_defaulted_trials"] = sum(bool(e.get("defaulted_any")) for e in eps)
    stop = []
    if out["parse_failed_rate"] > 0.05:
        stop.append("parse-failure rate > 5%")
    if out["defaulted_rate"] > 0.05:
        stop.append("default rate > 5%")
    if out["comprehension"] is not None and out["comprehension"] < 0.80:
        stop.append("comprehension < 80%")
    out["stop"] = stop
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True)
    ap.add_argument("--model", default="main")
    ap.add_argument("--wording", default=None)
    ap.add_argument("--cells", default=None, help="comma-separated cell ids to restrict to")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    a = ap.parse_args()
    cells = set(a.cells.split(",")) if a.cells else None
    s = summarize(a.exp, a.model, a.log_dir, a.wording, cells)
    fmt = {k: (f"{v:.2%}" if k.endswith("_rate") or k == "comprehension" and v is not None else v) for k, v in s.items()}
    print(json.dumps(fmt, indent=1))
    sys.exit(3 if s["stop"] else 0)
