"""
analysis_lib.py — data loading, statistics and figures for analysis.ipynb.

Main-analysis rules
  - decision rows only from completed attempts (attempt_id with an episode summary row)
  - pilot / smoke logs (logs/smoke/) are never read
  - Exp 2 trials with a defaulted call are excluded from the main analysis (count reported)
  - every cell is reported with and without comprehension failures and defaulted calls
  - overdraw rates are scored against BOTH pre-registered benchmarks (per_worker, contest_aware)

Primary outcome: Swarm A workers' round-1 overdraw rate (the benchmark's prediction is about
round 1 in every cell; later rounds depend on the realized history). All-rounds rates are
reported alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import params as P

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
FIG = ROOT / "results" / "figures"
MECH_ORDER = ["retaliation", "staged", "collateral", "bonus", "gateway"]
COND_ORDER = ["stable", "turnover", "renaming", "infrastructure"]
MODEL_LABEL = {"main": "Claude Haiku 5.5", "cross_family": "GPT-6 Luna", "frontier": "GPT-6 Sol",
               "dryrun": "dry run (scripted)"}


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load(exp: int, model: str, log_dir: Path = LOGS, wording: str | None = "A",
         cells: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(decisions, episodes) for completed attempts. wording='A' is the main analysis; 'B' the
    robustness slice; None keeps both. cells restricts to a set of cell ids."""
    eps = _jsonl(Path(log_dir) / f"episodes_exp{exp}_{model}.jsonl")
    dec = _jsonl(Path(log_dir) / f"exp{exp}_{model}.jsonl")
    if wording is not None:
        eps = [e for e in eps if e.get("wording", "A") == wording]
        dec = [r for r in dec if r.get("wording", "A") == wording]
    if cells is not None:
        eps = [e for e in eps if e["cell_id"] in cells]
        dec = [r for r in dec if r["cell_id"] in cells]
    done = {e["attempt_id"] for e in eps}
    dec = [r for r in dec if r["attempt_id"] in done]
    E = pd.DataFrame(eps)
    D = pd.DataFrame(dec)
    if len(D):
        D["llm"] = ~D["provider"].isin(["scripted"]) & D["provider"].notna()
        comp = D["comprehension"].apply(lambda c: c if isinstance(c, dict) else {})
        D["payoff_pass"] = comp.apply(lambda c: c.get("payoff_pass"))
        D["value_if_end_pass"] = comp.apply(lambda c: c.get("value_if_end_pass"))
        D["future_cost"] = comp.apply(lambda c: c.get("expected_future_cost_if_overdraw"))
        D["overdraw"] = (D["action"] == "overdraw").astype(float)
    if len(E):
        E["model"] = model
    return D, E


def benchmark() -> dict:
    return json.loads((ROOT / "results" / "benchmark.json").read_text())


def bench_table() -> pd.DataFrame:
    b = benchmark()
    rows = []
    for r in b["exp1"]:
        rows.append({
            "cell_id": f"{r['mechanism']}-{r['condition']}-g{r['g']}",
            "mechanism": r["mechanism"], "condition": r["condition"], "g": r["g"],
            "pw_action": r["predicted_action_round1"], "pw_ell": r["ell"], "pw_ender": r["predicted_ender"],
            "pw_welfare": r["expected_welfare"],
            "ca_action": r["contest_aware"]["predicted_action_round1"], "ca_ell": r["contest_aware"]["ell"],
            "ca_ender": r["contest_aware"]["predicted_ender"], "ca_welfare": r["contest_aware"]["expected_welfare"],
            "h8_noise_check": r["h8_noise_check"],
        })
    t = pd.DataFrame(rows)
    t["pw_pred"] = (t["pw_action"] == "overdraw").astype(float)
    t["ca_pred"] = (t["ca_action"] == "overdraw").astype(float)
    return t


# ---------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------


def cluster_bootstrap(df: pd.DataFrame, value: str, cluster: str = "episode", n: int = 2000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean of `value` with a 95% percentile CI, resampling clusters (episodes) with replacement."""
    if df.empty:
        return (np.nan, np.nan, np.nan)
    g = df.groupby(cluster)[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    boots = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return (s.sum() / c.sum(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def a_worker_rows(D: pd.DataFrame, rounds: str = "first", exclude: str = "none") -> pd.DataFrame:
    """Swarm A worker decisions. rounds: 'first' (round 1) or 'all'.
    exclude: 'none' | 'defaulted' | 'defaulted_or_comprehension' (drop defaulted calls and calls
    that failed the payoff comprehension check)."""
    w = D[(D["role"] == "worker") & (D["side"] == "A") & D["llm"]].copy()
    if rounds == "first":
        w = w[w["round"] == 1]
    if exclude in ("defaulted", "defaulted_or_comprehension"):
        w = w[~w["defaulted"].astype(bool)]
    if exclude == "defaulted_or_comprehension":
        w = w[w["payoff_pass"].astype(bool)]
    return w


def cell_rates(D: pd.DataFrame, rounds: str = "first", exclude: str = "none", n_boot: int = 2000) -> pd.DataFrame:
    """Overdraw rate per Exp 1 cell with cluster-bootstrap CI, joined to both benchmarks."""
    w = a_worker_rows(D, rounds, exclude)
    out = []
    for cid, grp in w.groupby("cell_id"):
        m, lo, hi = cluster_bootstrap(grp, "overdraw", n=n_boot)
        out.append({"cell_id": cid, "rate": m, "lo": lo, "hi": hi, "n_decisions": len(grp),
                    "n_episodes": grp["episode"].nunique()})
    R = pd.DataFrame(out)
    if R.empty:
        return R
    B = bench_table()
    R = R.merge(B, on="cell_id", how="left")
    R["gap_pw"] = R["rate"] - R["pw_pred"]
    R["gap_ca"] = R["rate"] - R["ca_pred"]
    R["match_pw"] = 1 - (R["rate"] - R["pw_pred"]).abs()
    R["match_ca"] = 1 - (R["rate"] - R["ca_pred"]).abs()
    return R.sort_values(["g", "mechanism", "condition"]).reset_index(drop=True)


def reliability_by_cell(D: pd.DataFrame) -> pd.DataFrame:
    L = D[D["llm"]]
    g = L.groupby("cell_id").agg(calls=("llm", "size"), defaulted=("defaulted", "mean"),
                                 parse_failed=("parse_failed", "mean"))
    w = L[L["role"] == "worker"].groupby("cell_id")["payoff_pass"].mean().rename("worker_payoff_pass")
    n = L[(L["role"] == "negotiator") & L["value_if_end_pass"].notna()].groupby("cell_id")["value_if_end_pass"] \
        .mean().rename("negotiator_value_pass")
    return g.join(w).join(n).reset_index()


def logistic_regression(D: pd.DataFrame, rounds: str = "first", models: dict | None = None):
    """overdraw ~ mechanism x condition x g (+ model FE when several models), SE clustered by
    episode. Falls back to a linear probability model when the logit separates (cells at 0/100%).
    Returns (kind, fitted result, n)."""
    import statsmodels.formula.api as smf
    import warnings

    frames = []
    if models:
        for name, d in models.items():
            w = a_worker_rows(d, rounds)
            w = w.assign(model_fe=name)
            frames.append(w)
    else:
        frames.append(a_worker_rows(D, rounds).assign(model_fe="m"))
    w = pd.concat(frames, ignore_index=True)
    w = w[w["mechanism"] != "gateway"]  # gateway exists only under infrastructure
    w["gfac"] = w["g"].astype(str)
    w["clu"] = w["model_fe"] + ":" + w["cell_id"] + ":" + w["episode"].astype(str)
    fe = " + C(model_fe)" if w["model_fe"].nunique() > 1 else ""
    formula = "overdraw ~ C(mechanism) * C(condition) * C(gfac)" + fe
    groups = pd.factorize(w["clu"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.logit(formula, data=w).fit(disp=0, maxiter=200, cov_type="cluster",
                                                 cov_kwds={"groups": groups})
            if not np.all(np.isfinite(res.bse)) or (res.bse > 50).any():
                raise ValueError("quasi-separation")
            return "logit", res, len(w)
        except Exception:
            res = smf.ols(formula, data=w).fit(cov_type="cluster", cov_kwds={"groups": groups})
            return "linear probability (logit separates)", res, len(w)


def km_table(E: pd.DataFrame) -> pd.DataFrame:
    """Deal survival: duration = rounds the deal survived; event = a negotiator ended it."""
    e = E.copy()
    e["event"] = e["ended_by"].notna().astype(int)
    e["duration"] = np.where(e["event"] == 1, e["ended_round"].fillna(0) - 1, e["rounds_played"])
    return e


# ---------------------------------------------------------------------------------
# Hypotheses (headline numbers, scored against both benchmarks)
# ---------------------------------------------------------------------------------


def _rate(R, mech, cond, g=10):
    r = R[(R["mechanism"] == mech) & (R["condition"] == cond) & (R["g"] == g)]
    return None if r.empty else r.iloc[0]


def _fmt(r):
    if r is None:
        return "n/a"
    return f"{r['rate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] (pw {r['pw_pred']:.0f}, ca {r['ca_pred']:.0f})"


def hypotheses(D: pd.DataFrame, E: pd.DataFrame, R: pd.DataFrame, exp2: pd.DataFrame | None = None,
               exp2_dec: pd.DataFrame | None = None) -> dict:
    """Headline numbers for H1-H8. R = cell_rates (round 1)."""
    H = {}
    # H1 friction: retaliation and staged hold under stable, fail under turnover (g=10)
    H["H1"] = {f"{m}: stable": _fmt(_rate(R, m, "stable")) for m in ("retaliation", "staged")}
    H["H1"].update({f"{m}: turnover": _fmt(_rate(R, m, "turnover")) for m in ("retaliation", "staged")})
    # H2 agency: under turnover, bonus restores compliance where the others fail
    H["H2"] = {f"{m}: turnover": _fmt(_rate(R, m, "turnover")) for m in ("bonus", "retaliation", "staged", "collateral")}
    # H3 insurance: collateral x turnover — overdraws, who ends, B's realized value
    ct = E[(E["cell_id"] == "collateral-turnover-g10")]
    ended = ct[ct["ended_by"].notna()]
    H["H3"] = {
        "overdraw (round 1)": _fmt(_rate(R, "collateral", "turnover")),
        "episodes ended": f"{len(ended)}/{len(ct[ct['T'] > 1])} with T>1",
        "ended by A / B / both": f"{(ended['ended_by'] == 'A').sum()} / {(ended['ended_by'] == 'B').sum()} / "
                                 f"{(ended['ended_by'] == 'both').sum()}  (both benchmarks: A ends)",
        "mean swarm scores A / B": f"{ct['swarm_scores'].apply(lambda s: s['A']).mean():.1f} / "
                                   f"{ct['swarm_scores'].apply(lambda s: s['B']).mean():.1f} "
                                   f"(benchmarks: 150 / 180)",
    }
    # H4 attribution: renaming x retaliation — B's freeze modes, units on offenders vs innocents
    H["H4"] = {}
    for cond in ("stable", "renaming", "turnover"):
        e = E[(E["mechanism"] == "retaliation") & (E["condition"] == cond)]
        fr = [f for fs in e["freezes"] for f in fs if f["by"] == "B"]
        modes = pd.Series([f["mode"] for f in fr]).value_counts().to_dict() if fr else {}
        on, inn = e["freeze_units_on_offenders"].sum(), e["freeze_units_on_innocents"].sum()
        H["H4"][cond] = (f"freeze decisions {len(fr)} {modes}; units on offenders {on:.1f}, on innocents {inn:.1f}"
                         + (f" ({inn / (on + inn):.0%} on innocents)" if on + inn else ""))
    H["H4"]["overdraw renaming g10"] = _fmt(_rate(R, "retaliation", "renaming"))
    # H5 placebo: renaming should not change staged / collateral / bonus
    H["H5"] = {}
    for m in ("staged", "collateral", "bonus"):
        a, b = _rate(R, m, "stable"), _rate(R, m, "renaming")
        if a is not None and b is not None:
            H["H5"][m] = f"renaming - stable = {b['rate'] - a['rate']:+.2f} (stable {a['rate']:.2f}, renaming {b['rate']:.2f}; benchmarks 0)"
    # H6 choice (Exp 2)
    if exp2 is not None and len(exp2):
        x = exp2[exp2["include_in_main_analysis"]]
        tab = pd.crosstab(x["condition"], x["proposed"], normalize="index").round(2)
        acc = x.assign(acc=(x["decision"] == "accept")).groupby("proposed")["acc"].mean().round(2).to_dict()
        H["H6"] = {"proposal shares by condition": tab.to_dict(orient="index"),
                   "B acceptance by proposed option": acc,
                   "excluded defaulted trials": int((~exp2["include_in_main_analysis"]).sum()),
                   "benchmark choice (per_worker / contest_aware)": rational_choice_table()}
    # H7 infrastructure: gateway holds under turnover + renaming
    H["H7"] = {"gateway x infrastructure g10": _fmt(_rate(R, "gateway", "infrastructure")),
               "for comparison retaliation x turnover g10": _fmt(_rate(R, "retaliation", "turnover"))}
    # H8 strategic, not noise
    H["H8"] = {}
    for (m, c), grp in R.groupby(["mechanism", "condition"]):
        r10, r30 = grp[grp["g"] == 10], grp[grp["g"] == 30]
        if len(r10) and len(r30):
            r10, r30 = r10.iloc[0], r30.iloc[0]
            tag = " [h8 cell]" if bool(r10["h8_noise_check"]) else ""
            H["H8"][f"{m} x {c}{tag}"] = (f"{r10['rate']:.2f} -> {r30['rate']:.2f} "
                                          f"(pw {r10['pw_pred']:.0f}->{r30['pw_pred']:.0f}, "
                                          f"ca {r10['ca_pred']:.0f}->{r30['ca_pred']:.0f})")
    h8 = R[R["h8_noise_check"] == True]  # noqa: E712
    if len(h8):
        d = h8[h8["g"] == 30]["rate"].mean() - h8[h8["g"] == 10]["rate"].mean()
        H["H8"]["mean rise in h8 cells"] = f"{d:+.2f} (both benchmarks: +1)"
    other = R[R["h8_noise_check"] == False]  # noqa: E712
    d2 = other[other["g"] == 30]["rate"].mean() - other[other["g"] == 10]["rate"].mean()
    H["H8"]["mean rise in other cells"] = f"{d2:+.2f} (per_worker: flips where g=10 complies; contest_aware: 0 under stable/renaming)"
    # benchmark fit summary
    H["fit"] = {
        "mean |rate - per_worker|": round(float((R["rate"] - R["pw_pred"]).abs().mean()), 3),
        "mean |rate - contest_aware|": round(float((R["rate"] - R["ca_pred"]).abs().mean()), 3),
    }
    return H


def rational_choice_table() -> dict:
    """Benchmark Exp 2 choice per condition and g: A's best option among those B accepts
    (B's expected score > contest value), under each benchmark (B's workers comply)."""
    b = benchmark()
    out = {}
    for bm in ("per_worker", "contest_aware"):
        res = {}
        rows = b["menu_grid"]
        for c in P.CONDITIONS:
            for g in P.G_LEVELS:
                cand = []
                for r in rows:
                    if r["condition"] != c or r["g"] != g:
                        continue
                    v = r if bm == "per_worker" else r["contest_aware"]
                    key = P.MECH_KEYS[r["mechanism"]]
                    cand.append((key, v["expected_score_A"], v["expected_score_B"]))
                ok = [x for x in cand if x[2] > P.CONTEST_VALUE + 1e-9] or cand
                best = max(x[1] for x in ok)
                res[f"{c} g{g}"] = sorted(k for k, a, _ in ok if abs(a - best) < 1e-6)
        out[bm] = res
    return out


# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------


def _ax_grid(ax, R, title, g=10):
    import matplotlib.pyplot as plt  # noqa: F401

    r = R[R["g"] == g]
    conds = [c for c in COND_ORDER if c in set(r["condition"])]
    mechs = [m for m in MECH_ORDER if m in set(r["mechanism"])]
    width = 0.8 / max(1, len(conds))
    colors = {"stable": "#3B6EA8", "turnover": "#C8553D", "renaming": "#6A994E", "infrastructure": "#8E6C8A"}
    for j, c in enumerate(conds):
        for i, m in enumerate(mechs):
            row = r[(r["mechanism"] == m) & (r["condition"] == c)]
            if row.empty:
                continue
            row = row.iloc[0]
            x = i + (j - (len(conds) - 1) / 2) * width
            ax.bar(x, row["rate"], width * 0.9, color=colors[c], label=c if i == 0 or m == "gateway" else None)
            ax.errorbar(x, row["rate"], yerr=[[row["rate"] - row["lo"]], [row["hi"] - row["rate"]]],
                        color="black", lw=1, capsize=2)
            ax.scatter(x, row["pw_pred"], marker="_", s=180, color="black", zorder=5,
                       label="per_worker benchmark" if (i == 0 and j == 0) else None)
            ax.scatter(x, row["ca_pred"], marker="o", s=28, facecolors="none", edgecolors="black", zorder=6,
                       label="contest_aware benchmark" if (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(mechs)))
    ax.set_xticklabels(mechs)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Swarm A overdraw rate, round 1")
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False)


def fig1(R: pd.DataFrame, path: Path, label: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    _ax_grid(axes[0], R, f"{label}: g = 10", g=10)
    _ax_grid(axes[1], R, f"{label}: g = 30", g=30)
    fig.suptitle("Figure 1. Overdraw rate by mechanism x condition (bars: 95% cluster-bootstrap CI; "
                 "dash: per_worker benchmark, circle: contest_aware)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig1b(panels: dict, path: Path):
    """Side-by-side g=10 grids for several models (dict label -> cell_rates)."""
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), sharey=True, squeeze=False)
    for ax, (label, R) in zip(axes[0], panels.items()):
        if R is None or R.empty:
            ax.set_title(f"{label}: no data")
            continue
        _ax_grid(ax, R, f"{label}: g = 10", g=10)
    fig.suptitle("Figure 1b. Overdraw rate at g = 10 across models", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2(E: pd.DataFrame, path: Path, label: str):
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    e = km_table(E)
    mechs = [m for m in MECH_ORDER if m in set(e["mechanism"])]
    fig, axes = plt.subplots(1, len(mechs), figsize=(3.3 * len(mechs), 3.4), sharey=True, squeeze=False)
    colors = {"stable": "#3B6EA8", "turnover": "#C8553D", "renaming": "#6A994E", "infrastructure": "#8E6C8A"}
    for ax, m in zip(axes[0], mechs):
        for c in COND_ORDER:
            for g, ls in ((10, "-"), (30, "--")):
                sub = e[(e["mechanism"] == m) & (e["condition"] == c) & (e["g"] == g)]
                if sub.empty:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(sub["duration"], sub["event"], label=f"{c} g{g}")
                kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[c], ls=ls, lw=1.4)
        ax.set_title(m)
        ax.set_xlabel("rounds survived")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, frameon=False)
    axes[0][0].set_ylabel("P(deal not ended)")
    fig.suptitle(f"Figure 2. Deal survival (Kaplan-Meier; game end by the continuation draw is censoring) - {label}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig3(R: pd.DataFrame, path: Path, label: str):
    import matplotlib.pyplot as plt

    a = R[R["g"] == 10].set_index(["mechanism", "condition"])
    b = R[R["g"] == 30].set_index(["mechanism", "condition"])
    j = a[["rate", "h8_noise_check"]].join(b[["rate"]], rsuffix="_30").dropna()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    for (m, c), row in j.iterrows():
        h8 = bool(row["h8_noise_check"])
        ax.scatter(row["rate"], row["rate_30"], s=60 if h8 else 25, color="#C8553D" if h8 else "#3B6EA8",
                   edgecolors="black" if h8 else "none", zorder=3)
        ax.annotate(f"{m[:5]}x{c[:4]}", (row["rate"], row["rate_30"]), fontsize=6, xytext=(3, 2),
                    textcoords="offset points")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("overdraw rate, g = 10")
    ax.set_ylabel("overdraw rate, g = 30")
    ax.set_title(f"Figure 3. g = 10 vs 30 per cell ({label})\nred = h8_noise_check cells "
                 "(both benchmarks: 0 -> 1)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4(E: pd.DataFrame, path: Path, label: str):
    import matplotlib.pyplot as plt

    e = E[E["mechanism"] == "retaliation"]
    rows = []
    for (c, g), grp in e.groupby(["condition", "g"]):
        rows.append({"cell": f"{c}\ng{g}", "offenders": grp["freeze_units_on_offenders"].sum() / len(grp),
                     "innocents": grp["freeze_units_on_innocents"].sum() / len(grp), "c": c, "g": g})
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    t["o"] = t["c"].map({c: i for i, c in enumerate(COND_ORDER)})
    t = t.sort_values(["o", "g"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(t))
    ax.bar(x, t["offenders"], color="#3B6EA8", label="on the worker(s) that overdrew")
    ax.bar(x, t["innocents"], bottom=t["offenders"], color="#C8553D", label="on workers that did not")
    ax.set_xticks(x)
    ax.set_xticklabels(t["cell"])
    ax.set_ylabel("freeze units placed per episode\n(units per round, summed over freezes)")
    ax.set_title(f"Figure 4. Who absorbs the freezes (retaliation cells, {label})", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig5(exp2: pd.DataFrame | None, exp3: pd.DataFrame | None, path: Path):
    import matplotlib.pyplot as plt

    keys = ["freezes", "staged_release", "deposits", "compliance_payment", "protocol_gateway", "none"]
    colors = dict(zip(keys, ["#3B6EA8", "#6A994E", "#E0A458", "#C8553D", "#8E6C8A", "#999999"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (title, df, col) in zip(axes, (("Exp 2: A's proposed option (bar) and B's acceptance (dot)", exp2, "proposed"),
                                           ("Exp 3: agreed option (share of episodes; grey = no agreement)", exp3,
                                            "agreed_mechanism"))):
        if df is None or df.empty:
            ax.set_title(title + "\n(no data)", fontsize=8)
            continue
        d = df.copy()
        if col == "proposed":
            d = d[d["include_in_main_analysis"]]
        d["label"] = d["condition"] + " g" + d["g"].astype(str)
        order = [f"{c} g{g}" for g in P.G_LEVELS for c in COND_ORDER]
        order = [o for o in order if o in set(d["label"])]
        for i, lab in enumerate(order):
            sub = d[d["label"] == lab]
            bottom = 0.0
            for k in keys + [None]:
                share = ((sub[col] == k) if k else sub[col].isna()).mean()
                if share:
                    ax.bar(i, share, bottom=bottom, color=colors.get(k, "#DDDDDD"))
                    bottom += share
            if col == "proposed":
                ax.scatter(i, (sub["decision"] == "accept").mean(), color="black", zorder=5)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        present = [k for k in keys if (d[col] == k).any()]
        handles = [Patch(color=colors[k], label=k) for k in present]
        if d[col].isna().any():
            handles.append(Patch(color="#DDDDDD", label="no agreement"))
        if col == "proposed":
            handles.append(Line2D([], [], marker="o", color="black", ls="", label="B accepts"))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=8)
        ax.legend(handles=handles, fontsize=6, frameon=False, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Figure 5. Mechanism choice by condition", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
