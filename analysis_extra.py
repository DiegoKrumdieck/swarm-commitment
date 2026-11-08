"""
analysis_extra.py — EXPLORATORY analyses (not pre-registered), added after the main results.

  contagion        turnover / infrastructure cells: overdraw by round (Fig 6a), P(overdraw in t | any A
                   overdraw in t-1) vs after a clean round (Fig 6b), per-cell LPM with episode FE
  horizon          how models read the continuation horizon: B's expected_value_if_continue after a clean
                   history (truth 300), end rate after a clean history, A workers' future-cost estimates
                   by round in stable cells (Fig 7)
  renaming         keyword classification of A worker reasoning (changing IDs / not identifiable / spread
                   freezes) in renaming and stable cells; B freeze modes by model and condition
  exit_vs_punish   retaliation x turnover: B's realized per-round value, implied continuation value vs 150,
                   B's end rate, freeze units on innocents
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import analysis_lib as A
import params as P

TURN = ("turnover", "infrastructure")
NON_STAGED = ("retaliation", "collateral", "bonus", "gateway")


def pays_now(mechanism: str, g: int) -> bool:
    """Does overdrawing raise the worker's own score this round? (+g vs bonus 15 / gateway 25)"""
    return g == 30 or mechanism in ("retaliation", "collateral", "staged")


def _per_round(E: pd.DataFrame) -> dict:
    return dict(zip(E["attempt_id"], E["overdraws_per_round"]))


def _clean_before(per_round: list, t: int) -> bool:
    """No overdraw by anyone in rounds 1..t-1."""
    return all(r["A"] == 0 and r["B"] == 0 for r in per_round[: t - 1])


# ---------------------------------------------------------------------------------
# 2. Contagion
# ---------------------------------------------------------------------------------


def contagion_frame(D: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    w = D[(D["role"] == "worker") & (D["side"] == "A") & D["llm"] & D["condition"].isin(TURN)].copy()
    pr = _per_round(E)
    w["prev_any"] = [
        (pr[a][t - 2]["A"] > 0) if t >= 2 else np.nan for a, t in zip(w["attempt_id"], w["round"])
    ]
    w["pays"] = [pays_now(m, g) for m, g in zip(w["mechanism"], w["g"])]
    return w


def contagion_table(D: pd.DataFrame, E: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    import warnings

    w = contagion_frame(D, E)
    w2 = w[w["round"] >= 2].copy()
    rows = []
    for cid, x in w2.groupby("cell_id"):
        after = x[x["prev_any"] == True]  # noqa: E712
        clean = x[x["prev_any"] == False]  # noqa: E712
        a = A.cluster_bootstrap(after, "overdraw", "attempt_id", n_boot) if len(after) else (np.nan,) * 3
        c = A.cluster_bootstrap(clean, "overdraw", "attempt_id", n_boot) if len(clean) else (np.nan,) * 3
        coef = se = p = np.nan
        if x["prev_any"].nunique() > 1 and x["overdraw"].nunique() > 1:
            xx = x.assign(prev=x["prev_any"].astype(float))
            groups = pd.factorize(xx["attempt_id"])[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = smf.ols("overdraw ~ prev + round + C(attempt_id)", data=xx).fit(
                        cov_type="cluster", cov_kwds={"groups": groups})
                    coef, se, p = res.params["prev"], res.bse["prev"], res.pvalues["prev"]
                except Exception:
                    pass
        rows.append({
            "cell_id": cid, "pays_now": bool(x["pays"].iloc[0]),
            "p_after_overdraw": a[0], "after_lo": a[1], "after_hi": a[2], "n_after": len(after),
            "p_after_clean": c[0], "clean_lo": c[1], "clean_hi": c[2], "n_clean": len(clean),
            "lpm_prev_coef": coef, "lpm_prev_se": se, "lpm_prev_p": p,
            "n_rows": len(x), "n_episodes": x["attempt_id"].nunique(),
        })
    return pd.DataFrame(rows).sort_values(["pays_now", "cell_id"], ascending=[False, True]).reset_index(drop=True)


def fig6a(frames: dict, path, max_round: int = 8):
    """frames: model label -> contagion_frame. Overdraw rate by round; cols: pays now / not."""
    import matplotlib.pyplot as plt

    labels = list(frames)
    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 3.6 * len(labels)), sharey=True, squeeze=False)
    for i, lab in enumerate(labels):
        w = frames[lab]
        for j, pays in enumerate((True, False)):
            ax = axes[i][j]
            x = w[w["pays"] == pays]
            for cid, grp in x.groupby("cell_id"):
                g = grp[grp["round"] <= max_round].groupby("round")["overdraw"].agg(["mean", "count"])
                g = g[g["count"] >= 6]
                ax.plot(g.index, g["mean"], marker="o", ms=3, lw=1.2, label=cid)
            ax.set_title(f"{lab}: overdraw {'pays the worker' if pays else 'does not pay'} this round", fontsize=9)
            ax.set_xlabel("round")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6, frameon=False, ncol=2)
        axes[i][0].set_ylabel("Swarm A overdraw rate")
    fig.suptitle("Figure 6a (exploratory). Overdraw rate by round, turnover and infrastructure cells "
                 "(points with >= 6 decisions)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig6b(tables: dict, path):
    """tables: model label -> contagion_table."""
    import matplotlib.pyplot as plt

    labels = list(tables)
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 4.4), sharey=True, squeeze=False)
    for ax, lab in zip(axes[0], labels):
        t = tables[lab]
        x = np.arange(len(t))
        for off, pre, col, name in ((-0.12, "after", "#C8553D", "after a round with an A overdraw"),
                                    (0.12, "clean", "#3B6EA8", "after a clean round")):
            m = t[f"p_after_{'overdraw' if pre == 'after' else 'clean'}"]
            lo, hi = t[f"{pre}_lo"], t[f"{pre}_hi"]
            ok = m.notna()
            ax.errorbar(x[ok] + off, m[ok], yerr=[(m - lo)[ok], (hi - m)[ok]], fmt="o", color=col, ms=4,
                        capsize=2, label=name)
        for k, (pays, cid) in enumerate(zip(t["pays_now"], t["cell_id"])):
            if not pays:
                ax.axvspan(k - 0.45, k + 0.45, color="#EEEEEE", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(t["cell_id"], rotation=40, ha="right", fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{lab} (grey: overdraw does not pay this round)", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("P(A worker overdraws in round t), t >= 2")
    fig.suptitle("Figure 6b (exploratory). Contagion: overdraw after an A overdraw in t-1 vs after a clean "
                 "round (95% episode-cluster bootstrap CI)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------
# 3. Horizon reading
# ---------------------------------------------------------------------------------


def horizon_tables(D: pd.DataFrame, E: pd.DataFrame, max_round: int = 8) -> dict:
    pr = _per_round(E)
    n = D[(D["role"] == "negotiator") & D["llm"] & (D["round"] >= 2)].copy()
    n["clean"] = [_clean_before(pr[a], t) for a, t in zip(n["attempt_id"], n["round"])]
    n["ev_cont"] = n["comprehension"].apply(lambda c: (c or {}).get("expected_value_if_continue"))
    n["end"] = (n["decision"] == "end").astype(float)
    nc = n[n["clean"]]
    bv = nc[(nc["side"] == "B") & nc["mechanism"].isin(NON_STAGED)]
    b_value = bv.groupby("round")["ev_cont"].agg(["mean", "median", "count"])
    end_rate = nc.groupby(["side", "round"])["end"].agg(["mean", "count"]).unstack(0)
    w = D[(D["role"] == "worker") & (D["side"] == "A") & D["llm"] & (D["condition"] == "stable")].copy()
    fut = w.groupby("round").agg(mean=("future_cost", "mean"), median=("future_cost", "median"),
                                 bench_ell=("ell_state", "mean"), count=("future_cost", "size"))
    fut_mech = w.groupby(["mechanism", "round"])["future_cost"].median().unstack(0)
    return {"b_continue_value_clean": b_value[b_value.index <= max_round],
            "end_rate_clean": end_rate[end_rate.index <= max_round],
            "future_cost_stable": fut[fut.index <= max_round],
            "future_cost_stable_by_mech": fut_mech[fut_mech.index <= max_round],
            "n_clean_negotiator_decisions": int(len(nc)),
            "overall_end_rate_clean": {s: float(nc[nc["side"] == s]["end"].mean()) for s in ("A", "B")}}


def fig7(horizons: dict, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for lab, h in horizons.items():
        b = h["b_continue_value_clean"]
        b = b[b["count"] >= 5]
        axes[0].plot(b.index, b["median"], marker="o", label=lab)
        er = h["end_rate_clean"]
        if ("count", "B") in er:
            m = (er[("mean", "A")].fillna(0) * er[("count", "A")].fillna(0)
                 + er[("mean", "B")].fillna(0) * er[("count", "B")].fillna(0)) / \
                (er[("count", "A")].fillna(0) + er[("count", "B")].fillna(0))
            cnt = er[("count", "A")].fillna(0) + er[("count", "B")].fillna(0)
            m = m[cnt >= 5]
            axes[1].plot(m.index, m, marker="o", label=lab)
        f = h["future_cost_stable"]
        f = f[f["count"] >= 5]
        axes[2].plot(f.index, f["median"], marker="o", label=lab)
    axes[0].axhline(300, color="black", ls="--", lw=1, label="truth (300)")
    axes[0].set_title("B's expected value if the deal continues,\nafter a fully clean history (median)", fontsize=9)
    axes[1].set_title("End rate after a fully clean history\n(A and B negotiator decisions)", fontsize=9)
    axes[2].set_title("A workers' expected future cost of overdrawing,\nstable cells (median)", fontsize=9)
    axes[2].axhline(20, color="black", ls=":", lw=1, label="per_worker ell (20-25)")
    for ax in axes:
        ax.set_xlabel("round")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Figure 7 (exploratory). How models read the continuation horizon", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------
# 4. Renaming: what workers say, and B's freeze modes
# ---------------------------------------------------------------------------------

RENAMING_PATTERNS = [
    r"\b(new|changing|changes?|changed|fresh|different|rotating|rotates?)\b[^.;]{0,40}\b(public )?ids?\b",
    r"\b(public )?ids?\b[^.;]{0,40}\b(change|changes|changing|rotate|rotates|differ|new each|every round|each round)\b",
    r"identif|anonym|untrace|traceab|untarget|targetab",
    r"can(?:no|')?t (?:be )?(?:tell|trace|target|identify|pin|know which|see which|single)",
    r"(?:not|never) (?:be )?(?:identif|trace|target|attribut|single)",
    r"which (?:one|worker|of )[^.;]{0,30}(?:overdrew|did|was)",
    r"\bonly (?:see |sees |the )?(?:the )?(?:count|number)\b|how many (?:of )?(?:our|my|the)? ?workers",
    r"\bspread\b|\bsplit\b[^.;]{0,25}freez",
    r"freez[^.;]{0,40}\b(all three|every (?:one|worker|active worker|\w+ worker)|each of the three|all (?:of )?(?:our|the|my) (?:active )?workers)\b",
]
RENAMING_RX = re.compile("|".join(f"(?:{p})" for p in RENAMING_PATTERNS), re.I)


def renaming_reasoning(models: dict, seed: int = 0) -> dict:
    """models: label -> D. Classify A worker reasoning in renaming and stable cells."""
    rows, samples = [], []
    for lab, D in models.items():
        w = D[(D["role"] == "worker") & (D["side"] == "A") & D["llm"] & D["condition"].isin(["renaming", "stable"])].copy()
        w["reasoning"] = w["parsed"].apply(lambda p: (p or {}).get("reasoning") or "")
        w["match"] = w["reasoning"].apply(lambda s: bool(RENAMING_RX.search(s)))
        w["model"] = lab
        for (cond, act), x in w.groupby(["condition", "action"]):
            rows.append({"model": lab, "condition": cond, "action": act, "n": len(x),
                         "mentions_ids_or_identifiability": float(x["match"].mean())})
        samples.append(w[w["condition"] == "renaming"][["model", "cell_id", "round", "action", "match", "reasoning"]])
    T = pd.DataFrame(rows)
    S = pd.concat(samples, ignore_index=True)
    rng = np.random.default_rng(seed)
    pick = lambda df: df.iloc[rng.choice(len(df), size=min(10, len(df)), replace=False)] if len(df) else df  # noqa: E731
    # draw matches across actions so overdraw reasoning is represented
    m = S[S["match"]]
    mo, mq = m[m["action"] == "overdraw"], m[m["action"] == "quota"]
    matches = pd.concat([pick(mo).head(5), pick(mq).head(10 - min(5, len(mo)))])
    nonm = S[~S["match"]]
    nmo, nmq = nonm[nonm["action"] == "overdraw"], nonm[nonm["action"] == "quota"]
    nonmatches = pd.concat([pick(nmo).head(5), pick(nmq).head(10 - min(5, len(nmo)))])
    return {"table": T, "matches": matches, "nonmatches": nonmatches}


def freeze_modes(models: dict) -> pd.DataFrame:
    """models: label -> E. B's freeze decisions (retaliation cells) by mode, model, condition, g."""
    rows = []
    for lab, E in models.items():
        e = E[E["mechanism"] == "retaliation"]
        for (c, g), grp in e.groupby(["condition", "g"]):
            fr = [f for fs in grp["freezes"] for f in fs if f["by"] == "B"]
            cnt = pd.Series([f["mode"] for f in fr], dtype=object).value_counts()
            rows.append({"model": lab, "condition": c, "g": g, "decisions": len(fr),
                         **{m: int(cnt.get(m, 0)) for m in ("targeted", "all", "spread", "none")},
                         "units_on_offenders": float(grp["freeze_units_on_offenders"].sum()),
                         "units_on_innocents": float(grp["freeze_units_on_innocents"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------
# 5. Exit vs punishment: retaliation x turnover
# ---------------------------------------------------------------------------------


def exit_vs_punish(D: pd.DataFrame, E: pd.DataFrame, model: str) -> pd.DataFrame:
    """B's per-round value under retaliation = 60 - 2g * (A overdraws) (B is never frozen in Exp 1).
    Its negotiator at round t sees last round's value v; the benchmark ends iff v / (1 - delta) < 150."""
    out = []
    for g in P.G_LEVELS:
        cid = f"retaliation-turnover-g{g}"
        e = E[E["cell_id"] == cid]
        if e.empty:
            continue
        base_b = P.DEFAULT_SPLIT[1]
        realized = [base_b - P.LOSS_MULT * g * r["A"] for _, x in e.iterrows()
                    for r in x["overdraws_per_round"][1:]]
        pr = _per_round(e)
        nb = D[(D["cell_id"] == cid) & (D["role"] == "negotiator") & (D["side"] == "B") & D["llm"]].copy()
        nb["v_last"] = [base_b - P.LOSS_MULT * g * pr[a][t - 2]["A"] for a, t in zip(nb["attempt_id"], nb["round"])]
        nb["implied"] = nb["v_last"] / (1 - P.DELTA)
        nb["bench_end"] = nb["implied"] < P.CONTEST_VALUE - 1e-9
        nb["end"] = nb["decision"] == "end"
        nb["ev_cont"] = nb["comprehension"].apply(lambda c: (c or {}).get("expected_value_if_continue"))
        be = nb[nb["bench_end"]]
        out.append({
            "model": model, "cell": cid, "episodes": len(e),
            "B_realized_value_per_round_r2plus": float(np.mean(realized)) if realized else np.nan,
            "n_rounds_r2plus": len(realized),
            "B_decisions": len(nb),
            "mean_implied_continuation_value": float(nb["implied"].mean()) if len(nb) else np.nan,
            "share_decisions_benchmark_says_end": float(nb["bench_end"].mean()) if len(nb) else np.nan,
            "B_end_rate": float(nb["end"].mean()) if len(nb) else np.nan,
            "B_end_rate_when_benchmark_says_end": float(be["end"].mean()) if len(be) else np.nan,
            "n_benchmark_says_end": len(be),
            "B_reported_ev_continue_mean": float(nb["ev_cont"].mean()) if len(nb) else np.nan,
            "freeze_units_on_innocents": float(e["freeze_units_on_innocents"].sum()),
            "freeze_units_on_offenders": float(e["freeze_units_on_offenders"].sum()),
            "freeze_decisions_B": sum(1 for fs in e["freezes"] for f in fs if f["by"] == "B"),
        })
    return pd.DataFrame(out)
