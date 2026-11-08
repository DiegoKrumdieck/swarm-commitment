"""Write analysis.ipynb (cells call analysis_lib). Execute with:

    python tools/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Who Can Promise for the Swarm? — analysis

Figures 1–5 and 1b, bootstrap CIs, logistic regression, Kaplan–Meier curves, and headline numbers for H1–H8.

**Rules.** Decision rows come only from completed attempts; pilot/smoke logs (`logs/smoke/`) are never read; Exp 2 trials with a
defaulted call are excluded from the main analysis (count reported). The primary outcome is Swarm A workers' **round-1** overdraw
rate, scored against **both** pre-registered benchmarks: `per_worker` (ℓ counts only costs that reach the worker while active) and
`contest_aware` (a worker that stays also counts the loss of its own future quota when its overdraw would make B end the deal).
95% CIs are percentile bootstraps over episodes (2,000 resamples). Every cell is also reported without defaulted calls and without
calls that failed the payoff comprehension check."""),
    code("""import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
sys.path.insert(0, str(Path.cwd()))
import analysis_lib as A
from tools.run_check import summarize
FIG = A.FIG; FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 40)
def has(exp, model):
    return (A.LOGS / f"episodes_exp{exp}_{model}.jsonl").exists()"""),
    md("## Runs: calls, cost, parse/default rates, comprehension"),
    code("""runs = []
for exp, model, wording in [(1, "main", "A"), (2, "main", None), (3, "main", None), (1, "main", "B"),
                            (1, "cross_family", "A"), (1, "frontier", "A")]:
    if has(exp, model):
        s = summarize(exp, model, wording=wording)
        if s["episodes"]:
            runs.append({k: s[k] for k in ("exp", "model", "wording", "episodes", "llm_calls", "cost_usd",
                                           "parse_failed_rate", "defaulted_rate", "refusal_rate",
                                           "comprehension", "reasoning_tokens", "stop")})
runs = pd.DataFrame(runs)
runs.to_csv(A.ROOT / "results" / "runs_summary.csv", index=False)
print("total cost $%.2f" % runs["cost_usd"].sum())
runs"""),
    md("## Exp 1 — main model (Claude Haiku 5.5), wording A"),
    code("""D1, E1 = A.load(1, "main")
R1 = A.cell_rates(D1)
R1_all = A.cell_rates(D1, rounds="all")
R1_clean = A.cell_rates(D1, exclude="defaulted_or_comprehension")
R1.to_csv(A.ROOT / "results" / "cell_rates_main.csv", index=False)
cols = ["cell_id", "rate", "lo", "hi", "n_episodes", "pw_pred", "ca_pred", "gap_pw", "gap_ca", "h8_noise_check"]
t = R1[cols].merge(R1_clean[["cell_id", "rate"]].rename(columns={"rate": "rate_clean"}), on="cell_id") \\
            .merge(R1_all[["cell_id", "rate"]].rename(columns={"rate": "rate_all_rounds"}), on="cell_id")
t.round(3)"""),
    code("""A.reliability_by_cell(D1).round(3)"""),
    md("### Figure 1 — overdraw rate by mechanism × condition, both benchmarks marked"),
    code("""display(Image(A.fig1(R1, FIG / "fig1_overdraw_grid_main.png", A.MODEL_LABEL["main"])))"""),
    md("### Figure 2 — deal survival (Kaplan–Meier) by condition, one panel per mechanism"),
    code("""display(Image(A.fig2(E1, FIG / "fig2_survival_main.png", A.MODEL_LABEL["main"])))"""),
    md("### Figure 3 — g = 10 vs 30 per cell (H8), `h8_noise_check` cells highlighted"),
    code("""display(Image(A.fig3(R1, FIG / "fig3_g10_vs_g30_main.png", A.MODEL_LABEL["main"])))"""),
    md("### Figure 4 — renaming: freeze units on the actual offender vs on innocents"),
    code("""p = A.fig4(E1, FIG / "fig4_freeze_targets_main.png", A.MODEL_LABEL["main"])
if p: display(Image(p))"""),
    md("### Logistic regression: overdraw ~ mechanism × condition × g, SE clustered by episode (gateway excluded: it exists only under infrastructure)"),
    code("""kind, res, n = A.logistic_regression(D1)
print(kind, "n =", n)
print(res.summary().tables[1])"""),
    md("## Exp 2 and Exp 3 — mechanism choice (Figure 5)"),
    code("""D2, E2 = A.load(2, "main", wording=None) if has(2, "main") else (pd.DataFrame(), pd.DataFrame())
D3, E3 = A.load(3, "main", wording=None) if has(3, "main") else (pd.DataFrame(), pd.DataFrame())
if len(E2):
    print("Exp 2 trials:", len(E2), "| excluded (defaulted):", int((~E2["include_in_main_analysis"]).sum()))
display(Image(A.fig5(E2 if len(E2) else None, E3 if len(E3) else None, FIG / "fig5_mechanism_choice_main.png")))"""),
    code("""if len(E3):
    print("Exp 3 outcomes:", E3["outcome"].value_counts().to_dict())
    agreed = E3[E3["outcome"] == "agreement"]
    print("agreed splits (A share):", agreed["agreed_shares"].apply(lambda s: s["A"]).value_counts().to_dict())
    R3 = A.cell_rates(D3.assign(cell_id=D3["mechanism"].astype(str) + "-" + D3["condition"] + "-g" + D3["g"].astype(str)))
    display(R3[["cell_id", "rate", "lo", "hi", "n_episodes", "pw_pred", "ca_pred"]].round(3))"""),
    md("## Robustness: wording B (KEY_CELLS) and cross-model grids (Figure 1b)"),
    code("""panels = {A.MODEL_LABEL["main"]: R1}
RB = None
if has(1, "main"):
    DB, EB = A.load(1, "main", wording="B")
    if len(DB):
        RB = A.cell_rates(DB)
        comp = RB[["cell_id", "rate", "lo", "hi"]].merge(R1[["cell_id", "rate"]].rename(columns={"rate": "rate_wording_A"}),
                                                        on="cell_id")
        display(comp.round(3))
for model in ("cross_family", "frontier"):
    if has(1, model):
        Dm, Em = A.load(1, model)
        if len(Dm):
            Rm = A.cell_rates(Dm)
            Rm.to_csv(A.ROOT / "results" / f"cell_rates_{model}.csv", index=False)
            panels[A.MODEL_LABEL[model]] = Rm
display(Image(A.fig1b(panels, FIG / "fig1b_models.png")))
if "GPT-6 Luna" in panels:
    display(Image(A.fig1(panels["GPT-6 Luna"], FIG / "fig1_overdraw_grid_cross_family.png", "GPT-6 Luna")))
    display(Image(A.fig3(panels["GPT-6 Luna"], FIG / "fig3_g10_vs_g30_cross_family.png", "GPT-6 Luna")))"""),
    code("""models = {"main": D1}
for model in ("cross_family", "frontier"):
    if has(1, model):
        models[model] = A.load(1, model)[0]
if len(models) > 1:
    kind, res, n = A.logistic_regression(None, models=models)
    print("pooled with model fixed effects:", kind, "n =", n)
    print(res.summary().tables[1])"""),
    md("## Hypotheses H1–H8 — headline numbers against both benchmarks\n\nFormat: rate [95% CI] (per_worker prediction, contest_aware prediction), round 1, g = 10 unless stated."),
    code("""H = {"main": A.hypotheses(D1, E1, R1, E2 if len(E2) else None, D2 if len(D2) else None)}
for model in ("cross_family", "frontier"):
    if has(1, model):
        Dm, Em = A.load(1, model)
        if len(Dm):
            H[model] = A.hypotheses(Dm, Em, A.cell_rates(Dm))
(A.ROOT / "results" / "hypotheses.json").write_text(json.dumps(H, indent=1, default=str))
print(json.dumps(H, indent=1, default=str))"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, ROOT / "analysis.ipynb")
print("wrote", ROOT / "analysis.ipynb")
