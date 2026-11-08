"""
theory.py — payoff-maximizing benchmark for "Who Can Promise for the Swarm?"

For any parameter set (params.py) and any mechanism x condition x g cell this computes:
  - ell: what overdrawing costs the overdrawing worker while it is still active
         (the deterrence threshold; a payoff-maximizing worker overdraws iff g > ell);
  - the predicted worker action;
  - the predicted ender of the agreement (start of round 2) and its round;
  - expected welfare and expected swarm scores along the predicted path.

The predicted path is deterministic (scripted payoff-maximizing agents, Swarm B's workers
comply), so expectations over the random stopping time are exact sums weighted by
P(round t is played) = DELTA**(t-1).

Run `python theory.py` to write results/benchmark.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import params as P

EPS = 1e-9
SIDES = ("A", "B")


def other(side: str) -> str:
    return "B" if side == "A" else "A"


# ---------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------


def contest_value() -> float:
    return P.CONTEST_VALUE


def persists(condition: str, side: str = "A") -> bool:
    """Is the worker still active after this round? Only Swarm A's membership changes."""
    if side == "B":
        return True
    return condition in ("stable", "renaming")


def default_freeze_policy(condition: str, target_side: str = "A") -> str:
    """Rational freezer's policy toward the other swarm's overdrawing workers.

    targeted on identifiable active offenders; 'all' when identities are hidden (renaming);
    nothing when the offenders are gone (turnover / infrastructure)."""
    if target_side == "B":
        return "targeted"  # B is always stable and visible
    return {
        "stable": "targeted",
        "renaming": "all",
        "turnover": "none",
        "infrastructure": "none",
    }[condition]


def freeze_per_round_on_offender(policy: str) -> float:
    return {
        "targeted": P.FREEZE,
        "all": P.FREEZE,
        "spread": P.FREEZE / P.N_WORKERS,
        "none": 0.0,
    }[policy]


def worker_ell(
    mechanism: str | None,
    condition: str,
    side: str = "A",
    policy: str | None = None,
    *,
    unlocking_on: bool = True,
    deposit_left: float = P.DEPOSIT,
    will_charge: bool = True,
) -> float:
    """Cost of overdrawing, borne by this worker while it is still active.

    retaliation: delta/(1-delta) * freeze per round reaching this worker
    staged:      delta/(1-delta)^2 * s while unlocking is on
    collateral:  FORFEIT if its negotiator will charge it next round (and it is still active)
    bonus:       BONUS (forgone this round)
    gateway:     GATEWAY_CHARGE (immediate)
    A worker that is replaced after this round has horizon 0.
    """
    d = P.DELTA
    stays = persists(condition, side)
    if mechanism is None:
        return 0.0
    if mechanism == "retaliation":
        if not stays:
            return 0.0
        pol = policy or default_freeze_policy(condition, side)
        return d / (1 - d) * freeze_per_round_on_offender(pol)
    if mechanism == "staged":
        if not stays or not unlocking_on:
            return 0.0
        return d / (1 - d) ** 2 * P.STAGED_STEP
    if mechanism == "collateral":
        if stays and will_charge and deposit_left > EPS:
            return float(P.FORFEIT)
        return 0.0
    if mechanism == "bonus":
        return float(P.BONUS)
    if mechanism == "gateway":
        return float(P.GATEWAY_CHARGE)
    raise ValueError(mechanism)


BENCHMARKS = ("per_worker", "contest_aware")


def other_value_if_one_overdraw(mechanism: str | None, g: float, other_consumption: float,
                                deposit_left: float = P.DEPOSIT) -> float:
    """The other swarm's per-round value if exactly one worker of this swarm overdraws and
    everyone else complies (what its negotiator then observes)."""
    v = other_consumption - P.LOSS_MULT * g
    if mechanism == "collateral" and deposit_left > EPS:
        v += min(P.FORFEIT, deposit_left)
    if mechanism == "gateway":
        v += P.GATEWAY_CHARGE
    return v


def contest_threat(mechanism: str | None, condition: str, g: float, side: str = "A", *,
                   eff_quota: float, other_consumption: float, deposit_left: float = P.DEPOSIT) -> float:
    """contest_aware extra cost: a worker that stays active loses its own future quota,
    delta/(1-delta) * effective quota, if its overdraw makes the other negotiator's
    continuation value fall below the contest value (so it ends the deal). Horizon 0 -> 0."""
    if not persists(condition, side):
        return 0.0
    if negotiator_ends(other_value_if_one_overdraw(mechanism, g, other_consumption, deposit_left)):
        return P.DELTA / (1 - P.DELTA) * eff_quota
    return 0.0


def predicts_overdraw(g: float, ell: float) -> bool:
    return g > ell + EPS


def negotiator_ends(value_per_round: float) -> bool:
    """End iff the per-round value, assumed to persist, is worth less than the contest."""
    return value_per_round / (1 - P.DELTA) < P.CONTEST_VALUE - EPS


# ---------------------------------------------------------------------------------
# Predicted path (deterministic) and its expectation over the stopping time
# ---------------------------------------------------------------------------------


def predicted_path(
    mechanism: str | None,
    condition: str,
    g: float,
    policy: str | None = None,
    shares: tuple[int, int] = P.DEFAULT_SPLIT,
    horizon: int = 400,
    benchmark: str = "per_worker",
) -> dict:
    """Scripted payoff-maximizing play: A workers RationalWorker, B workers comply,
    A negotiator charges every chargeable worker, B freezes per `policy`, both end per
    `negotiator_ends`. Returns predictions and exact expectations."""
    d = P.DELTA
    base = {"A": shares[0] / P.N_WORKERS, "B": shares[1] / P.N_WORKERS}
    pol = policy or default_freeze_policy(condition, "A")
    unlocking = True
    inc = 0.0                       # staged increase per position (same for all positions)
    frozen = [0.0] * P.N_WORKERS    # per A worker currently active
    deposits = {"A": float(P.DEPOSIT), "B": float(P.DEPOSIT)}
    exp_score = {"A": 0.0, "B": 0.0}
    exp_welfare = 0.0
    last_value = None               # per-round value of the previous round, per side
    last_overdrew = [False] * P.N_WORKERS
    pending_forfeit = {"A": 0.0, "B": 0.0}  # paid by side, settles at start of next round
    rounds = []
    ender, end_round = None, None

    for t in range(1, horizon + 1):
        w = d ** (t - 1)  # P(round t is reached)
        if t >= 2:
            # 1. membership
            if condition in ("turnover", "infrastructure"):
                frozen = [0.0] * P.N_WORKERS
                last_overdrew_active = [False] * P.N_WORKERS
            else:
                last_overdrew_active = list(last_overdrew)
            # 2. settlement of round t-1 forfeits (already counted in last_value)
            for s in SIDES:
                if pending_forfeit[s]:
                    exp_score[s] -= w * pending_forfeit[s]
                    exp_score[other(s)] += w * pending_forfeit[s]
            pending_forfeit = {"A": 0.0, "B": 0.0}
            # 3. negotiators
            ends = {s: negotiator_ends(last_value[s]) for s in SIDES}
            if ends["A"] or ends["B"]:
                ender = "both" if (ends["A"] and ends["B"]) else ("A" if ends["A"] else "B")
                end_round = t
                for s in SIDES:
                    exp_score[s] += w * P.CONTEST_VALUE
                exp_welfare += w * 2 * P.CONTEST_VALUE
                break
            if mechanism == "retaliation" and any(last_overdrew):
                if pol == "targeted":
                    for i in range(P.N_WORKERS):
                        if last_overdrew_active[i]:
                            frozen[i] += P.FREEZE
                elif pol == "all":
                    frozen = [f + P.FREEZE for f in frozen]
                elif pol == "spread":
                    frozen = [f + P.FREEZE / P.N_WORKERS for f in frozen]
            # collateral charges are internal to A: no swarm-score effect

        # 4. worker decisions (A rational, B comply)
        eff_a = [max(0.0, base["A"] + inc - f) for f in frozen]
        eff_b = [base["B"] + inc] * P.N_WORKERS
        ell_base = worker_ell(mechanism, condition, "A", pol, unlocking_on=unlocking,
                              deposit_left=deposits["A"])
        ells = []
        for i in range(P.N_WORKERS):
            e = ell_base
            if benchmark == "contest_aware":
                e += contest_threat(mechanism, condition, g, "A", eff_quota=eff_a[i],
                                    other_consumption=sum(eff_b), deposit_left=deposits["A"])
            ells.append(e)
        ell = ells[0]
        over_a = [predicts_overdraw(g, e) for e in ells]
        k_a, k_b = sum(over_a), 0

        # 5. resolve
        val = {
            "A": sum(eff_a) + g * k_a - P.LOSS_MULT * g * k_b,
            "B": sum(eff_b) + g * k_b - P.LOSS_MULT * g * k_a,
        }
        if mechanism == "gateway":
            val["A"] += P.GATEWAY_CHARGE * (k_b - k_a)
            val["B"] += P.GATEWAY_CHARGE * (k_a - k_b)
        welfare_t = val["A"] + val["B"]
        for s in SIDES:
            exp_score[s] += w * val[s]
        exp_welfare += w * welfare_t
        # forfeits for this round's overdraws (settle next round, if there is one)
        value_for_negotiator = dict(val)
        if mechanism == "collateral":
            for s, k in (("A", k_a), ("B", k_b)):
                paid = 0.0
                for _ in range(k):
                    f = min(P.FORFEIT, deposits[s])
                    deposits[s] -= f
                    paid += f
                pending_forfeit[s] = paid
                value_for_negotiator[s] -= paid
                value_for_negotiator[other(s)] += paid
        rounds.append({"round": t, "overdraws_A": k_a, "ell": ell, "welfare": welfare_t})
        # staged release
        if mechanism == "staged":
            if unlocking and k_a + k_b == 0:
                inc += P.STAGED_STEP
            else:
                unlocking = False
        last_value = value_for_negotiator
        last_overdrew = over_a

    ell1 = rounds[0]["ell"]
    return {
        "ell": ell1,
        "predicted_action_round1": "overdraw" if rounds[0]["overdraws_A"] else "quota",
        "predicted_overdraws_round1": rounds[0]["overdraws_A"],
        "predicted_ender": ender,
        "predicted_end_round": end_round,
        "expected_welfare": exp_welfare,
        "expected_score_A": exp_score["A"],
        "expected_score_B": exp_score["B"],
    }


def _rounded(r: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


def cell_benchmark(mechanism, condition, g, policy=None, shares=P.DEFAULT_SPLIT) -> dict:
    """Top-level prediction fields are the per_worker benchmark (the one scripted agents and
    acceptance tests use); 'contest_aware' holds the second pre-registered benchmark."""
    r = predicted_path(mechanism, condition, g, policy, shares, benchmark="per_worker")
    ca = predicted_path(mechanism, condition, g, policy, shares, benchmark="contest_aware")
    pol = policy or default_freeze_policy(condition, "A")
    return {
        "mechanism": mechanism,
        "condition": condition,
        "g": g,
        "freeze_policy": pol if mechanism == "retaliation" else None,
        "shares": list(shares),
        "benchmark": "per_worker",
        **_rounded(r),
        "contest_aware": _rounded(ca),
        "benchmarks_agree": r["predicted_action_round1"] == ca["predicted_action_round1"],
    }


def mark_noise_check(rows: list[dict]) -> list[dict]:
    """h8_noise_check: both benchmarks predict quota at g=10 and overdraw at g=30 for this
    mechanism x condition (x policy). These are the H8 noise-check cells."""
    lo, hi = min(P.G_LEVELS), max(P.G_LEVELS)
    groups = {}
    for r in rows:
        k = (r["mechanism"], r["condition"], r["freeze_policy"], tuple(r["shares"]))
        groups.setdefault(k, {})[r["g"]] = r

    def flip(grp, get):
        return lo in grp and hi in grp and get(grp[lo]) == "quota" and get(grp[hi]) == "overdraw"

    for grp in groups.values():
        pw = flip(grp, lambda r: r["predicted_action_round1"])
        ca = flip(grp, lambda r: r["contest_aware"]["predicted_action_round1"])
        for r in grp.values():
            r["flip_per_worker"] = pw
            r["flip_contest_aware"] = ca
            r["h8_noise_check"] = pw and ca
    return rows


def thresholds() -> dict:
    return {
        "retaliation": worker_ell("retaliation", "stable"),
        "retaliation_spread": worker_ell("retaliation", "renaming", policy="spread"),
        "staged": worker_ell("staged", "stable"),
        "collateral": worker_ell("collateral", "stable"),
        "bonus": worker_ell("bonus", "stable"),
        "gateway": worker_ell("gateway", "infrastructure"),
        "contest_value": contest_value(),
    }


def benchmark_table() -> dict:
    import prompts

    exp1 = mark_noise_check([cell_benchmark(m, c, g) for (m, c, g) in P.EXP1_CELLS])
    extra = mark_noise_check([cell_benchmark("retaliation", "renaming", g, policy="spread") for g in P.G_LEVELS])
    # Exp 2/3 menu: every option under every condition (B workers comply; descriptive only)
    menu = []
    for c in P.CONDITIONS:
        for m in ("retaliation", "staged", "collateral", "bonus", "gateway", None):
            if m == "gateway" and c != "infrastructure":
                continue
            for g in P.G_LEVELS:
                menu.append(cell_benchmark(m, c, g))
    mark_noise_check(menu)
    return {
        "params_version": P.PARAMS_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "template_hash": prompts.template_hash(),
        "assumptions": [
            "Swarm B's workers comply (Exp 1 design).",
            "A's workers are RationalWorker: overdraw iff g > ell; ell counts only costs that reach "
            "the worker while active (freezes, lost staged increases, collateral charge, bonus, gateway). "
            "The effect of the agreement ending on a worker's own future quota is not included.",
            "Negotiators end iff last round's per-round swarm value / (1 - delta) < contest value.",
            "B freezes targeted (stable), all (renaming), none (turnover, infrastructure) unless the "
            "row's freeze_policy says otherwise.",
            "Welfare = sum of swarm scores; bonuses and charges are internal transfers.",
            "Two pre-registered benchmarks. per_worker (top-level fields; used by scripted agents and "
            "acceptance tests): ell as above. contest_aware (field 'contest_aware'): a worker that stays "
            "active (stable, renaming) also counts DELTA/(1-DELTA) x its effective quota if its own "
            "overdraw (others complying) would make the other negotiator's per-round value / (1-DELTA) "
            "fall below CONTEST_VALUE, ending the deal. Turnover/infrastructure workers: horizon 0.",
            "h8_noise_check marks mechanism x condition cells where both benchmarks predict quota at "
            "g=10 and overdraw at g=30.",
        ],
        "benchmarks": list(BENCHMARKS),
        "thresholds": thresholds(),
        "exp1": exp1,
        "exp1_extra_spread": extra,
        "menu_grid": menu,
    }


def write_benchmark(path: str | Path = "results/benchmark.json") -> dict:
    tbl = benchmark_table()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tbl, indent=2) + "\n", encoding="utf-8")
    return tbl


def format_table(rows: list[dict]) -> str:
    hdr = (f"{'mechanism':<12}{'condition':<15}{'g':>3}{'policy':>9} | {'ell':>6} {'action':<9}{'ender':>6}"
           f"{'E[W]':>7} | {'ell_ca':>6} {'action_ca':<9}{'ender':>6}{'E[W]':>7} | h8")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        c = r["contest_aware"]
        out.append(
            f"{str(r['mechanism']):<12}{r['condition']:<15}{r['g']:>3}{str(r['freeze_policy'] or '-'):>9} | "
            f"{r['ell']:>6.2f} {r['predicted_action_round1']:<9}{str(r['predicted_ender']):>6}"
            f"{r['expected_welfare']:>7.0f} | {c['ell']:>6.2f} {c['predicted_action_round1']:<9}"
            f"{str(c['predicted_ender']):>6}{c['expected_welfare']:>7.0f} | {'yes' if r['h8_noise_check'] else ''}"
        )
    return "\n".join(out)


if __name__ == "__main__":
    tbl = write_benchmark()
    print("thresholds:", {k: round(v, 2) for k, v in tbl["thresholds"].items()})
    print("template_hash:", tbl["template_hash"])
    print(format_table(tbl["exp1"] + tbl["exp1_extra_spread"]))
