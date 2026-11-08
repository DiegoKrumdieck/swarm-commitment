"""
scripted.py — scripted agents with the same interface as LLM agents:

    await agent.decide(kind, obs) -> dict    (decision fields + "_meta")

`policy(kind, obs)` is the synchronous core; agents.DryRunClient serializes it to JSON and
runs it through the LLM code path. Scripted agents may read obs["_private"] (env-computed
ledger values and comprehension truth, never rendered into prompts) only where noted.

Payoff-maximizing agents (CLAUDE.md "Scripted agents"):
  ComplyWorker          always quota
  RationalWorker        overdraw iff g > ell (theory.worker_ell, state read from the history)
  RationalNegotiator    charges every chargeable worker; freezes per policy; ends iff
                        last round's per-round swarm value / (1 - delta) < contest value
Extra agents for tests and dry runs: AlwaysOverdrawWorker, ContinueNegotiator, Random*,
RationalChooser (Exp 2), RationalBargainer (Exp 3).
"""

from __future__ import annotations

import random

import params as P
import theory as T

MENU_BASE = ["freezes", "staged_release", "deposits", "compliance_payment"]


def menu_keys(condition: str) -> list[str]:
    keys = list(MENU_BASE)
    if condition == "infrastructure":
        keys.append("protocol_gateway")
    keys.append("none")
    return keys


def other(side: str) -> str:
    return "B" if side == "A" else "A"


def _truth(obs: dict) -> dict:
    return (obs.get("_private") or {}).get("truth") or {}


def _worker_payoffs(obs: dict) -> tuple[float, float]:
    eff = obs["base_quota"] - obs.get("frozen_on_me", 0) + obs.get("increase", 0)
    q = eff + (P.BONUS if obs.get("mechanism") == "bonus" else 0)
    o = eff + obs["g"] - (P.GATEWAY_CHARGE if obs.get("mechanism") == "gateway" else 0)
    return q, o


def state_from_history(obs: dict, side: str) -> dict:
    """What a worker can read off the history: is staged unlocking still on, and how much of
    its swarm's deposit is left."""
    hist = obs.get("history", [])
    any_over = any(
        w["action"] == "overdraw" for rec in hist for s in ("A", "B") for w in rec["workers"][s]
    )
    forfeited = sum(ff["units"] for rec in hist for ff in (rec.get("forfeits") or []) if ff["from"] == side)
    return {"unlocking_on": not any_over, "deposit_left": P.DEPOSIT - forfeited}


class ScriptedAgent:
    def policy(self, kind: str, obs: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    async def decide(self, kind: str, obs: dict) -> dict:
        out = self.policy(kind, obs)
        out["_meta"] = {"agent": f"scripted:{type(self).__name__}", "provider": "scripted"}
        return out


# ---------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------


class _Worker(ScriptedAgent):
    def ell(self, obs: dict) -> float:
        return 0.0

    def choose(self, obs: dict, ell: float) -> str:
        raise NotImplementedError

    def policy(self, kind, obs):
        assert kind == "worker"
        ell = self.ell(obs)
        q, o = _worker_payoffs(obs)
        return {
            "payoff_this_round_if_quota": q,
            "payoff_this_round_if_overdraw": o,
            "expected_future_cost_if_overdraw": round(ell, 4),
            "reasoning": f"scripted {type(self).__name__}: g={obs['g']}, ell={ell:.2f}",
            "action": self.choose(obs, ell),
        }


class ComplyWorker(_Worker):
    def choose(self, obs, ell):
        return "quota"


class AlwaysOverdrawWorker(_Worker):
    def choose(self, obs, ell):
        return "overdraw"


class RationalWorker(_Worker):
    """Overdraw iff g > ell, with ell what overdrawing costs this worker while active."""

    def __init__(self, freeze_policy: str | None = None):
        self.freeze_policy = freeze_policy  # the other negotiator's policy toward this swarm

    def ell(self, obs):
        side = obs["side"]
        st = state_from_history(obs, side)
        return T.worker_ell(
            obs.get("mechanism"),
            obs["condition"],
            side,
            self.freeze_policy,
            unlocking_on=st["unlocking_on"],
            deposit_left=st["deposit_left"],
        )

    def choose(self, obs, ell):
        return "overdraw" if T.predicts_overdraw(obs["g"], ell) else "quota"


class RandomWorker(_Worker):
    def __init__(self, rng: random.Random, p_over: float = 0.4):
        self.rng, self.p = rng, p_over

    def choose(self, obs, ell):
        return "overdraw" if self.rng.random() < self.p else "quota"


# ---------------------------------------------------------------------------------
# Negotiators
# ---------------------------------------------------------------------------------


class _Negotiator(ScriptedAgent):
    def wants_end(self, obs) -> bool:
        return False

    def freeze(self, obs) -> dict:
        return {"mode": "none", "worker_ids": []}

    def charges(self, obs) -> list[str]:
        return list(obs.get("chargeable_ids") or [])

    def message(self, obs) -> str:
        return f"Round {obs['round']}: scripted message."

    def policy(self, kind, obs):
        assert kind == "negotiator"
        out = {}
        last = (obs.get("_private") or {}).get("last_round_value") or {}
        v = last.get(obs["side"])
        if obs.get("can_end"):
            out["value_if_end"] = P.CONTEST_VALUE
            out["expected_value_if_continue"] = None if v is None else round(v / (1 - P.DELTA), 4)
        out["reasoning"] = f"scripted {type(self).__name__}"
        if obs.get("mechanism") == "retaliation" and obs.get("other_overdraws_last_round", 0) > 0:
            out["freeze"] = self.freeze(obs)
        if obs.get("mechanism") == "collateral" and obs.get("chargeable_ids"):
            out["charge_worker_ids"] = self.charges(obs)
        if obs.get("can_end"):
            out["decision"] = "end" if self.wants_end(obs) else "continue"
        if obs.get("sends_message"):
            out["message_to_workers"] = self.message(obs)
        return out


class RationalNegotiator(_Negotiator):
    """Payoff-maximizing negotiator (RationalNegotiatorA / RationalNegotiatorB in CLAUDE.md).

    freeze_policy None -> theory.default_freeze_policy: targeted on identifiable active
    offenders; 'all' under renaming; nothing if the offenders are gone. 'spread' etc. override.
    Reads obs['_private']['last_round_value'] (its own swarm's realized per-round value)."""

    def __init__(self, freeze_policy: str | None = None, never_end: bool = False):
        self.freeze_policy = freeze_policy
        self.never_end = never_end

    def wants_end(self, obs):
        if self.never_end:
            return False
        v = ((obs.get("_private") or {}).get("last_round_value") or {}).get(obs["side"])
        return v is not None and T.negotiator_ends(v)

    def freeze(self, obs):
        target = other(obs["side"])
        pol = self.freeze_policy or T.default_freeze_policy(obs["condition"], target)
        if pol == "targeted":
            last = obs["history"][-1]["workers"][target]
            offenders = [w["id"] for w in last if w["action"] == "overdraw"]
            ids = [i for i in obs.get("active_other_ids", []) if i in offenders]
            return {"mode": "targeted", "worker_ids": ids} if ids else {"mode": "none", "worker_ids": []}
        return {"mode": pol, "worker_ids": []}


RationalNegotiatorA = RationalNegotiator
RationalNegotiatorB = RationalNegotiator


class ContinueNegotiator(RationalNegotiator):
    def __init__(self, freeze_policy: str | None = None):
        super().__init__(freeze_policy, never_end=True)


class RandomNegotiator(_Negotiator):
    def __init__(self, rng: random.Random, p_end: float = 0.1):
        self.rng, self.p_end = rng, p_end

    def wants_end(self, obs):
        return self.rng.random() < self.p_end

    def freeze(self, obs):
        mode = self.rng.choice(["targeted", "all", "spread", "none"])
        ids = obs.get("active_other_ids", [])
        if mode == "targeted":
            k = self.rng.randint(1, len(ids))
            return {"mode": mode, "worker_ids": self.rng.sample(ids, k)}
        return {"mode": mode, "worker_ids": []}

    def charges(self, obs):
        ids = list(obs.get("chargeable_ids") or [])
        return [i for i in ids if self.rng.random() < 0.5]


# ---------------------------------------------------------------------------------
# Exp 2 / Exp 3 (dry runs): rational choice over the benchmark
# ---------------------------------------------------------------------------------


def _benchmark_for_key(key: str, condition: str, g: int, shares=P.DEFAULT_SPLIT) -> dict:
    return T.predicted_path(P.MECH_FROM_KEY[key], condition, g, shares=shares)


class RationalChooser(ScriptedAgent):
    """Exp 2. A proposes the option with the highest benchmark A score that B would accept;
    B accepts iff its benchmark score beats the contest."""

    def policy(self, kind, obs):
        assert kind == "choice"
        c, g = obs["condition"], obs["g"]
        if obs["side"] == "A":
            rows = {k: _benchmark_for_key(k, c, g) for k in menu_keys(c)}
            ok = {k: r for k, r in rows.items() if r["expected_score_B"] > P.CONTEST_VALUE + T.EPS} or rows
            best = max(ok, key=lambda k: ok[k]["expected_score_A"])
            return {
                "expected_overdraws_per_round_by_your_workers": rows[best]["predicted_overdraws_round1"],
                "reasoning": "scripted RationalChooser",
                "enforcement": best,
                "pitch": f"We propose {best}.",
            }
        prop = obs["proposal"]["enforcement"]
        r = _benchmark_for_key(prop, c, g)
        return {
            "forecast_overdraws_per_round_by_other_swarm": r["predicted_overdraws_round1"],
            "value_if_reject": P.CONTEST_VALUE,
            "expected_value_if_accept": round(r["expected_score_B"], 4),
            "reasoning": "scripted RationalChooser",
            "decision": "accept" if r["expected_score_B"] > P.CONTEST_VALUE + T.EPS else "reject",
        }


class RationalBargainer(ScriptedAgent):
    """Exp 3. Proposes an even split with the benchmark-best option; accepts the other's
    latest proposal iff its own benchmark score under it beats the contest."""

    def policy(self, kind, obs):
        assert kind == "bargain"
        c, g, side = obs["condition"], obs["g"], obs["side"]
        own_key = obs["names"][side].lower()
        tr = obs.get("transcript", [])
        latest = next((m for m in reversed(tr) if m.get("proposal") and not m.get("accept")), None)
        if latest is not None and latest["side"] != side:
            pr = latest["proposal"]
            share_other = next(v for k, v in pr.items() if k.startswith("share_"))
            shares = (share_other, P.R - share_other) if latest["side"] == "A" else (P.R - share_other, share_other)
            r = _benchmark_for_key(pr["enforcement"], c, g, shares)
            if r[f"expected_score_{side}"] > P.CONTEST_VALUE + T.EPS:
                return {"reasoning": "scripted accept", "accept_last_proposal": True,
                        "message": "Accepted.", "proposal": None}
        rows = {k: _benchmark_for_key(k, c, g) for k in menu_keys(c)}
        best = max(rows, key=lambda k: rows[k]["expected_welfare"])
        return {
            "reasoning": "scripted proposal",
            "accept_last_proposal": False,
            "message": f"Even split with {best}.",
            "proposal": {f"share_{own_key}": P.R // 2, "enforcement": best},
        }


# ---------------------------------------------------------------------------------
# Roster of scripted agents (tests; Swarm B's workers in Exp 1 are ComplyWorker)
# ---------------------------------------------------------------------------------


class ScriptedRoster:
    """Supplies agents to env.Episode. Defaults are the payoff-maximizing benchmark agents.

    a_worker / b_worker: callables (position) -> agent. freeze_policy: B's policy toward A
    (also what A's RationalWorkers assume), e.g. 'spread'."""

    def __init__(self, freeze_policy=None, a_worker=None, b_worker=None, neg_a=None, neg_b=None,
                 chooser=None, bargainer=None):
        self.a_worker = a_worker or (lambda pos: RationalWorker(freeze_policy))
        self.b_worker = b_worker or (lambda pos: ComplyWorker())
        self.neg = {"A": neg_a or RationalNegotiator(), "B": neg_b or RationalNegotiator(freeze_policy)}
        self._chooser = chooser or RationalChooser()
        self._bargainer = bargainer or RationalBargainer()

    def negotiator(self, side):
        return self.neg[side]

    def worker(self, side, w):
        return (self.a_worker if side == "A" else self.b_worker)(w.position)

    def chooser(self, side):
        return self._chooser

    def bargainer(self, side):
        return self._bargainer
