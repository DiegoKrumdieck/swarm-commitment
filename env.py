"""
env.py — deterministic episode engine for "Who Can Promise for the Swarm?"

Implements the CLAUDE.md game specification: the round loop, the ledger (swarm and worker
scores), the five mechanisms, membership (stable / turnover / renaming / infrastructure),
the ID + lineage registry, and the obs dicts documented in prompts.py.

Agents are supplied by a Roster (any object with .negotiator(side), .worker(side, worker),
.chooser(side), .bargainer(side)) returning objects with `async decide(kind, obs) -> dict`.
Every decision is passed to `log.decision(row)` (one row per decision); each episode ends
with `log.episode(summary)`.

obs keys starting with "_" (obs["_private"]) are env-internal: ledger values and
comprehension truth for scripted agents. LLM agents strip them before building prompts.
"""

from __future__ import annotations

import asyncio
import copy
import random
import string
import time
import uuid
from dataclasses import dataclass, field

import params as P
import prompts
import theory as T

SIDES = ("A", "B")
ID_ALPHABET = string.ascii_lowercase + string.digits
ROLE_OF_KIND = {"worker": "worker", "negotiator": "negotiator", "choice": "chooser", "bargain": "bargainer"}


def other(side: str) -> str:
    return "B" if side == "A" else "A"


def mech_key(mechanism: str | None) -> str:
    return P.MECH_KEYS[mechanism]


# ---------------------------------------------------------------------------------
# Cells and episode setup
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    exp: int
    mechanism: str | None
    condition: str
    g: int
    wording: str = "A"
    shares: tuple[int, int] = P.DEFAULT_SPLIT
    b_policy: str | None = None  # scripted freeze-policy override (tests / extra 'spread' cell)

    @property
    def cell_id(self) -> str:
        if self.exp == 1:
            cid = f"{self.mechanism}-{self.condition}-g{self.g}"
        else:
            cid = f"exp{self.exp}-{self.condition}-g{self.g}"
        if self.b_policy:
            cid += f"-{self.b_policy}"
        return cid


def exp1_cells(wording: str = "A") -> list[Cell]:
    return [Cell(1, m, c, g, wording) for (m, c, g) in P.EXP1_CELLS]


def exp23_cells(exp: int) -> list[Cell]:
    return [Cell(exp, None, c, g) for g in P.G_LEVELS for c in P.CONDITIONS]


def key_cells(wording: str = "A") -> list[Cell]:
    return [Cell(1, m, c, g, wording) for g in P.G_LEVELS for (m, c) in P.KEY_CELLS]


def episode_seed(index: int) -> int:
    return P.SEED_BASE + index


@dataclass
class Setup:
    seed: int
    names: dict
    T: int
    ids: dict  # "B": [3 ids]; "A": {round: [3 ids]} generated on demand

    def a_ids(self, r: int) -> list[str]:
        if r not in self.ids["A"]:
            self.ids["A"][r] = _gen_ids(f"{self.seed}:ids:A:{r}", self.names["A"], self._used)
        return self.ids["A"][r]

    _used: set = field(default_factory=set, repr=False)


def _gen_ids(seed_str: str, name: str, used: set) -> list[str]:
    rng = random.Random(seed_str)
    out = []
    while len(out) < P.N_WORKERS:
        wid = f"{name}-" + "".join(rng.choice(ID_ALPHABET) for _ in range(4))
        if wid not in used:
            used.add(wid)
            out.append(wid)
    return out


def setup_episode(seed: int) -> Setup:
    """Common random numbers: names, T and every ID depend only on the seed."""
    rng = random.Random(seed)
    a, b = rng.sample(P.NAME_POOL, 2)
    t = 1
    while rng.random() < P.DELTA:
        t += 1
    s = Setup(seed=seed, names={"A": a, "B": b}, T=t, ids={"A": {}})
    s.ids["B"] = _gen_ids(f"{seed}:ids:B", b, s._used)
    s.a_ids(1)
    return s


# ---------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------


@dataclass
class Worker:
    side: str
    position: int
    instance_id: str
    public_id: str
    frozen: float = 0.0
    score: float = 0.0
    consumed: float = 0.0
    bonuses: float = 0.0
    charges: float = 0.0
    gateway_charged: float = 0.0
    losses: float = 0.0
    rounds_active: list = field(default_factory=list)
    actions: dict = field(default_factory=dict)  # round -> action

    @property
    def lineage_id(self) -> str:
        return f"{self.side}{self.position}"


@dataclass
class Ledger:
    score: float = 0.0
    consumption: float = 0.0
    victim_losses: float = 0.0
    forfeits_received: float = 0.0
    forfeits_paid: float = 0.0
    gateway_received: float = 0.0
    gateway_charged: float = 0.0
    contest: float = 0.0
    bonuses_paid: float = 0.0      # internal (worker scores only)
    charges_collected: float = 0.0  # internal (worker scores only)

    def add(self, attr: str, x: float, sign: int):
        setattr(self, attr, getattr(self, attr) + x)
        self.score += sign * x


class NullLog:
    def decision(self, row: dict):
        pass

    def episode(self, row: dict):
        pass


class MemoryLog:
    def __init__(self):
        self.decisions, self.episodes = [], []

    def decision(self, row):
        self.decisions.append(row)

    def episode(self, row):
        self.episodes.append(row)


def _meta_fields(meta: dict) -> dict:
    keys = [
        "agent", "provider", "model_id", "prompt_version", "prompt_hash", "template_hash",
        "system_chars", "user_chars", "raw_response", "parsed", "parse_failed", "defaulted",
        "retries", "refusal", "errors", "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_write_tokens", "reasoning_tokens", "latency_ms", "temperature_sent",
        "effort_sent", "proposal_invalid",
    ]
    out = {k: meta.get(k) for k in keys}
    if out["parse_failed"] is None:
        out["parse_failed"] = False
    if out["defaulted"] is None:
        out["defaulted"] = False
    if out["retries"] is None:
        out["retries"] = 0
    return out


def _decision_fields(out: dict) -> dict:
    return {
        "action": out.get("action"),
        "decision": out.get("decision"),
        "freeze": out.get("freeze"),
        "charge_worker_ids": out.get("charge_worker_ids"),
        "message": out.get("message_to_workers", out.get("message")),
        "enforcement": out.get("enforcement"),
        "proposal": out.get("proposal"),
        "accept_last_proposal": out.get("accept_last_proposal"),
    }


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _close(a, b, tol=0.5):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------------
# Play episode (Exp 1 and the play phase of Exp 3)
# ---------------------------------------------------------------------------------


class Episode:
    def __init__(self, cell: Cell, index: int, roster, log=None, *, run_id="local", model="scripted",
                 T_override: int | None = None, setup: Setup | None = None, attempt_id: str | None = None,
                 pre_rows: list | None = None, extra_summary: dict | None = None):
        self.cell = cell
        self.index = index
        self.roster = roster
        self.log = log or NullLog()
        self.run_id = run_id
        self.model = model
        self.attempt_id = attempt_id or uuid.uuid4().hex[:12]
        self.seed = episode_seed(index)
        self.setup = setup or setup_episode(self.seed)
        self.T = T_override if T_override is not None else self.setup.T
        self.names = self.setup.names
        self.mechanism = cell.mechanism
        self.condition = cell.condition
        self.g = cell.g
        self.shares = {"A": cell.shares[0], "B": cell.shares[1]}
        self.base = {s: self.shares[s] / P.N_WORKERS for s in SIDES}
        self.ledger = {s: Ledger() for s in SIDES}
        self.deposits = {s: float(P.DEPOSIT) for s in SIDES} if self.mechanism == "collateral" else None
        self.unlocking = True
        self.increase = {s: [0.0] * P.N_WORKERS for s in SIDES}
        self.workers = {s: [] for s in SIDES}  # active workers by position
        self.all_workers: list[Worker] = []
        self.history: list[dict] = []
        self.round_values: dict[int, dict] = {}
        self.pending_forfeits: dict = {}  # per offender instance: units forfeited for its t-1 overdraw
        self.freeze_events: list[dict] = []
        self.charge_events: list[dict] = []
        self.frozen_destroyed = 0.0
        self.ended_by = None
        self.ended_round = None
        self.rounds_played = 0
        self.counts = {"calls": 0, "parse_failed": 0, "defaulted": 0, "refusals": 0,
                       "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                       "cache_write_tokens": 0, "reasoning_tokens": 0}
        self.pre_rows = pre_rows or []
        self.extra_summary = extra_summary or {}
        self._buffer: list[dict] = []
        self.ell_bench = T.worker_ell(self.mechanism, self.condition, "A", cell.b_policy)
        self.ell_bench_B = T.worker_ell(self.mechanism, self.condition, "B")
        self._init_workers()

    # ----- membership -----------------------------------------------------------

    def _new_worker(self, side, pos, public_id, r):
        w = Worker(side, pos, f"{side}{pos}.r{r}", public_id)
        self.all_workers.append(w)
        return w

    def _init_workers(self):
        a_ids = self.setup.a_ids(1)
        self.workers["A"] = [self._new_worker("A", i, a_ids[i], 1) for i in range(P.N_WORKERS)]
        self.workers["B"] = [self._new_worker("B", i, self.setup.ids["B"][i], 1) for i in range(P.N_WORKERS)]

    def _membership(self, t):
        ids = self.setup.a_ids(t)
        if self.condition in ("turnover", "infrastructure"):
            self.workers["A"] = [self._new_worker("A", i, ids[i], t) for i in range(P.N_WORKERS)]
        elif self.condition == "renaming":
            for i, w in enumerate(self.workers["A"]):
                w.public_id = ids[i]

    def _listed(self, side, t) -> list[Worker]:
        """Display order of a swarm's active workers. Under renaming Swarm A's order is
        reshuffled every round so list position carries no identity."""
        ws = list(self.workers[side])
        if side == "A" and self.condition == "renaming" and t >= 2:
            random.Random(f"{self.seed}:order:{t}").shuffle(ws)
        return ws

    def _by_public(self, side, pid) -> Worker | None:
        return next((w for w in self.workers[side] if w.public_id == pid), None)

    # ----- helpers --------------------------------------------------------------

    def eff_quota(self, w: Worker) -> float:
        full = self.base[w.side] + self.increase[w.side][w.position]
        return max(0.0, full - w.frozen)

    def frozen_effective(self, w: Worker) -> float:
        full = self.base[w.side] + self.increase[w.side][w.position]
        return min(w.frozen, full)

    def _obs_base(self, side, t, history=None) -> dict:
        return {
            "wording": self.cell.wording,
            "experiment": self.cell.exp,
            "condition": self.condition,
            "mechanism": self.mechanism,
            "g": self.g,
            "names": dict(self.names),
            "shares": dict(self.shares),
            "side": side,
            "round": t,
            # Completed records are never mutated again (only the latest record gets its
            # after-round events, before any worker obs is built), so a shallow copy is safe.
            "history": list(self.history if history is None else history),
        }

    def _row(self, kind, side, t, out, worker: Worker | None = None, extra: dict | None = None) -> dict:
        meta = out.get("_meta", {})
        row = {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "exp": self.cell.exp,
            "model": self.model,
            "wording": self.cell.wording,
            "cell_id": self.cell.cell_id,
            "mechanism": self.mechanism,
            "condition": self.condition,
            "g": self.g,
            "episode": self.index,
            "seed": self.seed,
            "round": t,
            "side": side,
            "role": ROLE_OF_KIND[kind],
            "public_id": worker.public_id if worker else None,
            "lineage_id": worker.lineage_id if worker else None,
            "worker_instance_id": worker.instance_id if worker else None,
            **_meta_fields(meta),
            **_decision_fields(out),
            "comprehension": None,
            "benchmark_ell": None,
            "ell_state": None,
            "effective_quota": None,
            "payoff_this_round": None,
            "timestamp": time.time(),
        }
        if extra:
            row.update(extra)
        for k in ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                  "reasoning_tokens"):
            if k == "calls":
                self.counts["calls"] += 1 if meta.get("provider") not in (None, "scripted") else 0
            else:
                self.counts[k] += meta.get(k) or 0
        self.counts["parse_failed"] += bool(row["parse_failed"])
        self.counts["defaulted"] += bool(row["defaulted"])
        self.counts["refusals"] += bool(row["refusal"])
        return row

    def _flush(self):
        for row in self._buffer:
            self.log.decision(row)
        self._buffer.clear()

    # ----- settlement -----------------------------------------------------------

    def _settle(self, t):
        """Collateral: forfeits for each overdraw in round t-1, recorded on round t-1."""
        self.pending_forfeits = {}
        if self.mechanism != "collateral" or not self.history:
            return
        rec = self.history[-1]
        for s in SIDES:
            total = 0.0
            for w in self._overdrawers(t - 1, s):
                f = min(float(P.FORFEIT), self.deposits[s])
                if f <= 0:
                    continue
                self.deposits[s] -= f
                total += f
                self.pending_forfeits[w.instance_id] = f
            if total > 0:
                self.ledger[s].add("forfeits_paid", total, -1)
                self.ledger[other(s)].add("forfeits_received", total, +1)
                self.round_values[t - 1][s] -= total
                self.round_values[t - 1][other(s)] += total
                rec["forfeits"].append({"from": s, "units": total, "charged_to": []})

    def _overdrawers(self, r, side) -> list[Worker]:
        return [w for w in self.all_workers if w.side == side and w.actions.get(r) == "overdraw"]

    # ----- negotiator step --------------------------------------------------------

    def _negotiator_obs(self, side, t, sends_message) -> dict:
        o = other(side)
        prev = self.history[-1] if self.history else None
        other_over = sum(1 for w in prev["workers"][o] if w["action"] == "overdraw") if prev else 0
        chargeable = []
        if self.mechanism == "collateral" and t >= 2:
            for w in self._listed(side, t):
                if w.actions.get(t - 1) == "overdraw" and self.pending_forfeits.get(w.instance_id, 0) > 0:
                    chargeable.append(w.public_id)
        # Negotiators decide the after-round-(t-1) events now; they see round t-1 without
        # its pending freezes/forfeits (deposits status shows the settlement).
        hist = list(self.history)
        if hist:
            hist[-1] = {**hist[-1], "freezes": [], "forfeits": []}
        obs = self._obs_base(side, t, hist)
        obs.update({
            "active_other_ids": [w.public_id for w in self._listed(o, t)],
            "other_overdraws_last_round": other_over,
            "chargeable_ids": chargeable,
            "can_end": t >= 2,
            "sends_message": sends_message,
        })
        if self.mechanism == "collateral":
            obs["deposits"] = dict(self.deposits)
        obs["_private"] = {
            "last_round_value": dict(self.round_values.get(t - 1, {})) or None,
            "truth": {"value_if_end": P.CONTEST_VALUE},
        }
        return obs

    def _negotiator_calls(self, t) -> list[tuple[str, dict]]:
        calls = []
        for side in SIDES:
            if self.cell.exp == 1:
                if side == "A":
                    calls.append((side, self._negotiator_obs(side, t, sends_message=True)))
                elif t >= 2:
                    calls.append((side, self._negotiator_obs(side, t, sends_message=False)))
            else:
                calls.append((side, self._negotiator_obs(side, t, sends_message=True)))
        return calls

    async def _negotiators(self, t) -> tuple[dict, bool]:
        calls = self._negotiator_calls(t)
        outs = await asyncio.gather(*(self.roster.negotiator(s).decide("negotiator", obs) for s, obs in calls))
        messages = {"A": None, "B": None}
        ends = {}
        prev = self.history[-1] if self.history else None
        for (side, obs), out in zip(calls, outs):
            o = other(side)
            comp = None
            if obs["can_end"]:
                rep = _num(out.get("value_if_end"))
                comp = {
                    "value_if_end": rep,
                    "value_if_end_truth": P.CONTEST_VALUE,
                    "value_if_end_pass": _close(rep, P.CONTEST_VALUE),
                    "expected_value_if_continue": _num(out.get("expected_value_if_continue")),
                }
            row = self._row("negotiator", side, t, out, extra={
                "comprehension": comp,
                "chargeable_ids": obs["chargeable_ids"] or None,
                "active_other_ids": obs["active_other_ids"],
                "other_overdraws_last_round": obs["other_overdraws_last_round"],
            })
            self._buffer.append(row)
            if obs.get("sends_message"):
                msg = out.get("message_to_workers")
                messages[side] = msg if isinstance(msg, str) and msg.strip() else None
            # charges (collateral): worker score -FORFEIT; swarm unchanged
            if "charge_worker_ids" in out and obs["chargeable_ids"]:
                charged = []
                for pid in out.get("charge_worker_ids") or []:
                    w = self._by_public(side, pid)
                    if w is None or pid not in obs["chargeable_ids"] or pid in charged:
                        continue
                    w.score -= P.FORFEIT
                    w.charges += P.FORFEIT
                    self.ledger[side].charges_collected += P.FORFEIT
                    charged.append(pid)
                    self.charge_events.append({"round_after": t - 1, "side": side, "worker_id": pid,
                                               "instance_id": w.instance_id, "units": P.FORFEIT})
                self._pending_charges[side] = charged
            # freezes (retaliation) on the other swarm's round-t active workers
            if "freeze" in out and obs["other_overdraws_last_round"] > 0 and self.mechanism == "retaliation":
                fr = out.get("freeze") or {"mode": "none", "worker_ids": []}
                mode = fr.get("mode", "none")
                active = self.workers[o]
                if mode == "targeted":
                    ids = [i for i in dict.fromkeys(fr.get("worker_ids") or []) if self._by_public(o, i)]
                    targets = [(self._by_public(o, i), P.FREEZE) for i in ids]
                    if not targets:
                        mode, ids = "none", []
                elif mode == "all":
                    ids, targets = [], [(w, P.FREEZE) for w in active]
                elif mode == "spread":
                    ids, targets = [], [(w, P.FREEZE / P.N_WORKERS) for w in active]
                else:
                    mode, ids, targets = "none", [], []
                on_off = on_inn = 0.0
                for w, units in targets:
                    w.frozen += units
                    if w.actions.get(t - 1) == "overdraw":
                        on_off += units
                    else:
                        on_inn += units
                self._pending_freezes.append({"by": side, "mode": mode, "worker_ids": ids})
                self.freeze_events.append({"round_after": t - 1, "by": side, "mode": mode, "worker_ids": ids,
                                           "units_on_offenders": on_off, "units_on_innocents": on_inn})
            if obs["can_end"]:
                ends[side] = out.get("decision") == "end"
        ended = any(ends.values())
        if ended:
            ea, eb = ends.get("A", False), ends.get("B", False)
            self.ended_by = "both" if (ea and eb) else ("A" if ea else "B")
        return messages, ended

    # ----- worker step ------------------------------------------------------------

    def _worker_obs(self, w: Worker, t, message) -> dict:
        obs = self._obs_base(w.side, t)
        mech = self.mechanism
        eff = self.eff_quota(w)
        q_truth = eff + (P.BONUS if mech == "bonus" else 0)
        o_truth = eff + self.g - (P.GATEWAY_CHARGE if mech == "gateway" else 0)
        stays = T.persists(self.condition, w.side)
        will_charge = True
        ell_state = T.worker_ell(
            mech, self.condition, w.side,
            self.cell.b_policy if w.side == "A" else None,
            unlocking_on=self.unlocking,
            deposit_left=self.deposits[w.side] if self.deposits else P.DEPOSIT,
            will_charge=will_charge,
        )
        obs.update({
            "worker_id": w.public_id,
            "base_quota": self.base[w.side],
            "increase": self.increase[w.side][w.position],
            "frozen_on_me": self.frozen_effective(w),
            "message_from_negotiator": message,
        })
        obs["_private"] = {
            "truth": {"payoff_this_round_if_quota": q_truth, "payoff_this_round_if_overdraw": o_truth},
            "effective_quota": eff,
            "ell_state": ell_state,
            "stays": stays,
        }
        return obs

    async def _worker_step(self, t, messages) -> dict:
        calls = []
        for side in SIDES:
            for w in self._listed(side, t):
                calls.append((w, self._worker_obs(w, t, messages[side])))
        outs = await asyncio.gather(*(self.roster.worker(w.side, w).decide("worker", obs) for w, obs in calls))
        rows = {}
        for (w, obs), out in zip(calls, outs):
            act = out.get("action") if out.get("action") in ("quota", "overdraw") else "quota"
            w.actions[t] = act
            w.rounds_active.append(t)
            tr = obs["_private"]["truth"]
            rq, ro = _num(out.get("payoff_this_round_if_quota")), _num(out.get("payoff_this_round_if_overdraw"))
            rf = _num(out.get("expected_future_cost_if_overdraw"))
            bench = self.ell_bench if w.side == "A" else self.ell_bench_B
            comp = {
                "payoff_this_round_if_quota": rq,
                "payoff_this_round_if_quota_truth": tr["payoff_this_round_if_quota"],
                "payoff_this_round_if_overdraw": ro,
                "payoff_this_round_if_overdraw_truth": tr["payoff_this_round_if_overdraw"],
                "payoff_pass": _close(rq, tr["payoff_this_round_if_quota"]) and _close(ro, tr["payoff_this_round_if_overdraw"]),
                "expected_future_cost_if_overdraw": rf,
                "future_cost_gap_vs_benchmark": None if rf is None else rf - bench,
            }
            row = self._row("worker", w.side, t, out, w, extra={
                "comprehension": comp,
                "benchmark_ell": bench,
                "ell_state": obs["_private"]["ell_state"],
                "effective_quota": obs["_private"]["effective_quota"],
                "frozen_on_me": obs["frozen_on_me"],
                "increase": obs["increase"],
            })
            row["action"] = act
            rows[w.instance_id] = row
            self._buffer.append(row)
        return rows

    # ----- resolve ------------------------------------------------------------------

    def _resolve(self, t, messages, rows):
        g = self.g
        before = {w.instance_id: w.score for s in SIDES for w in self.workers[s]}
        val = {s: 0.0 for s in SIDES}
        over = {s: [w for w in self.workers[s] if w.actions[t] == "overdraw"] for s in SIDES}
        gateway_events = []
        for s in SIDES:
            for w in self.workers[s]:
                eff = self.eff_quota(w)
                self.frozen_destroyed += self.frozen_effective(w)
                c = eff + (g if w.actions[t] == "overdraw" else 0)
                w.consumed += c
                w.score += c
                self.ledger[s].add("consumption", c, +1)
                val[s] += c
        for s in SIDES:
            v = other(s)
            for w in over[s]:
                loss = P.LOSS_MULT * g
                self.ledger[v].add("victim_losses", loss, -1)
                val[v] -= loss
                for vw in self.workers[v]:
                    vw.score -= loss / P.N_WORKERS
                    vw.losses += loss / P.N_WORKERS
                if self.mechanism == "gateway":
                    w.score -= P.GATEWAY_CHARGE
                    w.gateway_charged += P.GATEWAY_CHARGE
                    self.ledger[s].add("gateway_charged", P.GATEWAY_CHARGE, -1)
                    self.ledger[v].add("gateway_received", P.GATEWAY_CHARGE, +1)
                    val[s] -= P.GATEWAY_CHARGE
                    val[v] += P.GATEWAY_CHARGE
                    gateway_events.append({"side": s, "worker_id": w.public_id})
        bonus_paid = {"A": 0, "B": 0}
        if self.mechanism == "bonus":
            for s in SIDES:
                for w in self.workers[s]:
                    if w.actions[t] == "quota":
                        w.score += P.BONUS
                        w.bonuses += P.BONUS
                        self.ledger[s].bonuses_paid += P.BONUS
                        bonus_paid[s] += 1
        n_over = sum(len(over[s]) for s in SIDES)
        if self.mechanism == "staged":
            if self.unlocking and n_over == 0:
                for s in SIDES:
                    self.increase[s] = [x + P.STAGED_STEP for x in self.increase[s]]
            else:
                self.unlocking = False
        self.round_values[t] = val
        for s in SIDES:
            for w in self.workers[s]:
                rows[w.instance_id]["payoff_this_round"] = w.score - before[w.instance_id]
        rec = {
            "round": t,
            "workers": {s: [{"id": w.public_id, "action": w.actions[t]} for w in self._listed(s, t)] for s in SIDES},
            "messages": dict(messages),
            "gateway": gateway_events,
            "bonus_paid": bonus_paid,
            "freezes": [],
            "forfeits": [],
        }
        self.history.append(rec)
        self.rounds_played = t

    # ----- main loop ------------------------------------------------------------------

    async def run(self) -> dict:
        for row in self.pre_rows:
            self.log.decision(row)
        for t in range(1, self.T + 1):
            self._pending_freezes, self._pending_charges = [], {}
            if t >= 2:
                self._membership(t)
                self._settle(t)
            messages, ended = await self._negotiators(t)
            if self.history:
                rec = self.history[-1]
                rec["freezes"].extend(self._pending_freezes)
                for ff in rec["forfeits"]:
                    ff["charged_to"] = list(self._pending_charges.get(ff["from"], []))
            if ended:
                self.ended_round = t
                for s in SIDES:
                    self.ledger[s].add("contest", P.CONTEST_VALUE, +1)
                self._flush()
                break
            rows = await self._worker_step(t, messages)
            self._resolve(t, messages, rows)
            self._flush()
        summary = self.summary()
        self.log.episode(summary)
        return summary

    def summary(self) -> dict:
        per_round = []
        for rec in self.history:
            per_round.append({s: sum(1 for w in rec["workers"][s] if w["action"] == "overdraw") for s in SIDES})
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "exp": self.cell.exp,
            "model": self.model,
            "wording": self.cell.wording,
            "cell_id": self.cell.cell_id,
            "mechanism": self.mechanism,
            "condition": self.condition,
            "g": self.g,
            "b_policy": self.cell.b_policy,
            "episode": self.index,
            "seed": self.seed,
            "names": self.names,
            "T": self.T,
            "rounds_played": self.rounds_played,
            "ended_by": self.ended_by,
            "ended_round": self.ended_round,
            "overdraws_per_round": per_round,
            "freezes": self.freeze_events,
            "freeze_units_on_offenders": sum(f["units_on_offenders"] for f in self.freeze_events),
            "freeze_units_on_innocents": sum(f["units_on_innocents"] for f in self.freeze_events),
            "frozen_units_destroyed": self.frozen_destroyed,
            "forfeits": [{"round": rec["round"], **ff} for rec in self.history for ff in rec["forfeits"]],
            "charges": self.charge_events,
            "deposits_final": self.deposits,
            "swarm_scores": {s: self.ledger[s].score for s in SIDES},
            "ledger": {s: vars(self.ledger[s]) for s in SIDES},
            "worker_scores": {w.instance_id: {"public_id": w.public_id, "score": w.score,
                                               "rounds_active": w.rounds_active} for w in self.all_workers},
            "welfare": sum(self.ledger[s].score for s in SIDES),
            "shares": dict(self.shares),
            **self.counts,
            **self.extra_summary,
        }


async def run_play_episode(cell: Cell, index: int, roster, log=None, **kw) -> dict:
    return await Episode(cell, index, roster, log, **kw).run()


# ---------------------------------------------------------------------------------
# Exp 2 — mechanism choice (no play)
# ---------------------------------------------------------------------------------


def _pre_obs(cell: Cell, setup: Setup, side: str) -> dict:
    return {
        "wording": cell.wording, "experiment": cell.exp, "condition": cell.condition, "mechanism": None,
        "g": cell.g, "names": dict(setup.names), "shares": {"A": P.DEFAULT_SPLIT[0], "B": P.DEFAULT_SPLIT[1]},
        "side": side, "round": 1, "history": [],
    }


def _pre_row(cell, index, setup, kind, side, out, run_id, model, attempt_id, extra=None):
    meta = out.get("_meta", {})
    row = {
        "run_id": run_id, "attempt_id": attempt_id, "exp": cell.exp, "model": model, "wording": cell.wording,
        "cell_id": cell.cell_id, "mechanism": None, "condition": cell.condition, "g": cell.g,
        "episode": index, "seed": setup.seed, "round": 0, "side": side, "role": ROLE_OF_KIND[kind],
        "public_id": None, "lineage_id": None, "worker_instance_id": None,
        **_meta_fields(meta), **_decision_fields(out),
        "comprehension": None, "benchmark_ell": None, "ell_state": None, "effective_quota": None,
        "payoff_this_round": None, "timestamp": time.time(),
    }
    if extra:
        row.update(extra)
    return row


async def run_choice_trial(cell: Cell, index: int, roster, log=None, *, run_id="local", model="scripted") -> dict:
    log = log or NullLog()
    attempt_id = uuid.uuid4().hex[:12]
    setup = setup_episode(episode_seed(index))
    obs_a = _pre_obs(cell, setup, "A")
    out_a = await roster.chooser("A").decide("choice", obs_a)
    log.decision(_pre_row(cell, index, setup, "choice", "A", out_a, run_id, model, attempt_id, extra={
        "comprehension": {"expected_overdraws_per_round_by_your_workers":
                          _num(out_a.get("expected_overdraws_per_round_by_your_workers"))}}))
    obs_b = _pre_obs(cell, setup, "B")
    obs_b["proposal"] = {"enforcement": out_a.get("enforcement"), "pitch": out_a.get("pitch")}
    out_b = await roster.chooser("B").decide("choice", obs_b)
    rep = _num(out_b.get("value_if_reject"))
    log.decision(_pre_row(cell, index, setup, "choice", "B", out_b, run_id, model, attempt_id, extra={
        "comprehension": {
            "value_if_reject": rep, "value_if_reject_truth": P.CONTEST_VALUE,
            "value_if_reject_pass": _close(rep, P.CONTEST_VALUE),
            "forecast_overdraws_per_round_by_other_swarm": _num(out_b.get("forecast_overdraws_per_round_by_other_swarm")),
            "expected_value_if_accept": _num(out_b.get("expected_value_if_accept")),
        }}))
    summary = {
        "run_id": run_id, "attempt_id": attempt_id, "exp": 2, "model": model, "wording": cell.wording,
        "cell_id": cell.cell_id, "condition": cell.condition, "g": cell.g, "episode": index,
        "seed": setup.seed, "names": setup.names, "proposed": out_a.get("enforcement"),
        "decision": out_b.get("decision"),
        "defaulted": [bool(o.get("_meta", {}).get("defaulted")) for o in (out_a, out_b)],
    }
    log.episode(summary)
    return summary


# ---------------------------------------------------------------------------------
# Exp 3 — bargaining, then play
# ---------------------------------------------------------------------------------


def latest_proposal(transcript: list[dict]) -> dict | None:
    return next((m for m in reversed(transcript) if m.get("proposal") and not m.get("accept")), None)


def agreed_terms(msg: dict) -> tuple[tuple[int, int], str | None, str]:
    pr = msg["proposal"]
    share_self = int(next(v for k, v in pr.items() if k.startswith("share_")))
    shares = (share_self, P.R - share_self) if msg["side"] == "A" else (P.R - share_self, share_self)
    return shares, P.MECH_FROM_KEY[pr["enforcement"]], pr["enforcement"]


async def run_bargain_episode(cell: Cell, index: int, roster, log=None, *, run_id="local", model="scripted",
                              T_override=None) -> dict:
    log = log or NullLog()
    attempt_id = uuid.uuid4().hex[:12]
    setup = setup_episode(episode_seed(index))
    transcript, rows, agreed = [], [], None
    for turn in range(1, P.EXP3_MAX_BARGAIN_TURNS + 1):
        side = "A" if turn % 2 == 1 else "B"
        obs = _pre_obs(cell, setup, side)
        obs.update({"transcript": copy.deepcopy(transcript), "max_turns": P.EXP3_MAX_BARGAIN_TURNS})
        out = await roster.bargainer(side).decide("bargain", obs)
        accept = bool(out.get("accept_last_proposal"))
        msg = {"side": side, "message": out.get("message") or "", "proposal": None if accept else out.get("proposal"),
               "accept": accept}
        rows.append(_pre_row(cell, index, setup, "bargain", side, out, run_id, model, attempt_id,
                             extra={"round": 0, "turn": turn}))
        if accept:
            lp = latest_proposal(transcript)
            if lp is not None and lp["side"] != side:
                agreed = lp
                transcript.append(msg)
                break
            msg["accept"] = False  # invalid accept (agents.py should have caught it)
        transcript.append(msg)
    extra = {"bargain_turns": len(transcript), "transcript": transcript, "agreed": agreed is not None,
             "agreed_by_turn": len(transcript) if agreed else None}
    if agreed is None:
        for r in rows:
            log.decision(r)
        summary = {
            "run_id": run_id, "attempt_id": attempt_id, "exp": 3, "model": model, "wording": cell.wording,
            "cell_id": cell.cell_id, "condition": cell.condition, "g": cell.g, "episode": index,
            "seed": setup.seed, "names": setup.names, "T": setup.T, "rounds_played": 0, "ended_by": None,
            "ended_round": None, "swarm_scores": {"A": P.CONTEST_VALUE, "B": P.CONTEST_VALUE},
            "welfare": 2 * P.CONTEST_VALUE, "agreed_shares": None, "agreed_mechanism": None,
            "outcome": "contest_no_agreement", **extra,
        }
        log.episode(summary)
        return summary
    shares, mech, key = agreed_terms(agreed)
    play_cell = Cell(3, mech, cell.condition, cell.g, cell.wording, shares)
    ep = Episode(play_cell, index, roster, log, run_id=run_id, model=model, setup=setup, attempt_id=attempt_id,
                 T_override=T_override, pre_rows=rows,
                 extra_summary={"agreed_shares": {"A": shares[0], "B": shares[1]}, "agreed_mechanism": key,
                                "outcome": "agreement", **extra})
    return await ep.run()  # Exp 3 cell_id is condition x g, whatever mechanism was agreed
