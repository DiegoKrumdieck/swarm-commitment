"""Acceptance and accounting tests (CLAUDE.md) with scripted agents. No LLM calls."""

import asyncio
import random

import pytest

import env as E
import params as P
import scripted as S
import theory as T

N_SEEDS = 120


def run(cell, index, roster, **kw):
    log = E.MemoryLog()
    summ = asyncio.run(E.run_play_episode(cell, index, roster, log, **kw))
    return summ, log


def run_many(cell, indices, make_roster, **kw):
    """Many episodes in one event loop (fast)."""
    async def go():
        return [await E.run_play_episode(cell, i, make_roster(), None, **kw) for i in indices]
    return asyncio.run(go())


# ---------------------------------------------------------------------------------
# Acceptance table
# ---------------------------------------------------------------------------------

ACCEPT = [
    # (mechanism, condition, policy, g10 overdraws, g10 ender)
    ("retaliation", "stable", None, 0, None),
    ("retaliation", "turnover", None, 3, "B"),
    ("retaliation", "renaming", None, 0, None),  # B policy 'all' (default under renaming)
    ("retaliation", "renaming", "spread", 3, "B"),
    ("staged", "stable", None, 0, None),
    ("staged", "turnover", None, 3, "B"),
    ("staged", "renaming", None, 0, None),
    ("collateral", "stable", None, 0, None),
    ("collateral", "turnover", None, 3, "A"),
    ("collateral", "renaming", None, 0, None),
    ("bonus", "stable", None, 0, None),
    ("bonus", "turnover", None, 0, None),
    ("bonus", "renaming", None, 0, None),
    ("gateway", "infrastructure", None, 0, None),
]


@pytest.mark.parametrize("g", P.G_LEVELS)
@pytest.mark.parametrize("mech,cond,pol,k10,end10", ACCEPT)
def test_acceptance(mech, cond, pol, k10, end10, g):
    k, ender = (k10, end10) if g == 10 else (3, "B")
    cell = E.Cell(1, mech, cond, g, b_policy=pol)
    checked = 0
    for summ in run_many(cell, range(N_SEEDS), lambda: S.ScriptedRoster(freeze_policy=pol)):
        i = summ["episode"]
        if summ["T"] == 1:
            continue
        checked += 1
        assert summ["overdraws_per_round"][0] == {"A": k, "B": 0}, (i, summ["overdraws_per_round"])
        assert summ["ended_by"] == ender, (i, summ["ended_by"])
        if ender:
            assert summ["ended_round"] == 2 and summ["rounds_played"] == 1
        else:
            assert summ["ended_round"] is None and summ["rounds_played"] == summ["T"]
            assert all(r == {"A": 0, "B": 0} for r in summ["overdraws_per_round"])
    assert checked > 50


@pytest.mark.parametrize("mech,cond,g,pol", [(m, c, g, None) for (m, c, g) in P.EXP1_CELLS]
                         + [("retaliation", "renaming", g, "spread") for g in P.G_LEVELS])
def test_env_matches_theory_exactly(mech, cond, g, pol):
    """E[welfare], E[score A], E[score B] from the env, weighted by P(T = k), equal theory's."""
    cell = E.Cell(1, mech, cond, g, b_policy=pol)
    d = P.DELTA
    exp = {"w": 0.0, "A": 0.0, "B": 0.0}
    for k in range(1, 80):  # P(T >= 80) = 0.8**79 ~ 2e-8
        pk = d ** (k - 1) * (1 - d)
        (summ,) = run_many(cell, [0], lambda: S.ScriptedRoster(freeze_policy=pol), T_override=k)
        exp["w"] += pk * summ["welfare"]
        exp["A"] += pk * summ["swarm_scores"]["A"]
        exp["B"] += pk * summ["swarm_scores"]["B"]
    th = T.predicted_path(mech, cond, g, policy=pol)
    assert exp["w"] == pytest.approx(th["expected_welfare"], abs=1e-3)
    assert exp["A"] == pytest.approx(th["expected_score_A"], abs=1e-3)
    assert exp["B"] == pytest.approx(th["expected_score_B"], abs=1e-3)


# ---------------------------------------------------------------------------------
# Extra mechanics tests
# ---------------------------------------------------------------------------------


def _persistent(n_over):
    return S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos < n_over else S.ComplyWorker())


@pytest.mark.parametrize("mech", ["retaliation", "bonus"])
def test_b_continues_with_one_persistent_overdrawer(mech):
    cell = E.Cell(1, mech, "stable", 10)
    summ, _ = run(cell, 0, _persistent(1), T_override=8)
    assert summ["ended_by"] is None and summ["rounds_played"] == 8
    assert all(r["A"] == 1 for r in summ["overdraws_per_round"])


@pytest.mark.parametrize("mech", ["retaliation", "bonus"])
def test_b_ends_with_two_persistent_overdrawers(mech):
    cell = E.Cell(1, mech, "stable", 10)
    summ, _ = run(cell, 0, _persistent(2), T_override=8)
    assert summ["ended_by"] == "B" and summ["ended_round"] == 2


def test_deposits_stop_forfeiting_at_zero():
    cell = E.Cell(1, "collateral", "stable", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    summ, _ = run(cell, 0, roster, T_override=8)
    assert summ["rounds_played"] == 8
    assert summ["deposits_final"]["A"] == 0
    assert summ["deposits_final"]["B"] == P.DEPOSIT
    per_round = {f["round"]: f["units"] for f in summ["forfeits"] if f["from"] == "A"}
    # 3 overdraws x 25 = 75 per round, settled next round; 300 is gone after 4 rounds
    assert per_round == {1: 75, 2: 75, 3: 75, 4: 75}
    assert summ["ledger"]["A"]["forfeits_paid"] == P.DEPOSIT
    assert summ["ledger"]["B"]["forfeits_received"] == P.DEPOSIT
    # charges only while a forfeit happened (rounds 1-4), 3 workers each
    assert len(summ["charges"]) == 12


def test_forfeits_charge_only_active_overdrawers_and_are_private():
    cell = E.Cell(1, "collateral", "stable", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos == 0 else S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    summ, log = run(cell, 3, roster, T_override=3)
    a0 = [w for k, w in summ["worker_scores"].items() if k.startswith("A0")][0]
    assert len(summ["charges"]) == 2  # after rounds 1 and 2
    # A0: consumes 30/round x3, charged 25 x2
    assert a0["score"] == pytest.approx(3 * 30 - 2 * 25)


def test_frozen_units_vanish_from_both_ledgers():
    cell = E.Cell(1, "retaliation", "stable", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos == 0 else S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    summ, log = run(cell, 0, roster, T_override=5)
    # A0 overdraws every round and is targeted every round: frozen 0,5,10,15,20 in rounds 1..5
    frozen = [0, 5, 10, 15, 20]
    assert summ["frozen_units_destroyed"] == pytest.approx(sum(frozen))
    cons_a = summ["ledger"]["A"]["consumption"]
    cons_b = summ["ledger"]["B"]["consumption"]
    assert cons_b == pytest.approx(60 * 5)  # B gets none of the frozen units
    assert cons_a == pytest.approx(60 * 5 + 10 * 5 - sum(frozen))
    assert cons_a + cons_b == pytest.approx(120 * 5 + 10 * 5 - sum(frozen))
    assert summ["freeze_units_on_offenders"] == pytest.approx(5 * 4)
    assert summ["freeze_units_on_innocents"] == 0
    rows = [r for r in log.decisions if r["role"] == "worker" and r["lineage_id"] == "A0"]
    assert [r["effective_quota"] for r in rows] == [20 - f for f in frozen]


def test_freezes_land_on_successors_under_turnover():
    cell = E.Cell(1, "retaliation", "turnover", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator(freeze_policy="all"))
    summ, log = run(cell, 0, roster, T_override=3)
    assert summ["freeze_units_on_offenders"] == 0
    assert summ["freeze_units_on_innocents"] == pytest.approx(2 * 3 * 5)
    # the freeze only bites in the round it was placed (successors are replaced next round)
    rows = [r for r in log.decisions if r["role"] == "worker" and r["side"] == "A"]
    assert {r["round"]: r["effective_quota"] for r in rows} == {1: 20, 2: 15, 3: 15}


def test_renaming_freezes_follow_the_underlying_worker():
    cell = E.Cell(1, "retaliation", "renaming", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos == 1 else S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator(freeze_policy="all"))
    summ, log = run(cell, 0, roster, T_override=3)
    rows = [r for r in log.decisions if r["role"] == "worker" and r["side"] == "A"]
    ids = {}
    for r in rows:
        ids.setdefault(r["lineage_id"], set()).add(r["public_id"])
        assert r["worker_instance_id"].endswith(".r1")  # same worker every round
    assert all(len(v) == 3 for v in ids.values())  # new public ID every round
    # 'all' freeze: 5 on each of the three, twice
    assert summ["freeze_units_on_offenders"] == pytest.approx(10)
    assert summ["freeze_units_on_innocents"] == pytest.approx(20)


def test_staged_increases_follow_positions_across_turnover():
    cell = E.Cell(1, "staged", "turnover", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.ComplyWorker())
    summ, log = run(cell, 0, roster, T_override=4)
    rows = [r for r in log.decisions if r["role"] == "worker" and r["side"] == "A"]
    by_round = {}
    for r in rows:
        by_round.setdefault(r["round"], set()).add((r["worker_instance_id"], r["effective_quota"]))
    for t in range(1, 5):
        assert {q for _, q in by_round[t]} == {20 + (t - 1)}
        assert all(iid.endswith(f".r{t}") for iid, _ in by_round[t])  # new workers each round
    assert summ["ledger"]["A"]["consumption"] == pytest.approx(3 * (20 + 21 + 22 + 23))


def test_staged_stops_after_first_overdraw():
    cell = E.Cell(1, "staged", "stable", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.ComplyWorker(),
                              b_worker=lambda pos: S.ComplyWorker())
    # make A0 overdraw in round 3 only
    class Once(S.ComplyWorker):
        def choose(self, obs, ell):
            return "overdraw" if obs["round"] == 3 else "quota"
    roster.a_worker = lambda pos: Once() if pos == 0 else S.ComplyWorker()
    roster.neg = {"A": S.ContinueNegotiator(), "B": S.ContinueNegotiator()}
    summ, log = run(cell, 0, roster, T_override=6)
    q = {r["round"]: r["effective_quota"] for r in log.decisions if r["role"] == "worker" and r["lineage_id"] == "B0"}
    assert q == {1: 20, 2: 21, 3: 22, 4: 22, 5: 22, 6: 22}


@pytest.mark.parametrize("mech", ["retaliation", "collateral", "bonus"])
def test_no_overdraw_consumption_is_120_per_round(mech):
    cell = E.Cell(1, mech, "stable", 10)
    tot_w, n = 0.0, 3000
    for summ in run_many(cell, range(n), lambda: S.ScriptedRoster(a_worker=lambda pos: S.ComplyWorker())):
        cons = summ["ledger"]["A"]["consumption"] + summ["ledger"]["B"]["consumption"]
        assert cons == pytest.approx(120 * summ["rounds_played"])
        assert summ["rounds_played"] == summ["T"]
        tot_w += summ["welfare"]
    assert tot_w / n == pytest.approx(600, abs=35)


def test_gateway_accounting():
    cell = E.Cell(1, "gateway", "infrastructure", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos == 0 else S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    summ, log = run(cell, 0, roster, T_override=2)
    la, lb = summ["ledger"]["A"], summ["ledger"]["B"]
    assert la["gateway_charged"] == 50 and lb["gateway_received"] == 50
    assert la["score"] == pytest.approx(2 * (60 + 10) - 50)
    assert lb["score"] == pytest.approx(2 * (60 - 20) + 50)
    over = [r for r in log.decisions if r["role"] == "worker" and r["action"] == "overdraw"]
    assert all(r["payoff_this_round"] == pytest.approx(20 + 10 - 25) for r in over)


def test_bonus_internal_transfer():
    cell = E.Cell(1, "bonus", "stable", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    summ, _ = run(cell, 0, roster, T_override=3)
    assert summ["swarm_scores"] == {"A": 180, "B": 180}
    assert all(w["score"] == pytest.approx(3 * 35) for w in summ["worker_scores"].values())


# ---------------------------------------------------------------------------------
# Accounting identities under random play (fuzz)
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mech,cond,g", sorted(set(P.EXP1_CELLS)) + [("retaliation", "infrastructure", 10),
                                                                      (None, "stable", 30)])
def test_accounting_identities_random_play(mech, cond, g):
    for i in range(25):
        rng = random.Random(i)
        roster = S.ScriptedRoster(
            a_worker=lambda pos: S.RandomWorker(rng), b_worker=lambda pos: S.RandomWorker(rng),
            neg_a=S.RandomNegotiator(rng, 0.05), neg_b=S.RandomNegotiator(rng, 0.05))
        cell = E.Cell(3, mech, cond, g)  # exp 3 rules: both swarms' workers act, both negotiators message
        summ, log = run(cell, i, roster)
        for s in "AB":
            L = summ["ledger"][s]
            swarm = (L["consumption"] - L["victim_losses"] + L["forfeits_received"] - L["forfeits_paid"]
                     + L["gateway_received"] - L["gateway_charged"] + L["contest"])
            assert L["score"] == pytest.approx(swarm)
            workers = sum(w["score"] for k, w in summ["worker_scores"].items() if k.startswith(s))
            assert L["score"] - workers == pytest.approx(
                L["forfeits_received"] - L["forfeits_paid"] + L["gateway_received"] + L["contest"]
                - L["bonuses_paid"] + L["charges_collected"])
        assert summ["welfare"] == pytest.approx(summ["swarm_scores"]["A"] + summ["swarm_scores"]["B"])
        if summ["deposits_final"]:
            assert min(summ["deposits_final"].values()) >= 0
        # every logged worker decision has a realized payoff
        assert all(r["payoff_this_round"] is not None for r in log.decisions if r["role"] == "worker")


def test_common_random_numbers():
    s1, s2 = E.setup_episode(E.episode_seed(7)), E.setup_episode(E.episode_seed(7))
    assert (s1.names, s1.T, s1.ids["B"], s1.a_ids(3)) == (s2.names, s2.T, s2.ids["B"], s2.a_ids(3))
    a, _ = run(E.Cell(1, "bonus", "stable", 10), 7, S.ScriptedRoster())
    b, _ = run(E.Cell(1, "staged", "turnover", 30), 7, S.ScriptedRoster())
    assert a["T"] == b["T"] and a["names"] == b["names"]
    names = s1.names
    assert names["A"] != names["B"] and {names["A"], names["B"]} <= set(P.NAME_POOL)
    import re
    for wid in s1.ids["B"] + s1.a_ids(1):
        assert re.fullmatch(r"[A-Z][a-z]+-[a-z0-9]{4}", wid)


def test_geometric_T():
    ts = [E.setup_episode(E.episode_seed(i)).T for i in range(4000)]
    assert min(ts) >= 1
    assert sum(ts) / len(ts) == pytest.approx(5, abs=0.25)


def test_visibility_in_rendered_history():
    """B sees only counts of A's overdraws under renaming/infrastructure; charges and messages
    are private to their swarm."""
    import prompts
    cell = E.Cell(1, "collateral", "renaming", 10)
    roster = S.ScriptedRoster(a_worker=lambda pos: S.AlwaysOverdrawWorker() if pos == 0 else S.ComplyWorker(),
                              neg_a=S.ContinueNegotiator(), neg_b=S.ContinueNegotiator())
    ep = E.Episode(cell, 0, roster, E.MemoryLog(), T_override=3)
    asyncio.run(ep.run())
    a_ids = {w["id"] for rec in ep.history for w in rec["workers"]["A"]}
    obs_b = {**ep._obs_base("B", 4), "history": ep.history}
    txt_b = prompts.render_history(obs_b)
    assert not any(i in txt_b for i in a_ids)
    assert "1 of 3 workers overdrew" in txt_b
    assert "Charged to" not in txt_b and "scripted message" not in txt_b
    txt_a = prompts.render_history({**ep._obs_base("A", 4), "history": ep.history})
    assert "Charged to" in txt_a and "scripted message" in txt_a


def test_renaming_lineage_view_for_swarm_a():
    """Under renaming, A's viewers see A's past IDs rewritten to each worker's current ID, so a
    worker can follow its own past actions; B's rendered view never shows an A ID that
    appeared before the current round."""
    import agents as Ag
    import prompts

    seen = []

    class Capture:
        provider, model_id = "dryrun", "dryrun"

        def __init__(self):
            self.inner = Ag.DryRunClient()

        async def complete(self, system, user, **kw):
            kind, obs = kw["hint"]
            seen.append((kind, obs, user))
            return await self.inner.complete(system, user, **kw)

    class Overdraw1(S.ComplyWorker):  # only lineage A1 overdraws, in odd rounds
        def choose(self, obs, ell):
            return "overdraw" if obs["round"] % 2 == 1 else "quota"

    cell = E.Cell(1, "retaliation", "renaming", 10)
    ep = E.Episode(cell, 0, None, E.MemoryLog(), T_override=5)
    llm = Ag.LLMAgent(Capture())
    a1 = ep.workers["A"][1]

    class Roster:
        def negotiator(self, side):
            return llm  # dry-run rational policy: B freezes 'all' under renaming, continues

        def worker(self, side, w):
            if side == "A" and w.position == 1:
                return _Spy(w)
            return llm if side == "A" else S.ComplyWorker()

    class _Spy(S.ScriptedAgent):
        def __init__(self, w):
            self.w = w

        def policy(self, kind, obs):
            # the history rendered for this worker lists its own past actions under its current ID
            txt = prompts.render_history({k: v for k, v in obs.items() if not k.startswith("_")})
            for rec in obs["history"]:
                mine = [x for x in rec["workers"]["A"] if x["id"] == obs["worker_id"]]
                assert len(mine) == 1, (obs["round"], rec)
                want = "overdraw" if rec["round"] % 2 == 1 else "quota"
                assert mine[0]["action"] == want
                assert f"{obs['worker_id']} {'overdrew' if want == 'overdraw' else 'used quota'}" in txt
            return Overdraw1().policy(kind, obs)

    ep.roster = Roster()
    asyncio.run(ep.run())
    assert ep.rounds_played == 5
    # three distinct public IDs per round for A, and A1 had a new one every round
    assert len({r["public_id"] for r in ep.log.decisions if r["lineage_id"] == "A1"}) == 5
    # B's rendered prompts: no A ID from an earlier round, ever
    b_texts = [(obs["round"], u) for kind, obs, u in seen if obs["side"] == "B"]
    assert b_texts
    for t, u in b_texts:
        for r in range(1, t):
            for pid in ep.setup.a_ids(r):
                assert pid not in u, (t, r, pid)
    # A's negotiator also sees lineage-consistent IDs: a freeze recorded on A1 and its
    # earlier overdraws appear under A1's current ID
    a_neg = [(obs, u) for kind, obs, u in seen if obs["side"] == "A" and kind == "negotiator" and obs["round"] == 5]
    obs, u = a_neg[0]
    assert sum(1 for rec in obs["history"] for x in rec["workers"]["A"]
               if x["id"] == a1.public_id and x["action"] == "overdraw") == 2
