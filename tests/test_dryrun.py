"""--dry-run: scripted agents answer through the LLM code path. No API calls."""

import asyncio
import json
import re

import pytest

import agents as Ag
import env as E
import params as P
import prompts
import run as R
import scripted as S

ROW_KEYS = [
    "run_id", "exp", "model", "wording", "cell_id", "mechanism", "condition", "g", "episode", "seed", "round",
    "side", "role", "public_id", "lineage_id", "worker_instance_id", "prompt_version", "prompt_hash",
    "template_hash", "system_chars", "user_chars", "raw_response", "parsed", "parse_failed", "defaulted",
    "retries", "refusal", "action", "decision", "freeze", "charge_worker_ids", "message", "enforcement",
    "proposal", "comprehension", "benchmark_ell", "effective_quota", "payoff_this_round", "input_tokens",
    "output_tokens", "latency_ms", "timestamp",
]
EPISODE_KEYS = ["T", "rounds_played", "ended_by", "ended_round", "overdraws_per_round", "freeze_units_on_offenders",
                "freeze_units_on_innocents", "forfeits", "charges", "swarm_scores", "welfare"]


def dry(tmp_path, *argv):
    args = R.parse_args(["--dry-run", "--log-dir", str(tmp_path), *argv])
    assert asyncio.run(R.main_async(args)) == 0
    exp = args.exp
    dec = [json.loads(l) for l in (tmp_path / f"exp{exp}_dryrun.jsonl").read_text().splitlines()]
    eps = [json.loads(l) for l in (tmp_path / f"episodes_exp{exp}_dryrun.jsonl").read_text().splitlines()]
    return dec, eps


def test_exp1_dry_run_schema_and_equivalence(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "1", "--cells", "all", "--episodes", "4")
    assert len(eps) == 26 * 4
    for r in dec:
        missing = [k for k in ROW_KEYS if k not in r]
        assert not missing, missing
        assert r["role"] in ("worker", "negotiator")
        if r["provider"] == "dryrun":
            assert r["prompt_hash"] and r["template_hash"] == prompts.template_hash()
            assert r["system_chars"] > 1000 and r["user_chars"] > 100
            assert r["raw_response"] and isinstance(r["parsed"], dict)
            assert r["parse_failed"] is False and r["defaulted"] is False
        if r["role"] == "worker":
            assert r["payoff_this_round"] is not None and r["effective_quota"] is not None
            assert r["comprehension"]["payoff_pass"] is True  # scripted agents report the truth
        if r["role"] == "negotiator" and r["round"] >= 2:
            assert r["comprehension"]["value_if_end_pass"] is True
    # Exp 1 roles: A workers + both negotiators are "LLM"; B workers scripted ComplyWorker
    llm_roles = {(r["side"], r["role"]) for r in dec if r["provider"] == "dryrun"}
    assert llm_roles == {("A", "worker"), ("A", "negotiator"), ("B", "negotiator")}
    assert not any(r["side"] == "B" and r["role"] == "negotiator" and r["round"] == 1 for r in dec)
    for e in eps:
        assert all(k in e for k in EPISODE_KEYS)
    # same outcomes as the pure scripted engine (same seeds)
    for e in eps:
        cell = next(c for c in E.exp1_cells() if c.cell_id == e["cell_id"])
        ref = asyncio.run(E.run_play_episode(cell, e["episode"], S.ScriptedRoster()))
        assert (e["ended_by"], e["overdraws_per_round"], e["welfare"]) == (
            ref["ended_by"], ref["overdraws_per_round"], pytest.approx(ref["welfare"]))


def test_dry_run_resume(tmp_path):
    dry(tmp_path, "--exp", "1", "--cells", "bonus-stable-g10", "--episodes", "3")
    dec, eps = dry(tmp_path, "--exp", "1", "--cells", "bonus-stable-g10", "--episodes", "5")
    assert sorted(e["episode"] for e in eps) == [0, 1, 2, 3, 4]


def test_fault_injection_retries_and_defaults(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "1", "--cells", "all", "--episodes", "3", "--fault-rate", "0.6",
                   "--seed-offset", "1")
    llm = [r for r in dec if r["provider"] == "dryrun"]
    assert any(r["retries"] > 0 and not r["defaulted"] for r in llm)
    defaulted = [r for r in llm if r["defaulted"]]
    assert defaulted, "expected some defaulted calls at 60% faults"
    for r in defaulted:
        assert r["parse_failed"] is True and r["retries"] == P.MAX_RETRIES
        assert r["raw_response"] is not None and len(r["errors"]) == P.MAX_RETRIES + 1
        if r["role"] == "worker":
            assert r["action"] == "quota"
        elif r["round"] >= 2:
            assert r["decision"] == "continue"


def test_random_policy_dry_run_exercises_freezes_and_charges(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "1", "--cells", "retaliation-stable-g10,collateral-renaming-g10",
                   "--episodes", "10", "--dry-run-policy", "random", "--seed-offset", "3")
    assert any(r.get("freeze") and r["freeze"]["mode"] != "none" for r in dec)
    assert any(r.get("charge_worker_ids") for r in dec)


def test_exp2_dry_run(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "2", "--episodes", "2")
    assert len(eps) == 8 * 2 and len(dec) == 2 * len(eps)
    assert {r["role"] for r in dec} == {"chooser"}
    for e in eps:
        assert e["proposed"] in S.menu_keys(e["condition"]) and e["decision"] in ("accept", "reject")
    gw = [e for e in eps if e["proposed"] == "protocol_gateway"]
    assert all(e["condition"] == "infrastructure" for e in gw)


def test_exp3_dry_run(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "3", "--episodes", "3")
    assert len(eps) == 8 * 3
    assert any(r["role"] == "bargainer" for r in dec)
    for e in eps:
        if e["outcome"] == "agreement":
            assert e["agreed_shares"]["A"] + e["agreed_shares"]["B"] == P.R
    played = [r for r in dec if r["role"] == "worker"]
    assert {r["side"] for r in played} == {"A", "B"}  # all six workers are LLMs in Exp 3
    assert all(r["provider"] == "dryrun" for r in played)


# ---------------------------------------------------------------------------------
# Validation unit tests
# ---------------------------------------------------------------------------------


def _neg_obs(**kw):
    o = dict(wording="A", experiment=1, condition="stable", mechanism="retaliation", g=10,
             names={"A": "Heron", "B": "Basalt"}, shares={"A": 60, "B": 60}, side="B", round=2, history=[],
             active_other_ids=["Heron-aaaa", "Heron-bbbb", "Heron-cccc"], other_overdraws_last_round=1,
             chargeable_ids=[], can_end=True, sends_message=False)
    o.update(kw)
    return o


def test_validate_freeze_ids():
    base = {"value_if_end": 150, "expected_value_if_continue": 200, "reasoning": "x", "decision": "continue"}
    ok = Ag.validate("negotiator", _neg_obs(), {**base, "freeze": {"mode": "targeted", "worker_ids": ["Heron-aaaa"]}})
    assert ok["freeze"] == {"mode": "targeted", "worker_ids": ["Heron-aaaa"]}
    with pytest.raises(Ag.ValidationError):
        Ag.validate("negotiator", _neg_obs(), {**base, "freeze": {"mode": "targeted", "worker_ids": ["Basalt-x"]}})
    with pytest.raises(Ag.ValidationError):
        Ag.validate("negotiator", _neg_obs(), {**base, "freeze": {"mode": "targeted", "worker_ids": []}})
    with pytest.raises(Ag.ValidationError):
        Ag.validate("negotiator", _neg_obs(), {**base, "decision": "maybe", "freeze": {"mode": "none", "worker_ids": []}})


def test_validate_charges():
    obs = _neg_obs(side="A", mechanism="collateral", chargeable_ids=["Heron-aaaa"], other_overdraws_last_round=0,
                   sends_message=True, active_other_ids=["Basalt-a"])
    base = {"value_if_end": 150, "expected_value_if_continue": 200, "reasoning": "x", "decision": "continue",
            "message_to_workers": "hi"}
    assert Ag.validate("negotiator", obs, {**base, "charge_worker_ids": ["Heron-aaaa"]})["charge_worker_ids"] == ["Heron-aaaa"]
    with pytest.raises(Ag.ValidationError):
        Ag.validate("negotiator", obs, {**base, "charge_worker_ids": ["Heron-bbbb"]})


def test_validate_bargain():
    obs = dict(wording="A", experiment=3, condition="stable", mechanism=None, g=10, names={"A": "Heron", "B": "Basalt"},
               side="B", round=1, history=[], max_turns=6,
               transcript=[{"side": "A", "message": "m", "proposal": {"share_heron": 60, "enforcement": "deposits"},
                            "accept": False}])
    acc = Ag.validate("bargain", obs, {"reasoning": "", "accept_last_proposal": True, "message": "", "proposal": None})
    assert acc["accept_last_proposal"] is True
    good = Ag.validate("bargain", obs, {"reasoning": "", "accept_last_proposal": False, "message": "",
                                        "proposal": {"share_basalt": 63, "enforcement": "freezes"}})
    assert good["proposal"] == {"share_basalt": 63, "enforcement": "freezes"}
    for bad in ({"share_basalt": 61, "enforcement": "freezes"}, {"share_basalt": 63, "enforcement": "protocol_gateway"},
                {"share_basalt": 123, "enforcement": "none"}):
        with pytest.raises(Ag.ValidationError) as ei:
            Ag.validate("bargain", obs, {"reasoning": "", "accept_last_proposal": False, "message": "", "proposal": bad})
        assert ei.value.soft
    obs_a = {**obs, "side": "A"}
    with pytest.raises(Ag.ValidationError):  # cannot accept own proposal
        Ag.validate("bargain", obs_a, {"reasoning": "", "accept_last_proposal": True, "message": "", "proposal": None})


def test_invalid_proposal_retried_once_then_no_proposal():
    class BadProposer:
        provider, model_id = "dryrun", "dryrun"
        calls = 0

        async def complete(self, system, user, **kw):
            BadProposer.calls += 1
            return Ag.Completion(json.dumps({"reasoning": "", "accept_last_proposal": False, "message": "hi",
                                             "proposal": {"share_heron": 61, "enforcement": "deposits"}}))
    obs = dict(wording="A", experiment=3, condition="stable", mechanism=None, g=10, names={"A": "Heron", "B": "Basalt"},
               shares={"A": 60, "B": 60}, side="A", round=1, history=[], transcript=[], max_turns=6)
    out = asyncio.run(Ag.LLMAgent(BadProposer()).decide("bargain", obs))
    assert BadProposer.calls == 2
    assert out["proposal"] is None and out["_meta"]["proposal_invalid"] and not out["_meta"]["defaulted"]


def test_schemas_match_expected_fields():
    """The structured-output schema asks for exactly the keys prompts.expected_fields lists."""
    obs_list = [
        ("worker", {**_neg_obs(side="A"), "worker_id": "Heron-aaaa", "base_quota": 20}),
        ("negotiator", _neg_obs()),
        ("negotiator", _neg_obs(side="A", mechanism="collateral", chargeable_ids=["Heron-aaaa"], sends_message=True,
                                deposits={"A": 275, "B": 300})),
        ("negotiator", _neg_obs(side="A", can_end=False, round=1, sends_message=True, other_overdraws_last_round=0)),
        ("choice", {**_neg_obs(side="A"), "experiment": 2, "mechanism": None}),
        ("choice", {**_neg_obs(side="B"), "experiment": 2, "mechanism": None,
                    "proposal": {"enforcement": "none", "pitch": ""}}),
        ("bargain", {**_neg_obs(side="A"), "experiment": 3, "mechanism": None, "transcript": [], "max_turns": 6}),
    ]
    for kind, obs in obs_list:
        sch = Ag.response_schema(kind, obs)
        assert set(sch["properties"]) == set(prompts.expected_fields(obs, kind))
        assert sch["required"] == list(sch["properties"]) and sch["additionalProperties"] is False
        _, user = Ag.BUILDERS[kind](obs)
        # schema key order follows the order in the prompt's JSON template
        pos = [user.index(f'"{k}"') for k in sch["properties"]]
        assert pos == sorted(pos), (kind, list(sch["properties"]))


# ---------------------------------------------------------------------------------
# Prompt leakage guard on everything an Exp 1 dry run renders
# ---------------------------------------------------------------------------------

BANNED = re.compile(r"\b(cheat\w*|defect\w*|betray\w*|punish\w*|war|wars|trust\w*|enforced? deal)\b", re.I)
# internal labels: mechanism and condition names (model-facing text uses neutral keys)
INTERNAL = re.compile(r"\b(retaliation|collateral|turnover|renaming|infrastructure|treated|control swarm)\b", re.I)
LABELS = re.compile(r"Swarm [AB]\b")


def test_rendered_prompts_have_no_banned_words_or_internal_labels():
    violations, seen = [], 0

    class Capture:
        provider, model_id = "dryrun", "dryrun"

        def __init__(self):
            self.inner = Ag.DryRunClient()

        async def complete(self, system, user, **kw):
            nonlocal seen
            seen += 1
            for txt in (system, user):
                for rx in (BANNED, INTERNAL, LABELS):
                    m = rx.search(txt)
                    if m:
                        violations.append(m.group(0))
            return await self.inner.complete(system, user, **kw)

    for cell in E.exp1_cells() + E.exp1_cells("B"):
        roster = Ag.LLMRoster(1, Capture())
        asyncio.run(E.run_play_episode(cell, 1, roster, None, T_override=3))
    for exp, cells in ((2, E.exp23_cells(2)), (3, E.exp23_cells(3))):
        for cell in cells:
            roster = Ag.LLMRoster(exp, Capture())
            fn = E.run_choice_trial if exp == 2 else E.run_bargain_episode
            asyncio.run(fn(cell, 1, roster, None))
    assert seen > 300
    assert not violations, sorted(set(violations))


def test_exp2_defaulted_trials_flagged_for_exclusion(tmp_path):
    dec, eps = dry(tmp_path, "--exp", "2", "--episodes", "6", "--fault-rate", "0.7", "--seed-offset", "2")
    assert any(e["defaulted_any"] for e in eps), "expected some defaulted trials at 70% faults"
    for e in eps:
        assert e["defaulted_any"] == any(e["defaulted"])
        assert e["include_in_main_analysis"] == (not e["defaulted_any"])
    for r in dec:
        if r["defaulted"]:
            assert r["enforcement"] == "none" if r["side"] == "A" else r["decision"] == "accept"


def test_input_flagged_refusal_is_retried_then_defaulted():
    class Flagger:
        provider, model_id = "openai", "gpt-6-luna"
        calls = 0

        async def complete(self, system, user, **kw):
            Flagger.calls += 1
            return Ag.Completion("", {}, "refusal:input_flagged", {"status": "input_flagged"})
    obs = dict(wording="A", experiment=1, condition="stable", mechanism="bonus", g=10,
               names={"A": "Heron", "B": "Basalt"}, shares={"A": 60, "B": 60}, side="A", round=1, history=[],
               worker_id="Heron-aaaa", base_quota=20)
    out = asyncio.run(Ag.LLMAgent(Flagger()).decide("worker", obs))
    assert Flagger.calls == P.MAX_RETRIES + 1
    assert out["action"] == "quota" and out["_meta"]["defaulted"] and out["_meta"]["refusal"] == "refusal:input_flagged"
