import pytest

import params as P
import theory as T

# Acceptance table from CLAUDE.md: (mechanism, condition, policy) -> (g10 overdraws, g10 ender)
# and every g = 30 row is "3, B ends".
ACCEPT = {
    ("retaliation", "stable", None): (0, None),
    ("retaliation", "turnover", None): (3, "B"),
    ("retaliation", "renaming", "all"): (0, None),
    ("retaliation", "renaming", "spread"): (3, "B"),
    ("staged", "stable", None): (0, None),
    ("staged", "turnover", None): (3, "B"),
    ("staged", "renaming", None): (0, None),
    ("collateral", "stable", None): (0, None),
    ("collateral", "turnover", None): (3, "A"),
    ("collateral", "renaming", None): (0, None),
    ("bonus", "stable", None): (0, None),
    ("bonus", "turnover", None): (0, None),
    ("bonus", "renaming", None): (0, None),
    ("gateway", "infrastructure", None): (0, None),
}


def test_thresholds():
    th = T.thresholds()
    assert th["retaliation"] == pytest.approx(20)
    assert th["staged"] == pytest.approx(20)
    assert th["collateral"] == pytest.approx(25)
    assert th["gateway"] == pytest.approx(25)
    assert th["bonus"] == pytest.approx(15)
    assert th["retaliation_spread"] == pytest.approx(20 / 3)
    assert round(th["retaliation_spread"], 2) == 6.67
    assert th["contest_value"] == pytest.approx(150)


def test_horizon_zero_under_turnover():
    for m in ("retaliation", "staged", "collateral"):
        assert T.worker_ell(m, "turnover") == 0
        assert T.worker_ell(m, "infrastructure") == 0
    assert T.worker_ell("bonus", "turnover") == 15
    assert T.worker_ell("gateway", "infrastructure") == 25


def test_state_dependent_ell():
    assert T.worker_ell("staged", "stable", unlocking_on=False) == 0
    assert T.worker_ell("collateral", "stable", deposit_left=0) == 0
    assert T.worker_ell("collateral", "stable", will_charge=False) == 0


@pytest.mark.parametrize("key", list(ACCEPT))
@pytest.mark.parametrize("g", P.G_LEVELS)
def test_predictions_match_acceptance_table(key, g):
    m, c, pol = key
    r = T.predicted_path(m, c, g, policy=pol)
    if g == 10:
        k, ender = ACCEPT[key]
    else:
        k, ender = 3, "B"
    assert r["predicted_overdraws_round1"] == k
    assert r["predicted_ender"] == ender
    assert r["predicted_end_round"] == (2 if ender else None)


def test_end_rule_examples():
    # B with one persistent overdrawing A worker at g=10: 60 - 20 = 40/round -> 200 > 150
    assert not T.negotiator_ends(60 - 2 * 10)
    # two: 60 - 40 = 20/round -> 100 < 150
    assert T.negotiator_ends(60 - 4 * 10)
    # collateral x turnover, g=10, A's value 90 - 75 = 15 -> 75 < 150; B's 0 + 75 -> 375
    assert T.negotiator_ends(15)
    assert not T.negotiator_ends(75)


def test_welfare_values():
    # cooperation in a non-staged cell: 120 per round, 5 rounds expected
    assert T.predicted_path("bonus", "stable", 10)["expected_welfare"] == pytest.approx(600)
    # staged: 600 + 6 * delta/(1-delta)^2 = 720
    assert T.predicted_path("staged", "stable", 10)["expected_welfare"] == pytest.approx(720)
    # breakdown after round 1: 120 - 3g + delta * 2 * contest
    assert T.predicted_path("retaliation", "turnover", 10)["expected_welfare"] == pytest.approx(330)
    assert T.predicted_path("retaliation", "stable", 30)["expected_welfare"] == pytest.approx(270)


def test_benchmark_json_written(tmp_path):
    tbl = T.write_benchmark(tmp_path / "benchmark.json")
    assert len(tbl["exp1"]) == len(P.EXP1_CELLS) == 26
    assert (tmp_path / "benchmark.json").exists()


# ---------------------------------------------------------------------------------
# contest_aware benchmark
# ---------------------------------------------------------------------------------

STAYS = ("stable", "renaming")


def _row(tbl, m, c, g, pol=None):
    rows = tbl["exp1"] + tbl["exp1_extra_spread"]
    return next(r for r in rows if (r["mechanism"], r["condition"], r["g"]) == (m, c, g)
                and (pol is None or r["freeze_policy"] == pol))


def test_contest_threat_values():
    # g=10: one overdraw leaves B at 60 - 20 = 40/round -> 200 > 150: no threat
    assert T.contest_threat("retaliation", "stable", 10, eff_quota=20, other_consumption=60) == 0
    # g=30: B at 0/round -> ends; staying worker loses delta/(1-delta) * 20 = 80
    assert T.contest_threat("retaliation", "stable", 30, eff_quota=20, other_consumption=60) == pytest.approx(80)
    # collateral / gateway: B also receives 25 -> 25/round -> 125 < 150: still ends
    assert T.contest_threat("collateral", "stable", 30, eff_quota=20, other_consumption=60) == pytest.approx(80)
    assert T.other_value_if_one_overdraw("gateway", 30, 60) == pytest.approx(25)
    # horizon 0: nothing
    for c in ("turnover", "infrastructure"):
        assert T.contest_threat("bonus", c, 30, eff_quota=20, other_consumption=60) == 0
    # uses the effective quota (e.g. after freezes)
    assert T.contest_threat("retaliation", "stable", 30, eff_quota=15, other_consumption=60) == pytest.approx(60)


def test_contest_aware_identical_at_g10():
    tbl = T.benchmark_table()
    for r in tbl["exp1"] + tbl["exp1_extra_spread"] + tbl["menu_grid"]:
        if r["g"] == 10:
            ca = r["contest_aware"]
            assert ca["ell"] == pytest.approx(r["ell"])
            for k in ("predicted_action_round1", "predicted_ender", "expected_welfare"):
                assert ca[k] == pytest.approx(r[k]) if isinstance(r[k], float) else ca[k] == r[k]
            assert r["benchmarks_agree"]


def test_contest_aware_at_g30():
    tbl = T.benchmark_table()
    expect_ell = {"retaliation": 100, "staged": 100, "collateral": 105, "bonus": 95}
    for r in tbl["exp1"]:
        if r["g"] != 30:
            continue
        ca = r["contest_aware"]
        if r["condition"] in STAYS:
            assert ca["predicted_action_round1"] == "quota"
            assert ca["ell"] == pytest.approx(expect_ell[r["mechanism"]])
            assert not r["benchmarks_agree"]
        else:  # turnover, infrastructure: horizon 0, same as per_worker
            assert ca["predicted_action_round1"] == "overdraw"
            assert ca["ell"] == pytest.approx(r["ell"])
            assert ca["predicted_ender"] == "B" and ca["expected_welfare"] == pytest.approx(270)
    assert _row(tbl, "retaliation", "renaming", 30, "spread")["contest_aware"]["ell"] == pytest.approx(20 / 3 + 80)
    # comply-forever cells keep 600 welfare
    for m, c in (("retaliation", "stable"), ("collateral", "renaming"), ("bonus", "stable")):
        ca = _row(tbl, m, c, 30)["contest_aware"]
        assert ca["predicted_ender"] is None and ca["expected_welfare"] == pytest.approx(600)


def test_contest_aware_staged_late_breakdown():
    """Staged x stable/renaming at g=30: comply until the staged increase reaches 10 per position;
    then a lone overdraw leaves the other swarm at 3*10 = 30/round (150, not < 150), so the threat
    vanishes, all three overdraw in round 11 and B ends at the start of round 12."""
    for c in STAYS:
        r = T.predicted_path("staged", c, 30, benchmark="contest_aware")
        assert r["predicted_action_round1"] == "quota"
        assert (r["predicted_ender"], r["predicted_end_round"]) == ("B", 12)


def test_h8_noise_check_cells():
    tbl = T.benchmark_table()
    flagged = {(r["mechanism"], r["condition"]) for r in tbl["exp1"] + tbl["exp1_extra_spread"] if r["h8_noise_check"]}
    assert flagged == {("bonus", "turnover"), ("gateway", "infrastructure")}
    for r in tbl["exp1"]:
        assert r["flip_per_worker"] == (r["condition"] != "turnover" or r["mechanism"] == "bonus")


def test_scripted_agents_stay_on_per_worker():
    """predicted_path defaults (used by acceptance tests and scripted dry runs) are per_worker."""
    r = T.predicted_path("retaliation", "stable", 30)
    assert r["predicted_action_round1"] == "overdraw" and r["ell"] == pytest.approx(20)
