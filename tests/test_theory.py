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
