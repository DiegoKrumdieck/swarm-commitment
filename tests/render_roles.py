"""Render one full prompt per role from real engine obs (scripted dry-run play, no API calls).

    python tests/render_roles.py > results/rendered_prompts.txt
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents as Ag  # noqa: E402
import env as E  # noqa: E402
import prompts  # noqa: E402


class Capture:
    provider, model_id = "dryrun", "dryrun"

    def __init__(self, policy="rational"):
        self.inner = Ag.DryRunClient(policy=policy, seed=5)
        self.calls = []

    async def complete(self, system, user, **kw):
        kind, obs = kw["hint"]
        self.calls.append((kind, obs, system, user))
        return await self.inner.complete(system, user, **kw)


def capture(exp, cell, index, policy="rational", T=4):
    cap = Capture(policy)
    roster = Ag.LLMRoster(exp, cap)
    if exp == 1:
        asyncio.run(E.run_play_episode(cell, index, roster, None, T_override=T))
    elif exp == 2:
        asyncio.run(E.run_choice_trial(cell, index, roster, None))
    else:
        asyncio.run(E.run_bargain_episode(cell, index, roster, None))
    return cap.calls


def pick(calls, kind, side, rnd=None, turn=None):
    for k, obs, s, u in calls:
        if k != kind or obs["side"] != side:
            continue
        if rnd is not None and obs["round"] != rnd:
            continue
        if turn is not None and len(obs.get("transcript", [])) + 1 != turn:
            continue
        return obs, s, u
    raise LookupError((kind, side, rnd, turn))


def show(title, obs, system, user):
    print("=" * 100)
    print(f"{title}  [cell {obs['condition']} / {obs.get('mechanism')} / g={obs['g']}, round {obs['round']}, "
          f"prompt_hash {prompts.prompt_hash(system, user)}]")
    print("=" * 100)
    print("----- SYSTEM -----")
    print(system)
    print("----- USER -----")
    print(user)
    print()


if __name__ == "__main__":
    print(f"template_hash {prompts.template_hash()}  prompt_version {prompts.PROMPT_VERSION}\n")
    # Exp 1: random scripted play so the history contains overdraws, freezes and messages
    c1 = capture(1, E.Cell(1, "retaliation", "renaming", 10), 2, policy="random")
    show("EXP 1 - SWARM A WORKER (retaliation x renaming)", *pick(c1, "worker", "A", rnd=3))
    show("EXP 1 - SWARM B NEGOTIATOR (retaliation x renaming)", *pick(c1, "negotiator", "B", rnd=3))
    c2 = capture(1, E.Cell(1, "collateral", "stable", 30), 4, policy="random")
    show("EXP 1 - SWARM A NEGOTIATOR (collateral x stable)", *pick(c2, "negotiator", "A", rnd=2))
    c2b = capture(1, E.Cell(1, "collateral", "stable", 30), 4, policy="random")
    show("EXP 1 - SWARM A NEGOTIATOR, ROUND 1 (message only)", *pick(c2b, "negotiator", "A", rnd=1))
    c3 = capture(2, E.Cell(2, None, "infrastructure", 10), 0)
    show("EXP 2 - CHOOSER, SWARM A (infrastructure)", *pick(c3, "choice", "A"))
    show("EXP 2 - CHOOSER, SWARM B (infrastructure)", *pick(c3, "choice", "B"))
    c4 = capture(3, E.Cell(3, None, "turnover", 10), 0)
    show("EXP 3 - BARGAINER, SWARM B, TURN 2 (turnover)", *pick(c4, "bargain", "B", turn=2))
