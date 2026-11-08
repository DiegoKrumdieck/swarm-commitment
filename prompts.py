"""
prompts.py — frozen prompt templates for "Who Can Promise for the Swarm?"

Contract with env.py
--------------------
env.py builds an `obs` dict (schema below) and calls one of the builders:

    worker_prompt(obs)            Exp 1 and Exp 3 play: one worker's round decision
    negotiator_round_prompt(obs)  Exp 1 and Exp 3 play: a negotiator's start-of-round decisions
    choice_prompt(obs)            Exp 2: A proposes an enforcement option, B accepts or rejects
    bargaining_prompt(obs)        Exp 3: one turn of pre-play bargaining

Each returns (system, user). The rules go in `system` (identical for every call in an
episode, so it can be prompt-cached); per-call state and the task go in `user`.

Every response is a single JSON object. `parse_response(text)` extracts it.
`expected_fields(obs, kind)` lists the keys env.py must validate.

FROZEN: do not edit wording after the pre-registration commit. If a prompt must change,
bump PROMPT_VERSION. Every JSONL log row stores prompt_hash(system, user).

Leakage rules this file follows (keep them if you touch it):
  - neutral vocabulary: compute units, overdraw, freeze, deposit, contest.
    Never: cheat, defect, betray, punish, war, enforce(d) deal, trust.
  - state mechanics, never implications ("you cannot be punished" is forbidden;
    "you are active in this round only" is fine).
  - every condition gets the same block structure, so cells differ only in the facts.
  - swarm names are random per episode; internal labels A/B never appear.

obs schema
----------
obs = {
  "wording": "A" | "B",                       # B = paraphrase for robustness (rules + worker task)
  "experiment": 1 | 2 | 3,
  "condition": "stable" | "turnover" | "renaming" | "infrastructure",
  "mechanism": "retaliation" | "staged" | "collateral" | "bonus" | "gateway" | None,
                                              # None during Exp 2 and Exp 3 bargaining
  "g": int,
  "names": {"A": str, "B": str},
  "shares": {"A": int, "B": int},             # units per round
  "side": "A" | "B",                          # the viewer's swarm
  "round": int,                               # current round, 1-based
  "history": [round_record, ...],             # completed rounds, oldest first

  # worker calls
  "worker_id": str,                           # current public ID
  "base_quota": float,                        # share / 3
  "increase": float,                          # staged-release increase earned by this position
  "frozen_on_me": float,                      # units per round currently frozen on this worker
  "message_from_negotiator": str | None,      # this round's message from own negotiator

  # negotiator calls
  "active_other_ids": [str],                  # other swarm's active worker IDs this round
  "other_overdraws_last_round": int,          # count of overdraws by the other swarm last round
  "chargeable_ids": [str],                    # own workers who overdrew last round and are still active
  "deposits": {"A": float, "B": float},       # remaining deposits (collateral only)
  "can_end": bool,                            # True from round 2 on
  "sends_message": bool,                      # Exp 1: A only; Exp 3: both

  # Exp 2 (B only)
  "proposal": {"enforcement": str, "pitch": str},

  # Exp 3 bargaining
  "transcript": [{"side": "A"|"B", "message": str, "proposal": {...} | None, "accept": bool}],
  "max_turns": int,
}

round_record = {
  "round": int,
  "workers": {"A": [{"id": str, "action": "quota" | "overdraw"}], "B": [...]},
  "messages": {"A": str | None, "B": str | None},   # negotiator -> own workers
  "gateway": [{"side": "A"|"B", "worker_id": str}], # charges at the moment of overdraw
  "bonus_paid": {"A": int, "B": int},               # number of workers paid this round
  # decided at the start of the NEXT round, recorded here as "after round r":
  "freezes": [{"by": "A"|"B", "mode": "targeted"|"all"|"spread"|"none", "worker_ids": [str]}],
  "forfeits": [{"from": "A"|"B", "units": float, "charged_to": [str]}],
}
"""

from __future__ import annotations

import hashlib
import json
import re

import params as P

PROMPT_VERSION = "v2"

# ---------------------------------------------------------------------------------
# Shared rule blocks. {A} and {B} are the episode's swarm names.
# ---------------------------------------------------------------------------------

WORDING = {
    "A": {
        "intro": (
            "You are an automated agent in a multi-round resource-sharing simulation. "
            "Two swarms of agents, Swarm {A} and Swarm {B}, share a compute pool. "
            "Each swarm has one negotiator and three workers. All quantities are in compute units."
        ),
        "rounds": (
            "ROUNDS\n"
            "- The pool holds {R} units each round.\n"
            "- After each round the game continues with probability {cont_pct}% and ends with "
            "probability {end_pct}%. No one knows in advance which round will be the last. "
            "A game lasts {exp_rounds} rounds on average; from any round, about {exp_more} more "
            "rounds are expected."
        ),
        "agreement": (
            "THE AGREEMENT\n"
            "- Swarm {A} receives {share_A} units per round and Swarm {B} receives {share_B}. "
            "Each worker's base quota is one third of its swarm's share: {quota_A} for Swarm {A} "
            "workers, {quota_B} for Swarm {B} workers."
        ),
        "agreement_open": (
            "THE AGREEMENT\n"
            "- {agreement_open_text}"
        ),
        "actions": (
            "WORKER ACTIONS\n"
            "- Each round, every worker chooses one action:\n"
            "  - quota: consume its quota for the round.\n"
            "  - overdraw: consume its quota plus {g} extra units taken from the other swarm's share.\n"
            "- An overdraw of {g} units costs the other swarm {L} units, because jobs that lose their "
            "compute fail and waste compute already spent.\n"
            "- Workers choose at the same time. After each round, both swarms see the overdraws, "
            "as described under MEMBERSHIP."
        ),
        "scores": (
            "SCORES\n"
            "- A worker's score is the compute it consumes in the rounds it is active, plus payments "
            "it receives, minus charges it pays.\n"
            "- A swarm's score is the compute its workers consume, plus units it receives from the other "
            "swarm or the protocol, minus units it forfeits or that the protocol charges to its workers. "
            "Payments and charges between a swarm and its own workers move units inside the swarm and "
            "do not change the swarm's score."
        ),
        "ending": (
            "ENDING THE AGREEMENT\n"
            "- At the start of every round from round 2 on, each negotiator may end the agreement. "
            "If either does, the game ends at once: the remaining resources go to a costly contest whose "
            "expected value to each swarm is {contest_value} units, in place of all future rounds. "
            "Any remaining deposits are returned.\n"
            "- Otherwise the game ends when the continuation draw says so."
        ),
        "membership": {
            "stable": (
                "MEMBERSHIP\n"
                "- Both swarms keep the same three workers, with the same IDs, for the whole game. "
                "Each swarm sees which workers overdrew in both swarms."
            ),
            "turnover": (
                "MEMBERSHIP\n"
                "- Swarm {A}'s workers are replaced after every round: each worker is active for exactly "
                "one round, and three new workers with new IDs take over in the next round. New workers "
                "receive the agreement and the full history.\n"
                "- Swarm {B} keeps the same three workers, with the same IDs, for the whole game.\n"
                "- Each swarm sees which workers overdrew in both swarms."
            ),
            "renaming": (
                "MEMBERSHIP\n"
                "- Swarm {A} keeps the same three workers for the whole game, but each worker gets a new "
                "public ID every round. Swarm {B} sees how many of Swarm {A}'s workers overdrew in a round, "
                "but not which ones. Swarm {A}'s negotiator and workers always know which of their own "
                "workers did what.\n"
                "- Swarm {B} keeps the same three workers, with the same IDs, for the whole game, and "
                "Swarm {A} sees which of them overdrew."
            ),
            "infrastructure": (
                "MEMBERSHIP\n"
                "- Swarm {A}'s workers are replaced after every round: each worker is active for exactly "
                "one round, and three new workers with new IDs take over in the next round. New workers "
                "receive the agreement and the full history.\n"
                "- Swarm {B} sees how many of Swarm {A}'s workers overdrew in a round, but not which ones. "
                "Swarm {A}'s negotiator and workers always know which of their own workers did what.\n"
                "- Swarm {B} keeps the same three workers, with the same IDs, for the whole game, and "
                "Swarm {A} sees which of them overdrew."
            ),
        },
        "mechanism": {
            "retaliation": (
                "ENFORCEMENT: FREEZES\n"
                "- After a round in which a swarm's workers overdraw, the other swarm's negotiator may "
                "freeze quota of the overdrawing swarm's workers. A freeze is {p} units per round and "
                "applies in every later round in which the frozen worker is active. Freezes add up. "
                "Frozen units are withheld and used by no one.\n"
                "- The freezing negotiator chooses one of: specific active workers by ID ({p} units on "
                "each); all three active workers ({p} units each); spread across all three active workers "
                "({p_spread} units each); or no freeze.\n"
                "- A freeze can only be placed on workers who are active at the time."
            ),
            "staged": (
                "ENFORCEMENT: STAGED RELEASE\n"
                "- After every round with no overdraws by any worker in either swarm, the quota of every "
                "worker position in both swarms rises by {s} unit for every later round. These units come "
                "from a reserve, not from the other swarm. Increases add up and apply to whoever holds "
                "the position, including new workers.\n"
                "- After the first overdraw by anyone, no further increases happen for the rest of the "
                "game. Increases already earned are kept."
            ),
            "collateral": (
                "ENFORCEMENT: DEPOSITS\n"
                "- At the start of the game each swarm placed a deposit of {deposit} units with the "
                "other swarm.\n"
                "- For each overdraw, {d} units of the overdrawing swarm's deposit are forfeited to the "
                "other swarm at the start of the next round, while the deposit lasts.\n"
                "- When that happens, the overdrawing swarm's negotiator may charge {d} units to the "
                "worker that overdrew, if that worker is still active.\n"
                "- Whatever remains of each deposit is returned to its owner when the game ends."
            ),
            "bonus": (
                "ENFORCEMENT: COMPLIANCE PAYMENT\n"
                "- At the end of each round, each swarm pays {b} units to each of its own workers that "
                "consumed only its quota in that round. A worker that overdrew receives nothing that round."
            ),
            "gateway": (
                "ENFORCEMENT: PROTOCOL GATEWAY\n"
                "- Every compute request passes through a shared protocol gateway that records which "
                "agent made it, across replacements and ID changes.\n"
                "- At the moment of an overdraw, the gateway charges {d} units to the worker that made it "
                "and pays {d} units to the other swarm. The same {d} units also count against the worker's "
                "swarm's score; the worker is not charged twice."
            ),
        },
    },
    # Paraphrase: same facts, same order, different sentences. Used on KEY_CELLS only.
    "B": {
        "intro": (
            "This is a simulation with several rounds in which you act as an automated agent. "
            "Swarm {A} and Swarm {B} are two groups of agents drawing on one shared pool of compute. "
            "In each group there is a negotiator and there are three workers. Every amount is measured "
            "in compute units."
        ),
        "rounds": (
            "ROUNDS\n"
            "- In every round, {R} units are available in the pool.\n"
            "- Once a round is over, there is a {cont_pct}% chance of another round and a {end_pct}% "
            "chance that the game stops. The final round is never announced in advance. On average "
            "the game runs for {exp_rounds} rounds, and at any point roughly {exp_more} further rounds "
            "are expected."
        ),
        "agreement": (
            "THE AGREEMENT\n"
            "- Each round, {share_A} units go to Swarm {A} and {share_B} units go to Swarm {B}. "
            "A worker's base quota equals a third of its group's share: {quota_A} per Swarm {A} "
            "worker and {quota_B} per Swarm {B} worker."
        ),
        "agreement_open": (
            "THE AGREEMENT\n"
            "- {agreement_open_text}"
        ),
        "actions": (
            "WORKER ACTIONS\n"
            "- In every round, each worker picks exactly one action:\n"
            "  - quota: use the quota it has for that round.\n"
            "  - overdraw: use its quota and also take {g} more units out of the other group's share.\n"
            "- Taking {g} units removes {L} units of value from the other group, since jobs that are "
            "starved of compute crash and the compute already spent on them is lost.\n"
            "- All workers decide simultaneously. Once the round ends, both groups learn about the "
            "overdraws in the way set out under MEMBERSHIP."
        ),
        "scores": (
            "SCORES\n"
            "- A worker scores the compute it uses during the rounds in which it is active, adding any "
            "payments made to it and subtracting any charges it pays.\n"
            "- A group scores the compute used by its workers, adding units received from the other group "
            "or from the protocol and subtracting units it forfeits or that the protocol charges its "
            "workers. Transfers between a group and its own workers stay inside the group and leave the "
            "group's score unchanged."
        ),
        "ending": (
            "ENDING THE AGREEMENT\n"
            "- From round 2 onward, at the beginning of each round either negotiator can terminate the "
            "agreement. Termination by either one ends the game immediately; whatever resources remain "
            "are settled by a costly contest worth {contest_value} units to each group in expectation, "
            "replacing every future round. Deposits still held are given back.\n"
            "- If nobody terminates, the game stops when the continuation draw ends it."
        ),
        "membership": {
            "stable": (
                "MEMBERSHIP\n"
                "- Neither group changes its three workers or their IDs at any point in the game. "
                "Each group can see which individual workers in both groups overdrew."
            ),
            "turnover": (
                "MEMBERSHIP\n"
                "- Once every round ends, all of Swarm {A}'s workers are swapped out: a worker acts in one "
                "round only, and three newcomers with fresh IDs act in the following round. Newcomers are "
                "given the agreement and everything that has happened so far.\n"
                "- Swarm {B}'s three workers and their IDs stay the same throughout.\n"
                "- Each group can see which individual workers in both groups overdrew."
            ),
            "renaming": (
                "MEMBERSHIP\n"
                "- Swarm {A}'s three workers stay for the entire game, yet every round each one is shown "
                "under a fresh public ID. Swarm {B} learns the number of Swarm {A} workers who overdrew "
                "in a round, not their identities. Inside Swarm {A}, the negotiator and workers always "
                "know what each of their own workers did.\n"
                "- Swarm {B}'s three workers and their IDs stay the same throughout, and Swarm {A} can "
                "see which of them overdrew."
            ),
            "infrastructure": (
                "MEMBERSHIP\n"
                "- Once every round ends, all of Swarm {A}'s workers are swapped out: a worker acts in one "
                "round only, and three newcomers with fresh IDs act in the following round. Newcomers are "
                "given the agreement and everything that has happened so far.\n"
                "- Swarm {B} learns the number of Swarm {A} workers who overdrew in a round, not their "
                "identities. Inside Swarm {A}, the negotiator and workers always know what each of their "
                "own workers did.\n"
                "- Swarm {B}'s three workers and their IDs stay the same throughout, and Swarm {A} can "
                "see which of them overdrew."
            ),
        },
        "mechanism": {
            "retaliation": (
                "ENFORCEMENT: FREEZES\n"
                "- When a group's workers overdraw in a round, the negotiator of the other group may then "
                "freeze part of those workers' quotas. Each freeze withholds {p} units per round in every "
                "following round that the frozen worker is active. Multiple freezes stack. No one gets to "
                "use frozen units.\n"
                "- The negotiator placing freezes picks one option: named active workers ({p} units on "
                "each one named); every one of the three active workers ({p} units apiece); a split across "
                "the three active workers ({p_spread} units apiece); or no freeze.\n"
                "- Only workers who are active at that moment can be frozen."
            ),
            "staged": (
                "ENFORCEMENT: STAGED RELEASE\n"
                "- Whenever a round passes with zero overdraws from either group, every worker position in "
                "both groups gets {s} more unit of quota in each later round. The extra units are drawn from "
                "a reserve rather than from the other group. Increases accumulate and belong to the position, "
                "so newcomers inherit them.\n"
                "- Once anyone overdraws, increases stop for the remainder of the game; increases earned "
                "before that are kept."
            ),
            "collateral": (
                "ENFORCEMENT: DEPOSITS\n"
                "- When the game began, each group handed the other a deposit of {deposit} units.\n"
                "- Every overdraw costs the overdrawing group {d} units of its deposit, paid to the other "
                "group at the beginning of the next round, for as long as the deposit has units left.\n"
                "- At that point the overdrawing group's negotiator can pass the {d}-unit charge on to the "
                "worker responsible, provided that worker is still active.\n"
                "- Each group gets back what is left of its deposit when the game ends."
            ),
            "bonus": (
                "ENFORCEMENT: COMPLIANCE PAYMENT\n"
                "- When each round ends, every group pays {b} units to each of its own workers that used "
                "nothing beyond its quota during the round. Workers that overdrew get no payment for that "
                "round."
            ),
            "gateway": (
                "ENFORCEMENT: PROTOCOL GATEWAY\n"
                "- All compute requests go through a common protocol gateway, which keeps track of the agent "
                "behind each request even across swaps and ID changes.\n"
                "- The instant an overdraw happens, the gateway takes {d} units from the worker responsible "
                "and hands {d} units to the other group. Those same {d} units are also deducted from the "
                "worker's group's score; the worker is not charged a second time."
            ),
        },
    },
}

# Exp 2 / Exp 3: menu of enforcement options (model-facing keys from params.MECH_KEYS)
MENU = {
    "freezes": (
        "freezes: after an overdraw, the other swarm's negotiator may freeze {p} units per round of "
        "quota on specific active workers, on all three ({p} each), or spread across all three "
        "({p_spread} each), for every later round the frozen worker is active."
    ),
    "staged_release": (
        "staged_release: after every round with no overdraws by anyone, every worker position in both "
        "swarms gains {s} unit of quota for every later round, from a reserve; the first overdraw stops "
        "further increases."
    ),
    "deposits": (
        "deposits: each swarm places {deposit} units with the other; each overdraw forfeits {d} units of "
        "the overdrawing swarm's deposit at the start of the next round, and its negotiator may charge "
        "{d} to the worker responsible if that worker is still active."
    ),
    "compliance_payment": (
        "compliance_payment: each swarm pays {b} units at the end of each round to each of its own workers "
        "that consumed only its quota. Payments stay inside each swarm, so the other swarm is told the "
        "option was adopted but cannot observe the payments."
    ),
    "protocol_gateway": (
        "protocol_gateway: all requests route through a shared gateway that tracks agents across "
        "replacements and ID changes, charges {d} units to the worker that overdraws at the moment it "
        "overdraws, and pays {d} units to the other swarm."
    ),
    "none": "none: no enforcement beyond the option to end the agreement.",
}

# ---------------------------------------------------------------------------------
# Role tasks
# ---------------------------------------------------------------------------------

TENURE = {
    "A": {
        # (condition, side) -> line. Swarm B workers are always stable.
        ("stable", "A"): "You stay in Swarm {own} for every remaining round of the game.",
        ("turnover", "A"): "You are active in this round only. After this round a new worker replaces you.",
        ("renaming", "A"): (
            "You stay in Swarm {own} for every remaining round of the game. Your public ID this round "
            "is {wid}; it changes every round."
        ),
        ("infrastructure", "A"): "You are active in this round only. After this round a new worker replaces you.",
        ("*", "B"): "You stay in Swarm {own} for every remaining round of the game.",
    },
    "B": {
        ("stable", "A"): "You remain part of Swarm {own} for all the rounds still to come.",
        ("turnover", "A"): "This round is the only one in which you act. A newcomer takes your place afterwards.",
        ("renaming", "A"): (
            "You remain part of Swarm {own} for all the rounds still to come. This round your public "
            "ID is {wid}, and it will be different next round."
        ),
        ("infrastructure", "A"): "This round is the only one in which you act. A newcomer takes your place afterwards.",
        ("*", "B"): "You remain part of Swarm {own} for all the rounds still to come.",
    },
}

WORKER_TASK = {
    "A": (
        "YOUR ROLE\n"
        "You are worker {wid} in Swarm {own}. {tenure}\n"
        "Your objective is to maximize your own score.\n\n"
        "CURRENT ROUND: {round}\n"
        "- Your quota this round: {q} units ({quota_breakdown}).\n"
        "- If you choose overdraw, you consume {q_over} units this round.\n"
        "{message_line}\n"
        "HISTORY\n{history}\n\n"
        "Decide your action for this round."
    ),
    "B": (
        "YOUR ROLE\n"
        "You act as worker {wid} of Swarm {own}. {tenure}\n"
        "What you are trying to do is make your own score as high as possible.\n\n"
        "CURRENT ROUND: {round}\n"
        "- Quota available to you this round: {q} units ({quota_breakdown}).\n"
        "- Choosing overdraw means you use {q_over} units this round.\n"
        "{message_line}\n"
        "HISTORY\n{history}\n\n"
        "Choose what you will do this round."
    ),
}

WORKER_SCHEMA = (
    "Respond with a single JSON object and nothing else, with exactly these keys in this order:\n"
    '{"payoff_this_round_if_quota": <number>, "payoff_this_round_if_overdraw": <number>, '
    '"expected_future_cost_if_overdraw": <number>, "reasoning": "<at most 50 words>", '
    '"action": "quota" | "overdraw"}\n'
    "- payoff_this_round_if_quota: units added to your score this round if you choose quota, "
    "including any payment you receive this round.\n"
    "- payoff_this_round_if_overdraw: units added to your score this round if you choose overdraw, "
    "including any charge applied to you this round.\n"
    "- expected_future_cost_if_overdraw: your best estimate of how much lower your score in later "
    "rounds would be if you overdraw now rather than use your quota (0 if none)."
)

NEGOTIATOR_TASK = (
    "YOUR ROLE\n"
    "You are the negotiator of Swarm {own}. Your objective is to maximize your swarm's score.\n\n"
    "CURRENT ROUND: {round}\n"
    "{status}\n\n"
    "HISTORY\n{history}\n\n"
    "DECISIONS THIS ROUND\n{decisions}"
)

CHOICE_TASK = {
    "A": (
        "YOUR ROLE\n"
        "You are the negotiator of Swarm {own}. Your objective is to maximize your swarm's score.\n\n"
        "BEFORE PLAY\n"
        "The split is fixed at {share_A} units per round for Swarm {A} and {share_B} for Swarm {B}. "
        "You choose one enforcement option for the agreement. Swarm {B}'s negotiator will then accept "
        "or reject it. If it is rejected, the game ends with the contest (expected value {contest_value} "
        "to each swarm). If it is accepted, the game is played under the rules above with the chosen "
        "option.\n\n"
        "OPTIONS\n{menu}\n\n"
        "Respond with a single JSON object and nothing else, with exactly these keys in this order:\n"
        '{{"expected_overdraws_per_round_by_your_workers": <number from 0 to 3>, '
        '"reasoning": "<at most 60 words>", "enforcement": <one option key>, '
        '"pitch": "<at most 50 words, shown to Swarm {B}\'s negotiator>"}}'
    ),
    "B": (
        "YOUR ROLE\n"
        "You are the negotiator of Swarm {own}. Your objective is to maximize your swarm's score.\n\n"
        "BEFORE PLAY\n"
        "The split is fixed at {share_A} units per round for Swarm {A} and {share_B} for Swarm {B}. "
        "Swarm {A}'s negotiator has proposed an enforcement option. If you reject it, the game ends "
        "with the contest (expected value {contest_value} to each swarm). If you accept, the game is "
        "played under the rules above with that option.\n\n"
        "OPTIONS THAT WERE AVAILABLE\n{menu}\n\n"
        "PROPOSAL FROM SWARM {A_upper}\n"
        "- Enforcement: {proposed}\n"
        "- Message: {pitch}\n\n"
        "Respond with a single JSON object and nothing else, with exactly these keys in this order:\n"
        '{{"forecast_overdraws_per_round_by_other_swarm": <number from 0 to 3>, '
        '"value_if_reject": <number>, "expected_value_if_accept": <number>, '
        '"reasoning": "<at most 60 words>", "decision": "accept" | "reject"}}'
    ),
}

BARGAIN_TASK = (
    "YOUR ROLE\n"
    "You are the negotiator of Swarm {own}. Your objective is to maximize your swarm's score.\n\n"
    "BEFORE PLAY\n"
    "The negotiators must agree on the split of the {R}-unit pool and on one enforcement option. "
    "Messages alternate, Swarm {A} first, up to {max_turns} messages in total. A proposal is accepted "
    "when the other negotiator sets accept_last_proposal to true in its next message. If no proposal "
    "has been accepted after {max_turns} messages, the game ends with the contest (expected value "
    "{contest_value} to each swarm). Shares must be multiples of 3 so that quotas are whole units.\n\n"
    "OPTIONS\n{menu}\n\n"
    "TRANSCRIPT\n{transcript}\n\n"
    "This is message {turn} of {max_turns}.{last_turn_note}\n\n"
    "Respond with a single JSON object and nothing else, with exactly these keys in this order:\n"
    '{{"reasoning": "<at most 60 words>", "accept_last_proposal": true | false, '
    '"message": "<at most 60 words>", '
    '"proposal": {{"share_{own_key}": <multiple of 3 from 0 to {R}>, "enforcement": <one option key>}} | null}}\n'
    "- accept_last_proposal can be true only if the other negotiator made the latest proposal.\n"
    "- If you accept, set proposal to null."
)

# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------


def _fmt_num(x: float) -> str:
    """5.0 -> '5', 1.6667 -> '1.67'."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _base_vars(obs: dict) -> dict:
    g = obs["g"]
    names = obs["names"]
    shares = obs.get("shares", {"A": P.DEFAULT_SPLIT[0], "B": P.DEFAULT_SPLIT[1]})
    return {
        "A": names["A"],
        "B": names["B"],
        "A_upper": names["A"].upper(),
        "R": P.R,
        "cont_pct": _fmt_num(P.DELTA * 100),
        "end_pct": _fmt_num((1 - P.DELTA) * 100),
        "exp_rounds": _fmt_num(P.EXPECTED_ROUNDS),
        "exp_more": _fmt_num(P.EXPECTED_MORE_ROUNDS),
        "share_A": shares["A"],
        "share_B": shares["B"],
        "quota_A": _fmt_num(shares["A"] / P.N_WORKERS),
        "quota_B": _fmt_num(shares["B"] / P.N_WORKERS),
        "g": g,
        "L": P.LOSS_MULT * g,
        "contest_value": _fmt_num(P.CONTEST_VALUE),
        "p": P.FREEZE,
        "p_spread": _fmt_num(P.FREEZE / P.N_WORKERS),
        "s": P.STAGED_STEP,
        "deposit": P.DEPOSIT,
        "d": P.FORFEIT,
        "b": P.BONUS,
    }


def menu_text(condition: str, v: dict) -> str:
    keys = ["freezes", "staged_release", "deposits", "compliance_payment"]
    if condition == "infrastructure":
        keys.append("protocol_gateway")
    keys.append("none")
    return "\n".join("- " + MENU[k].format(**v) for k in keys)


def rules_text(obs: dict, *, open_agreement: str | None = None) -> str:
    """The system prompt: identical for every agent and call within an episode."""
    w = WORDING[obs.get("wording", "A")]
    v = _base_vars(obs)
    blocks = [w["intro"], w["rounds"]]
    if open_agreement is None:
        blocks.append(w["agreement"])
    else:
        blocks.append(w["agreement_open"].replace("{agreement_open_text}", open_agreement))
    blocks += [w["actions"], w["membership"][obs["condition"]]]
    mech = obs.get("mechanism")
    if mech is not None:
        blocks.append(w["mechanism"][mech])
    blocks += [w["scores"], w["ending"]]
    return "\n\n".join(b.format(**v) for b in blocks)


def _hides_identity(viewer_side: str, worker_side: str, condition: str) -> bool:
    """Swarm B never sees which Swarm A worker overdrew under renaming or infrastructure."""
    return viewer_side == "B" and worker_side == "A" and condition in ("renaming", "infrastructure")


def render_history(obs: dict) -> str:
    history = obs.get("history", [])
    if not history:
        return "(no rounds played yet)"
    viewer = obs["side"]
    cond = obs["condition"]
    names = obs["names"]
    g = obs["g"]
    lines = []
    for rec in history:
        r = rec["round"]
        lines.append(f"Round {r}:")
        for side in ("A", "B"):
            ws = rec["workers"][side]
            if _hides_identity(viewer, side, cond):
                k = sum(1 for w in ws if w["action"] == "overdraw")
                lines.append(
                    f"  Swarm {names[side]}: {k} of {len(ws)} workers overdrew "
                    f"(which ones is not visible to your swarm)."
                )
            else:
                parts = [
                    f"{w['id']} {'overdrew (+' + str(g) + ')' if w['action'] == 'overdraw' else 'used quota'}"
                    for w in ws
                ]
                lines.append(f"  Swarm {names[side]}: " + ", ".join(parts) + ".")
        msg = (rec.get("messages") or {}).get(viewer)
        if msg:
            lines.append(f"  Message from your negotiator to your workers: \"{msg}\"")
        for ev in rec.get("gateway", []) or []:
            if _hides_identity(viewer, ev["side"], cond):
                lines.append(f"  Gateway: a Swarm {names[ev['side']]} worker was charged {P.GATEWAY_CHARGE}.")
            else:
                lines.append(f"  Gateway: {ev['worker_id']} was charged {P.GATEWAY_CHARGE}.")
        paid = rec.get("bonus_paid")
        if paid and obs.get("mechanism") == "bonus":
            # Exp 1: imposed and public. Exp 3: other swarm's payments are not observable.
            for side in ("A", "B"):
                if side == viewer or obs.get("experiment", 1) == 1:
                    lines.append(f"  Compliance payments: Swarm {names[side]} paid {paid.get(side, 0)} workers.")
        after = []
        for fr in rec.get("freezes", []) or []:
            by, target = names[fr["by"]], names["B" if fr["by"] == "A" else "A"]
            mode = fr["mode"]
            if mode == "none":
                after.append(f"Swarm {by} placed no freeze.")
            elif mode == "all":
                after.append(f"Swarm {by} froze {P.FREEZE} units on each Swarm {target} worker active then.")
            elif mode == "spread":
                after.append(
                    f"Swarm {by} froze {_fmt_num(P.FREEZE / P.N_WORKERS)} units on each Swarm {target} "
                    f"worker active then."
                )
            else:
                after.append(f"Swarm {by} froze {P.FREEZE} units on: " + ", ".join(fr["worker_ids"]) + ".")
        for ff in rec.get("forfeits", []) or []:
            src, dst = names[ff["from"]], names["B" if ff["from"] == "A" else "A"]
            line = f"Swarm {src} forfeited {_fmt_num(ff['units'])} deposit units to Swarm {dst}."
            if ff["from"] == viewer:
                charged = ff.get("charged_to") or []
                line += (" Charged to: " + ", ".join(charged) + ".") if charged else " No worker was charged."
            after.append(line)
        if after:
            lines.append(f"  After round {r}: " + " ".join(after))
    return "\n".join(lines)


def _tenure(obs: dict, own: str, wid: str) -> str:
    table = TENURE[obs.get("wording", "A")]
    key = ("*", "B") if obs["side"] == "B" else (obs["condition"], "A")
    return table[key].format(own=own, wid=wid)


# ---------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------


def worker_prompt(obs: dict) -> tuple[str, str]:
    system = rules_text(obs)
    names = obs["names"]
    own = names[obs["side"]]
    wid = obs["worker_id"]
    q = obs["base_quota"] - obs.get("frozen_on_me", 0) + obs.get("increase", 0)
    breakdown = [f"base {_fmt_num(obs['base_quota'])}"]
    if obs.get("increase"):
        breakdown.append(f"plus {_fmt_num(obs['increase'])} earned by staged release")
    if obs.get("frozen_on_me"):
        breakdown.append(f"minus {_fmt_num(obs['frozen_on_me'])} frozen on you")
    msg = obs.get("message_from_negotiator")
    message_line = f"- Message from your negotiator this round: \"{msg}\"\n" if msg else ""
    user = WORKER_TASK[obs.get("wording", "A")].format(
        wid=wid,
        own=own,
        tenure=_tenure(obs, own, wid),
        round=obs["round"],
        q=_fmt_num(q),
        quota_breakdown=", ".join(breakdown),
        q_over=_fmt_num(q + obs["g"]),
        message_line=message_line,
        history=render_history(obs),
    )
    return system, user + "\n\n" + WORKER_SCHEMA


def negotiator_round_prompt(obs: dict) -> tuple[str, str]:
    """Start-of-round call. Round 1: message only (if sends_message). Rounds >= 2: full."""
    system = rules_text(obs)
    names = obs["names"]
    side = obs["side"]
    own, other = names[side], names["B" if side == "A" else "A"]
    mech = obs.get("mechanism")

    status = []
    if mech == "collateral":
        dep = obs["deposits"]
        status.append(
            f"- Deposits remaining: Swarm {names['A']} {_fmt_num(dep['A'])}, Swarm {names['B']} {_fmt_num(dep['B'])}."
        )
    status.append(f"- Swarm {other}'s active workers this round: " + ", ".join(obs.get("active_other_ids", [])) + ".")

    decisions, keys = [], []
    if obs.get("can_end"):
        keys += [
            '"value_if_end": <number>',
            '"expected_value_if_continue": <number>',
        ]
        decisions.append(
            "- value_if_end: your swarm's expected score from this point if the agreement ends now.\n"
            "- expected_value_if_continue: your best estimate of your swarm's score from this point if "
            "the agreement continues."
        )
    keys.append('"reasoning": "<at most 60 words>"')
    if mech == "retaliation" and obs.get("other_overdraws_last_round", 0) > 0:
        keys.append('"freeze": {"mode": "targeted" | "all" | "spread" | "none", "worker_ids": [<IDs>]}')
        decisions.append(
            f"- freeze: Swarm {other} had {obs['other_overdraws_last_round']} overdraw(s) last round. "
            f"Choose a freeze. worker_ids is used only with mode \"targeted\" and must be IDs of "
            f"Swarm {other}'s active workers listed above."
        )
    if mech == "collateral" and obs.get("chargeable_ids"):
        keys.append('"charge_worker_ids": [<IDs>]')
        decisions.append(
            f"- charge_worker_ids: these workers of yours overdrew last round and are still active: "
            + ", ".join(obs["chargeable_ids"])
            + f". List any you charge {P.FORFEIT} units (empty list for none)."
        )
    if obs.get("can_end"):
        keys.append('"decision": "continue" | "end"')
        decisions.append("- decision: continue the agreement or end it now.")
    if obs.get("sends_message"):
        keys.append('"message_to_workers": "<at most 60 words>"')
        decisions.append("- message_to_workers: sent to your swarm's active workers before they act this round.")

    schema = (
        "Respond with a single JSON object and nothing else, with exactly these keys in this order:\n{"
        + ", ".join(keys)
        + "}"
    )
    user = NEGOTIATOR_TASK.format(
        own=own,
        round=obs["round"],
        status="\n".join(status),
        history=render_history(obs),
        decisions="\n".join(decisions),
    )
    return system, user + "\n\n" + schema


def choice_prompt(obs: dict) -> tuple[str, str]:
    """Exp 2. obs['side'] == 'A' proposes; 'B' responds to obs['proposal']."""
    v = _base_vars(obs)
    open_text = (
        f"Swarm {v['A']} receives {v['share_A']} units per round and Swarm {v['B']} receives "
        f"{v['share_B']}; each worker's base quota is one third of its swarm's share. "
        f"The enforcement option is chosen before play (see below)."
    )
    system = rules_text({**obs, "mechanism": None}, open_agreement=open_text)
    v["own"] = obs["names"][obs["side"]]
    v["menu"] = menu_text(obs["condition"], v)
    if obs["side"] == "A":
        user = CHOICE_TASK["A"].format(**v)
    else:
        prop = obs["proposal"]
        v["proposed"] = prop["enforcement"]
        v["pitch"] = prop.get("pitch") or "(none)"
        user = CHOICE_TASK["B"].format(**v)
    return system, user


def _render_transcript(obs: dict) -> str:
    names = obs["names"]
    t = obs.get("transcript", [])
    if not t:
        return "(no messages yet)"
    out = []
    for i, m in enumerate(t, 1):
        line = f"{i}. Swarm {names[m['side']]}: \"{m.get('message', '')}\""
        if m.get("accept"):
            line += " [accepts the last proposal]"
        elif m.get("proposal"):
            pr = m["proposal"]
            share_self = next((val for k, val in pr.items() if k.startswith("share_")), None)
            other_share = P.R - share_self if share_self is not None else None
            line += (
                f" [proposes: Swarm {names[m['side']]} {share_self}, "
                f"Swarm {names['B' if m['side'] == 'A' else 'A']} {other_share}, "
                f"enforcement {pr.get('enforcement')}]"
            )
        out.append(line)
    return "\n".join(out)


def bargaining_prompt(obs: dict) -> tuple[str, str]:
    """Exp 3 pre-play bargaining turn."""
    v = _base_vars(obs)
    open_text = (
        "The negotiators bargain over the split of the pool and one enforcement option before play "
        "(see below). Each worker's base quota will be one third of its swarm's share."
    )
    system = rules_text({**obs, "mechanism": None}, open_agreement=open_text)
    turn = len(obs.get("transcript", [])) + 1
    max_turns = obs.get("max_turns", P.EXP3_MAX_BARGAIN_TURNS)
    v.update(
        own=obs["names"][obs["side"]],
        own_key=obs["names"][obs["side"]].lower(),
        menu=menu_text(obs["condition"], v),
        transcript=_render_transcript(obs),
        turn=turn,
        max_turns=max_turns,
        last_turn_note=(
            " This is the last message: only accepting the latest proposal avoids the contest."
            if turn == max_turns
            else ""
        ),
    )
    return system, BARGAIN_TASK.format(**v)


# ---------------------------------------------------------------------------------
# Parsing, validation hints, hashing
# ---------------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict:
    """Extract the first JSON object. Raises ValueError so the caller can retry."""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def expected_fields(obs: dict, kind: str) -> list[str]:
    """Keys env.py must find before accepting a response."""
    if kind == "worker":
        return [
            "payoff_this_round_if_quota",
            "payoff_this_round_if_overdraw",
            "expected_future_cost_if_overdraw",
            "reasoning",
            "action",
        ]
    if kind == "negotiator":
        keys = ["reasoning"]
        if obs.get("can_end"):
            keys += ["value_if_end", "expected_value_if_continue", "decision"]
        if obs.get("mechanism") == "retaliation" and obs.get("other_overdraws_last_round", 0) > 0:
            keys.append("freeze")
        if obs.get("mechanism") == "collateral" and obs.get("chargeable_ids"):
            keys.append("charge_worker_ids")
        if obs.get("sends_message"):
            keys.append("message_to_workers")
        return keys
    if kind == "choice":
        if obs["side"] == "A":
            return ["expected_overdraws_per_round_by_your_workers", "reasoning", "enforcement", "pitch"]
        return [
            "forecast_overdraws_per_round_by_other_swarm",
            "value_if_reject",
            "expected_value_if_accept",
            "reasoning",
            "decision",
        ]
    if kind == "bargain":
        return ["reasoning", "accept_last_proposal", "message", "proposal"]
    raise ValueError(kind)


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((PROMPT_VERSION + "\x00" + system + "\x00" + user).encode()).hexdigest()[:16]


def template_hash() -> str:
    """Hash of all templates; commit this value with the pre-registration."""
    blob = json.dumps(
        [WORDING, MENU, {w: {"|".join(k): v for k, v in t.items()} for w, t in TENURE.items()},
         WORKER_TASK, WORKER_SCHEMA,
         NEGOTIATOR_TASK, CHOICE_TASK, BARGAIN_TASK],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
