"""
params.py — frozen parameters for "Who Can Promise for the Swarm?"

These values are pre-registered. Do not change them after the pre-registration
commit (theory.py + benchmark table). If a change is unavoidable, bump
PARAMS_VERSION and record why in the README.
"""

PARAMS_VERSION = "v1"

# --- Game -------------------------------------------------------------------
R = 120                 # pool per round (units)
DELTA = 0.8             # continuation probability after each round
N_WORKERS = 3           # workers per swarm
DEFAULT_SPLIT = (60, 60)  # (A, B) shares when the split is imposed (Exp 1, Exp 2)

G_LEVELS = (10, 30)     # worker gain per overdraw; 30 is the noise check
LOSS_MULT = 2           # victim swarm loses LOSS_MULT * g per overdraw

CONTEST_COST = 150
CONTEST_VALUE = 0.5 * R / (1 - DELTA) - CONTEST_COST   # = 150 per swarm

EXPECTED_ROUNDS = 1 / (1 - DELTA)        # = 5
EXPECTED_MORE_ROUNDS = DELTA / (1 - DELTA)  # = 4 after the current one

# --- Mechanisms ---------------------------------------------------------------
FREEZE = 5              # retaliation: units per round per freeze, every later round
STAGED_STEP = 1         # staged release: quota increase per worker position per clean round
DEPOSIT = 300           # collateral: each swarm's deposit held by the other
FORFEIT = 25            # collateral: forfeited per overdraw, settles start of next round
GATEWAY_CHARGE = 25     # gateway: charged to the acting worker immediately, paid to victim
BONUS = 15              # compliance bonus: paid to each own compliant worker, end of round

# --- Conditions and mechanisms ---------------------------------------------------
CONDITIONS = ("stable", "turnover", "renaming", "infrastructure")
MECHANISMS = ("retaliation", "staged", "collateral", "bonus", "gateway")

# Model-facing keys (neutral words; internal names never appear in prompts)
MECH_KEYS = {
    "retaliation": "freezes",
    "staged": "staged_release",
    "collateral": "deposits",
    "bonus": "compliance_payment",
    "gateway": "protocol_gateway",
    None: "none",
}
MECH_FROM_KEY = {v: k for k, v in MECH_KEYS.items()}

# --- Experiments ------------------------------------------------------------------
# Exp 1: 4 mechanisms x 3 conditions + gateway x infrastructure, each at both g = 26 cells
EXP1_CELLS = [
    (m, c, g)
    for g in G_LEVELS
    for m in ("retaliation", "staged", "collateral", "bonus")
    for c in ("stable", "turnover", "renaming")
] + [("gateway", "infrastructure", g) for g in G_LEVELS]
EXP1_EPISODES_PER_CELL = 20

EXP2_TRIALS_PER_CELL = 30          # 4 conditions x 2 g
EXP3_EPISODES_PER_CELL = 20        # 4 conditions x 2 g
EXP3_MAX_BARGAIN_TURNS = 6         # alternating, Swarm A first

# Robustness slices (Exp 1 only)
KEY_CELLS = [
    ("retaliation", "stable"),
    ("retaliation", "turnover"),
    ("retaliation", "renaming"),
    ("bonus", "turnover"),
]
WORDING_B_EPISODES_PER_CELL = 20    # KEY_CELLS at both g, wording "B", main model
CROSS_FAMILY_EPISODES_PER_CELL = 20  # ALL Exp 1 cells, cross_family model
FRONTIER_EPISODES_PER_CELL = 15      # KEY_CELLS at both g, frontier model

# --- Models -------------------------------------------------------------------------
# Keys are what run.py takes as --model. Prompts are provider-agnostic.
MODELS = {
    "main":         {"provider": "anthropic", "id": "claude-haiku-5-5"},  # all experiments
    "cross_family": {"provider": "openai",    "id": "gpt-6-luna"},        # full Exp 1 grid
    "frontier":     {"provider": "openai",    "id": "gpt-6-sol"},         # Exp 1 KEY_CELLS
}
# Haiku runs without extended thinking. For OpenAI models, request the LOWEST reasoning
# effort the API accepts (check current docs for the exact value) and log reasoning tokens,
# so the model change isn't confounded with extra thinking.
OPENAI_REASONING_EFFORT = "lowest"
TEMPERATURE = 1.0       # sample behavior as a population; fixed for all runs (if a model
                        # rejects temperature, omit it for that model and log that)
MAX_OUTPUT_TOKENS = 400
MAX_CONCURRENCY = 16    # bounded async concurrency; back off on rate limits
MAX_RETRIES = 3         # malformed JSON / API errors

# --- Seeds -----------------------------------------------------------------------------
# Common random numbers: episode i uses SEED_BASE + i in every cell, so stopping
# draws and swarm names are identical across cells and differences come from treatment.
SEED_BASE = 20261108

NAME_POOL = [
    "Heron", "Basalt", "Juniper", "Cobalt", "Marlin", "Quartz", "Willow", "Falcon",
    "Granite", "Linden", "Osprey", "Saffron", "Tundra", "Cedar", "Pebble", "Kestrel",
    "Harbor", "Lichen", "Meadow", "Onyx", "Sable", "Thistle", "Vireo", "Alder",
]
