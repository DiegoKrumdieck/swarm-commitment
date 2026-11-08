"""
agents.py — one interface for LLM and scripted agents:

    await agent.decide(kind, obs) -> dict      kind in {"worker", "negotiator", "choice", "bargain"}

The returned dict holds the validated decision fields plus "_meta" (raw response, hashes,
tokens, retries, parse/default/refusal flags) that env.py writes to the JSONL log.

Layers
  provider adapter  `async complete(system, user, *, schema, name) -> Completion`
                    AnthropicClient, OpenAIClient, DryRunClient (scripted policy -> JSON text,
                    optional fault injection, so --dry-run exercises the same code path)
  LLMAgent          build prompt (prompts.py), call, parse, validate, retry, default
  Roster            which agent plays which role in an experiment

No keys are logged. Keys come from .env (python-dotenv); missing key -> fail fast.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field

import params as P
import prompts
import scripted as S

# Prices per 1M tokens (USD), from the provider docs on 2026-11-08.
# Anthropic Haiku 5.5 (prompts <= 100K tokens): input 0.10, output 0.50, cache read 0.1x, 5-min write 1.25x.
PRICING = {
    "claude-haiku-5-5": {"input": 0.10, "cache_read": 0.01, "cache_write": 0.125, "output": 0.50},
    "gpt-6-luna": {"input": 0.10, "cache_read": 0.01, "cache_write": 0.10, "output": 0.50},
    "gpt-6-sol": {"input": 2.00, "cache_read": 0.20, "cache_write": 2.00, "output": 10.00},
    "dryrun": {"input": 0.0, "cache_read": 0.0, "cache_write": 0.0, "output": 0.0},
}

# OpenAI: lowest reasoning effort accepted by gpt-6-luna and gpt-6-sol (docs: none|low|medium|high|xhigh|max).
OPENAI_LOWEST_EFFORT = {"gpt-6-luna": "none", "gpt-6-sol": "none"}

KINDS = ("worker", "negotiator", "choice", "bargain")
TEMPLATE_HASH = prompts.template_hash()


class FatalProviderError(RuntimeError):
    """Non-retryable (bad request, auth, unknown model): stop the run."""


class TransientProviderError(RuntimeError):
    """Network error / 429 / 5xx that survived the SDK's own backoff: counts as a retry."""


class ValidationError(ValueError):
    def __init__(self, msg: str, soft: bool = False):
        super().__init__(msg)
        self.soft = soft  # Exp 3: only the proposal is invalid


@dataclass
class Completion:
    text: str
    usage: dict = field(default_factory=dict)
    refusal: str | None = None
    meta: dict = field(default_factory=dict)


def _save_flagged(model_id: str, system: str, user: str, error: str):
    """Keep every prompt a provider's input moderation declined (logs/flagged/, gitignored)."""
    import hashlib

    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "flagged")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:12]
    with open(os.path.join(d, f"{model_id}_{h}.json"), "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "error": error[:500], "system": system, "user": user}, f, indent=1)


def cost_usd(model_id: str, row: dict) -> float:
    p = PRICING.get(model_id)
    if p is None:
        return 0.0
    inp = row.get("input_tokens") or 0
    cr = row.get("cache_read_tokens") or 0
    cw = row.get("cache_write_tokens") or 0
    out = row.get("output_tokens") or 0
    return (inp * p["input"] + cr * p["cache_read"] + cw * p["cache_write"] + out * p["output"]) / 1e6


# ---------------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------------


def require_key(provider: str):
    from dotenv import load_dotenv

    load_dotenv()
    var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[provider]
    v = os.environ.get(var, "")
    if not v or v.endswith("..."):
        raise SystemExit(f"{var} is not set. Copy .env.example to .env and fill in the key for {provider}.")


class AnthropicClient:
    """Claude Haiku 5.5 via the Messages API.

    - thinking disabled (no extended thinking; accepted at effort <= high)
    - temperature: the SDK (>= 2.x, API version 2026-10-21) rejects any sampling parameter
      on this model, so it is omitted; the model samples at its default (1.0 = TEMPERATURE).
    - system = rules, marked cache_control ephemeral (minimum cacheable prefix 512 tokens)
    - structured output via output_config.format json_schema
    - refusals: HTTP 200 with stop_reason "refusal" (+ stop_details.category); no fallback exists
    """

    provider = "anthropic"

    def __init__(self, model_id: str):
        import anthropic

        require_key("anthropic")
        self.anthropic = anthropic
        self.model_id = model_id
        self.client = anthropic.AsyncAnthropic(max_retries=6)  # SDK backs off on 408/409/429/5xx

    async def complete(self, system, user, *, schema=None, name="response", hint=None) -> Completion:
        a = self.anthropic
        kw = dict(
            model=self.model_id,
            max_tokens=P.MAX_OUTPUT_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},
        )
        if schema is not None:
            kw["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            r = await self.client.messages.create(**kw)
        except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError) as e:
            raise FatalProviderError(f"{type(e).__name__}: {str(e)[:300]}") from e
        except a.APIError as e:
            raise TransientProviderError(f"{type(e).__name__}: {str(e)[:200]}") from e
        u = r.usage
        usage = {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "reasoning_tokens": 0,
        }
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        refusal = None
        if r.stop_reason == "refusal":
            sd = getattr(r, "stop_details", None)
            refusal = f"refusal:{getattr(sd, 'category', None)}"
        return Completion(text, usage, refusal, {"stop_reason": r.stop_reason, "temperature_sent": None,
                                                 "effort_sent": None, "request_id": getattr(r, "_request_id", None)})


class OpenAIClient:
    """GPT-6 Luna / Sol via the Responses API.

    - reasoning.effort = lowest accepted ("none"); reasoning tokens logged separately
    - temperature = TEMPERATURE (accepted at effort none; if a 400 names temperature, it is
      dropped for the rest of the run and logged as temperature_sent = None)
    - rules in the system message (automatic prompt caching on the shared prefix)
    - structured output via text.format json_schema (strict)
    - refusals: output content item of type "refusal"
    """

    provider = "openai"

    def __init__(self, model_id: str):
        import openai

        require_key("openai")
        self.openai = openai
        self.model_id = model_id
        self.client = openai.AsyncOpenAI(max_retries=6)
        self.effort = OPENAI_LOWEST_EFFORT.get(model_id, "none")
        self.send_temperature = True

    async def complete(self, system, user, *, schema=None, name="response", hint=None) -> Completion:
        o = self.openai
        kw = dict(
            model=self.model_id,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            reasoning={"effort": self.effort},
            max_output_tokens=P.MAX_OUTPUT_TOKENS,
        )
        if self.send_temperature:
            kw["temperature"] = P.TEMPERATURE
        if schema is not None:
            kw["text"] = {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}
        try:
            r = await self.client.responses.create(**kw)
        except o.BadRequestError as e:
            if self.send_temperature and "temperature" in str(e).lower():
                self.send_temperature = False
                return await self.complete(system, user, schema=schema, name=name, hint=hint)
            msg = str(e)
            if "flagged" in msg.lower() or "usage policy" in msg.lower():
                # OpenAI's input moderation declined the prompt: a refusal, not a malformed request.
                # Logged as refusal "input_flagged"; LLMAgent retries, then defaults.
                _save_flagged(self.model_id, system, user, msg)
                return Completion("", {}, "refusal:input_flagged", {
                    "status": "input_flagged", "effort_sent": self.effort,
                    "temperature_sent": P.TEMPERATURE if self.send_temperature else None})
            raise FatalProviderError(f"BadRequestError: {msg[:300]}") from e
        except (o.AuthenticationError, o.PermissionDeniedError, o.NotFoundError) as e:
            raise FatalProviderError(f"{type(e).__name__}: {str(e)[:300]}") from e
        except o.APIError as e:
            raise TransientProviderError(f"{type(e).__name__}: {str(e)[:200]}") from e
        u = r.usage
        itd = getattr(u, "input_tokens_details", None)
        otd = getattr(u, "output_tokens_details", None)
        cached = getattr(itd, "cached_tokens", 0) or 0
        usage = {
            "input_tokens": (u.input_tokens or 0) - cached,
            "output_tokens": u.output_tokens or 0,
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
            "reasoning_tokens": getattr(otd, "reasoning_tokens", 0) or 0,
        }
        refusal, parts = None, []
        for item in r.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for c in item.content or []:
                if getattr(c, "type", None) == "refusal":
                    refusal = "refusal:" + (getattr(c, "refusal", "") or "")[:200]
                elif getattr(c, "type", None) == "output_text":
                    parts.append(c.text)
        meta = {"status": r.status, "temperature_sent": P.TEMPERATURE if self.send_temperature else None,
                "effort_sent": self.effort}
        if r.status == "incomplete":
            meta["incomplete_reason"] = getattr(getattr(r, "incomplete_details", None), "reason", None)
        return Completion("".join(parts), usage, refusal, meta)


class DryRunClient:
    """Stands in for an LLM: runs a scripted policy on the call's obs and returns its JSON as text.
    fault_rate > 0 injects malformed responses (prose, truncated JSON, missing keys, bad values)
    to exercise parsing, validation, retries and defaults."""

    provider = "dryrun"
    model_id = "dryrun"

    def __init__(self, fault_rate: float = 0.0, seed: int = 0, policy: str = "rational"):
        self.fault_rate = fault_rate
        self.rng = random.Random(seed)
        self.policy = policy

    def _agent(self, kind, obs):
        if kind == "worker":
            if self.policy == "random":
                return S.RandomWorker(self.rng)
            return S.RationalWorker() if obs["side"] == "A" else S.ComplyWorker()
        if kind == "negotiator":
            return S.RandomNegotiator(self.rng, 0.1) if self.policy == "random" else S.RationalNegotiator()
        if kind == "choice":
            return S.RationalChooser()
        return S.RationalBargainer()

    async def complete(self, system, user, *, schema=None, name="response", hint=None) -> Completion:
        kind, obs = hint
        out = self._agent(kind, obs).policy(kind, obs)
        text = json.dumps(out)
        if self.fault_rate and self.rng.random() < self.fault_rate:
            fault = self.rng.choice(["prose", "truncated", "missing", "badvalue", "wrapped"])
            if fault == "prose":
                text = "I will use my quota this round."
            elif fault == "truncated":
                text = text[: len(text) // 2]
            elif fault == "missing":
                d = dict(out)
                d.pop(self.rng.choice(list(d)), None)
                text = json.dumps(d)
            elif fault == "badvalue":
                d = dict(out)
                for k in ("action", "decision", "enforcement"):
                    if k in d:
                        d[k] = "maybe"
                text = json.dumps(d)
            else:
                text = "Here is my answer:\n```json\n" + text + "\n```"
        usage = {"input_tokens": (len(system) + len(user)) // 4, "output_tokens": len(text) // 4,
                 "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0}
        await asyncio.sleep(0)
        return Completion(text, usage, None, {"temperature_sent": None, "effort_sent": None})


def make_client(model_key: str, *, dry_run=False, fault_rate=0.0, seed=0, policy="rational"):
    if dry_run:
        return DryRunClient(fault_rate, seed, policy)
    spec = P.MODELS[model_key]
    if spec["provider"] == "anthropic":
        return AnthropicClient(spec["id"])
    if spec["provider"] == "openai":
        return OpenAIClient(spec["id"])
    raise ValueError(spec)


# ---------------------------------------------------------------------------------
# Prompts, response schemas, validation, defaults
# ---------------------------------------------------------------------------------

BUILDERS = {
    "worker": prompts.worker_prompt,
    "negotiator": prompts.negotiator_round_prompt,
    "choice": prompts.choice_prompt,
    "bargain": prompts.bargaining_prompt,
}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


NUM = {"type": "number"}
STR = {"type": "string"}


def _freeze_asked(obs):
    return obs.get("mechanism") == "retaliation" and obs.get("other_overdraws_last_round", 0) > 0


def _charge_asked(obs):
    return obs.get("mechanism") == "collateral" and bool(obs.get("chargeable_ids"))


def response_schema(kind: str, obs: dict) -> dict:
    """JSON schema with the keys in the order the prompt lists them."""
    if kind == "worker":
        return _obj({
            "payoff_this_round_if_quota": NUM,
            "payoff_this_round_if_overdraw": NUM,
            "expected_future_cost_if_overdraw": NUM,
            "reasoning": STR,
            "action": {"type": "string", "enum": ["quota", "overdraw"]},
        })
    if kind == "negotiator":
        props = {}
        if obs.get("can_end"):
            props["value_if_end"] = NUM
            props["expected_value_if_continue"] = NUM
        props["reasoning"] = STR
        if _freeze_asked(obs):
            props["freeze"] = _obj({
                "mode": {"type": "string", "enum": ["targeted", "all", "spread", "none"]},
                "worker_ids": {"type": "array", "items": STR},
            })
        if _charge_asked(obs):
            props["charge_worker_ids"] = {"type": "array", "items": STR}
        if obs.get("can_end"):
            props["decision"] = {"type": "string", "enum": ["continue", "end"]}
        if obs.get("sends_message"):
            props["message_to_workers"] = STR
        return _obj(props)
    if kind == "choice":
        if obs["side"] == "A":
            return _obj({
                "expected_overdraws_per_round_by_your_workers": NUM,
                "reasoning": STR,
                "enforcement": {"type": "string", "enum": S.menu_keys(obs["condition"])},
                "pitch": STR,
            })
        return _obj({
            "forecast_overdraws_per_round_by_other_swarm": NUM,
            "value_if_reject": NUM,
            "expected_value_if_accept": NUM,
            "reasoning": STR,
            "decision": {"type": "string", "enum": ["accept", "reject"]},
        })
    if kind == "bargain":
        own_key = obs["names"][obs["side"]].lower()
        prop = _obj({
            f"share_{own_key}": {"type": "integer"},
            "enforcement": {"type": "string", "enum": S.menu_keys(obs["condition"])},
        })
        return _obj({
            "reasoning": STR,
            "accept_last_proposal": {"type": "boolean"},
            "message": STR,
            "proposal": {"anyOf": [prop, {"type": "null"}]},
        })
    raise ValueError(kind)


def _as_num(x, key):
    if isinstance(x, bool):
        raise ValidationError(f"{key} is not a number")
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{key} is not a number: {x!r}")


def _as_enum(x, key, allowed):
    v = str(x).strip().lower() if x is not None else None
    if v not in allowed:
        raise ValidationError(f"{key}={x!r} not in {allowed}")
    return v


def _as_str(x, key):
    if x is None:
        return ""
    if not isinstance(x, str):
        raise ValidationError(f"{key} is not a string")
    return x


def validate(kind: str, obs: dict, parsed) -> dict:
    """Return the normalized decision or raise ValidationError."""
    if not isinstance(parsed, dict):
        raise ValidationError("response is not a JSON object")
    missing = [k for k in prompts.expected_fields(obs, kind) if k not in parsed]
    if missing:
        raise ValidationError(f"missing keys: {missing}")
    out = {}
    if kind == "worker":
        for k in ("payoff_this_round_if_quota", "payoff_this_round_if_overdraw", "expected_future_cost_if_overdraw"):
            out[k] = _as_num(parsed[k], k)
        out["reasoning"] = _as_str(parsed["reasoning"], "reasoning")
        out["action"] = _as_enum(parsed["action"], "action", ("quota", "overdraw"))
        return out
    if kind == "negotiator":
        out["reasoning"] = _as_str(parsed["reasoning"], "reasoning")
        if obs.get("can_end"):
            out["value_if_end"] = _as_num(parsed["value_if_end"], "value_if_end")
            out["expected_value_if_continue"] = _as_num(parsed["expected_value_if_continue"],
                                                        "expected_value_if_continue")
            out["decision"] = _as_enum(parsed["decision"], "decision", ("continue", "end"))
        if _freeze_asked(obs):
            fr = parsed["freeze"]
            if not isinstance(fr, dict):
                raise ValidationError("freeze is not an object")
            mode = _as_enum(fr.get("mode"), "freeze.mode", ("targeted", "all", "spread", "none"))
            ids = fr.get("worker_ids") or []
            if not isinstance(ids, list):
                raise ValidationError("freeze.worker_ids is not a list")
            ids = list(dict.fromkeys(str(i).strip() for i in ids))
            if mode == "targeted":
                active = set(obs.get("active_other_ids") or [])
                bad = [i for i in ids if i not in active]
                if not ids or bad:
                    raise ValidationError(f"targeted freeze needs active IDs; got {ids}")
            else:
                ids = []
            out["freeze"] = {"mode": mode, "worker_ids": ids}
        if _charge_asked(obs):
            ids = parsed["charge_worker_ids"] or []
            if not isinstance(ids, list):
                raise ValidationError("charge_worker_ids is not a list")
            ids = list(dict.fromkeys(str(i).strip() for i in ids))
            bad = [i for i in ids if i not in set(obs["chargeable_ids"])]
            if bad:
                raise ValidationError(f"charge IDs not chargeable: {bad}")
            out["charge_worker_ids"] = ids
        if obs.get("sends_message"):
            out["message_to_workers"] = _as_str(parsed["message_to_workers"], "message_to_workers")
        return out
    if kind == "choice":
        out["reasoning"] = _as_str(parsed["reasoning"], "reasoning")
        if obs["side"] == "A":
            out["expected_overdraws_per_round_by_your_workers"] = _as_num(
                parsed["expected_overdraws_per_round_by_your_workers"], "expected_overdraws")
            out["enforcement"] = _as_enum(parsed["enforcement"], "enforcement", tuple(S.menu_keys(obs["condition"])))
            out["pitch"] = _as_str(parsed["pitch"], "pitch")
        else:
            for k in ("forecast_overdraws_per_round_by_other_swarm", "value_if_reject", "expected_value_if_accept"):
                out[k] = _as_num(parsed[k], k)
            out["decision"] = _as_enum(parsed["decision"], "decision", ("accept", "reject"))
        return out
    if kind == "bargain":
        out["reasoning"] = _as_str(parsed["reasoning"], "reasoning")
        acc = parsed["accept_last_proposal"]
        if isinstance(acc, str) and acc.strip().lower() in ("true", "false"):
            acc = acc.strip().lower() == "true"
        if not isinstance(acc, bool):
            raise ValidationError("accept_last_proposal is not a boolean")
        out["message"] = _as_str(parsed["message"], "message")
        if acc:
            from env import latest_proposal

            lp = latest_proposal(obs.get("transcript", []))
            if lp is None or lp["side"] == obs["side"]:
                raise ValidationError("accept_last_proposal true but the other side made no latest proposal")
            out["accept_last_proposal"] = True
            out["proposal"] = None
            return out
        out["accept_last_proposal"] = False
        pr = parsed["proposal"]
        if pr is None:
            out["proposal"] = None
            return out
        own_key = f"share_{obs['names'][obs['side']].lower()}"
        try:
            if not isinstance(pr, dict):
                raise ValueError("proposal is not an object")
            share = pr.get(own_key)
            if isinstance(share, bool) or share is None:
                raise ValueError(f"{own_key} missing")
            share_f = float(share)
            if share_f != int(share_f) or int(share_f) % 3 or not 0 <= int(share_f) <= P.R:
                raise ValueError(f"{own_key}={share} not a multiple of 3 in [0, {P.R}]")
            enf = str(pr.get("enforcement", "")).strip().lower()
            if enf not in S.menu_keys(obs["condition"]):
                raise ValueError(f"enforcement {enf!r} not in menu")
        except (TypeError, ValueError) as e:
            err = ValidationError(f"invalid proposal: {e}", soft=True)
            err.partial = out
            raise err
        out["proposal"] = {own_key: int(share_f), "enforcement": enf}
        return out
    raise ValueError(kind)


def default_decision(kind: str, obs: dict) -> dict:
    """Used after MAX_RETRIES failures; the row is flagged defaulted = true."""
    if kind == "worker":
        return {"payoff_this_round_if_quota": None, "payoff_this_round_if_overdraw": None,
                "expected_future_cost_if_overdraw": None, "reasoning": None, "action": "quota"}
    if kind == "negotiator":
        out = {"reasoning": None}
        if obs.get("can_end"):
            out.update(value_if_end=None, expected_value_if_continue=None, decision="continue")
        if _freeze_asked(obs):
            out["freeze"] = {"mode": "none", "worker_ids": []}
        if _charge_asked(obs):
            out["charge_worker_ids"] = []
        if obs.get("sends_message"):
            out["message_to_workers"] = None
        return out
    if kind == "choice":
        if obs["side"] == "A":
            return {"expected_overdraws_per_round_by_your_workers": None, "reasoning": None,
                    "enforcement": "none", "pitch": ""}
        return {"forecast_overdraws_per_round_by_other_swarm": None, "value_if_reject": None,
                "expected_value_if_accept": None, "reasoning": None, "decision": "accept"}
    if kind == "bargain":
        return {"reasoning": None, "accept_last_proposal": False, "message": "", "proposal": None}
    raise ValueError(kind)


# ---------------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------------


class LLMAgent:
    def __init__(self, client, semaphore: asyncio.Semaphore | None = None):
        self.client = client
        self.sem = semaphore or asyncio.Semaphore(P.MAX_CONCURRENCY)

    async def decide(self, kind: str, obs: dict) -> dict:
        clean = {k: v for k, v in obs.items() if not k.startswith("_")}
        system, user = BUILDERS[kind](clean)
        schema = response_schema(kind, clean)
        meta = {
            "agent": f"llm:{self.client.provider}",
            "provider": self.client.provider,
            "model_id": self.client.model_id,
            "prompt_version": prompts.PROMPT_VERSION,
            "prompt_hash": prompts.prompt_hash(system, user),
            "template_hash": TEMPLATE_HASH,
            "system_chars": len(system),
            "user_chars": len(user),
            "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "errors": [],
            "raw_response": None,
            "raw_responses": [],
            "refusal": None,
            "proposal_invalid": False,
        }
        t0 = time.perf_counter()
        decision, soft_used, attempt = None, 0, 0
        for attempt in range(P.MAX_RETRIES + 1):
            try:
                async with self.sem:
                    comp = await self.client.complete(system, user, schema=schema, name=kind, hint=(kind, obs))
            except TransientProviderError as e:  # network / 429 / 5xx after SDK backoff
                meta["errors"].append(f"api:{type(e).__name__}:{str(e)[:160]}")
                await asyncio.sleep(min(30, 2 ** attempt))
                continue
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
                meta[k] += comp.usage.get(k, 0) or 0
            meta["raw_response"] = comp.text
            meta["raw_responses"].append(comp.text)
            meta["temperature_sent"] = comp.meta.get("temperature_sent")
            meta["effort_sent"] = comp.meta.get("effort_sent")
            if comp.meta.get("stop_reason"):
                meta["stop_reason"] = comp.meta["stop_reason"]
            if comp.refusal:
                meta["refusal"] = comp.refusal
                meta["errors"].append(comp.refusal)
                continue
            try:
                parsed = prompts.parse_response(comp.text)
            except (ValueError, json.JSONDecodeError) as e:
                meta["errors"].append(f"parse:{str(e)[:120]}")
                continue
            meta["parsed"] = parsed
            try:
                decision = validate(kind, clean, parsed)
                break
            except ValidationError as e:
                meta["errors"].append(f"invalid:{e}")
                if e.soft:
                    # Exp 3: invalid proposal -> retry once, then treat as "no proposal"
                    if soft_used >= 1:
                        decision = {**e.partial, "proposal": None}
                        meta["proposal_invalid"] = True
                        break
                    soft_used += 1
                continue
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        meta["retries"] = attempt
        if len(meta["raw_responses"]) <= 1:
            meta.pop("raw_responses")
        if decision is None:
            decision = default_decision(kind, clean)
            meta["parse_failed"] = True
            meta["defaulted"] = True
        else:
            meta["parse_failed"] = False
            meta["defaulted"] = False
        decision["_meta"] = meta
        return decision


# ---------------------------------------------------------------------------------
# Rosters
# ---------------------------------------------------------------------------------


class LLMRoster:
    """Exp 1: A's workers and both negotiators are LLMs; B's workers are ComplyWorker.
    Exp 2/3: every role is an LLM. One model per episode."""

    def __init__(self, exp: int, client, semaphore=None):
        self.exp = exp
        self.llm = LLMAgent(client, semaphore)
        self.comply = S.ComplyWorker()

    def negotiator(self, side):
        return self.llm

    def worker(self, side, w):
        if self.exp == 1 and side == "B":
            return self.comply
        return self.llm

    def chooser(self, side):
        return self.llm

    def bargainer(self, side):
        return self.llm
