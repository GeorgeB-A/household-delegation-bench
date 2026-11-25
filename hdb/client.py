"""Minimal model clients. One interface: chat(messages, tools=None, json_schema=None) -> dict(text, tool_calls, usage).

openai     any OpenAI-compatible Chat Completions endpoint (OpenAI, OpenRouter, vLLM, Together, a local server ...);
           set --base-url and the key variable for non-OpenAI endpoints
anthropic  Anthropic Messages API
mock       random valid answers, for testing the pipeline without an API key"""
from __future__ import annotations
import json, os, random, time

TRANSIENT = ("rate", "429", "overload", "529", "500", "502", "503", "504", "timeout", "timed out", "connection")


def _retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # transport errors only; content is never re-requested after a valid reply
            if i == tries - 1 or not any(t in str(e).lower() for t in TRANSIENT):
                raise
            time.sleep(min(60, 2 ** (i + 1) + random.random()))


class OpenAIClient:
    def __init__(self, model, base_url=None, api_key_env="OPENAI_API_KEY", temperature=None, max_tokens=1200, extra=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))
        self.model, self.temperature, self.max_tokens, self.extra = model, temperature, max_tokens, extra or {}

    def chat(self, messages, tools=None, json_schema=None):
        kw = dict(model=self.model, messages=messages, max_completion_tokens=self.max_tokens, **self.extra)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if tools:
            kw["tools"] = tools
        if json_schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": json_schema}}
        r = _retry(lambda: self.client.chat.completions.create(**kw))
        msg = r.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {"_unparsed": tc.function.arguments}
            calls.append({"name": tc.function.name, "args": args})
        u = r.usage
        return {"text": msg.content or "", "tool_calls": calls,
                "usage": {"in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None)}}


class AnthropicClient:
    def __init__(self, model, api_key_env="ANTHROPIC_API_KEY", temperature=None, max_tokens=1200, thinking_budget=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get(api_key_env))
        self.model, self.temperature, self.max_tokens, self.thinking = model, temperature, max_tokens, thinking_budget

    def chat(self, messages, tools=None, json_schema=None):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conv = []
        for m in messages:  # merge consecutive same-role turns (the API requires alternation)
            if m["role"] == "system":
                continue
            if conv and conv[-1]["role"] == m["role"]:
                conv[-1]["content"] += "\n\n" + m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=conv)
        if self.temperature is not None and not self.thinking:
            kw["temperature"] = self.temperature
        if self.thinking:
            kw["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
        if tools:
            kw["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                            "input_schema": t["function"]["parameters"]} for t in tools]
        if json_schema:
            kw["extra_body"] = {"output_config": {"format": {"type": "json_schema", "schema": json_schema}}}
        r = _retry(lambda: self.client.messages.create(**kw))
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        calls = [{"name": b.name, "args": dict(b.input)} for b in r.content if getattr(b, "type", "") == "tool_use"]
        return {"text": text, "tool_calls": calls, "usage": {"in": r.usage.input_tokens, "out": r.usage.output_tokens}}


class MockClient:
    """Answers at random within the allowed options. Use it to test the pipeline."""
    def __init__(self, seed=0, **_):
        self.rng = random.Random(seed)

    def chat(self, messages, tools=None, json_schema=None):
        if tools:
            t = self.rng.choice(tools)["function"]
            args = {"entry_id": "m05", "note_to_household": "mock"}
            if t["name"] == "update_memory":
                args["new_text"] = "mock rewrite"
            return {"text": "", "tool_calls": [{"name": t["name"], "args": args}], "usage": {}}
        props = json_schema["properties"]
        out = {k: self.rng.choice(v["enum"]) for k, v in props.items()}
        if "choice" in out:
            if out["choice"] == "act":
                out["action"] = self.rng.choice([a for a in props["action"]["enum"] if a != "none"]); out["question"] = "none"
            else:
                out["question"] = self.rng.choice([q for q in props["question"]["enum"] if q != "none"]); out["action"] = "none"
        return {"text": json.dumps(out), "tool_calls": [], "usage": {}}


def make_client(provider, model, **kw):
    if provider == "openai":
        return OpenAIClient(model, **kw)
    if provider == "anthropic":
        kw.pop("base_url", None)
        return AnthropicClient(model, **{k: v for k, v in kw.items() if k in ("api_key_env", "temperature", "max_tokens", "thinking_budget")})
    if provider == "mock":
        return MockClient()
    raise ValueError(f"unknown provider {provider}")
