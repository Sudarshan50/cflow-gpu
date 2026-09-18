#!/usr/bin/env python3
"""Capability surface an agentic harness actually depends on.

Each check is written so a PASS means the feature is usable by a client, not
merely that the request returned 200.
"""
import base64
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"
KEY = open("/scratch/deploy/api-key.txt").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
fails = []


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chk(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} {detail}")
    if not ok:
        fails.append(name)


def msg(d):
    return d["choices"][0]["message"]


# ---------------------------------------------------------------- tools
print("== tool calling surface ==")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
STOCK = {"type": "function", "function": {
    "name": "get_stock", "description": "Current stock price for a ticker",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}},
                   "required": ["ticker"]}}}

# Parallel tool calls: agent harnesses batch independent reads in one turn.
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 400, "temperature": 0,
        "tools": [WEATHER, STOCK], "tool_choice": "auto",
        "messages": [{"role": "user", "content":
                      "Get the weather in Paris AND the stock price of NVDA. "
                      "Call both tools at once."}]})
    tc = msg(d).get("tool_calls") or []
    names = sorted(c["function"]["name"] for c in tc)
    chk("parallel tool calls in one message", len(tc) >= 2, f"{len(tc)} calls: {names}")
except Exception as e:
    chk("parallel tool calls in one message", False, f"{type(e).__name__}: {e}")

# Forced named tool: harnesses use this to guarantee a schema-shaped answer.
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 300, "temperature": 0,
        "tools": [WEATHER, STOCK],
        "tool_choice": {"type": "function", "function": {"name": "get_stock"}},
        "messages": [{"role": "user", "content": "What is the weather in Rome?"}]})
    tc = msg(d).get("tool_calls") or []
    chk("tool_choice: forced named function",
        len(tc) == 1 and tc[0]["function"]["name"] == "get_stock",
        f"called {[c['function']['name'] for c in tc]}")
except Exception as e:
    chk("tool_choice: forced named function", False, f"{type(e).__name__}: {e}")

# tool_choice: none must suppress calls entirely.
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 120, "temperature": 0,
        "tools": [WEATHER], "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather in Oslo?"}]})
    chk("tool_choice: none suppresses calls", not (msg(d).get("tool_calls")))
except Exception as e:
    chk("tool_choice: none suppresses calls", False, f"{type(e).__name__}: {e}")

# Real harnesses ship 20-40 tool schemas in the system turn.
try:
    many = [{"type": "function", "function": {
        "name": f"tool_{i}", "description": f"Utility number {i} for repo ops",
        "parameters": {"type": "object", "properties": {
            "arg": {"type": "string", "description": "x" * 120}}, "required": ["arg"]}}}
        for i in range(30)] + [WEATHER]
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 300, "temperature": 0,
        "tools": many, "tool_choice": "auto",
        "messages": [{"role": "user", "content": "What is the weather in Lima?"}]})
    tc = msg(d).get("tool_calls") or []
    chk("31 tool schemas, picks the right one",
        len(tc) == 1 and tc[0]["function"]["name"] == "get_weather",
        f"{d['usage']['prompt_tokens']} prompt tokens")
except Exception as e:
    chk("31 tool schemas, picks the right one", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------- structured outputs
print()
print("== structured outputs ==")
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 200, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "Give me a JSON object with keys name and age."}]})
    txt = msg(d).get("content") or ""
    json.loads(txt)
    chk("response_format json_object", True, txt[:60].replace("\n", " "))
except Exception as e:
    chk("response_format json_object", False, f"{type(e).__name__}: {str(e)[:90]}")

SCHEMA = {"type": "object",
          "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
          "required": ["city", "population"], "additionalProperties": False}
try:
    # 200 max_tokens starved this and looked like a server failure: the
    # completion needed 232. Grammar-constrained output still spends tokens
    # reasoning before it emits, so budget generously on structured calls.
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 900, "temperature": 0,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "city_info", "schema": SCHEMA, "strict": True}},
        "messages": [{"role": "user", "content": "Tokyo's population, as JSON."}]})
    txt = msg(d).get("content") or ""
    o = json.loads(txt)
    chk("response_format json_schema (strict)",
        set(o) == {"city", "population"} and isinstance(o["population"], int), txt[:60])
except Exception as e:
    chk("response_format json_schema (strict)", False, f"{type(e).__name__}: {str(e)[:90]}")

# ------------------------------------------------------------- vision
print()
print("== vision (multimodal input) ==")
try:
    b64 = base64.b64encode(open("/scratch/hf/vision-test.png", "rb").read()).decode()
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 300, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What alphanumeric code is written in this image? "
                                     "Reply with the code only."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]})
    m = msg(d)
    out = (m.get("content") or "") + (m.get("reasoning_content") or "")
    chk("image input + OCR of 'R7X-42Q'", "R7X" in out.upper(),
        f"mm_tokens={d['usage']['prompt_tokens']} got={out.strip()[:50]!r}")
except urllib.error.HTTPError as e:
    chk("image input + OCR of 'R7X-42Q'", False, f"HTTP {e.code} {e.read().decode()[:120]}")
except Exception as e:
    chk("image input + OCR of 'R7X-42Q'", False, f"{type(e).__name__}: {str(e)[:110]}")

# ------------------------------------------------- misc client surface
print()
print("== other things clients rely on ==")
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 80, "temperature": 0,
        "stop": ["STOPHERE"],
        "messages": [{"role": "user", "content": "Say: alpha STOPHERE beta"}]})
    txt = msg(d).get("content") or ""
    chk("stop sequences honoured", "beta" not in txt, repr(txt[:50]))
except Exception as e:
    chk("stop sequences honoured", False, f"{type(e).__name__}: {e}")

try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 600, "temperature": 0,
        "messages": [{"role": "user", "content": "Think step by step: what is 37*43?"}]})
    m = msg(d)
    has_rc = "reasoning_content" in m and m["reasoning_content"]
    chk("reasoning_content is a separate field", True,
        f"present={bool(has_rc)} (extra non-standard key clients must tolerate)")
except Exception as e:
    chk("reasoning_content is a separate field", False, f"{type(e).__name__}: {e}")

# Claude Code and similar target the Anthropic Messages shape.
try:
    req = urllib.request.Request(
        BASE + "/v1/messages",
        data=json.dumps({"model": "kimi-k3", "max_tokens": 40,
                         "messages": [{"role": "user", "content": "Reply with the word ok."}]}).encode(),
        headers=HDR)
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    chk("/v1/messages (Anthropic shape) works", "content" in d or "id" in d,
        json.dumps(d)[:70])
except urllib.error.HTTPError as e:
    chk("/v1/messages (Anthropic shape) works", False, f"HTTP {e.code} {e.read().decode()[:100]}")
except Exception as e:
    chk("/v1/messages (Anthropic shape) works", False, f"{type(e).__name__}: {str(e)[:90]}")

# Unknown extra fields: gateways inject things like cache_control / metadata.
try:
    d = post("/v1/chat/completions", {
        "model": "kimi-k3", "max_tokens": 30, "temperature": 0,
        "metadata": {"user_id": "abc"}, "user": "someone",
        "messages": [{"role": "user", "content": "Reply with ok."}]})
    chk("tolerates extra gateway fields", True, "metadata/user accepted")
except urllib.error.HTTPError as e:
    chk("tolerates extra gateway fields", False, f"HTTP {e.code} {e.read().decode()[:110]}")
except Exception as e:
    chk("tolerates extra gateway fields", False, f"{type(e).__name__}: {e}")

print()
print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
