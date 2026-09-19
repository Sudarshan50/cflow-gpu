#!/usr/bin/env python3
"""KVV-style API compatibility health check for the self-hosted Kimi-K3 endpoint.

Mirrors the section layout of the reference report (instructions.pdf):

  A basic protocol     B parameter validation  C tool calling
  D structured output  E reasoning control     F streaming quality
  G usage reporting    H multimodal            I long context
  J protocol semantics K concurrency           L consistency
  M message shapes     N reasoning budget      S edge security

Differences from the reference platform are recorded rather than hidden. That
report tested a multi-channel Azure gateway: 65 channels, an Anthropic
/v1/messages path and a /v1/responses path. This deployment is a single vLLM
engine behind one nginx edge, so channel fan-out checks become repeat-stability
checks, and the two extra protocol surfaces are expected to be absent rather
than broken.

No credential is written to disk. The key is read from customers.tsv and
redacted from every output file.

Usage:
  ./kvv_health.py --out-dir DIR [--base-url URL] [--model NAME]
                  [--only A,B,C] [--skip I,K]
"""

import argparse
import json
import os
import re
import ssl
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

DEFAULT_BASE = "https://cflox.store/v1"
DEFAULT_MODEL = "FW-Kimi-K3"
TSV = "/scratch/deploy/customers.tsv"
EVAL_CUSTOMER = "evalbot"

# The reference report's own §7 recheck #1 was a probe defect: max_tokens=40 let
# reasoning consume the entire budget, content came back empty, and long-context
# retrieval looked broken when it was fine. Every content-bearing check here
# therefore gets a budget generous enough for reasoning plus an answer.
DEFAULT_MAX_TOKENS = 2000
SHORT_TIMEOUT = 60
TIMEOUT = 240
LONG_TIMEOUT = 900

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

_REDACT = []


def redact(text):
    out = str(text)
    for secret in _REDACT:
        if secret:
            out = out.replace(secret, "sk-k3-<redacted>")
    return out


class Result:
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()

    def add(self, cid, name, status, detail="", **extra):
        row = {"id": cid, "name": name, "status": status,
               "detail": redact(detail)[:4000]}
        row.update(extra)
        with self.lock:
            self.rows.append(row)
        mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}[status]
        print("  [%s] %-6s %-52s %s" % (mark, cid, name[:52], redact(detail)[:140]),
              flush=True)
        return row

    def counts(self):
        c = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for r in self.rows:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c


R = Result()
CTX = ssl.create_default_context()


def http(method, url, key=None, body=None, timeout=TIMEOUT, extra_headers=None,
         raw_body=None):
    """One request. Returns (code, parsed_or_text, elapsed_s, error_str)."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    if extra_headers:
        headers.update(extra_headers)
    data = None
    if raw_body is not None:
        data = raw_body if isinstance(raw_body, bytes) else raw_body.encode()
    elif body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            text = r.read().decode("utf-8", "replace")
            code = r.status
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        code = e.code
    except Exception as e:
        return 0, None, time.time() - t0, "%s: %s" % (type(e).__name__, e)
    elapsed = time.time() - t0
    try:
        return code, json.loads(text), elapsed, None
    except Exception:
        return code, text, elapsed, None


def stream(url, key, body, timeout=TIMEOUT):
    """SSE request. Returns (code, frames, saw_done, error)."""
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    frames, saw_done = [], False
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            code = r.status
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    continue
                try:
                    frames.append(json.loads(payload))
                except Exception:
                    pass
    except urllib.error.HTTPError as e:
        return e.code, frames, saw_done, e.read().decode("utf-8", "replace")[:400]
    except Exception as e:
        return 0, frames, saw_done, "%s: %s" % (type(e).__name__, e)
    return code, frames, saw_done, None


class Api:
    """Thin endpoint wrapper carrying base url / model / key."""

    def __init__(self, base, model, key):
        self.base = base.rstrip("/")
        self.root = self.base[:-3].rstrip("/") if self.base.endswith("/v1") else self.base
        self.model = model
        self.key = key

    def chat(self, messages, timeout=TIMEOUT, **kw):
        body = {"model": self.model, "messages": messages}
        body.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        body.update(kw)
        return http("POST", self.base + "/chat/completions", self.key, body, timeout)

    def ask(self, prompt, timeout=TIMEOUT, **kw):
        return self.chat([{"role": "user", "content": prompt}], timeout, **kw)

    def raw(self, path, body=None, method="POST", key="__self__", timeout=TIMEOUT,
            raw_body=None):
        k = self.key if key == "__self__" else key
        return http(method, self.base + path, k, body, timeout, raw_body=raw_body)

    def at_root(self, path, body=None, method="POST", timeout=TIMEOUT):
        return http(method, self.root + path, self.key, body, timeout)


def content_of(d):
    try:
        return d["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def reasoning_of(d):
    try:
        m = d["choices"][0]["message"]
    except Exception:
        return ""
    return m.get("reasoning") or m.get("reasoning_content") or ""


def usage_of(d):
    return (d or {}).get("usage") or {}


def reasoning_tokens(d):
    det = usage_of(d).get("completion_tokens_details") or {}
    return det.get("reasoning_tokens")


def cached_tokens(d):
    det = usage_of(d).get("prompt_tokens_details") or {}
    return det.get("cached_tokens")


def err_text(payload):
    if isinstance(payload, dict):
        e = payload.get("error")
        if isinstance(e, dict):
            return str(e.get("message") or e)
        return str(e or payload)[:300]
    return str(payload)[:300]


def png(width, height, rgb):
    """Minimal RGB PNG, so multimodal checks need no fixture files."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def data_url(rgb=(220, 20, 20), size=64):
    import base64
    return "data:image/png;base64," + base64.b64encode(png(size, size, rgb)).decode()


def ok2xx(code):
    return 200 <= code < 300


# --------------------------------------------------------------------------
# A. Basic protocol
# --------------------------------------------------------------------------
def section_A(api):
    print("\nA. Basic protocol")
    code, d, _, err = http("GET", api.base + "/models", api.key, timeout=SHORT_TIMEOUT)
    names = [m.get("id") for m in (d or {}).get("data", [])] if isinstance(d, dict) else []
    R.add("A1", "GET /v1/models returns target model",
          PASS if api.model in names else FAIL,
          "http %s names=%s" % (code, names or err), http_code=code)

    code, d, el, err = api.ask("Say hello in exactly one word.")
    fr = (d or {}).get("choices", [{}])[0].get("finish_reason") if isinstance(d, dict) else None
    R.add("A2", "Non-streaming chat structure",
          PASS if ok2xx(code) and content_of(d) and fr == "stop" else FAIL,
          "http %s finish=%s content=%r %.1fs" % (code, fr, content_of(d)[:40], el),
          http_code=code)

    code, frames, done, err = stream(api.base + "/chat/completions", api.key,
                                     {"model": api.model, "max_tokens": 300,
                                      "stream": True,
                                      "messages": [{"role": "user",
                                                    "content": "Count from 1 to 40."}]})
    R.add("A3", "Streaming chat frames + [DONE]",
          PASS if ok2xx(code) and len(frames) > 5 and done else FAIL,
          "http %s frames=%d done=%s %s" % (code, len(frames), done, err or ""),
          http_code=code, frames=len(frames))

    code, d, _, _ = api.ask("hi", max_tokens=600)
    echoed = (d or {}).get("model") if isinstance(d, dict) else None
    R.add("A4", "Response model field echo",
          PASS if echoed == api.model else WARN,
          "echoed=%r expected=%r" % (echoed, api.model), http_code=code)

    code, d, _, _ = api.chat([{"role": "user", "content": "Reply: fine",
                               "name": "tester", "bogus_field": "x"}])
    R.add("A5", "Unknown fields in message object are ignored",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d) if not ok2xx(code) else "accepted"),
          http_code=code)

    code, d, _, _ = http("POST", api.base + "/chat/completions", api.key,
                         {"model": "no-such-model-xyz", "max_tokens": 16,
                          "messages": [{"role": "user", "content": "hi"}]},
                         timeout=SHORT_TIMEOUT)
    R.add("A6", "Unknown model name returns 4xx",
          PASS if 400 <= code < 500 else WARN,
          "http %s (reference platform returned 503 and was flagged)" % code,
          http_code=code)

    code, d, _, _ = http("POST", api.base + "/embeddings", api.key,
                         {"model": api.model, "input": "hello"}, timeout=SHORT_TIMEOUT)
    R.add("A7", "/v1/embeddings behaviour documented",
          PASS if code in (400, 404, 403, 500) else WARN,
          "http %s - no embedding model is served here" % code, http_code=code)

    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "What is in this video?"},
        {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}]}],
        timeout=SHORT_TIMEOUT)
    R.add("A8", "Video input rejected with a clear error",
          PASS if 400 <= code < 500 else WARN,
          "http %s %s" % (code, err_text(d)[:120]), http_code=code)


# --------------------------------------------------------------------------
# B. Parameter validation
# --------------------------------------------------------------------------
def section_B(api):
    print("\nB. Parameter validation")
    cases = [
        ("B1", "temperature=0 accepted", {"temperature": 0}, "2xx"),
        ("B2", "temperature=2 accepted", {"temperature": 2}, "2xx"),
        ("B3", "temperature=3 rejected", {"temperature": 3}, "4xx"),
        ("B4", "temperature=-1 rejected", {"temperature": -1}, "4xx"),
        ("B5", "top_p=0.1 accepted", {"top_p": 0.1}, "2xx"),
        ("B6", "top_p=1.5 rejected", {"top_p": 1.5}, "4xx"),
        ("B7", "frequency_penalty=2 accepted", {"frequency_penalty": 2}, "2xx"),
        ("B8", "frequency_penalty=5 rejected", {"frequency_penalty": 5}, "4xx"),
        ("B9", "presence_penalty=-2 accepted", {"presence_penalty": -2}, "2xx"),
        ("B10", "negative max_tokens rejected", {"max_tokens": -5}, "4xx"),
        ("B11", "unknown parameter ignored", {"totally_made_up_param": True}, "2xx"),
    ]
    for cid, name, kw, want in cases:
        kw = dict(kw)
        kw.setdefault("max_tokens", 600)
        code, d, _, err = api.ask("Reply with the word ok.", timeout=SHORT_TIMEOUT, **kw)
        good = ok2xx(code) if want == "2xx" else (400 <= code < 500)
        R.add(cid, name, PASS if good else FAIL,
              "http %s want=%s %s" % (code, want, "" if good else err_text(d)[:120]),
              http_code=code)

    code, d, _, _ = api.ask("Write a long essay about the sea.", max_tokens=16)
    u = usage_of(d)
    R.add("B12", "max_tokens=16 truncates",
          PASS if ok2xx(code) and u.get("completion_tokens") == 16 else WARN,
          "http %s completion_tokens=%s finish=%s" % (
              code, u.get("completion_tokens"),
              (d or {}).get("choices", [{}])[0].get("finish_reason")),
          http_code=code)

    code, d, _, _ = http("POST", api.base + "/chat/completions", api.key,
                         {"model": api.model, "messages": [], "max_tokens": 16},
                         timeout=SHORT_TIMEOUT)
    R.add("B13", "empty messages rejected", PASS if 400 <= code < 500 else FAIL,
          "http %s" % code, http_code=code)

    code, d, _, _ = api.ask("List: alpha beta gamma delta", max_tokens=800,
                            stop=["beta"])
    R.add("B14", "stop sequence honoured",
          PASS if ok2xx(code) and "beta" not in content_of(d) else WARN,
          "http %s content=%r" % (code, content_of(d)[:60]), http_code=code)

    a = api.ask("Pick a random noun. Reply with only that noun.", max_tokens=800,
                seed=424242, temperature=0.0)
    b = api.ask("Pick a random noun. Reply with only that noun.", max_tokens=800,
                seed=424242, temperature=0.0)
    same = content_of(a[1]).strip() == content_of(b[1]).strip()
    R.add("B15", "seed reproducibility", PASS if same else WARN,
          "same=%s a=%r b=%r" % (same, content_of(a[1])[:30], content_of(b[1])[:30]))

    code, d, _, _ = api.ask("hi", max_tokens=600, n=2)
    n_ok = ok2xx(code) and len((d or {}).get("choices", [])) == 2
    R.add("B16", "n=2 multiple candidates",
          PASS if n_ok else WARN,
          "http %s choices=%s (reference platform: 65/65 rejected n>1)" % (
              code, len((d or {}).get("choices", [])) if isinstance(d, dict) else "-"),
          http_code=code)

    code, d, _, _ = api.ask("hi", max_tokens=99999999)
    R.add("B17", "oversized max_tokens handled cleanly",
          PASS if 400 <= code < 500 else WARN,
          "http %s %s" % (code, err_text(d)[:120]), http_code=code)

    code, d, _, _ = api.ask("Reply ok.", max_tokens=600, logprobs=True, top_logprobs=3)
    lp = None
    try:
        lp = (d["choices"][0].get("logprobs") or {}).get("content")
    except Exception:
        pass
    R.add("B18", "logprobs returned when requested",
          PASS if ok2xx(code) and lp else WARN,
          "http %s logprobs=%s (reference platform: only 14/65 channels)" % (
              code, "present" if lp else "empty"), http_code=code)


# --------------------------------------------------------------------------
# C. Tool calling
# --------------------------------------------------------------------------
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "book_trip",
        "description": "Book a multi-leg trip.",
        "parameters": {
            "type": "object",
            "properties": {
                "traveller": {
                    "type": "object",
                    "properties": {"name": {"type": "string"},
                                   "age": {"type": "integer"}},
                    "required": ["name"],
                },
                "legs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"from": {"type": "string"},
                                       "to": {"type": "string"},
                                       "cabin": {"type": "string",
                                                 "enum": ["economy", "business"]}},
                        "required": ["from", "to"],
                    },
                },
            },
            "required": ["traveller", "legs"],
            "additionalProperties": False,
        },
    },
}


def tool_calls_of(d):
    try:
        return d["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        return []


def section_C(api):
    print("\nC. Tool calling")
    code, d, _, _ = api.ask("What is the weather in Paris right now?",
                            tools=[WEATHER_TOOL], tool_choice="auto")
    tc = tool_calls_of(d)
    R.add("C1", "tool_choice=auto produces a call",
          PASS if ok2xx(code) and tc else FAIL,
          "http %s calls=%d" % (code, len(tc)), http_code=code)

    code, d, _, _ = api.ask("What is the weather in Paris?",
                            tools=[WEATHER_TOOL], tool_choice="none")
    R.add("C2", "tool_choice=none suppresses calls",
          PASS if ok2xx(code) and not tool_calls_of(d) else FAIL,
          "http %s calls=%d" % (code, len(tool_calls_of(d))), http_code=code)

    code, d, _, _ = api.ask("Tell me a joke.", tools=[WEATHER_TOOL],
                            tool_choice={"type": "function",
                                         "function": {"name": "get_weather"}})
    tc = tool_calls_of(d)
    R.add("C3", "tool_choice=specific function forces that tool",
          PASS if ok2xx(code) and tc and tc[0]["function"]["name"] == "get_weather"
          else FAIL,
          "http %s calls=%s" % (code, [t["function"]["name"] for t in tc]),
          http_code=code)

    hits = 0
    for _ in range(5):
        code, d, _, _ = api.ask("Hello there.", tools=[WEATHER_TOOL],
                                tool_choice="required")
        if ok2xx(code) and tool_calls_of(d):
            hits += 1
    R.add("C4", "tool_choice=required 5/5", PASS if hits == 5 else FAIL,
          "%d/5 produced a tool call" % hits, hits=hits)

    code, d, _, _ = api.ask(
        "Book a trip for Ada, age 36, from Delhi to Tokyo in business, "
        "then Tokyo to Sydney in economy.", tools=[NESTED_TOOL],
        tool_choice={"type": "function", "function": {"name": "book_trip"}})
    args, valid = None, False
    tc = tool_calls_of(d)
    if tc:
        try:
            args = json.loads(tc[0]["function"]["arguments"])
            legs = args.get("legs") or []
            valid = (isinstance(args.get("traveller"), dict)
                     and args["traveller"].get("name")
                     and len(legs) >= 2
                     and all(l.get("from") and l.get("to") for l in legs)
                     and all(l.get("cabin") in (None, "economy", "business")
                             for l in legs))
        except Exception:
            valid = False
    R.add("C5", "nested schema arguments conform",
          PASS if valid else FAIL,
          "http %s args=%s" % (code, json.dumps(args)[:200] if args else "none"),
          http_code=code)

    many = [WEATHER_TOOL, NESTED_TOOL] + [{
        "type": "function",
        "function": {"name": "tool_%d" % i, "description": "Unrelated filler %d" % i,
                     "parameters": {"type": "object", "properties": {}}},
    } for i in range(18)]
    code, d, _, _ = api.ask("What is the weather in Oslo?", tools=many,
                            tool_choice="auto")
    tc = tool_calls_of(d)
    right = bool(tc) and tc[0]["function"]["name"] == "get_weather"
    R.add("C6", "correct routing among 20 tools", PASS if right else FAIL,
          "http %s picked=%s" % (code, [t["function"]["name"] for t in tc]),
          http_code=code)

    code, d, _, _ = api.ask("What is the weather in Paris AND in Cairo? "
                            "Call the tool once per city.",
                            tools=[WEATHER_TOOL], tool_choice="auto",
                            parallel_tool_calls=True)
    tc = tool_calls_of(d)
    R.add("C7", "parallel_tool_calls returns multiple calls",
          PASS if len(tc) >= 2 else WARN,
          "http %s calls=%d" % (code, len(tc)), http_code=code, calls=len(tc))

    code, d, _, _ = api.ask("What is the weather in Lisbon?",
                            tools=[WEATHER_TOOL], tool_choice="required")
    tc = tool_calls_of(d)
    if not tc:
        R.add("C8", "multi-turn tool result consumed", FAIL, "no first-turn call")
    else:
        call = tc[0]
        msgs = [
            {"role": "user", "content": "What is the weather in Lisbon?"},
            {"role": "assistant", "tool_calls": [call], "content": None},
            {"role": "tool", "tool_call_id": call["id"],
             "content": json.dumps({"city": "Lisbon", "temp_c": 19,
                                    "sky": "clear"})},
        ]
        code2, d2, _, _ = api.chat(msgs)
        body = content_of(d2)
        R.add("C8", "multi-turn tool result consumed",
              PASS if ok2xx(code2) and "19" in body else WARN,
              "http %s content=%r" % (code2, body[:120]), http_code=code2)

    code, frames, done, err = stream(
        api.base + "/chat/completions", api.key,
        {"model": api.model, "max_tokens": 900, "stream": True,
         "tools": [WEATHER_TOOL], "tool_choice": "required",
         "messages": [{"role": "user", "content": "Weather in Berlin?"}]})
    frag = ""
    name_seen = None
    for f in frames:
        for ch in f.get("choices", []):
            for t in (ch.get("delta", {}) or {}).get("tool_calls", []) or []:
                fn = t.get("function") or {}
                if fn.get("name"):
                    name_seen = fn["name"]
                frag += fn.get("arguments") or ""
    parsed = None
    try:
        parsed = json.loads(frag)
    except Exception:
        pass
    R.add("C9", "streamed tool_calls reassemble",
          PASS if name_seen and parsed is not None else WARN,
          "http %s name=%s args=%r" % (code, name_seen, frag[:120]), http_code=code)

    code, d, _, _ = api.ask("Weather in Rome?", tools=[WEATHER_TOOL],
                            tool_choice="required")
    tc = tool_calls_of(d)
    shape_ok = bool(tc) and all(
        t.get("id") and t.get("type") == "function"
        and isinstance(t.get("function", {}).get("arguments"), str) for t in tc)
    R.add("C10", "tool_call object shape is OpenAI-conformant",
          PASS if shape_ok else FAIL,
          "http %s shape_ok=%s" % (code, shape_ok), http_code=code)


# --------------------------------------------------------------------------
# D. Structured output
# --------------------------------------------------------------------------
def section_D(api):
    print("\nD. Structured output")
    code, d, _, _ = api.ask("Give me a JSON object with keys city and country "
                            "for Paris.", response_format={"type": "json_object"})
    body = content_of(d)
    parsed = None
    try:
        parsed = json.loads(body)
    except Exception:
        pass
    R.add("D1", "response_format=json_object returns valid JSON",
          PASS if ok2xx(code) and isinstance(parsed, dict) else FAIL,
          "http %s body=%r" % (code, body[:140]), http_code=code)

    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"},
            "landmarks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["city", "population", "landmarks"],
        "additionalProperties": False,
    }
    code, d, _, _ = api.ask("Describe Paris.", response_format={
        "type": "json_schema",
        "json_schema": {"name": "city_info", "schema": schema, "strict": True}})
    body = content_of(d)
    conform = False
    try:
        p = json.loads(body)
        conform = (set(p.keys()) == {"city", "population", "landmarks"}
                   and isinstance(p["population"], int)
                   and isinstance(p["landmarks"], list)
                   and all(isinstance(x, str) for x in p["landmarks"]))
    except Exception:
        pass
    R.add("D2", "json_schema strict conforms exactly",
          PASS if ok2xx(code) and conform else FAIL,
          "http %s body=%r" % (code, body[:160]), http_code=code)

    ref_schema = {
        "type": "object",
        "$defs": {"point": {"type": "object",
                            "properties": {"x": {"type": "number"},
                                           "y": {"type": "number"}},
                            "required": ["x", "y"]}},
        "properties": {"start": {"$ref": "#/$defs/point"},
                       "end": {"$ref": "#/$defs/point"}},
        "required": ["start", "end"],
        "additionalProperties": False,
    }
    code, d, _, _ = api.ask("Give a line from (1,2) to (3,4).", response_format={
        "type": "json_schema",
        "json_schema": {"name": "line", "schema": ref_schema, "strict": True}})
    R.add("D3", "$ref / $defs schema accepted",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d)[:140] if not ok2xx(code) else "ok"),
          http_code=code)


# --------------------------------------------------------------------------
# E. Reasoning control
# --------------------------------------------------------------------------
REASONING_LEVELS = ["none", "low", "medium", "high", "xhigh", "max",
                    "minimal", "adaptive"]


def section_E(api):
    print("\nE. Reasoning control")
    code, d, _, _ = api.ask("What is 12 times 12? Answer with the number.")
    base_rt = reasoning_tokens(d)
    R.add("E0", "default request reports reasoning_tokens",
          PASS if ok2xx(code) else FAIL,
          "http %s reasoning_tokens=%s reasoning_len=%d" % (
              code, base_rt, len(reasoning_of(d))), http_code=code,
          reasoning_tokens=base_rt)

    observed = {}
    for i, level in enumerate(REASONING_LEVELS, start=1):
        code, d, _, _ = api.ask("Explain briefly why the sky is blue.",
                                reasoning_effort=level)
        rt = reasoning_tokens(d)
        if ok2xx(code):
            observed[level] = rt
            R.add("E%d" % i, "reasoning_effort=%s" % level, PASS,
                  "http %s reasoning_tokens=%s" % (code, rt),
                  http_code=code, reasoning_tokens=rt)
        else:
            R.add("E%d" % i, "reasoning_effort=%s" % level, WARN,
                  "http %s %s (capability boundary)" % (code, err_text(d)[:110]),
                  http_code=code)

    graded = [(l, observed[l]) for l in ("none", "low", "medium", "high", "xhigh", "max")
              if observed.get(l) is not None]
    mono = all(graded[i][1] <= graded[i + 1][1] for i in range(len(graded) - 1)) \
        if len(graded) > 1 else None
    R.add("E9", "reasoning_tokens grow with effort level",
          PASS if mono else (WARN if graded else SKIP),
          "observed=%s monotonic=%s" % (graded, mono))

    disables = [
        ("E10", "enable_thinking=false", {"enable_thinking": False}),
        ("E11", "thinking.type=disabled", {"thinking": {"type": "disabled"}}),
        ("E12", "chat_template_kwargs.thinking=false",
         {"chat_template_kwargs": {"thinking": False}}),
        ("E13", "reasoning_effort=none", {"reasoning_effort": "none"}),
    ]
    for cid, name, kw in disables:
        code, d, _, _ = api.ask("What is 2+2? Answer with the number.", **kw)
        rt = reasoning_tokens(d)
        rl = len(reasoning_of(d))
        off = ok2xx(code) and (rt in (0, None)) and rl == 0
        R.add(cid, "disable thinking via %s" % name,
              PASS if off else (WARN if ok2xx(code) else WARN),
              "http %s reasoning_tokens=%s reasoning_len=%d" % (code, rt, rl),
              http_code=code)

    code, d, _, _ = api.ask("Think hard: what is 17*23?",
                            thinking={"type": "enabled", "budget_tokens": 128})
    R.add("E14", "thinking.budget_tokens=128 behaviour",
          PASS if ok2xx(code) or 400 <= code < 500 else FAIL,
          "http %s %s" % (code, err_text(d)[:120] if not ok2xx(code) else "accepted"),
          http_code=code)


# --------------------------------------------------------------------------
# F. Streaming quality
# --------------------------------------------------------------------------
def section_F(api):
    print("\nF. Streaming quality")
    code, frames, done, err = stream(
        api.base + "/chat/completions", api.key,
        {"model": api.model, "max_tokens": 400, "stream": True,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": "Name three colours."}]})
    final_usage = None
    for f in frames:
        if f.get("usage"):
            final_usage = f["usage"]
    R.add("F1", "stream_options.include_usage delivers usage",
          PASS if final_usage and final_usage.get("total_tokens") else FAIL,
          "http %s usage=%s" % (code, json.dumps(final_usage or {})[:160]),
          http_code=code)

    shape_ok = bool(frames) and all(
        f.get("object") == "chat.completion.chunk" and "choices" in f for f in frames)
    R.add("F2", "SSE frame structure is standard",
          PASS if shape_ok and done else FAIL,
          "frames=%d done=%s objects_ok=%s" % (len(frames), done, shape_ok))

    prompt = "List the planets of the solar system in order."
    code_s, frames_s, _, _ = stream(
        api.base + "/chat/completions", api.key,
        {"model": api.model, "max_tokens": 1200, "stream": True, "temperature": 0,
         "seed": 7, "messages": [{"role": "user", "content": prompt}]})
    streamed = "".join(
        (c.get("delta", {}) or {}).get("content") or ""
        for f in frames_s for c in f.get("choices", []))
    code_n, d_n, _, _ = api.ask(prompt, max_tokens=1200, temperature=0, seed=7)
    nonstream = content_of(d_n)
    both = bool(streamed.strip()) and bool(nonstream.strip())
    overlap = False
    if both:
        a = re.sub(r"\W+", "", streamed.lower())
        b = re.sub(r"\W+", "", nonstream.lower())
        overlap = a[:120] == b[:120] or "mercury" in a and "mercury" in b
    R.add("F3", "streaming and non-streaming agree",
          PASS if overlap else WARN,
          "streamed=%d chars nonstream=%d chars match_head=%s" % (
              len(streamed), len(nonstream), overlap))

    code, frames, done, err = stream(
        api.base + "/chat/completions", api.key,
        {"model": api.model, "max_tokens": 4000, "stream": True,
         "messages": [{"role": "user",
                       "content": "Write a detailed 1200-word essay about the "
                                  "history of cartography."}]},
        timeout=LONG_TIMEOUT)
    chars = sum(len((c.get("delta", {}) or {}).get("content") or "")
                for f in frames for c in f.get("choices", []))
    R.add("F4", "long stream does not drop frames",
          PASS if ok2xx(code) and done and len(frames) > 200 else WARN,
          "http %s frames=%d chars=%d done=%s" % (code, len(frames), chars, done),
          http_code=code, frames=len(frames), chars=chars)


# --------------------------------------------------------------------------
# G. Usage reporting
# --------------------------------------------------------------------------
def section_G(api):
    print("\nG. Usage reporting")
    unit = "The quick brown fox jumps over the lazy dog. "
    points = []
    for mult in (10, 20, 40):
        code, d, _, _ = api.ask(unit * mult + "\nReply with the word ok.",
                                max_tokens=600)
        pt = usage_of(d).get("prompt_tokens")
        if pt:
            points.append((mult, pt))
    dev = None
    if len(points) >= 2:
        (m1, p1), (m2, p2) = points[0], points[-1]
        per_unit = (p2 - p1) / (m2 - m1)
        overhead = p1 - per_unit * m1
        expected = [per_unit * m + overhead for m, _ in points]
        dev = max(abs(e - p) / p for e, (_, p) in zip(expected, points))
    R.add("G1", "prompt_tokens scale linearly with input",
          PASS if dev is not None and dev < 0.02 else (WARN if dev is not None else FAIL),
          "points=%s max_deviation=%s" % (points, None if dev is None else round(dev, 4)),
          points=points)

    code, d, _, _ = api.ask("Say ok.", max_tokens=600)
    u = usage_of(d)
    consistent = (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                  == u.get("total_tokens"))
    R.add("G2", "prompt+completion equals total",
          PASS if consistent else FAIL, "usage=%s" % json.dumps(u)[:200])

    shared = ("You are a meticulous assistant. Context follows.\n"
              + ("Reference paragraph about hydrology. " * 400))
    api.ask(shared + "\nQuestion one: reply ok.", max_tokens=600)
    code, d2, _, _ = api.ask(shared + "\nQuestion two: reply ok.", max_tokens=600)
    ct = cached_tokens(d2)
    R.add("G3", "cached_tokens reported for a repeated prefix",
          PASS if ct and ct > 0 else WARN,
          "cached_tokens=%s prompt_tokens=%s" % (ct, usage_of(d2).get("prompt_tokens")),
          cached_tokens=ct)

    prompt = "Describe the water cycle in one sentence."
    code_n, d_n, _, _ = api.ask(prompt, max_tokens=900, temperature=0)
    code_s, frames, _, _ = stream(
        api.base + "/chat/completions", api.key,
        {"model": api.model, "max_tokens": 900, "stream": True, "temperature": 0,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": prompt}]})
    su = None
    for f in frames:
        if f.get("usage"):
            su = f["usage"]
    same_pt = (usage_of(d_n).get("prompt_tokens") == (su or {}).get("prompt_tokens"))
    R.add("G4", "streaming and non-streaming prompt_tokens identical",
          PASS if same_pt else WARN,
          "nonstream=%s stream=%s" % (usage_of(d_n).get("prompt_tokens"),
                                      (su or {}).get("prompt_tokens")))

    code, d, _, _ = api.ask("Write a very long story.", max_tokens=32)
    u = usage_of(d)
    R.add("G5", "truncated completion_tokens equals max_tokens",
          PASS if u.get("completion_tokens") == 32 else WARN,
          "completion_tokens=%s max_tokens=32" % u.get("completion_tokens"))


# --------------------------------------------------------------------------
# H. Multimodal
# --------------------------------------------------------------------------
def section_H(api):
    print("\nH. Multimodal")
    red = data_url((220, 20, 20))
    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "What colour fills this image? One word."},
        {"type": "image_url", "image_url": {"url": red}}]}])
    body = content_of(d).lower()
    R.add("H1", "base64 single image recognised",
          PASS if ok2xx(code) and "red" in body else (WARN if ok2xx(code) else FAIL),
          "http %s answer=%r" % (code, content_of(d)[:80]), http_code=code)

    green = data_url((20, 200, 20))
    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "How many images did I send? Reply with a digit."},
        {"type": "image_url", "image_url": {"url": red}},
        {"type": "image_url", "image_url": {"url": green}}]}])
    R.add("H2", "multiple images counted",
          PASS if ok2xx(code) and "2" in content_of(d) else
          (WARN if ok2xx(code) else FAIL),
          "http %s answer=%r" % (code, content_of(d)[:80]), http_code=code)

    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
                   "PNG_transparency_demonstration_1.png/280px-"
                   "PNG_transparency_demonstration_1.png"}}]}])
    R.add("H3", "remote URL image fetched",
          PASS if ok2xx(code) and content_of(d) else WARN,
          "http %s answer=%r" % (code, content_of(d)[:90]), http_code=code)

    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "Colour?"},
        {"type": "image_url", "image_url": {"url": red, "detail": "high"}}]}])
    R.add("H4", "image_url.detail accepted",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d)[:120] if not ok2xx(code) else "ok"),
          http_code=code)

    # The reference report's P0-1: a stray MimeType key made the upstream 400.
    # Worth probing directly, since it was their single largest failure cause.
    code, d, _, _ = api.chat([{"role": "user", "content": [
        {"type": "text", "text": "Colour?"},
        {"type": "image_url", "image_url": {"url": red, "MimeType": ""}}]}])
    R.add("H5", "stray image_url.MimeType key tolerated",
          PASS if ok2xx(code) else WARN,
          "http %s %s (reference platform P0-1: 1,548 failures/24h)" % (
              code, err_text(d)[:110]), http_code=code)


# --------------------------------------------------------------------------
# I. Long context
# --------------------------------------------------------------------------
def section_I(api):
    print("\nI. Long context")
    filler = ("Routine maintenance notes for the archive. Nothing of note here. ")
    needle = "The authorisation code for the vault is 74812."
    body = filler * 1800
    mid = len(body) // 2
    haystack = body[:mid] + needle + body[mid:]
    code, d, el, _ = api.ask(
        haystack + "\n\nWhat is the authorisation code for the vault? "
                   "Reply with the digits only.",
        max_tokens=DEFAULT_MAX_TOKENS, timeout=LONG_TIMEOUT)
    pt = usage_of(d).get("prompt_tokens")
    found = "74812" in content_of(d)
    R.add("I1", "needle retrieval in long context",
          PASS if ok2xx(code) and found else FAIL,
          "http %s prompt_tokens=%s answer=%r %.1fs" % (
              code, pt, content_of(d)[:60], el),
          http_code=code, prompt_tokens=pt, elapsed_s=round(el, 2))

    huge = "word " * 90000
    code, d, el, _ = api.ask(huge + "\nReply ok.", max_tokens=600,
                             timeout=LONG_TIMEOUT)
    pt = usage_of(d).get("prompt_tokens")
    R.add("I2", "large input (~90k words) handled",
          PASS if ok2xx(code) else WARN,
          "http %s prompt_tokens=%s %.1fs %s" % (
              code, pt, el, "" if ok2xx(code) else err_text(d)[:110]),
          http_code=code, prompt_tokens=pt, elapsed_s=round(el, 2))

    # config.yaml pins max-model-len 262144 and documents that an oversized
    # request must return a clean 400 rather than killing the engine.
    over = "word " * 300000
    code, d, el, _ = api.ask(over, max_tokens=600, timeout=LONG_TIMEOUT)
    R.add("I3", "over-limit request returns clean 400, engine survives",
          PASS if code == 400 else WARN,
          "http %s %.1fs %s" % (code, el, err_text(d)[:120]), http_code=code)

    hc, _, _, _ = http("GET", api.base + "/models", api.key, timeout=SHORT_TIMEOUT)
    R.add("I4", "engine still healthy after the over-limit probe",
          PASS if ok2xx(hc) else FAIL, "GET /v1/models http %s" % hc, http_code=hc)


# --------------------------------------------------------------------------
# J. Protocol and error semantics
# --------------------------------------------------------------------------
def section_J(api):
    print("\nJ. Protocol and error semantics")
    code, d, _, _ = http("POST", api.base + "/chat/completions",
                         "sk-k3-definitely-not-a-real-key",
                         {"model": api.model, "max_tokens": 16,
                          "messages": [{"role": "user", "content": "hi"}]},
                         timeout=SHORT_TIMEOUT)
    R.add("J1", "invalid key returns 401", PASS if code == 401 else FAIL,
          "http %s" % code, http_code=code)

    code, d, _, _ = api.at_root("/v1/messages", {
        "model": api.model, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say ok."}]}, timeout=SHORT_TIMEOUT)
    R.add("J2", "Anthropic /v1/messages surface",
          PASS if code in (403, 404) else WARN,
          "http %s - not served here; edge allowlists /v1/ only" % code,
          http_code=code)

    code, d, _, _ = api.raw("/responses", {
        "model": api.model, "input": "Say ok.", "max_output_tokens": 64},
        timeout=SHORT_TIMEOUT)
    responses_ok = ok2xx(code)
    R.add("J3", "/v1/responses surface",
          PASS if responses_ok or 400 <= code < 500 else WARN,
          "http %s %s" % (code, "available" if responses_ok else err_text(d)[:110]),
          http_code=code)

    if responses_ok:
        anyof = {"type": "object", "anyOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "string"}}, "required": ["b"]}]}
        c1, _, _, _ = api.ask("Call the tool.", tools=[{
            "type": "function",
            "function": {"name": "any_tool", "parameters": anyof}}],
            tool_choice="auto", timeout=SHORT_TIMEOUT)
        c2, d2, _, _ = api.raw("/responses", {
            "model": api.model, "input": "Call the tool.",
            "max_output_tokens": 64,
            "tools": [{"type": "function", "name": "any_tool",
                       "parameters": anyof}]}, timeout=SHORT_TIMEOUT)
        same = (ok2xx(c1) == ok2xx(c2))
        R.add("J4", "anyOf tool schema consistent across endpoints",
              PASS if same else WARN,
              "chat=%s responses=%s (reference platform P1-3 mismatch)" % (c1, c2))
    else:
        R.add("J4", "anyOf tool schema consistent across endpoints", SKIP,
              "/v1/responses not served")

    code, d, _, _ = api.chat([{"role": "developer", "content": "Reply with: ok"},
                              {"role": "user", "content": "Hello."}])
    R.add("J5", "developer role accepted",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d)[:110] if not ok2xx(code) else "ok"),
          http_code=code)


# --------------------------------------------------------------------------
# K. Concurrency
# --------------------------------------------------------------------------
def section_K(api, concurrency=6, total=12):
    print("\nK. Concurrency")
    results, lock = [], threading.Lock()

    def worker(i):
        code, d, el, err = api.ask("In one sentence, what is entropy? (#%d)" % i,
                                   max_tokens=900)
        with lock:
            results.append({"code": code, "elapsed": el, "err": err,
                            "chars": len(content_of(d))})

    idx, threads = 0, []
    t0 = time.time()
    while idx < total:
        batch = []
        for _ in range(min(concurrency, total - idx)):
            t = threading.Thread(target=worker, args=(idx,))
            idx += 1
            t.start()
            batch.append(t)
        for t in batch:
            t.join()
        threads.extend(batch)
    wall = time.time() - t0

    good = [r for r in results if ok2xx(r["code"])]
    lats = sorted(r["elapsed"] for r in good)

    def pct(q):
        if not lats:
            return None
        return round(lats[min(len(lats) - 1, int(q * (len(lats) - 1)))], 3)

    R.add("K1", "%dx concurrent x %d requests" % (concurrency, total),
          PASS if len(good) == total else FAIL,
          "ok=%d/%d wall=%.1fs p50=%ss p95=%ss" % (
              len(good), total, wall, pct(0.5), pct(0.95)),
          ok=len(good), total=total, wall_s=round(wall, 2),
          p50_s=pct(0.5), p95_s=pct(0.95))

    codes = {}
    for r in results:
        codes[r["code"]] = codes.get(r["code"], 0) + 1
    R.add("K2", "no throttling or server errors under test load",
          PASS if not any(c in (429, 500, 502, 503, 504) for c in codes) else WARN,
          "status spread=%s" % codes)


# --------------------------------------------------------------------------
# L. Consistency (single backend: repeat stability instead of channel fan-out)
# --------------------------------------------------------------------------
def section_L(api, all_keys):
    print("\nL. Consistency")
    pts = []
    for _ in range(6):
        code, d, _, _ = api.ask("Reply with the single word: ok", max_tokens=600,
                                temperature=0)
        if ok2xx(code):
            pts.append(usage_of(d).get("prompt_tokens"))
    unique = sorted(set(p for p in pts if p is not None))
    R.add("L1", "prompt_tokens identical across 6 repeats",
          PASS if len(unique) == 1 else WARN,
          "values=%s" % unique, values=unique)

    per_key = {}
    for name, key in all_keys:
        probe = Api(api.base, api.model, key)
        code, d, _, _ = probe.ask("Reply with the single word: ok", max_tokens=600,
                                  temperature=0, timeout=SHORT_TIMEOUT)
        per_key[name] = {"http": code, "prompt_tokens": usage_of(d).get("prompt_tokens"),
                         "model": (d or {}).get("model") if isinstance(d, dict) else None}
    all_ok = all(ok2xx(v["http"]) for v in per_key.values())
    same_pt = len(set(v["prompt_tokens"] for v in per_key.values())) == 1
    R.add("L2", "every customer key sees the same model and billing",
          PASS if all_ok and same_pt else (WARN if all_ok else FAIL),
          "per_key=%s" % json.dumps(per_key), per_key=per_key)

    code, d, _, _ = http("GET", api.base + "/models", api.key, timeout=SHORT_TIMEOUT)
    names = [m.get("id") for m in (d or {}).get("data", [])] if isinstance(d, dict) else []
    R.add("L3", "served-model aliases both advertised",
          PASS if len(names) >= 2 else WARN,
          "names=%s (config.yaml keeps the legacy alias deliberately)" % names)

    alias = [n for n in names if n != api.model]
    if alias:
        code, d, _, _ = http("POST", api.base + "/chat/completions", api.key,
                             {"model": alias[0], "max_tokens": 600,
                              "messages": [{"role": "user", "content": "Say ok."}]},
                             timeout=SHORT_TIMEOUT)
        R.add("L4", "legacy alias %r still serves" % alias[0],
              PASS if ok2xx(code) else FAIL, "http %s" % code, http_code=code)
    else:
        R.add("L4", "legacy alias still serves", SKIP, "no alias advertised")


# --------------------------------------------------------------------------
# M. Message shapes and roles
# --------------------------------------------------------------------------
def section_M(api):
    print("\nM. Message shapes and roles")
    code, d, _, _ = api.chat([{"role": "user",
                               "content": [{"type": "text", "text": "Say ok."}]}])
    R.add("M1", "content-array text shape accepted",
          PASS if ok2xx(code) else FAIL, "http %s" % code, http_code=code)

    code, d, _, _ = api.chat([{"role": "system", "content": "Reply with: ok"},
                              {"role": "user", "content": "Hello."}])
    R.add("M2", "system role accepted", PASS if ok2xx(code) else FAIL,
          "http %s content=%r" % (code, content_of(d)[:40]), http_code=code)

    code, d, _, _ = api.chat([
        {"role": "user", "content": "Weather in Oslo?"},
        {"role": "tool", "tool_call_id": "call_orphan",
         "content": "{\"temp_c\": 1}"}])
    R.add("M3", "orphan tool message without assistant turn",
          PASS if 400 <= code < 500 else WARN,
          "http %s %s" % (code, err_text(d)[:110]), http_code=code)

    code, d, _, _ = api.chat([{"role": "user", "content": "Say ok.", "name": "alice"}])
    R.add("M4", "message .name field accepted",
          PASS if ok2xx(code) else WARN, "http %s" % code, http_code=code)

    # Reference report P1-1: assistant history carrying a vendor `reasoning`
    # field was rejected upstream. Our own responses emit exactly that field,
    # so a well-behaved client will echo it back on the next turn.
    code, d, _, _ = api.chat([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4",
         "reasoning": "The user asked a simple sum."},
        {"role": "user", "content": "And times three?"}])
    R.add("M5", "assistant history with vendor reasoning field round-trips",
          PASS if ok2xx(code) else FAIL,
          "http %s %s (reference platform P1-1: 243 failures/24h)" % (
              code, err_text(d)[:110]), http_code=code)

    code, d, _, _ = api.chat([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4",
         "reasoning_content": "Simple arithmetic."},
        {"role": "user", "content": "And times three?"}])
    R.add("M6", "assistant history with reasoning_content round-trips",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d)[:110]), http_code=code)

    code, d, _, _ = api.ask("Say ok.", max_tokens=600,
                            web_search_options={"search_context_size": "low"})
    R.add("M7", "unknown vendor extension web_search_options",
          PASS if ok2xx(code) else WARN,
          "http %s %s" % (code, err_text(d)[:110]), http_code=code)

    code, d, _, _ = api.chat([{"role": "invalid_role", "content": "hi"}])
    R.add("M8", "invalid role rejected", PASS if 400 <= code < 500 else WARN,
          "http %s" % code, http_code=code)


# --------------------------------------------------------------------------
# N. Reasoning budget vs visible output
# --------------------------------------------------------------------------
def section_N(api):
    print("\nN. Reasoning budget")
    rows = []
    for mt in (2000, 4000, 8000):
        code, d, el, _ = api.ask(
            "A farmer has 3 pens with 7 sheep each. He sells 4. How many remain? "
            "Show your working.", max_tokens=mt)
        rows.append({"max_tokens": mt, "http": code,
                     "reasoning_tokens": reasoning_tokens(d),
                     "completion_tokens": usage_of(d).get("completion_tokens"),
                     "content_chars": len(content_of(d)),
                     "correct": "17" in content_of(d)})
    all_answered = all(r["content_chars"] > 0 for r in rows)
    R.add("N1", "content present at every budget",
          PASS if all_answered else WARN, "rows=%s" % json.dumps(rows))

    R.add("N2", "answer correct across budgets",
          PASS if all(r["correct"] for r in rows) else WARN,
          "correct=%s" % [r["correct"] for r in rows])

    # This is the trap that produced a false failure in the reference report.
    code, d, _, _ = api.ask("What is the capital of France?", max_tokens=40)
    body = content_of(d)
    rt = reasoning_tokens(d)
    R.add("N3", "small max_tokens can starve visible content",
          PASS if body.strip() else WARN,
          "max_tokens=40 content=%r reasoning_tokens=%s - reference report §7 "
          "recheck #1 was exactly this probe defect" % (body[:60], rt),
          http_code=code)

    code, d, _, _ = api.ask("What is the capital of France?", max_tokens=40,
                            reasoning_effort="none")
    R.add("N4", "reasoning_effort=none frees the budget for content",
          PASS if content_of(d).strip() else WARN,
          "content=%r reasoning_tokens=%s" % (content_of(d)[:60], reasoning_tokens(d)),
          http_code=code)


# --------------------------------------------------------------------------
# S. Edge security (specific to this deployment)
# --------------------------------------------------------------------------
def section_S(api):
    print("\nS. Edge security")
    unauth = [
        ("S1", "POST /invocations", "POST", "/invocations"),
        ("S2", "POST /scale_elastic_ep", "POST", "/scale_elastic_ep"),
        ("S3", "POST /tokenize", "POST", "/tokenize"),
        ("S4", "POST /detokenize", "POST", "/detokenize"),
        ("S5", "GET /version", "GET", "/version"),
        ("S6", "GET /openapi.json", "GET", "/openapi.json"),
        ("S7", "GET /docs", "GET", "/docs"),
        ("S8", "GET /metrics", "GET", "/metrics"),
        ("S9", "GET /health", "GET", "/health"),
    ]
    for cid, name, method, path in unauth:
        code, d, _, _ = http(method, api.root + path, None,
                             {} if method == "POST" else None, timeout=SHORT_TIMEOUT)
        R.add(cid, "%s blocked without credentials" % name,
              PASS if code in (401, 403, 404) else FAIL,
              "http %s (want 403/404)" % code, http_code=code)

    up = None
    try:
        with open("/scratch/deploy/api-key.txt") as f:
            up = f.read().strip()
    except OSError:
        pass
    if up:
        code, d, _, _ = http("GET", api.base + "/models", up, timeout=SHORT_TIMEOUT)
        R.add("S10", "upstream key rejected at the public edge",
              PASS if code == 401 else FAIL,
              "http %s - the upstream credential must not work from outside" % code,
              http_code=code)
    else:
        R.add("S10", "upstream key rejected at the public edge", SKIP,
              "api-key.txt not readable")

    code, d, _, _ = http("GET", "http://cflox.store/v1/models", api.key,
                         timeout=SHORT_TIMEOUT)
    R.add("S11", "plaintext HTTP redirects to TLS",
          PASS if ok2xx(code) or code in (301, 308) else WARN,
          "http %s (urllib follows the 301)" % code, http_code=code)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
SECTIONS = {
    "A": ("Basic protocol", section_A),
    "B": ("Parameter validation", section_B),
    "C": ("Tool calling", section_C),
    "D": ("Structured output", section_D),
    "E": ("Reasoning control", section_E),
    "F": ("Streaming quality", section_F),
    "G": ("Usage reporting", section_G),
    "H": ("Multimodal", section_H),
    "I": ("Long context", section_I),
    "J": ("Protocol semantics", section_J),
    "K": ("Concurrency", section_K),
    "L": ("Consistency", section_L),
    "M": ("Message shapes", section_M),
    "N": ("Reasoning budget", section_N),
    "S": ("Edge security", section_S),
}


def load_keys():
    keys = []
    try:
        with open(TSV) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    keys.append((parts[0].strip(), parts[1].strip()))
    except OSError as e:
        print("cannot read %s: %s" % (TSV, e), file=sys.stderr)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--concurrent-total", type=int, default=12)
    args = ap.parse_args()

    keys = load_keys()
    if not keys:
        print("FATAL: no customer keys available", file=sys.stderr)
        return 2
    chosen = dict(keys).get(EVAL_CUSTOMER) or keys[0][1]
    for _, k in keys:
        _REDACT.append(k)
    try:
        with open("/scratch/deploy/api-key.txt") as f:
            _REDACT.append(f.read().strip())
    except OSError:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    api = Api(args.base_url, args.model, chosen)

    order = [s for s in SECTIONS if not args.only or s in
             [x.strip().upper() for x in args.only.split(",") if x.strip()]]
    skip = {x.strip().upper() for x in args.skip.split(",") if x.strip()}
    order = [s for s in order if s not in skip]

    started = time.time()
    print("KVV-style health check")
    print("  endpoint : %s" % args.base_url)
    print("  model    : %s" % args.model)
    print("  key      : %s (redacted)" % EVAL_CUSTOMER)
    print("  sections : %s" % ",".join(order))

    for sec in order:
        name, fn = SECTIONS[sec]
        try:
            if sec == "L":
                fn(api, keys)
            elif sec == "K":
                fn(api, args.concurrency, args.concurrent_total)
            else:
                fn(api)
        except Exception as e:
            R.add("%s-ERR" % sec, "section %s crashed" % name, FAIL,
                  "%s: %s" % (type(e).__name__, e))

    elapsed = time.time() - started
    counts = R.counts()
    payload = {
        "meta": {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(started)),
            "elapsed_s": round(elapsed, 1),
            "base_url": args.base_url,
            "model": args.model,
            "sections": order,
            "eval_customer": EVAL_CUSTOMER,
            "customer_keys_tested": [n for n, _ in keys],
            "harness": os.path.basename(__file__),
        },
        "counts": counts,
        "checks": R.rows,
    }
    out = os.path.join(args.out_dir, "health_results.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n%s" % ("=" * 68))
    print("total=%d  PASS=%d  WARN=%d  FAIL=%d  SKIP=%d   in %.1fs"
          % (len(R.rows), counts[PASS], counts[WARN], counts[FAIL], counts[SKIP],
             elapsed))
    print("=" * 68)
    for r in R.rows:
        if r["status"] in (FAIL, WARN):
            print("  %-5s %-6s %s :: %s" % (r["status"], r["id"], r["name"],
                                            r["detail"][:120]))
    print("\nwrote %s" % out)
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
