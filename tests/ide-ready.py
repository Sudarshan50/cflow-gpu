#!/usr/bin/env python3
"""Does the public endpoint behave like a hosted provider to an IDE agent?

agentic-test.py covers the capability surface against 127.0.0.1:8001. This one
goes through nginx + a real customer key over TLS, and covers the parts an IDE
leans on that a non-streaming test cannot see: SSE deltas, assembled streaming
tool calls, the tool-result round trip, usage accounting, and what an
oversized request does.
"""
import concurrent.futures as cf
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

# This test deliberately goes through the public edge, so it needs a real
# CUSTOMER key (from customers.tsv), not the upstream key in api-key.txt.
#   K3_PUBLIC_URL=https://cflox.store/v1 K3_CUSTOMER_KEY=sk-k3-... ./ide-ready.py
BASE = os.environ.get("K3_PUBLIC_URL", "https://cflox.store/v1")
KEY = os.environ.get("K3_CUSTOMER_KEY", "")
if not KEY:
    sys.exit("K3_CUSTOMER_KEY is unset. Pass a key from customers.tsv:\n"
             "  K3_CUSTOMER_KEY=$(awk -F'\\t' '!/^#/&&NF{print $2; exit}' "
             "/scratch/deploy/customers.tsv) ./ide-ready.py")
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
CTX = ssl.create_default_context()
fails = []


def chk(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {detail}", flush=True)
    if not ok:
        fails.append(name)


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read())


def stream(body, timeout=600):
    """Return (ttft_s, chunks, total_s). Raises on HTTP error."""
    body = dict(body, stream=True)
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.time()
    ttft, chunks = None, []
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            chunks.append(json.loads(payload))
    return ttft, chunks, time.time() - t0


READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the repo",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}

print("== streaming (what the IDE renders token by token) ==")
try:
    ttft, chunks, total = stream({
        "model": "kimi-k3", "max_tokens": 200, "temperature": 0,
        "messages": [{"role": "user", "content": "Count from 1 to 20, space separated."}]})
    text = "".join((c["choices"][0]["delta"].get("content") or "")
                   for c in chunks if c.get("choices"))
    reasoning = "".join((c["choices"][0]["delta"].get("reasoning_content") or "")
                        for c in chunks if c.get("choices"))
    chk("SSE deltas stream incrementally", len(chunks) > 10,
        f"{len(chunks)} chunks, ttft={ttft:.2f}s, total={total:.2f}s")
    chk("streamed text is non-empty", bool(text.strip() or reasoning.strip()),
        f"content={len(text)}ch reasoning={len(reasoning)}ch")
    fr = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    chk("terminal finish_reason present", any(f for f in fr), f"{[f for f in fr if f]}")
except Exception as e:
    chk("SSE deltas stream incrementally", False, f"{type(e).__name__}: {str(e)[:120]}")

# Usage in stream: IDEs show cost/context meters from this.
try:
    _, chunks, _ = stream({
        "model": "kimi-k3", "max_tokens": 40, "temperature": 0,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Say ok."}]})
    usage = next((c["usage"] for c in chunks if c.get("usage")), None)
    chk("stream_options.include_usage returns usage", bool(usage),
        json.dumps(usage) if usage else "no usage chunk")
except Exception as e:
    chk("stream_options.include_usage returns usage", False, f"{type(e).__name__}: {str(e)[:120]}")

print()
print("== tool calling over the wire ==")
# Streaming tool calls must be assemblable from deltas by index.
try:
    _, chunks, _ = stream({
        "model": "kimi-k3", "max_tokens": 400, "temperature": 0,
        "tools": [READ_FILE], "tool_choice": "auto",
        "messages": [{"role": "user", "content": "Read the file src/main.py using the tool."}]})
    acc = {}
    for c in chunks:
        if not c.get("choices"):
            continue
        for tc in c["choices"][0]["delta"].get("tool_calls") or []:
            slot = acc.setdefault(tc["index"], {"name": "", "args": "", "id": ""})
            slot["id"] += tc.get("id") or ""
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            slot["args"] += fn.get("arguments") or ""
    ok = len(acc) == 1
    args = {}
    if ok:
        slot = acc[0]
        ok = slot["name"] == "read_file" and bool(slot["id"])
        try:
            args = json.loads(slot["args"])
        except Exception:
            ok = False
    chk("streamed tool_call assembles from deltas", ok,
        f"{ {i: (s['name'], s['args']) for i, s in acc.items()} }")
    chk("streamed tool args parse as JSON", args.get("path") == "src/main.py",
        f"args={args}")
    fr = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    chk("finish_reason == tool_calls", "tool_calls" in fr, f"{[f for f in fr if f]}")
except Exception as e:
    chk("streamed tool_call assembles from deltas", False, f"{type(e).__name__}: {str(e)[:120]}")

# The round trip: assistant tool_call -> role:tool result -> final answer.
try:
    first = post({"model": "kimi-k3", "max_tokens": 400, "temperature": 0,
                  "tools": [READ_FILE], "tool_choice": "auto",
                  "messages": [{"role": "user",
                                "content": "Read src/version.txt and tell me the version."}]})
    am = first["choices"][0]["message"]
    tc = (am.get("tool_calls") or [])[0]
    second = post({"model": "kimi-k3", "max_tokens": 400, "temperature": 0,
                   "tools": [READ_FILE],
                   "messages": [
                       {"role": "user",
                        "content": "Read src/version.txt and tell me the version."},
                       {"role": "assistant", "content": am.get("content"),
                        "tool_calls": am["tool_calls"]},
                       {"role": "tool", "tool_call_id": tc["id"],
                        "content": "4.7.1-beta"}]})
    m2 = second["choices"][0]["message"]
    out = (m2.get("content") or "") + (m2.get("reasoning_content") or "")
    chk("tool result round trip -> final answer", "4.7.1" in out,
        f"said {out.strip()[:70]!r}")
except Exception as e:
    chk("tool result round trip -> final answer", False, f"{type(e).__name__}: {str(e)[:120]}")

print()
print("== limits and failure modes an IDE will hit ==")
# Oversized prompt must be a clean 400, never a dead engine.
try:
    post({"model": "kimi-k3", "max_tokens": 20,
          "messages": [{"role": "user", "content": "word " * 200000}]}, timeout=300)
    chk("over-limit prompt -> clean 4xx", False, "request was accepted (!)")
except urllib.error.HTTPError as e:
    body = e.read().decode()[:130]
    chk("over-limit prompt -> clean 4xx", 400 <= e.code < 500, f"HTTP {e.code} {body}")
except Exception as e:
    chk("over-limit prompt -> clean 4xx", False, f"{type(e).__name__}: {str(e)[:110]}")

# Engine still alive after that rejection.
try:
    d = post({"model": "kimi-k3", "max_tokens": 10, "temperature": 0,
              "messages": [{"role": "user", "content": "Reply with ok."}]})
    chk("engine healthy after rejection", bool(d["choices"]), "200 OK")
except Exception as e:
    chk("engine healthy after rejection", False, f"{type(e).__name__}: {str(e)[:110]}")

# Parallel sessions: an IDE opens several agent turns at once.
def one(i):
    t0 = time.time()
    ttft, chunks, total = stream({
        "model": "kimi-k3", "max_tokens": 150, "temperature": 0,
        "messages": [{"role": "user", "content": f"In one sentence, what is a {['mutex','closure','B-tree','JIT','GIL','ABI','vtable','epoll'][i]}?"}]})
    return ttft, total, len(chunks)

try:
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(8)))
    ttfts = sorted(r[0] for r in res)
    chk("8 concurrent streams all complete", all(r[2] > 3 for r in res),
        f"ttft min/med/max = {ttfts[0]:.2f}/{ttfts[4]:.2f}/{ttfts[-1]:.2f}s")
except Exception as e:
    chk("8 concurrent streams all complete", False, f"{type(e).__name__}: {str(e)[:110]}")

print()
print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
