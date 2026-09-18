#!/usr/bin/env python3
"""Does the 256k window actually work, and does the model USE it?

Accepting a long prompt is not the same as attending over it. Each case buries
a unique fact ("needle") at a controlled depth and asks for it back, so a
failure to retrieve is distinguishable from a failure to accept.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8001"
KEY = open("/scratch/deploy/api-key.txt").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def chat(messages, max_tokens=64, timeout=900, **kw):
    body = {"model": "kimi-k3", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0, **kw}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


# Filler that is semantically inert but tokenises densely, so the needle cannot
# be found by "the only interesting sentence" heuristics.
FILLER = ("The quarterly logistics report notes routine variance in regional "
          "throughput across distribution centres. ")


def build(target_tokens, needle, depth=0.5):
    # ~15 tokens per filler sentence; calibrate by measuring, not guessing.
    approx = max(1, int(target_tokens / 15))
    body = [FILLER] * approx
    at = int(len(body) * depth)
    body.insert(at, f" IMPORTANT: the authorization code is {needle}. ")
    return "".join(body)


print(f"{'target':>9} {'actual':>9} {'depth':>6} {'TTFT-ish':>9} {'found':>6}  verdict")
print("-" * 68)

# The 200k case MUST return a clean 400, not a crash. That is the whole point
# of capping max-model-len at a verified length.
CASES = [(20_000, 0.5), (65_000, 0.5), (125_000, 0.1), (125_000, 0.5),
         (125_000, 0.95), (200_000, 0.5)]
results = []
for i, (target, depth) in enumerate(CASES):
    needle = f"{7000 + i * 913}-QX"
    prompt = build(target, needle, depth)
    q = (f"\n\nQuestion: what is the authorization code stated in the text above? "
         f"Reply with only the code.")
    try:
        # 48 tokens was too few: K3 is a reasoning model and spent the whole
        # budget before emitting any content, which read as a false negative.
        d, dt = chat([{"role": "user", "content": prompt + q}], max_tokens=256)
        used = d["usage"]["prompt_tokens"]
        msg = d["choices"][0]["message"]
        out = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        ok = needle in out
        results.append(ok)
        print(f"{target:>9,} {used:>9,} {depth:>6.2f} {dt:>8.1f}s {str(ok):>6}  "
              f"{'' if ok else repr(out[:60])}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:160]
        results.append(False)
        print(f"{target:>9,} {'-':>9} {depth:>6.2f} {'-':>9} {'ERR':>6}  HTTP {e.code} {body}")
    except Exception as e:
        results.append(False)
        print(f"{target:>9,} {'-':>9} {depth:>6.2f} {'-':>9} {'ERR':>6}  {type(e).__name__}: {e}")

print()
print(f"needle retrieval: {sum(results)}/{len(results)}")

# Multi-turn tool loop: the actual agentic pattern, where a tool RESULT is fed
# back and the model must continue rather than re-call the tool.
print()
print("=== agentic tool loop (call -> result -> continue) ===")
TOOLS = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from the repository",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]
try:
    d, dt = chat([{"role": "user",
                   "content": "Read the file config/db.yaml and tell me the port number in it."}],
                 max_tokens=300, tools=TOOLS, tool_choice="auto")
    m = d["choices"][0]["message"]
    tc = m.get("tool_calls")
    print(f"  turn 1: finish={d['choices'][0]['finish_reason']} tool_calls={bool(tc)}")
    if tc:
        call = tc[0]
        print(f"    -> {call['function']['name']}({call['function']['arguments']})")
        d2, _ = chat([
            {"role": "user",
             "content": "Read the file config/db.yaml and tell me the port number in it."},
            {"role": "assistant", "content": m.get("content") or "", "tool_calls": tc},
            {"role": "tool", "tool_call_id": call["id"],
             "content": "host: localhost\nport: 54329\nuser: admin"},
        ], max_tokens=200, tools=TOOLS)
        m2 = d2["choices"][0]["message"]
        txt = (m2.get("content") or "")
        print(f"  turn 2: finish={d2['choices'][0]['finish_reason']}")
        print(f"    -> {txt[:150]}")
        print(f"  round-trip used the tool result: {'54329' in txt}")
    else:
        print("  NO TOOL CALL - agentic use would break")
except Exception as e:
    print(f"  ERR {type(e).__name__}: {e}")
