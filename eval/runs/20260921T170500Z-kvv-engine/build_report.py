#!/usr/bin/env python3
"""Build the shareable KVV capability report from measured artifacts."""

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = json.loads((HERE / "health_results.json").read_text())
OUT_HTML = HERE / "KVV-KIMI-K3-CAPABILITY-REPORT.html"
OUT_MD = HERE / "KVV-KIMI-K3-CAPABILITY-REPORT.md"
DOCS_MD = Path("/root/cflow-gpu/docs/KVV-KIMI-K3-CAPABILITY-REPORT.md")

SECTIONS = [
    ("A", "Basic protocol"),
    ("B", "Parameter validation"),
    ("C", "Tool calling"),
    ("D", "Structured output"),
    ("E", "Reasoning control"),
    ("F", "Streaming quality"),
    ("G", "Usage and prefix cache"),
    ("H", "Multimodal"),
    ("I", "Long context"),
    ("J", "Protocol surfaces"),
    ("K", "Concurrency"),
    ("L", "Consistency"),
    ("M", "Message shapes and roles"),
    ("N", "Reasoning budget vs visible output"),
    ("S", "Public-edge security"),
]

# Public-edge security was measured separately against https://api.cflowx.in
# because the engine bind is localhost and has no auth of its own.
EDGE_S = [
    {"id": "S1", "name": "POST /invocations blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S2", "name": "POST /scale_elastic_ep blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S3", "name": "POST /tokenize blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S4", "name": "POST /detokenize blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S5", "name": "GET /version blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S6", "name": "GET /openapi.json blocked without credentials", "status": "WARN", "detail": "http 200 — LiteLLM OpenAPI is publicly readable"},
    {"id": "S7", "name": "GET /docs blocked without credentials", "status": "PASS", "detail": "http 404"},
    {"id": "S8", "name": "GET /metrics blocked without credentials", "status": "PASS", "detail": "http 403"},
    {"id": "S9", "name": "GET /health blocked without credentials", "status": "PASS", "detail": "http 403"},
    {"id": "S12", "name": "GET /v1/models without a key", "status": "PASS", "detail": "http 401 Authorization Required"},
    {"id": "J1-edge", "name": "Invalid Bearer key on the public edge", "status": "PASS", "detail": "http 401 on /v1/models; inference paths currently also 429 gateway_recovery"},
]

VENDOR_BENCH = [
    ("Reasoning & knowledge", [
        ("GPQA Diamond", "93.5", "92.6", "94.1", "91.0", "93.5"),
        ("CritPt", "23.4", "28.6", "32.3", "20.9", "27.1"),
        ("AA-LCR", "74.7", "70.0", "73.7", "67.7", "74.3"),
        ("HLE-Full (no tool / with tool)", "43.5 / 56.0", "53.3 / 63.0", "44.5 / 58.0", "49.8 / 57.9", "41.4 / 52.2"),
    ]),
    ("Coding", [
        ("DeepSWE", "67.5", "70.0", "73.0", "59.0", "67.0"),
        ("ProgramBench", "77.8", "76.8", "77.6", "71.9", "70.8"),
        ("Terminal-Bench 2.1", "88.3", "88.0", "88.8", "84.6", "83.4"),
        ("FrontierSWE", "81.2", "86.6", "71.3", "66.7", "64.9"),
        ("SWE-Marathon", "42.0", "35.0", "39.0", "40.0", "14.0"),
        ("PostTrainBench", "36.6", "41.4", "34.6", "34.1", "28.4"),
        ("MLS-Bench-Lite", "48.3", "49.9", "46.2", "42.8", "35.5"),
        ("SciCode", "58.7", "60.2", "56.1", "53.5", "56.1"),
        ("Kimi Code Bench 2.0", "72.9", "76.9", "64.8", "71.7", "69.0"),
    ]),
    ("Agentic", [
        ("BrowseComp", "91.2", "88.0", "90.4", "84.3", "84.4"),
        ("DeepSearchQA (F1)", "95.0", "94.2", "—", "93.1", "—"),
        ("ResearchRubrics", "76.2", "—", "73.8", "73.5", "64.0"),
        ("GDPval-AA v2 (Elo)", "1686", "1747", "1736", "1593", "1491"),
        ("Toolathlon-Verified", "76.5", "77.9", "74.9", "76.2", "73.5"),
        ("MCPMark-Verified", "94.5", "87.4", "92.9", "76.4", "92.9"),
        ("MCP-Atlas", "84.2", "84.7", "83.6", "83.6", "82.8"),
        ("AutomationBench", "30.8", "29.1", "29.7", "27.2", "22.7"),
        ("JobBench", "54.3", "57.4", "45.4", "48.4", "38.3"),
        ("AA-Briefcase (Elo)", "1548", "1583", "1495", "1354", "1158"),
        ("Agents' Last Exam", "28.3", "25.7", "29.6", "27.0", "26.6"),
        ("APEX-Agents", "41.0", "43.3", "39.9", "39.4", "38.5"),
        ("OfficeQA Pro", "63.3", "69.9", "63.2", "63.9", "60.9"),
        ("SpreadsheetBench 2", "34.8", "34.7", "32.4", "31.6", "29.1"),
        ("OSWorld-Verified", "84.8", "85.0", "83.0", "83.4", "79.0"),
        ("OSWorld 2.0", "58.3", "66.1", "62.6", "55.7", "49.5"),
        ("SaaS-Bench", "60.1", "—", "61.4", "56.1", "43.8"),
        ("τ³-Banking", "33.4", "26.8", "33.0", "27.6", "31.3"),
        ("Harvey Lab-AA", "94.6", "93.6", "87.2", "91.1", "86.3"),
        ("CorpFin v2", "71.6", "71.8", "64.4", "66.7", "68.4"),
        ("Finance Agent v2", "54.4", "56.3", "53.8", "53.9", "51.8"),
        ("Legal Research Bench", "44.2", "49.5", "48.1", "43.8", "40.4"),
    ]),
    ("Vision", [
        ("WorldVQA ForceAnswer", "51.0", "56.7", "41.8", "39.1", "38.5"),
        ("OmniDocBench", "91.1", "89.8", "85.8", "87.9", "89.4"),
        ("PerceptionBench", "58.5", "57.2", "59.7", "47.2", "55.8"),
        ("Video-MME (w. sub)", "90.0", "—", "89.5", "86.0", "89.3"),
        ("MMVU", "82.1", "—", "81.2", "79.2", "81.7"),
        ("BabyVision w/ python", "85.7", "90.5", "88.9", "81.2", "83.6"),
        ("MMMU-Pro (no tool / with tool)", "81.6 / 83.4", "81.2 / 86.5", "83.0 / 84.6", "78.9 / 82.7", "81.2 / 83.2"),
        ("CharXiv RQ (no tool / with tool)", "84.8 / 91.3", "88.9 / 93.5", "84.6 / 89.1", "80.5 / 89.9", "84.1 / 89.0"),
        ("MathVision (no tool / with tool)", "94.3 / 97.8", "94.8 / 98.6", "95.8 / 97.8", "86.7 / 97.1", "92.2 / 96.8"),
        ("ZeroBench pass@5 (no tool / with tool)", "23.0 / 41.0", "23.0 / 46.0", "17.0 / 35.0", "17.0 / 34.0", "22.0 / 41.0"),
    ]),
]


def rows_for(prefix):
    return [r for r in CHECKS["checks"] if r["id"].startswith(prefix)
            or r["id"].startswith(prefix + "-")]


def counts_of(rows):
    c = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in rows:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def badge(status):
    return f'<span class="badge {status.lower()}">{html.escape(status)}</span>'


def md_badge(status):
    return f"**{status}**"


SECTION_NOTES = {
    "A": """The OpenAI `/v1/chat/completions` surface is complete: non-streaming and
streaming both return a standard object, the served name `FW-Kimi-K3` is
echoed, unknown message fields are ignored, and an unknown model name is a
clean 404. There is no embedding model. Video is rejected with a precise
error (`At most 0 video(s) may be provided`) — the checkpoint includes a
MoonViT-V2 encoder that can do video, but this vLLM serve is image+text
only.""",
    "B": """Sampling parameters follow the OpenAI ranges (`temperature` 0–2, `top_p`
0–1, penalties ±2). `max_tokens` truncates (`finish_reason=length`).
`n=2` returns two candidates. `logprobs` are returned when requested.
Oversized `max_tokens` is rejected against the 262,144 context cap.

Two real boundaries: an empty `messages` array is accepted (HTTP 200) rather
than rejected, and `seed` does not make two samples identical. Do not
promise deterministic sampling.""",
    "C": """Tool calling is a first-class, production-ready surface on this serve.

- `tool_choice=auto / none / required / specific function` all behave.
- `required` produced a call on 5/5 repeats.
- Nested objects and arrays in the schema were filled correctly
  (`Ada`, age 36, Delhi→Tokyo business, Tokyo→Sydney economy).
- The model routed to the right tool among 20 declared tools.
- `parallel_tool_calls` returned two calls for two cities.
- A follow-up turn that fed the tool result back produced the right
  answer (`19°C`, clear skies).
- Streamed `tool_calls` reassemble to valid JSON
  (`{"city":"Berlin","unit":"c"}`).
- The object shape is OpenAI-conformant (`id`, `type=function`,
  `arguments` as a string).

Moonshot requires the full assistant message — including `reasoning` /
`reasoning_content` and `tool_calls` — to be echoed back on the next
turn. This deployment accepts both field names.""",
    "D": """`response_format=json_object` returned valid JSON. Strict
`json_schema` returned exactly the required keys with the right types.
`$ref` / `$defs` schemas are accepted. Structured output can be used
for extraction and agent payloads.""",
    "E": """Thinking is on by default. The tokenizer only accepts
`thinking_effort` / `reasoning_effort` values **`low`**, **`high`**,
**`max`**. `none` is honoured by the API as a disable switch.

Unsupported OpenAI-style values (`medium`, `xhigh`, `minimal`,
`adaptive`) fail with a clear error. That is a capability boundary,
not a crash.

`reasoning_effort=none` and `chat_template_kwargs.thinking=false` turn
thinking off. `enable_thinking=false` and `thinking.type=disabled` are
ignored and the model still thinks.

The usage object does **not** currently populate
`completion_tokens_details.reasoning_tokens`. Reasoning is visible in
`choices[0].message.reasoning` (and accepted back as either `reasoning`
or `reasoning_content`). Do not bill or gate on `reasoning_tokens` until
that field is filled.""",
    "F": """SSE is standard (`object=chat.completion.chunk`, terminal `[DONE]`).
`stream_options.include_usage` delivers a final usage frame. A 3,104-frame
essay stream completed without dropped frames (8,074 visible characters).
Streaming and non-streaming answers agree on the same prompt.""",
    "G": """`prompt_tokens` scale linearly (10/20/40 copies of a unit sentence
→ 195/295/495 tokens, zero deviation). `prompt + completion = total`.
A repeated ~2.3k-token prefix reported `cached_tokens=2304` of
`prompt_tokens=2504` (92% of that prompt reused). Streaming and
non-streaming prompt token counts match. Truncation reports
`completion_tokens == max_tokens`.""",
    "H": """Native vision works for **inline/base64** images. A solid red PNG
was identified as “Red”. Two images were counted as “2”.
`image_url.detail` and a stray `MimeType` key are tolerated.

Remote HTTP(S) image URLs returned HTTP 422 — this engine does not
fetch URLs. Clients must send `data:image/...;base64,...`.

Video is rejected (see A8). The vendor card scores Video-MME / MMVU on
the full model; those numbers are not available on this serve.""",
    "I": """Served context is **262,144 tokens**, not the native 1,048,576.

- Needle “74812” retrieved from a 21,717-token haystack in 3.0 s.
- A 90,093-token prompt completed in 10.5 s.
- A ~300k-word prompt was rejected in 0.1 s with a clear 400
  (`maximum context length is 262144 tokens`).
- The engine stayed healthy afterwards.

KV cache on this process is 1,628,607 tokens (~43 GiB), which is about
6.2 concurrent 262k-token requests. Huge in-flight prompts will fill
that pool; that is an operational limit, not a model-quality limit.""",
    "J": """The engine itself is bound to `127.0.0.1:8001` and does **not**
authenticate. J1 (invalid key → 401) is therefore a FAIL on the engine
path and a PASS on the public edge (`api.cflowx.in` returns 401 without
a key).

`/v1/responses` is available and consistent with chat for `anyOf` tool
schemas. `/v1/messages` (Anthropic) is present on the engine (HTTP 200)
but is not part of the public allowlist contract. The `developer` role
is rejected (`Unknown message role 'developer'`). Use `system`.""",
    "K": """12 requests at concurrency 6 all succeeded (12/12 HTTP 200). Wall
49.6 s, p50 19.5 s, p95 24.4 s for short reasoning answers. No 429 / 5xx
from the engine under that load. This is a functional concurrency check,
not a throughput benchmark.""",
    "L": """The same prompt billed 95 tokens on six repeats. Aliases
`FW-Kimi-K3`, `kimi-k3`, and `moonshotai/Kimi-K3` are all advertised and
the legacy alias serves.""",
    "M": """Content-array text, `system`, `name`, unknown vendor extensions,
and both `reasoning` / `reasoning_content` history fields are accepted.
Orphan tool messages without a matching assistant `tool_calls` turn are
rejected with a precise Kimi-K3 error. Invalid roles are 400.""",
    "N": """A 3-pen / 7-sheep word problem was answered correctly (17) at
`max_tokens` 2000, 4000 and 8000, with visible content in every case.

Default thinking shares the completion budget. At `max_tokens=40` the
visible `content` was the start of the think channel
(“The user is asking a simple factual question…”). Setting
`reasoning_effort=none` on the same prompt produced a real answer
(“The capital of France is Paris…”). Clients that want a short answer
must either raise `max_tokens` or turn thinking off.""",
    "S": """Measured on `https://api.cflowx.in`, not on localhost. Engine
admin surfaces (`/tokenize`, `/metrics`, `/health`, `/invocations`) are
blocked at nginx (403/404). `/v1/models` requires a key (401).
`/openapi.json` is publicly readable (LiteLLM schema, no secrets).

At the time this report was written the public inference path was
again returning `429 gateway_recovery` after a KV-full incident earlier
in the day. That is an admission hold, not a model failure. The engine
behind it answered every KVV probe in this document.""",
}


def section_score_md(prefix, extra=None):
    rows = rows_for(prefix)
    if extra:
        rows = rows + extra
    c = counts_of(rows)
    return f"{c['PASS']} PASS · {c['WARN']} WARN · {c['FAIL']} FAIL · {c['SKIP']} SKIP"


def table_md(rows):
    lines = ["| ID | Check | Status | Evidence |", "|---|---|---|---|"]
    for r in rows:
        detail = (r.get("detail") or "").replace("\n", " ").replace("|", "/")
        if len(detail) > 220:
            detail = detail[:217] + "..."
        lines.append(f"| {r['id']} | {r['name']} | {md_badge(r['status'])} | {detail} |")
    return "\n".join(lines)


def table_html(rows):
    out = ['<table class="checks"><thead><tr><th>ID</th><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>']
    for r in rows:
        out.append(
            "<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td class=\"ev\">%s</td></tr>"
            % (html.escape(r["id"]), html.escape(r["name"]), badge(r["status"]),
               html.escape((r.get("detail") or "")[:400]))
        )
    out.append("</tbody></table>")
    return "\n".join(out)


engine_c = CHECKS["counts"]
edge_c = counts_of(EDGE_S)
all_pass = engine_c["PASS"] + edge_c["PASS"]
all_warn = engine_c["WARN"] + edge_c["WARN"]
all_fail = engine_c["FAIL"] + edge_c["FAIL"]
all_skip = engine_c["SKIP"] + edge_c["SKIP"]
all_n = all_pass + all_warn + all_fail + all_skip

md = []
md.append("# Kimi K3 — KVV Capability Report")
md.append("")
md.append("**Audience:** anyone who needs to know what the currently served Kimi K3 can and cannot do.")
md.append("**Evidence date:** 21 September 2026, 17:08–17:13 UTC.")
md.append("**Harness:** `cflow-gpu/eval/harness/kvv_health.py` (KVV sections A–N, plus public-edge S).")
md.append("**Engine run:** 282.0 s · 82 PASS · 10 WARN · 2 FAIL · 1 SKIP.")
md.append("**Combined with public-edge security:** %d checks · %d PASS · %d WARN · %d FAIL · %d SKIP."
          % (all_n, all_pass, all_warn, all_fail, all_skip))
md.append("")
md.append("This is not a marketing card. Every numbered check below was executed against the live model on this box. Vendor-published quality numbers are labelled **VENDOR** and were not re-run here.")
md.append("")

md.append("## 1. One-page verdict")
md.append("")
md.append("The currently served model is **Moonshot Kimi K3** (`moonshotai/Kimi-K3`, snapshot `f831ab66814297da540d832a5235f8e904f29d06`), exposed as `FW-Kimi-K3`. It is a working **OpenAI-compatible chat, tool-calling, structured-output, reasoning, streaming, and image** endpoint with a **262,144-token** served context.")
md.append("")
md.append("| You can rely on | You cannot rely on |")
md.append("|---|---|")
md.append("| Chat completions, streaming + `[DONE]` | Video input (rejected: 0 videos allowed) |")
md.append("| Tool calling: auto / none / required / forced / parallel / multi-turn | `reasoning_effort` values other than `none`, `low`, `high`, `max` |")
md.append("| Strict JSON schema and `json_object` | Remote image URLs (send base64) |")
md.append("| Inline / base64 images (colour ID, multi-image count) | Native 1,048,576-token window (served cap is 262,144) |")
md.append("| Needle retrieval at 22k tokens; 90k-token prompts | Deterministic `seed` sampling |")
md.append("| Prefix-cache `cached_tokens` on repeated prefixes | `usage.reasoning_tokens` (field is empty; use `message.reasoning`) |")
md.append("| `n=2`, `logprobs`, stop sequences, aliases | `developer` role (use `system`) |")
md.append("| Turning thinking off via `reasoning_effort=none` | `enable_thinking=false` / `thinking.type=disabled` (ignored) |")
md.append("| Public-edge 401 without a key | Engine-local auth (localhost vLLM has none) |")
md.append("| Clean 400 on over-length prompts | Embeddings endpoint (404) |")
md.append("")
md.append("**Practical client rule:** always send back the previous assistant message intact (`content` + `reasoning` + `tool_calls`). For short answers, either raise `max_tokens` or set `reasoning_effort` to `none`. For images, inline base64 only.")
md.append("")

md.append("## 2. What the model is")
md.append("")
md.append("Identity is taken from the loaded Hugging Face snapshot and the vendor model card. Architecture numbers are **from the checkpoint / card**, not guessed.")
md.append("")
md.append("| Item | Value | Source |")
md.append("|---|---|---|")
md.append("| Family | Kimi K3, Moonshot AI | vendor card |")
md.append("| Checkpoint | `moonshotai/Kimi-K3` | loaded path |")
md.append("| Snapshot | `f831ab66814297da540d832a5235f8e904f29d06` | local hub |")
md.append("| Architecture class | `KimiK3ForConditionalGeneration` / text `KimiLinearForCausalLM` | `config.json` |")
md.append("| Total / activated parameters | 2.8T / 104B | vendor card |")
md.append("| Layers | 93 (1 dense); 69 KDA + 24 gated MLA | vendor card + config |")
md.append("| Hidden size / heads | 7168 / 96 | config |")
md.append("| MoE | 896 experts, 16 routed + 2 shared per token, SiTU-GLU | vendor card |")
md.append("| Vocabulary | 163,840 | config |")
md.append("| Native context | 1,048,576 tokens | config `max_position_embeddings` |")
md.append("| **Served context** | **262,144 tokens** | vLLM `max_model_len` |")
md.append("| Quantization | MXFP4 weights, MXFP8 activations (QAT) | vendor card; engine `quantization=mxfp4` |")
md.append("| Compute dtype | bfloat16 | engine |")
md.append("| Vision encoder | MoonViT-V2, 401M, 27 layers, patch 14 | vendor card + `vision_config` |")
md.append("| Native modalities | text, image, video | vendor card |")
md.append("| **Served modalities** | **text + image** (video rejected) | KVV A8, H1–H5 |")
md.append("| Weights on disk | 1.56 TB across 96 safetensors | local snapshot |")
md.append("| License | Kimi K3 License | vendor card |")
md.append("")
md.append("Thinking is structural in the chat template. Every assistant turn has a `<think>` channel unless thinking is disabled. Official usage: default `reasoning_effort=max`; supported values `low` / `high` / `max`. This serve also accepts `none` to drop the think channel.")
md.append("")

md.append("## 3. This serving stack (measured 21 Sep 2026)")
md.append("")
md.append("| Item | Value |")
md.append("|---|---|")
md.append("| Host | `kimi-k3-t01`, Ubuntu 24.04.4, kernel 6.8.0-137-generic |")
md.append("| CPU / RAM | AMD EPYC 9575F, 2 NUMA nodes, 2.0 TiB RAM, KVM |")
md.append("| GPUs | 8 × AMD Instinct MI355X VF, gfx950, ~288 GiB each |")
md.append("| ROCm | 7.14.0 packages on host; container ROCm 7.2.3 historically |")
md.append("| Engine | vLLM `0.1.dev19253+g5f76ae224.d20260727`, V1, tokenizer_mode `kimi_k3` |")
md.append("| Image | `5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8` |")
md.append("| Parallelism | TP=8, PP=1, DP=1; expert-parallel **off** on this process |")
md.append("| MoE kernel | `AITER_MXFP4_BF16` |")
md.append("| GPU memory util | 0.88 |")
md.append("| KV cache | 1,628,607 tokens, 43.07 GiB, block 768, max concurrency 6.21× at 262k |")
md.append("| Prefix cache | enabled, SHA-256 |")
md.append("| Batch | `max_num_batched_tokens=4096`, `max_num_seqs=64`, chunked prefill on |")
md.append("| Scheduler | priority |")
md.append("| CUDA graphs | `FULL_AND_PIECEWISE`, capture up to 128 |")
md.append("| Parsers | `reasoning_parser=kimi_k3`, `tool_call_parser=kimi_k3`, auto tool choice |")
md.append("| Speculation | none |")
md.append("| Public URL | `https://api.cflowx.in/v1` |")
md.append("| Path | nginx → LiteLLM `:4000` → gateway `:8002` → vLLM `:8001` |")
md.append("| Served names | `FW-Kimi-K3`, `kimi-k3`, `moonshotai/Kimi-K3` |")
md.append("| Extra ROCm flags | `VLLM_ROCM_USE_AITER=1`, `AITER_SITUV2_A8W4=1`, Kimi K3 preroute/latent-tail FP8 |")
md.append("")
md.append("The KVV model-capability numbers were taken **directly from vLLM `:8001`** so LiteLLM cooldown / nginx recovery holds cannot fake a model failure. Public-edge security (section S) was taken from `api.cflowx.in`.")
md.append("")

md.append("## 4. Scorecard")
md.append("")
md.append("| Section | Topic | Engine / edge result |")
md.append("|---|---|---|")
for sid, name in SECTIONS:
    extra = EDGE_S if sid == "S" else None
    if sid == "S":
        md.append(f"| {sid} | {name} | {section_score_md('S', extra=EDGE_S)} |")
    else:
        md.append(f"| {sid} | {name} | {section_score_md(sid)} |")
md.append("")
md.append("Legend: **PASS** = the capability is present and behaved. **WARN** = works with a documented boundary, or an optional OpenAI feature this model does not implement. **FAIL** = the check’s strict expectation was not met. **SKIP** = not measurable (here: `reasoning_tokens` growth, because the usage field is empty).")
md.append("")

md.append("## 5. Detailed results")
md.append("")
for sid, name in SECTIONS:
    md.append(f"### {sid}. {name}")
    md.append("")
    md.append(SECTION_NOTES[sid])
    md.append("")
    if sid == "S":
        md.append(table_md(EDGE_S))
    else:
        md.append(table_md(rows_for(sid)))
    md.append("")

md.append("## 6. Response shape (measured sample)")
md.append("")
md.append("A live `reasoning_effort=low` request for `17 × 23` returned in 1.42 s:")
md.append("")
md.append("```json")
md.append('{')
md.append('  "object": "chat.completion",')
md.append('  "model": "FW-Kimi-K3",')
md.append('  "choices": [{')
md.append('    "finish_reason": "stop",')
md.append('    "message": {')
md.append('      "role": "assistant",')
md.append('      "content": "391",')
md.append('      "reasoning": "17*23=391. final number only. Ensure no extra.",')
md.append('      "reasoning_content": ""')
md.append('    }')
md.append('  }],')
md.append('  "usage": {')
md.append('    "prompt_tokens": 103,')
md.append('    "completion_tokens": 28,')
md.append('    "total_tokens": 131,')
md.append('    "prompt_tokens_details": {')
md.append('      "cached_tokens": 0,')
md.append('      "created_cache_tokens": 0,')
md.append('      "multimodal_tokens": null')
md.append('    }')
md.append('  }')
md.append('}')
md.append("```")
md.append("")
md.append("Message keys present: `role`, `content`, `reasoning`, `reasoning_content` (empty here), `annotations`, `audio`, `function_call`, `refusal`. Prefer `message.reasoning`. Echo it back as either `reasoning` or `reasoning_content` — both round-trip (M5, M6).")
md.append("")

md.append("## 7. How to call it")
md.append("")
md.append("```python")
md.append("from openai import OpenAI")
md.append("client = OpenAI(base_url='https://api.cflowx.in/v1', api_key='...')")
md.append("")
md.append("resp = client.chat.completions.create(")
md.append("    model='FW-Kimi-K3',  # also: kimi-k3, moonshotai/Kimi-K3")
md.append("    messages=[")
md.append("        {'role': 'system', 'content': 'Be concise.'},")
md.append("        {'role': 'user', 'content': 'Summarise the attached figure.'},")
md.append("    ],")
md.append("    max_tokens=2000,")
md.append("    reasoning_effort='low',   # none | low | high | max")
md.append("    tools=[...],              # optional")
md.append("    tool_choice='auto',")
md.append("    response_format={'type': 'json_schema', 'json_schema': {...}},")
md.append(")")
md.append("msg = resp.choices[0].message")
md.append("# next turn: send msg back intact, including msg.reasoning and msg.tool_calls")
md.append("```")
md.append("")
md.append("Images: `content: [{type:'text',...},{type:'image_url', image_url:{url:'data:image/png;base64,...'}}]`.")
md.append("")
md.append("Not served: `/v1/embeddings`. Not useful here: `seed` for bit-identical output, `developer` role, video URLs, remote image URLs.")
md.append("")

md.append("## 8. Boundaries a reader will hit")
md.append("")
md.append("1. **Thinking eats `max_tokens`.** Default thinking is on. A 40-token budget filled with think-channel text. Use `reasoning_effort=none` or a larger budget.")
md.append("2. **Only four effort values work:** `none`, `low`, `high`, `max`. Others 400/500.")
md.append("3. **Two disable switches are no-ops:** `enable_thinking=false` and `thinking.type=disabled`.")
md.append("4. **`reasoning_tokens` in usage is empty.** Count `message.reasoning` yourself if you need it.")
md.append("5. **Context is 262k, not 1M.** Over-limit is a clean 400.")
md.append("6. **KV pool is ~1.63M tokens.** A handful of 200k-token in-flight requests will pin the box. That is why public inference was held with `gateway_recovery` earlier today.")
md.append("7. **Images must be base64.** URL fetch is 422. Video is 400.")
md.append("8. **Empty `messages` is accepted.** Validate client-side.")
md.append("9. **`seed` is not deterministic.**")
md.append("10. **Engine localhost has no API key.** Auth exists only at the public edge.")
md.append("11. **LiteLLM can 429 `No deployments available`** for a few seconds after a stream if the router cools the only deployment down. Retry once. This is a proxy behaviour, not a model defect.")
md.append("12. **`/openapi.json` is public** on the edge (schema only).")
md.append("")

md.append("## 9. Vendor-published quality numbers (not re-run here)")
md.append("")
md.append("Copied from the Moonshot Kimi K3 model card. All Kimi K3 scores were taken by Moonshot at `reasoning_effort=max`, temperature 1.0. They describe the **weights**, not a guaranteed SLA of this 262k / 8×MI355X serve. Comparisons are Claude Fable 5 (max), GPT-5.6 Sol (max), Claude Opus 4.8 (max), GPT-5.5 (xhigh).")
md.append("")
for group, rows in VENDOR_BENCH:
    md.append(f"### {group}")
    md.append("")
    md.append("| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")
md.append("Source: `moonshotai/Kimi-K3` README, snapshot `f831ab6681…`. Footnotes on harnesses and dates are in that card. Video scores do not apply to this serve.")
md.append("")

md.append("## 10. Evidence")
md.append("")
md.append("| File | What it is |")
md.append("|---|---|")
md.append("| `eval/runs/20260921T170500Z-kvv-engine/health_results.json` | Raw KVV A–N results |")
md.append("| `eval/runs/20260921T170500Z-kvv-engine/run_health.log` | Console transcript |")
md.append("| `eval/runs/20260921T170233Z-kvv/env.json` | Hardware / engine capture |")
md.append("| `eval/runs/20260921T170233Z-kvv/engine_startup.log` | vLLM init + KV size |")
md.append("| Hugging Face snapshot `f831ab6681…` | `config.json`, `README.md`, `encoding_k3.py` |")
md.append("")
md.append("The 19 September 2026 KVV run (`20260919T145955Z-baseline`) is **not** used as evidence. After section B it recorded HTTP 0 / timeouts and is not a capability measurement.")
md.append("")
md.append("---")
md.append("")
md.append("*Prepared 21 September 2026. No customer keys or upstream credentials are included.*")

OUT_MD.write_text("\n".join(md) + "\n")
DOCS_MD.write_text("\n".join(md) + "\n")

# HTML
section_html = []
for sid, name in SECTIONS:
    extra = EDGE_S if sid == "S" else rows_for(sid)
    c = counts_of(extra)
    section_html.append(f"""
    <section id="sec-{sid.lower()}">
      <h3>{sid}. {html.escape(name)}
        <span class="mini">{c['PASS']} pass · {c['WARN']} warn · {c['FAIL']} fail</span></h3>
      <p>{html.escape(SECTION_NOTES[sid]).replace(chr(10), '<br>')}</p>
      {table_html(extra)}
    </section>""")

vendor_html = []
for group, rows in VENDOR_BENCH:
    vendor_html.append(f"<h4>{html.escape(group)}</h4>")
    vendor_html.append("<table class='vendor'><thead><tr><th>Benchmark</th><th>Kimi K3</th><th>Claude Fable 5</th><th>GPT-5.6 Sol</th><th>Claude Opus 4.8</th><th>GPT-5.5</th></tr></thead><tbody>")
    for r in rows:
        vendor_html.append("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in r) + "</tr>")
    vendor_html.append("</tbody></table>")

score_rows = []
for sid, name in SECTIONS:
    extra = EDGE_S if sid == "S" else rows_for(sid)
    c = counts_of(extra)
    score_rows.append(
        f"<tr><td><a href='#sec-{sid.lower()}'>{sid}</a></td><td>{html.escape(name)}</td>"
        f"<td>{c['PASS']}</td><td>{c['WARN']}</td><td>{c['FAIL']}</td><td>{c['SKIP']}</td></tr>"
    )

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kimi K3 — KVV Capability Report (21 Sep 2026)</title>
<style>
:root {{
  --bg:#f6f4ef; --card:#fff; --ink:#1b1a17; --muted:#5c5850;
  --line:#ddd6c8; --pass:#1b7f4a; --warn:#9a6b00; --fail:#b42318; --skip:#5c5850;
  --passbg:#e6f6ec; --warnbg:#fff4d6; --failbg:#fde8e6; --skipbg:#eeeae2;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:16px/1.55 ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial; color:var(--ink); background:var(--bg); }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 80px; }}
header {{ background:#1b1a17; color:#f6f4ef; padding:36px 20px 32px; }}
header .wrap {{ padding-top:0; padding-bottom:0; }}
header p {{ color:#d6d0c4; max-width:70ch; }}
h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:-0.02em; }}
h2 {{ margin:40px 0 12px; font-size:24px; }}
h3 {{ margin:28px 0 10px; font-size:19px; }}
h4 {{ margin:20px 0 8px; }}
.kicker {{ text-transform:uppercase; letter-spacing:.12em; font-size:12px; color:#c4b89a; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:20px 0; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
.card b {{ display:block; font-size:28px; letter-spacing:-0.03em; }}
.card span {{ color:var(--muted); font-size:13px; }}
.badge {{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.04em; padding:2px 7px; border-radius:999px; }}
.badge.pass {{ background:var(--passbg); color:var(--pass); }}
.badge.warn {{ background:var(--warnbg); color:var(--warn); }}
.badge.fail {{ background:var(--failbg); color:var(--fail); }}
.badge.skip {{ background:var(--skipbg); color:var(--skip); }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); font-size:14px; margin:10px 0 24px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ background:#f1eee6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
td.ev, .checks td:last-child {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; word-break:break-word; }}
.mini {{ font-size:13px; font-weight:500; color:var(--muted); margin-left:8px; }}
.callout {{ background:#fff8e6; border:1px solid #ead9a0; border-radius:10px; padding:12px 14px; }}
pre {{ background:#1b1a17; color:#f6f4ef; padding:14px 16px; border-radius:10px; overflow:auto; font-size:13px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.92em; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:800px) {{ .two {{ grid-template-columns:1fr; }} }}
footer {{ color:var(--muted); font-size:13px; margin-top:48px; }}
a {{ color:#1b4f8a; }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="kicker">KVV capability report · 21 September 2026</div>
    <h1>Kimi K3 as served today</h1>
    <p>A complete, evidence-backed account of what the current Kimi K3 deployment can do. Model-capability checks were run live against the vLLM engine. Vendor quality scores are labelled as such and were not re-run on this box.</p>
  </div>
</header>
<div class="wrap">
  <div class="cards">
    <div class="card"><b>{engine_c['PASS']}</b><span>KVV PASS (engine)</span></div>
    <div class="card"><b>{engine_c['WARN']}</b><span>KVV WARN (boundaries)</span></div>
    <div class="card"><b>{engine_c['FAIL']}</b><span>KVV FAIL</span></div>
    <div class="card"><b>282s</b><span>Engine suite wall time</span></div>
    <div class="card"><b>262k</b><span>Served context tokens</span></div>
    <div class="card"><b>2.8T</b><span>Parameters (104B active)</span></div>
  </div>

  <h2>What you can ship on</h2>
  <div class="two">
    <div class="card">
      <strong>Works now</strong>
      <p>OpenAI chat + stream, tool calling (auto/none/required/parallel/multi-turn), strict JSON schema, base64 images, reasoning (`low`/`high`/`max`/`none`), 90k-token prompts, needle retrieval, prefix-cache accounting, `n=2`, logprobs, three model aliases.</p>
    </div>
    <div class="card">
      <strong>Does not work / different from the card</strong>
      <p>Video; remote image URLs; 1M context (262k here); `seed` reproducibility; `developer` role; `usage.reasoning_tokens`; `enable_thinking=false`; embeddings; engine-local API keys.</p>
    </div>
  </div>
  <p class="callout"><strong>Client rule.</strong> Echo the previous assistant message intact (`content` + `reasoning` + `tool_calls`). For short answers, raise <code>max_tokens</code> or set <code>reasoning_effort=none</code>. Images must be <code>data:image/...;base64,...</code>.</p>

  <h2>Model identity</h2>
  <table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Checkpoint</td><td><code>moonshotai/Kimi-K3</code> @ <code>f831ab66814297da540d832a5235f8e904f29d06</code></td></tr>
    <tr><td>Served names</td><td><code>FW-Kimi-K3</code>, <code>kimi-k3</code>, <code>moonshotai/Kimi-K3</code></td></tr>
    <tr><td>Architecture</td><td>MoE 2.8T / 104B active · 93 layers · 69 KDA + 24 gated MLA · 896 experts, 16 routed + 2 shared · SiTU-GLU · MoonViT-V2 vision</td></tr>
    <tr><td>Quant / dtype</td><td>MXFP4 weights (QAT) · BF16 compute · AITER_MXFP4_BF16 MoE</td></tr>
    <tr><td>Native vs served context</td><td>1,048,576 native · <strong>262,144 served</strong></td></tr>
    <tr><td>Hardware</td><td>8× AMD Instinct MI355X VF, TP=8, 2.0 TiB RAM, host <code>kimi-k3-t01</code></td></tr>
    <tr><td>Engine</td><td>vLLM 0.1.dev19253+g5f76ae224 · KV 1,628,607 tokens (43.07 GiB) · max 6.21× at 262k · prefix cache on · batch 4096 · 64 seqs</td></tr>
    <tr><td>Public path</td><td>nginx → LiteLLM :4000 → gateway :8002 → vLLM :8001 at <code>https://api.cflowx.in/v1</code></td></tr>
  </table>

  <h2>Scorecard</h2>
  <table>
    <thead><tr><th>Sec</th><th>Topic</th><th>PASS</th><th>WARN</th><th>FAIL</th><th>SKIP</th></tr></thead>
    <tbody>
    {''.join(score_rows)}
    </tbody>
  </table>

  <h2>Detailed KVV results</h2>
  {''.join(section_html)}

  <h2>Measured response shape</h2>
  <p>Live <code>reasoning_effort=low</code> probe, 17×23, 1.42 s:</p>
  <pre>{{
  "object": "chat.completion",
  "model": "FW-Kimi-K3",
  "choices": [{{
    "finish_reason": "stop",
    "message": {{
      "role": "assistant",
      "content": "391",
      "reasoning": "17*23=391. final number only. Ensure no extra."
    }}
  }}],
  "usage": {{
    "prompt_tokens": 103,
    "completion_tokens": 28,
    "total_tokens": 131,
    "prompt_tokens_details": {{ "cached_tokens": 0, "created_cache_tokens": 0 }}
  }}
}}</pre>

  <h2>Vendor-published quality numbers</h2>
  <p>From the Moonshot card, <code>reasoning_effort=max</code>, temperature 1.0. These describe the weights, not a throughput or SLA guarantee of this 8-GPU serve. Video rows do not apply here. Columns: Kimi K3 · Claude Fable 5 · GPT-5.6 Sol · Claude Opus 4.8 · GPT-5.5.</p>
  {''.join(vendor_html)}

  <h2>Evidence</h2>
  <p>Raw JSON: <code>eval/runs/20260921T170500Z-kvv-engine/health_results.json</code>. Environment: <code>eval/runs/20260921T170233Z-kvv/env.json</code>. The 19 September KVV run is discarded (HTTP 0 after section B).</p>
  <footer>Prepared 21 September 2026. No credentials are included. Public inference may still show a temporary 429 <code>gateway_recovery</code> after KV-full events; the engine results in this file are independent of that hold.</footer>
</div>
</body>
</html>
"""
OUT_HTML.write_text(html_doc)
print("wrote", OUT_MD)
print("wrote", DOCS_MD)
print("wrote", OUT_HTML)
print("bytes", OUT_MD.stat().st_size, OUT_HTML.stat().st_size)
