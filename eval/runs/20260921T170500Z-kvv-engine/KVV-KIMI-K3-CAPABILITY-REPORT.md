# Kimi K3 — KVV Capability Report

**Audience:** anyone who needs to know what the currently served Kimi K3 can and cannot do.
**Evidence date:** 21 September 2026, 17:08–17:13 UTC.
**Harness:** `cflow-gpu/eval/harness/kvv_health.py` (KVV sections A–N, plus public-edge S).
**Engine run:** 282.0 s · 82 PASS · 10 WARN · 2 FAIL · 1 SKIP.
**Combined with public-edge security:** 106 checks · 92 PASS · 11 WARN · 2 FAIL · 1 SKIP.

This is not a marketing card. Every numbered check below was executed against the live model on this box. Vendor-published quality numbers are labelled **VENDOR** and were not re-run here.

## 1. One-page verdict

The currently served model is **Moonshot Kimi K3** (`moonshotai/Kimi-K3`, snapshot `f831ab66814297da540d832a5235f8e904f29d06`), exposed as `FW-Kimi-K3`. It is a working **OpenAI-compatible chat, tool-calling, structured-output, reasoning, streaming, and image** endpoint with a **262,144-token** served context.

| You can rely on | You cannot rely on |
|---|---|
| Chat completions, streaming + `[DONE]` | Video input (rejected: 0 videos allowed) |
| Tool calling: auto / none / required / forced / parallel / multi-turn | `reasoning_effort` values other than `none`, `low`, `high`, `max` |
| Strict JSON schema and `json_object` | Remote image URLs (send base64) |
| Inline / base64 images (colour ID, multi-image count) | Native 1,048,576-token window (served cap is 262,144) |
| Needle retrieval at 22k tokens; 90k-token prompts | Deterministic `seed` sampling |
| Prefix-cache `cached_tokens` on repeated prefixes | `usage.reasoning_tokens` (field is empty; use `message.reasoning`) |
| `n=2`, `logprobs`, stop sequences, aliases | `developer` role (use `system`) |
| Turning thinking off via `reasoning_effort=none` | `enable_thinking=false` / `thinking.type=disabled` (ignored) |
| Public-edge 401 without a key | Engine-local auth (localhost vLLM has none) |
| Clean 400 on over-length prompts | Embeddings endpoint (404) |

**Practical client rule:** always send back the previous assistant message intact (`content` + `reasoning` + `tool_calls`). For short answers, either raise `max_tokens` or set `reasoning_effort` to `none`. For images, inline base64 only.

## 2. What the model is

Identity is taken from the loaded Hugging Face snapshot and the vendor model card. Architecture numbers are **from the checkpoint / card**, not guessed.

| Item | Value | Source |
|---|---|---|
| Family | Kimi K3, Moonshot AI | vendor card |
| Checkpoint | `moonshotai/Kimi-K3` | loaded path |
| Snapshot | `f831ab66814297da540d832a5235f8e904f29d06` | local hub |
| Architecture class | `KimiK3ForConditionalGeneration` / text `KimiLinearForCausalLM` | `config.json` |
| Total / activated parameters | 2.8T / 104B | vendor card |
| Layers | 93 (1 dense); 69 KDA + 24 gated MLA | vendor card + config |
| Hidden size / heads | 7168 / 96 | config |
| MoE | 896 experts, 16 routed + 2 shared per token, SiTU-GLU | vendor card |
| Vocabulary | 163,840 | config |
| Native context | 1,048,576 tokens | config `max_position_embeddings` |
| **Served context** | **262,144 tokens** | vLLM `max_model_len` |
| Quantization | MXFP4 weights, MXFP8 activations (QAT) | vendor card; engine `quantization=mxfp4` |
| Compute dtype | bfloat16 | engine |
| Vision encoder | MoonViT-V2, 401M, 27 layers, patch 14 | vendor card + `vision_config` |
| Native modalities | text, image, video | vendor card |
| **Served modalities** | **text + image** (video rejected) | KVV A8, H1–H5 |
| Weights on disk | 1.56 TB across 96 safetensors | local snapshot |
| License | Kimi K3 License | vendor card |

Thinking is structural in the chat template. Every assistant turn has a `<think>` channel unless thinking is disabled. Official usage: default `reasoning_effort=max`; supported values `low` / `high` / `max`. This serve also accepts `none` to drop the think channel.

## 3. This serving stack (measured 21 Sep 2026)

| Item | Value |
|---|---|
| Host | `kimi-k3-t01`, Ubuntu 24.04.4, kernel 6.8.0-137-generic |
| CPU / RAM | AMD EPYC 9575F, 2 NUMA nodes, 2.0 TiB RAM, KVM |
| GPUs | 8 × AMD Instinct MI355X VF, gfx950, ~288 GiB each |
| ROCm | 7.14.0 packages on host; container ROCm 7.2.3 historically |
| Engine | vLLM `0.1.dev19253+g5f76ae224.d20260727`, V1, tokenizer_mode `kimi_k3` |
| Image | `5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8` |
| Parallelism | TP=8, PP=1, DP=1; expert-parallel **off** on this process |
| MoE kernel | `AITER_MXFP4_BF16` |
| GPU memory util | 0.88 |
| KV cache | 1,628,607 tokens, 43.07 GiB, block 768, max concurrency 6.21× at 262k |
| Prefix cache | enabled, SHA-256 |
| Batch | `max_num_batched_tokens=4096`, `max_num_seqs=64`, chunked prefill on |
| Scheduler | priority |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture up to 128 |
| Parsers | `reasoning_parser=kimi_k3`, `tool_call_parser=kimi_k3`, auto tool choice |
| Speculation | none |
| Public URL | `https://api.cflowx.in/v1` |
| Path | nginx → LiteLLM `:4000` → gateway `:8002` → vLLM `:8001` |
| Served names | `FW-Kimi-K3`, `kimi-k3`, `moonshotai/Kimi-K3` |
| Extra ROCm flags | `VLLM_ROCM_USE_AITER=1`, `AITER_SITUV2_A8W4=1`, Kimi K3 preroute/latent-tail FP8 |

The KVV model-capability numbers were taken **directly from vLLM `:8001`** so LiteLLM cooldown / nginx recovery holds cannot fake a model failure. Public-edge security (section S) was taken from `api.cflowx.in`.

## 4. Scorecard

| Section | Topic | Engine / edge result |
|---|---|---|
| A | Basic protocol | 8 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| B | Parameter validation | 16 PASS · 1 WARN · 1 FAIL · 0 SKIP |
| C | Tool calling | 10 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| D | Structured output | 3 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| E | Reasoning control | 8 PASS · 6 WARN · 0 FAIL · 1 SKIP |
| F | Streaming quality | 4 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| G | Usage and prefix cache | 5 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| H | Multimodal | 4 PASS · 1 WARN · 0 FAIL · 0 SKIP |
| I | Long context | 4 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| J | Protocol surfaces | 2 PASS · 2 WARN · 1 FAIL · 0 SKIP |
| K | Concurrency | 2 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| L | Consistency | 4 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| M | Message shapes and roles | 8 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| N | Reasoning budget vs visible output | 4 PASS · 0 WARN · 0 FAIL · 0 SKIP |
| S | Public-edge security | 10 PASS · 1 WARN · 0 FAIL · 0 SKIP |

Legend: **PASS** = the capability is present and behaved. **WARN** = works with a documented boundary, or an optional OpenAI feature this model does not implement. **FAIL** = the check’s strict expectation was not met. **SKIP** = not measurable (here: `reasoning_tokens` growth, because the usage field is empty).

## 5. Detailed results

### A. Basic protocol

The OpenAI `/v1/chat/completions` surface is complete: non-streaming and
streaming both return a standard object, the served name `FW-Kimi-K3` is
echoed, unknown message fields are ignored, and an unknown model name is a
clean 404. There is no embedding model. Video is rejected with a precise
error (`At most 0 video(s) may be provided`) — the checkpoint includes a
MoonViT-V2 encoder that can do video, but this vLLM serve is image+text
only.

| ID | Check | Status | Evidence |
|---|---|---|---|
| A1 | GET /v1/models returns target model | **PASS** | http 200 names=['FW-Kimi-K3', 'kimi-k3', 'moonshotai/Kimi-K3'] |
| A2 | Non-streaming chat structure | **PASS** | http 200 finish=stop content='Hello' 2.4s |
| A3 | Streaming chat frames + [DONE] | **PASS** | http 200 frames=301 done=True  |
| A4 | Response model field echo | **PASS** | echoed='FW-Kimi-K3' expected='FW-Kimi-K3' |
| A5 | Unknown fields in message object are ignored | **PASS** | http 200 accepted |
| A6 | Unknown model name returns 4xx | **PASS** | http 404 (reference platform returned 503 and was flagged) |
| A7 | /v1/embeddings behaviour documented | **PASS** | http 404 - no embedding model is served here |
| A8 | Video input rejected with a clear error | **PASS** | http 400 At most 0 video(s) may be provided in one prompt. (parameter=video) |

### B. Parameter validation

Sampling parameters follow the OpenAI ranges (`temperature` 0–2, `top_p`
0–1, penalties ±2). `max_tokens` truncates (`finish_reason=length`).
`n=2` returns two candidates. `logprobs` are returned when requested.
Oversized `max_tokens` is rejected against the 262,144 context cap.

Two real boundaries: an empty `messages` array is accepted (HTTP 200) rather
than rejected, and `seed` does not make two samples identical. Do not
promise deterministic sampling.

| ID | Check | Status | Evidence |
|---|---|---|---|
| B1 | temperature=0 accepted | **PASS** | http 200 want=2xx  |
| B2 | temperature=2 accepted | **PASS** | http 200 want=2xx  |
| B3 | temperature=3 rejected | **PASS** | http 400 want=4xx  |
| B4 | temperature=-1 rejected | **PASS** | http 400 want=4xx  |
| B5 | top_p=0.1 accepted | **PASS** | http 200 want=2xx  |
| B6 | top_p=1.5 rejected | **PASS** | http 400 want=4xx  |
| B7 | frequency_penalty=2 accepted | **PASS** | http 200 want=2xx  |
| B8 | frequency_penalty=5 rejected | **PASS** | http 400 want=4xx  |
| B9 | presence_penalty=-2 accepted | **PASS** | http 200 want=2xx  |
| B10 | negative max_tokens rejected | **PASS** | http 400 want=4xx  |
| B11 | unknown parameter ignored | **PASS** | http 200 want=2xx  |
| B12 | max_tokens=16 truncates | **PASS** | http 200 completion_tokens=16 finish=length |
| B13 | empty messages rejected | **FAIL** | http 200 |
| B14 | stop sequence honoured | **PASS** | http 200 content='The user has just provided a list: "alpha ' |
| B15 | seed reproducibility | **WARN** | same=False a='Lantern' b='Lighthouse' |
| B16 | n=2 multiple candidates | **PASS** | http 200 choices=2 (reference platform: 65/65 rejected n>1) |
| B17 | oversized max_tokens handled cleanly | **PASS** | http 400 max_tokens=99999999 cannot be greater than max_model_len=max_total_tokens=262144. Please request fewer output tokens. (p |
| B18 | logprobs returned when requested | **PASS** | http 200 logprobs=present (reference platform: only 14/65 channels) |

### C. Tool calling

Tool calling is a first-class, production-ready surface on this serve.

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
turn. This deployment accepts both field names.

| ID | Check | Status | Evidence |
|---|---|---|---|
| C1 | tool_choice=auto produces a call | **PASS** | http 200 calls=1 |
| C2 | tool_choice=none suppresses calls | **PASS** | http 200 calls=0 |
| C3 | tool_choice=specific function forces that tool | **PASS** | http 200 calls=['get_weather'] |
| C4 | tool_choice=required 5/5 | **PASS** | 5/5 produced a tool call |
| C5 | nested schema arguments conform | **PASS** | http 200 args={"traveller": {"name": "Ada", "age": 36}, "legs": [{"from": "Delhi", "to": "Tokyo", "cabin": "business"}, {"from": "Tokyo", "to": "Sydney", "cabin": "economy"}]} |
| C6 | correct routing among 20 tools | **PASS** | http 200 picked=['get_weather'] |
| C7 | parallel_tool_calls returns multiple calls | **PASS** | http 200 calls=2 |
| C8 | multi-turn tool result consumed | **PASS** | http 200 content='The current weather in Lisbon is **19°C** with **clear skies** — a lovely day! ☀️' |
| C9 | streamed tool_calls reassemble | **PASS** | http 200 name=get_weather args='{"city": "Berlin", "unit": "c"}' |
| C10 | tool_call object shape is OpenAI-conformant | **PASS** | http 200 shape_ok=True |

### D. Structured output

`response_format=json_object` returned valid JSON. Strict
`json_schema` returned exactly the required keys with the right types.
`$ref` / `$defs` schemas are accepted. Structured output can be used
for extraction and agent payloads.

| ID | Check | Status | Evidence |
|---|---|---|---|
| D1 | response_format=json_object returns valid JSON | **PASS** | http 200 body='{"city": "Paris", "country": "France"}' |
| D2 | json_schema strict conforms exactly | **PASS** | http 200 body='{"city":"Paris","population":2100000,"landmarks":["Eiffel Tower","Louvre Museum","Notre-Dame Cathedral","Arc de Triomphe","Sacré-Cœur Basilica","Champs-Élysées"' |
| D3 | $ref / $defs schema accepted | **PASS** | http 200 ok |

### E. Reasoning control

Thinking is on by default. The tokenizer only accepts
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
that field is filled.

| ID | Check | Status | Evidence |
|---|---|---|---|
| E0 | default request reports reasoning_tokens | **PASS** | http 200 reasoning_tokens=None reasoning_len=165 |
| E1 | reasoning_effort=none | **PASS** | http 200 reasoning_tokens=None |
| E2 | reasoning_effort=low | **PASS** | http 200 reasoning_tokens=None |
| E3 | reasoning_effort=medium | **WARN** | http 500 Unsupported thinking_effort='medium'; supported values are ['high', 'low', 'max']. (capability boundary) |
| E4 | reasoning_effort=high | **PASS** | http 200 reasoning_tokens=None |
| E5 | reasoning_effort=xhigh | **WARN** | http 500 Unsupported thinking_effort='xhigh'; supported values are ['high', 'low', 'max']. (capability boundary) |
| E6 | reasoning_effort=max | **PASS** | http 200 reasoning_tokens=None |
| E7 | reasoning_effort=minimal | **WARN** | http 500 Unsupported thinking_effort='minimal'; supported values are ['high', 'low', 'max']. (capability boundary) |
| E8 | reasoning_effort=adaptive | **WARN** | http 400 1 validation error:   {'type': 'literal_error', 'loc': ('body', 'reasoning_effort'), 'msg': "Input should be ' (capability boundary) |
| E9 | reasoning_tokens grow with effort level | **SKIP** | observed=[] monotonic=None |
| E10 | disable thinking via enable_thinking=false | **WARN** | http 200 reasoning_tokens=None reasoning_len=119 |
| E11 | disable thinking via thinking.type=disabled | **WARN** | http 200 reasoning_tokens=None reasoning_len=142 |
| E12 | disable thinking via chat_template_kwargs.thinking=false | **PASS** | http 200 reasoning_tokens=None reasoning_len=0 |
| E13 | disable thinking via reasoning_effort=none | **PASS** | http 200 reasoning_tokens=None reasoning_len=0 |
| E14 | thinking.budget_tokens=128 behaviour | **PASS** | http 200 accepted |

### F. Streaming quality

SSE is standard (`object=chat.completion.chunk`, terminal `[DONE]`).
`stream_options.include_usage` delivers a final usage frame. A 3,104-frame
essay stream completed without dropped frames (8,074 visible characters).
Streaming and non-streaming answers agree on the same prompt.

| ID | Check | Status | Evidence |
|---|---|---|---|
| F1 | stream_options.include_usage delivers usage | **PASS** | http 200 usage={"prompt_tokens": 92, "total_tokens": 165, "completion_tokens": 73, "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0}} |
| F2 | SSE frame structure is standard | **PASS** | frames=64 done=True objects_ok=True |
| F3 | streaming and non-streaming agree | **PASS** | streamed=426 chars nonstream=425 chars match_head=True |
| F4 | long stream does not drop frames | **PASS** | http 200 frames=3104 chars=8074 done=True |

### G. Usage and prefix cache

`prompt_tokens` scale linearly (10/20/40 copies of a unit sentence
→ 195/295/495 tokens, zero deviation). `prompt + completion = total`.
A repeated ~2.3k-token prefix reported `cached_tokens=2304` of
`prompt_tokens=2504` (92% of that prompt reused). Streaming and
non-streaming prompt token counts match. Truncation reports
`completion_tokens == max_tokens`.

| ID | Check | Status | Evidence |
|---|---|---|---|
| G1 | prompt_tokens scale linearly with input | **PASS** | points=[(10, 195), (20, 295), (40, 495)] max_deviation=0.0 |
| G2 | prompt+completion equals total | **PASS** | usage={"prompt_tokens": 91, "total_tokens": 229, "completion_tokens": 138, "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0, "multimodal_tokens": null}} |
| G3 | cached_tokens reported for a repeated prefix | **PASS** | cached_tokens=2304 prompt_tokens=2504 |
| G4 | streaming and non-streaming prompt_tokens identical | **PASS** | nonstream=96 stream=96 |
| G5 | truncated completion_tokens equals max_tokens | **PASS** | completion_tokens=32 max_tokens=32 |

### H. Multimodal

Native vision works for **inline/base64** images. A solid red PNG
was identified as “Red”. Two images were counted as “2”.
`image_url.detail` and a stray `MimeType` key are tolerated.

Remote HTTP(S) image URLs returned HTTP 422 — this engine does not
fetch URLs. Clients must send `data:image/...;base64,...`.

Video is rejected (see A8). The vendor card scores Video-MME / MMVU on
the full model; those numbers are not available on this serve.

| ID | Check | Status | Evidence |
|---|---|---|---|
| H1 | base64 single image recognised | **PASS** | http 200 answer='Red' |
| H2 | multiple images counted | **PASS** | http 200 answer='2' |
| H3 | remote URL image fetched | **WARN** | http 422 answer='' |
| H4 | image_url.detail accepted | **PASS** | http 200 ok |
| H5 | stray image_url.MimeType key tolerated | **PASS** | http 200 {'id': 'chatcmpl-bc20a533420014f5', 'object': 'chat.completion', 'created': 1790010676, 'model': 'FW-Kimi-K3', (reference platform P0-1: 1,548 failures/24h) |

### I. Long context

Served context is **262,144 tokens**, not the native 1,048,576.

- Needle “74812” retrieved from a 21,717-token haystack in 3.0 s.
- A 90,093-token prompt completed in 10.5 s.
- A ~300k-word prompt was rejected in 0.1 s with a clear 400
  (`maximum context length is 262144 tokens`).
- The engine stayed healthy afterwards.

KV cache on this process is 1,628,607 tokens (~43 GiB), which is about
6.2 concurrent 262k-token requests. Huge in-flight prompts will fill
that pool; that is an operational limit, not a model-quality limit.

| ID | Check | Status | Evidence |
|---|---|---|---|
| I1 | needle retrieval in long context | **PASS** | http 200 prompt_tokens=21717 answer='74812' 3.0s |
| I2 | large input (~90k words) handled | **PASS** | http 200 prompt_tokens=90093 10.5s  |
| I3 | over-limit request returns clean 400, engine survives | **PASS** | http 400 0.1s This model's maximum context length is 262144 tokens. However, you requested 600 output tokens and your prompt contains  |
| I4 | engine still healthy after the over-limit probe | **PASS** | GET /v1/models http 200 |

### J. Protocol surfaces

The engine itself is bound to `127.0.0.1:8001` and does **not**
authenticate. J1 (invalid key → 401) is therefore a FAIL on the engine
path and a PASS on the public edge (`api.cflowx.in` returns 401 without
a key).

`/v1/responses` is available and consistent with chat for `anyOf` tool
schemas. `/v1/messages` (Anthropic) is present on the engine (HTTP 200)
but is not part of the public allowlist contract. The `developer` role
is rejected (`Unknown message role 'developer'`). Use `system`.

| ID | Check | Status | Evidence |
|---|---|---|---|
| J1 | invalid key returns 401 | **FAIL** | http 200 |
| J2 | Anthropic /v1/messages surface | **WARN** | http 200 - not served here; edge allowlists /v1/ only |
| J3 | /v1/responses surface | **PASS** | http 200 available |
| J4 | anyOf tool schema consistent across endpoints | **PASS** | chat=200 responses=200 (reference platform P1-3 mismatch) |
| J5 | developer role accepted | **WARN** | http 400 Unknown message role 'developer' at index 0. |

### K. Concurrency

12 requests at concurrency 6 all succeeded (12/12 HTTP 200). Wall
49.6 s, p50 19.5 s, p95 24.4 s for short reasoning answers. No 429 / 5xx
from the engine under that load. This is a functional concurrency check,
not a throughput benchmark.

| ID | Check | Status | Evidence |
|---|---|---|---|
| K1 | 6x concurrent x 12 requests | **PASS** | ok=12/12 wall=49.6s p50=19.513s p95=24.422s |
| K2 | no throttling or server errors under test load | **PASS** | status spread={200: 12} |

### L. Consistency

The same prompt billed 95 tokens on six repeats. Aliases
`FW-Kimi-K3`, `kimi-k3`, and `moonshotai/Kimi-K3` are all advertised and
the legacy alias serves.

| ID | Check | Status | Evidence |
|---|---|---|---|
| L1 | prompt_tokens identical across 6 repeats | **PASS** | values=[95] |
| L2 | every customer key sees the same model and billing | **PASS** | per_key={"evalbot": {"http": 200, "prompt_tokens": 95, "model": "FW-Kimi-K3"}} |
| L3 | served-model aliases both advertised | **PASS** | names=['FW-Kimi-K3', 'kimi-k3', 'moonshotai/Kimi-K3'] (config.yaml keeps the legacy alias deliberately) |
| L4 | legacy alias 'kimi-k3' still serves | **PASS** | http 200 |

### M. Message shapes and roles

Content-array text, `system`, `name`, unknown vendor extensions,
and both `reasoning` / `reasoning_content` history fields are accepted.
Orphan tool messages without a matching assistant `tool_calls` turn are
rejected with a precise Kimi-K3 error. Invalid roles are 400.

| ID | Check | Status | Evidence |
|---|---|---|---|
| M1 | content-array text shape accepted | **PASS** | http 200 |
| M2 | system role accepted | **PASS** | http 200 content='ok' |
| M3 | orphan tool message without assistant turn | **PASS** | http 400 Kimi K3 tool messages need a resolvable tool name: carry `tool`/`name`, or match a preceding assistant tool_ca |
| M4 | message .name field accepted | **PASS** | http 200 |
| M5 | assistant history with vendor reasoning field round-trips | **PASS** | http 200 {'id': 'chatcmpl-809dbd3cb65e9185', 'object': 'chat.completion', 'created': 1790010761, 'model': 'FW-Kimi-K3', (reference platform P1-1: 243 failures/24h) |
| M6 | assistant history with reasoning_content round-trips | **PASS** | http 200 {'id': 'chatcmpl-9dd0139dc615d69f', 'object': 'chat.completion', 'created': 1790010762, 'model': 'FW-Kimi-K3', |
| M7 | unknown vendor extension web_search_options | **PASS** | http 200 {'id': 'chatcmpl-b06f8dd07cffa169', 'object': 'chat.completion', 'created': 1790010763, 'model': 'FW-Kimi-K3', |
| M8 | invalid role rejected | **PASS** | http 400 |

### N. Reasoning budget vs visible output

A 3-pen / 7-sheep word problem was answered correctly (17) at
`max_tokens` 2000, 4000 and 8000, with visible content in every case.

Default thinking shares the completion budget. At `max_tokens=40` the
visible `content` was the start of the think channel
(“The user is asking a simple factual question…”). Setting
`reasoning_effort=none` on the same prompt produced a real answer
(“The capital of France is Paris…”). Clients that want a short answer
must either raise `max_tokens` or turn thinking off.

| ID | Check | Status | Evidence |
|---|---|---|---|
| N1 | content present at every budget | **PASS** | rows=[{"max_tokens": 2000, "http": 200, "reasoning_tokens": null, "completion_tokens": 136, "content_chars": 197, "correct": true}, {"max_tokens": 4000, "http": 200, "reasoning_tokens": null, "completion_tokens": 98, ... |
| N2 | answer correct across budgets | **PASS** | correct=[True, True, True] |
| N3 | small max_tokens can starve visible content | **PASS** | max_tokens=40 content='The user is asking a simple factual question: What is the ca' reasoning_tokens=None - reference report §7 recheck #1 was exactly this probe defect |
| N4 | reasoning_effort=none frees the budget for content | **PASS** | content="The capital of France is Paris. It's not only the political " reasoning_tokens=None |

### S. Public-edge security

Measured on `https://api.cflowx.in`, not on localhost. Engine
admin surfaces (`/tokenize`, `/metrics`, `/health`, `/invocations`) are
blocked at nginx (403/404). `/v1/models` requires a key (401).
`/openapi.json` is publicly readable (LiteLLM schema, no secrets).

At the time this report was written the public inference path was
again returning `429 gateway_recovery` after a KV-full incident earlier
in the day. That is an admission hold, not a model failure. The engine
behind it answered every KVV probe in this document.

| ID | Check | Status | Evidence |
|---|---|---|---|
| S1 | POST /invocations blocked without credentials | **PASS** | http 404 |
| S2 | POST /scale_elastic_ep blocked without credentials | **PASS** | http 404 |
| S3 | POST /tokenize blocked without credentials | **PASS** | http 404 |
| S4 | POST /detokenize blocked without credentials | **PASS** | http 404 |
| S5 | GET /version blocked without credentials | **PASS** | http 404 |
| S6 | GET /openapi.json blocked without credentials | **WARN** | http 200 — LiteLLM OpenAPI is publicly readable |
| S7 | GET /docs blocked without credentials | **PASS** | http 404 |
| S8 | GET /metrics blocked without credentials | **PASS** | http 403 |
| S9 | GET /health blocked without credentials | **PASS** | http 403 |
| S12 | GET /v1/models without a key | **PASS** | http 401 Authorization Required |
| J1-edge | Invalid Bearer key on the public edge | **PASS** | http 401 on /v1/models; inference paths currently also 429 gateway_recovery |

## 6. Response shape (measured sample)

A live `reasoning_effort=low` request for `17 × 23` returned in 1.42 s:

```json
{
  "object": "chat.completion",
  "model": "FW-Kimi-K3",
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "391",
      "reasoning": "17*23=391. final number only. Ensure no extra.",
      "reasoning_content": ""
    }
  }],
  "usage": {
    "prompt_tokens": 103,
    "completion_tokens": 28,
    "total_tokens": 131,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "created_cache_tokens": 0,
      "multimodal_tokens": null
    }
  }
}
```

Message keys present: `role`, `content`, `reasoning`, `reasoning_content` (empty here), `annotations`, `audio`, `function_call`, `refusal`. Prefer `message.reasoning`. Echo it back as either `reasoning` or `reasoning_content` — both round-trip (M5, M6).

## 7. How to call it

```python
from openai import OpenAI
client = OpenAI(base_url='https://api.cflowx.in/v1', api_key='...')

resp = client.chat.completions.create(
    model='FW-Kimi-K3',  # also: kimi-k3, moonshotai/Kimi-K3
    messages=[
        {'role': 'system', 'content': 'Be concise.'},
        {'role': 'user', 'content': 'Summarise the attached figure.'},
    ],
    max_tokens=2000,
    reasoning_effort='low',   # none | low | high | max
    tools=[...],              # optional
    tool_choice='auto',
    response_format={'type': 'json_schema', 'json_schema': {...}},
)
msg = resp.choices[0].message
# next turn: send msg back intact, including msg.reasoning and msg.tool_calls
```

Images: `content: [{type:'text',...},{type:'image_url', image_url:{url:'data:image/png;base64,...'}}]`.

Not served: `/v1/embeddings`. Not useful here: `seed` for bit-identical output, `developer` role, video URLs, remote image URLs.

## 8. Boundaries a reader will hit

1. **Thinking eats `max_tokens`.** Default thinking is on. A 40-token budget filled with think-channel text. Use `reasoning_effort=none` or a larger budget.
2. **Only four effort values work:** `none`, `low`, `high`, `max`. Others 400/500.
3. **Two disable switches are no-ops:** `enable_thinking=false` and `thinking.type=disabled`.
4. **`reasoning_tokens` in usage is empty.** Count `message.reasoning` yourself if you need it.
5. **Context is 262k, not 1M.** Over-limit is a clean 400.
6. **KV pool is ~1.63M tokens.** A handful of 200k-token in-flight requests will pin the box. That is why public inference was held with `gateway_recovery` earlier today.
7. **Images must be base64.** URL fetch is 422. Video is 400.
8. **Empty `messages` is accepted.** Validate client-side.
9. **`seed` is not deterministic.**
10. **Engine localhost has no API key.** Auth exists only at the public edge.
11. **LiteLLM can 429 `No deployments available`** for a few seconds after a stream if the router cools the only deployment down. Retry once. This is a proxy behaviour, not a model defect.
12. **`/openapi.json` is public** on the edge (schema only).

## 9. Vendor-published quality numbers (not re-run here)

Copied from the Moonshot Kimi K3 model card. All Kimi K3 scores were taken by Moonshot at `reasoning_effort=max`, temperature 1.0. They describe the **weights**, not a guaranteed SLA of this 262k / 8×MI355X serve. Comparisons are Claude Fable 5 (max), GPT-5.6 Sol (max), Claude Opus 4.8 (max), GPT-5.5 (xhigh).

### Reasoning & knowledge

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 |
|---|---:|---:|---:|---:|---:|
| GPQA Diamond | 93.5 | 92.6 | 94.1 | 91.0 | 93.5 |
| CritPt | 23.4 | 28.6 | 32.3 | 20.9 | 27.1 |
| AA-LCR | 74.7 | 70.0 | 73.7 | 67.7 | 74.3 |
| HLE-Full (no tool / with tool) | 43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 |

### Coding

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 |
|---|---:|---:|---:|---:|---:|
| DeepSWE | 67.5 | 70.0 | 73.0 | 59.0 | 67.0 |
| ProgramBench | 77.8 | 76.8 | 77.6 | 71.9 | 70.8 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | 88.8 | 84.6 | 83.4 |
| FrontierSWE | 81.2 | 86.6 | 71.3 | 66.7 | 64.9 |
| SWE-Marathon | 42.0 | 35.0 | 39.0 | 40.0 | 14.0 |
| PostTrainBench | 36.6 | 41.4 | 34.6 | 34.1 | 28.4 |
| MLS-Bench-Lite | 48.3 | 49.9 | 46.2 | 42.8 | 35.5 |
| SciCode | 58.7 | 60.2 | 56.1 | 53.5 | 56.1 |
| Kimi Code Bench 2.0 | 72.9 | 76.9 | 64.8 | 71.7 | 69.0 |

### Agentic

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 |
|---|---:|---:|---:|---:|---:|
| BrowseComp | 91.2 | 88.0 | 90.4 | 84.3 | 84.4 |
| DeepSearchQA (F1) | 95.0 | 94.2 | — | 93.1 | — |
| ResearchRubrics | 76.2 | — | 73.8 | 73.5 | 64.0 |
| GDPval-AA v2 (Elo) | 1686 | 1747 | 1736 | 1593 | 1491 |
| Toolathlon-Verified | 76.5 | 77.9 | 74.9 | 76.2 | 73.5 |
| MCPMark-Verified | 94.5 | 87.4 | 92.9 | 76.4 | 92.9 |
| MCP-Atlas | 84.2 | 84.7 | 83.6 | 83.6 | 82.8 |
| AutomationBench | 30.8 | 29.1 | 29.7 | 27.2 | 22.7 |
| JobBench | 54.3 | 57.4 | 45.4 | 48.4 | 38.3 |
| AA-Briefcase (Elo) | 1548 | 1583 | 1495 | 1354 | 1158 |
| Agents' Last Exam | 28.3 | 25.7 | 29.6 | 27.0 | 26.6 |
| APEX-Agents | 41.0 | 43.3 | 39.9 | 39.4 | 38.5 |
| OfficeQA Pro | 63.3 | 69.9 | 63.2 | 63.9 | 60.9 |
| SpreadsheetBench 2 | 34.8 | 34.7 | 32.4 | 31.6 | 29.1 |
| OSWorld-Verified | 84.8 | 85.0 | 83.0 | 83.4 | 79.0 |
| OSWorld 2.0 | 58.3 | 66.1 | 62.6 | 55.7 | 49.5 |
| SaaS-Bench | 60.1 | — | 61.4 | 56.1 | 43.8 |
| τ³-Banking | 33.4 | 26.8 | 33.0 | 27.6 | 31.3 |
| Harvey Lab-AA | 94.6 | 93.6 | 87.2 | 91.1 | 86.3 |
| CorpFin v2 | 71.6 | 71.8 | 64.4 | 66.7 | 68.4 |
| Finance Agent v2 | 54.4 | 56.3 | 53.8 | 53.9 | 51.8 |
| Legal Research Bench | 44.2 | 49.5 | 48.1 | 43.8 | 40.4 |

### Vision

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Claude Opus 4.8 | GPT-5.5 |
|---|---:|---:|---:|---:|---:|
| WorldVQA ForceAnswer | 51.0 | 56.7 | 41.8 | 39.1 | 38.5 |
| OmniDocBench | 91.1 | 89.8 | 85.8 | 87.9 | 89.4 |
| PerceptionBench | 58.5 | 57.2 | 59.7 | 47.2 | 55.8 |
| Video-MME (w. sub) | 90.0 | — | 89.5 | 86.0 | 89.3 |
| MMVU | 82.1 | — | 81.2 | 79.2 | 81.7 |
| BabyVision w/ python | 85.7 | 90.5 | 88.9 | 81.2 | 83.6 |
| MMMU-Pro (no tool / with tool) | 81.6 / 83.4 | 81.2 / 86.5 | 83.0 / 84.6 | 78.9 / 82.7 | 81.2 / 83.2 |
| CharXiv RQ (no tool / with tool) | 84.8 / 91.3 | 88.9 / 93.5 | 84.6 / 89.1 | 80.5 / 89.9 | 84.1 / 89.0 |
| MathVision (no tool / with tool) | 94.3 / 97.8 | 94.8 / 98.6 | 95.8 / 97.8 | 86.7 / 97.1 | 92.2 / 96.8 |
| ZeroBench pass@5 (no tool / with tool) | 23.0 / 41.0 | 23.0 / 46.0 | 17.0 / 35.0 | 17.0 / 34.0 | 22.0 / 41.0 |

Source: `moonshotai/Kimi-K3` README, snapshot `f831ab6681…`. Footnotes on harnesses and dates are in that card. Video scores do not apply to this serve.

## 10. Evidence

| File | What it is |
|---|---|
| `eval/runs/20260921T170500Z-kvv-engine/health_results.json` | Raw KVV A–N results |
| `eval/runs/20260921T170500Z-kvv-engine/run_health.log` | Console transcript |
| `eval/runs/20260921T170233Z-kvv/env.json` | Hardware / engine capture |
| `eval/runs/20260921T170233Z-kvv/engine_startup.log` | vLLM init + KV size |
| Hugging Face snapshot `f831ab6681…` | `config.json`, `README.md`, `encoding_k3.py` |

The 19 September 2026 KVV run (`20260919T145955Z-baseline`) is **not** used as evidence. After section B it recorded HTTP 0 / timeouts and is not a capability measurement.

---

*Prepared 21 September 2026. No customer keys or upstream credentials are included.*
