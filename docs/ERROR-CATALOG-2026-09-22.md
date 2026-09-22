# System Error Catalog — 22 September 2026

**System:** `api.cflowx.in` — nginx → LiteLLM → gateway → vLLM, plus the dashboard, service logs, and deployment scripts.

**Snapshot:** 22 September 2026. HTTP counts cover 00:00–17:35 UTC; the gateway diagnostic observation below was taken at 17:36 UTC. References to “current” behavior describe this inspection, not a continuously updated status.

**Current finding:** at **17:36 UTC**, the gateway had recorded **8 admission timeouts caused by `workload budget: reserved_tokens`**. The engine was healthy, with approximately **2% active KV usage**.

Below is the verified error-family catalog. Messages that differ only by token counts, IDs, or limits are combined.

## 1. Errors actually recorded today

From retained `/v1/` nginx logs, **00:00–17:35 UTC**:

| HTTP status | Recorded responses | Meaning in this system |
|---|---:|---|
| **400** | 191 | Invalid request, unsupported input, parameter validation, or context overflow. |
| **401** | 31 | Missing, malformed, or invalid authentication. |
| **403** | 11 | Access/ownership restrictions; today’s detailed records include rejected Responses IDs. |
| **404** | 2 | Requested route was unavailable. |
| **429** | 3,137 | Tenant quotas, concurrency limits, or gateway admission limits. |
| **500** | 86 | Internal exceptions, compatibility failures, or upstream connection/stream handling failures. |
| **502** | 91 | An upstream service refused the connection, disconnected, or failed to answer. |
| **503** | 3,521 | Admission restrictions under earlier configurations, edge throttling, or temporary service unavailability. |
| **Total errors** | **7,070** | Out of **17,542 recorded API responses**; the other **10,472 returned 200**. |

These records include tests/probes and several configurations deployed during the day. A streaming request can also fail **after HTTP 200 has already been sent**.

## 2. Authentication and routing errors

| HTTP / error | Message or condition | Why it happens |
|---|---|---|
| **401** | Missing `Authorization` | nginx rejects a `/v1/` request with an empty authorization header. |
| **401** | `Malformed API Key passed in. Ensure Key has Bearer prefix` | The authorization header does not have the format expected by LiteLLM. |
| **401** | `LiteLLM Virtual Key expected … expected to start with 'sk-'` | A placeholder, another provider’s credential, or another invalid credential format was sent to the public API. |
| **401** | `Invalid proxy server token passed` / `KeyNotFoundError` | The supplied virtual key cannot be found in LiteLLM’s cache/database. Deleted or obsolete keys produce this condition. |
| **401** | `Authentication Error - Expired Key` | The key’s expiry time has passed. |
| **401** | `Key is blocked` / blocked team or deactivated user | Authentication finds the identity, but its account/key state prevents use. |
| **403** | `… not allowed to access model` | The key, user, team, or project is not authorized for the requested model. |
| **403** | Forbidden management route | The credential lacks the role or route permission needed for an administrative operation. |
| **403** | `This response id was not issued by this proxy` | LiteLLM cannot establish ownership of a supplied Responses API ID. This occurred in today’s records. |
| **403** | `local diagnostics only` | A non-loopback caller requests the gateway’s diagnostic endpoints. |
| **403** | Public `/health` or `/metrics` denied | nginx restricts these endpoints to loopback access. |
| **400** | `Invalid model name passed in` | LiteLLM cannot resolve the requested model name. |
| **404** | `The model … does not exist` | A request reaches vLLM with an unsupported model name. |
| **404** | `not found` / `invalid_request` | The requested path is not implemented by that component. The gateway itself accepts Chat/Completions paths; public Responses requests require LiteLLM’s conversion. |
| **405** | `Method Not Allowed` | A framework-managed endpoint exists, but the client used an unsupported HTTP method. |

## 3. Request-body, token, and parameter errors

| HTTP / error | Message or condition | Why it happens |
|---|---|---|
| **400** | `Invalid JSON payload` / `invalid JSON` | The request contains malformed, incomplete, or incorrectly encoded JSON. |
| **400** | `invalid Content-Length` | The gateway cannot parse the declared body length as an integer. |
| **400** | `missing or oversized body` | At the gateway, the declared body length is missing/zero or exceeds **64 MiB**. |
| **400** | `body must be a JSON object` | The top-level JSON is an array, string, number, or `null` rather than an object. |
| **413** | Request body too large | nginx rejects a body exceeding its **64 MiB** limit before it reaches the gateway. |
| **400** | Unsupported media type / JSON required | A JSON-only engine endpoint receives a different `Content-Type`. |
| **400** | `Missing required parameter: 'messages'` | A Chat Completions request omits its messages. |
| **400** | `Missing required parameter: 'input'` | A request to an endpoint requiring `input`, such as embeddings, omits it. |
| **400** | `prompt of … tokens leaves no room for output` | The prompt leaves fewer than **64 output tokens** after the gateway’s **256-token reserve** within the **262,144-token** window. |
| **400** | `ContextWindowExceededError` / maximum context length exceeded | The engine’s actual input/output accounting exceeds its context window. |
| **400** | `n/best_of must be the integer 1 on this replica` | The request asks for multiple candidates, or supplies a non-integer value. This gateway permits exactly one. |
| **400** | `temperature must be …` | Temperature is nonnumeric, non-finite, negative, or greater than **2**. |
| **400** | `top_p must be in (0, 1]` | `top_p` is zero, negative, or greater than one. |
| **400** | `presence_penalty must be in [-2, 2]` | Presence penalty is outside the supported range. |
| **400** | `frequency_penalty must be in [-2, 2]` | Frequency penalty is outside the supported range. |
| **400** | Invalid `repetition_penalty` | The engine requires a finite value greater than zero. |
| **400** | Invalid `top_k` / `min_p` | The engine receives an invalid type/range. `min_p` must be within **[0,1]**; `top_k` must satisfy the engine’s integer rules. |
| **400** | Invalid `min_tokens` / `max_tokens` relationship | Minimum generation is negative or exceeds the effective maximum. The gateway normally supplies/clamps the maximum first. |
| **400** | Invalid `stop` type | `stop` is neither a string nor a list of strings. |
| **400** | `stop cannot contain an empty string` | An empty stop sequence was supplied. |
| **400** | Stop strings require detokenization | Stop strings were requested while detokenization was disabled. |
| **400** | `Token id … is out of vocabulary` | A tokenized prompt contains an ID outside the model’s vocabulary. |
| **400** | Negative or malformed prompt-token array | Prompt token IDs have invalid values or the prompt does not match an accepted string/token-array structure. |
| **400** | Invalid `allowed_token_ids`, `logit_bias`, or `logprob_token_ids` | These fields contain invalid/out-of-vocabulary IDs, an invalid empty allowlist, or inconsistent settings. |
| **400** | `Requested sample/prompt logprobs … greater than max allowed` | The requested number of log probabilities exceeds the engine’s configured limit. |
| **400** | `when using top_logprobs, logprobs must be set to true` | Detailed log probabilities were requested without enabling log probabilities. |
| **400** | Invalid logprob options | Negative/incorrectly typed values, incompatible token-ID options, or prompt log probabilities requested with streaming. |
| **400** | `Stream options can only be defined when stream=True` | Streaming-only options accompany a nonstreaming request. |
| **400** | `tools must not be an empty array` | The request explicitly supplies an empty tools list. |
| **400** | `When using tool_choice, tools must be set` | Tool selection is requested without supplying tool definitions. |
| **400** | `Invalid value for tool_choice` | The value is not a supported choice such as `auto`, `none`, `required`, or a valid named function. |
| **400** | Invalid/missing function name in `tool_choice` | A named tool choice has malformed function data or does not match any declared tool. |
| **400** | Missing `json_schema` | `response_format.type` is `json_schema`, but its schema definition is absent. |
| **400** | Response-format validation errors | Unsupported format type, missing schema name, non-object schema, or incorrectly typed `strict` value. |
| **400** | Structured-output constraints missing or conflicting | No usable constraint was supplied, multiple exclusive constraints were supplied, or the constraints conflict with a named tool choice. |
| **400** | `Invalid … structural_tag specification` | The structural-tag format or grammar fails validation. |
| **400** | Conflicting generation-prompt flags | Both `continue_final_message` and `add_generation_prompt` are enabled. |
| **400** | Content-part / `ValidatorIterator` validation errors | A message’s content list contains strings/scalars where typed content objects are required. |
| **400** | `Extra data: line … column …` | A JSON parser encounters trailing data after a JSON value. The retained message does not identify the exact offending field. |
| **400** | `At most 0 audio(s) may be provided in one prompt` | The active model adapter does not accept audio input. |
| **400**, direct engine | Unsupported video input | The native vLLM adapter is image-only. Public video support works by converting video into image frames before inference. |
| **400** | `Invalid purpose: video` | A file upload uses a purpose not accepted by LiteLLM’s file API. |
| **400** | Unsupported/custom logits processor | The request supplies a processor that the server does not allow or cannot resolve. |
| **400**, HTML body | `Bad request syntax ('{…}POST …')` | An earlier rejected request left unread body bytes on a reused connection; the HTTP parser interpreted them as part of the next request. This signature occurred today. |

Sources include `redesign/gateway/server.py:187`, `redesign/gateway/clamping.py:40`, the installed vLLM request/sampling validators, and LiteLLM’s retained error metadata.

## 4. Admission, capacity, and tokenizer errors

**Current gateway behavior:** ordinary admission limits return **429**. Engine-distress rejection returns **503**. Earlier deployments returned 503 for several ordinary capacity conditions.

| Current HTTP | Error / reason | Exact trigger |
|---|---|---|
| **429** | Tenant `max_parallel_requests` limit | Too many outstanding requests use the same configured tenant/key allowance. This includes requests waiting downstream. |
| **429** | Tenant request-rate limit | The configured requests-per-minute allowance is exhausted. |
| **429** | Tenant token-rate limit | The configured token allowance for the rate-limit window is exhausted. |
| **429** | `BudgetExceededError` | A configured key/user/team/project or other applicable spending budget is exhausted. |
| **429** | `shared execution capacity: N` | The adaptive shared execution pool has no available slot. Its hard ceiling is **64**; pressure can lower it, and paused starts produce a limit of zero. |
| **429** | `workload budget: reserved_tokens` | Accepting the request would exceed the aggregate token-reservation ceiling: currently **92% of measured KV-token capacity**. |
| **429** | `workload budget: kv_headroom` | Existing measured/reserved usage plus the incoming reservation would exceed the projected KV limit. |
| **429** | `workload budget: untracked_engine_work` | For heavy requests, the engine reports more running/waiting work than this gateway owns. The gateway cannot safely account for that additional work. |
| **429** | `workload budget: engine_health_unavailable` | Throughput-mode admission cannot obtain the engine-health snapshot needed to make a reservation. |
| **429** | `queue_full` | The gateway’s waiting queue has reached **128 requests**. |
| **429** | `customer_queue_full` | The gateway customer label has reached its configured waiting allowance, currently **128**. This is separate from LiteLLM tenant quotas. |
| **429** | `queued_token_limit` | Queued request-shape accounting would exceed **67,108,864 tokens**. |
| **429** | `queued_byte_limit` | Accounted queued request bodies would exceed **1 GiB**. |
| **429**, or **503** for underlying distress | `queue_timeout` / admission deadline exceeded | A request cannot obtain admission within the **60-second** deadline. The response retains the last blocking policy reason. |
| **503** | `engine distressed: kv_usage …` | Sheddable classes encounter KV usage above **97%**, or above **90% with more than 8 engine waiters**. |
| **503** | `engine distressed: preemptions …/min` | The circuit breaker observes more than **1 preemption/minute** and sheds eligible classes. |
| **503** | `Prompt inspection admission deadline exceeded` — `tokenizer_busy` | No tokenizer-inspection slot becomes available before the deadline. Current inspection concurrency is **4**. |
| **400** | `Active model tokenizer returned HTTP 4xx` — `tokenizer_rejected` | The active tokenizer rejects the request’s model, messages, template, or other input. |
| **503** | `Prompt inspection is temporarily unavailable` — `tokenizer_unavailable` | The tokenizer connection fails/times out, returns a server error, or returns an unusable/oversized response. |
| **503** | `Invalid prompt inspection response` | Returned token IDs have invalid types or values. |
| **503**, nginx | Edge connection limit | More than **256 concurrent requests** are associated with one IP or authorization value. |

### Why `reserved_tokens` can fail while GPU usage is low

The measured KV pool was **1,641,413 tokens**, giving a reservation ceiling of **1,510,099 tokens**.

Some formats—including multimodal and certain constrained requests—use a conservative **262,144-token prompt reservation**, plus output allowance. Consequently, roughly five such reservations can consume the allowance even when their actual prompts are much smaller.

Also, **JSON `error.code` is separate from HTTP status**: values such as `reject_budget` and `queue_timeout` are application codes.

Sources: `redesign/gateway/admission.py:65`, `policy.py:111`, `workload.py:182`, `inspection.py:117`, `server.py:434`, and the live nginx throughput include.

## 5. Internal, compatibility, connection, and streaming errors

| HTTP / surface | Error / message | Why it happens |
|---|---|---|
| **500** | `Invalid tool choice … Got=<class 'int'>` | LiteLLM receives numeric `tool_choice` where a string/object is expected and raises an internal compatibility exception. |
| **500** | `'int' object is not iterable` | A malformed request supplies an integer where LiteLLM expects an iterable structure, such as messages, tools, or content. |
| **500 / internal exception** | `'int' object has no attribute 'get'` / `'str' object has no attribute 'get'` | A request field is a scalar where conversion/routing code expects a JSON object. |
| **500** | `Invalid user message at index …` | LiteLLM cannot convert a malformed user message to the provider’s Chat format. |
| **500** | `Unmapped prompt format` | The completion prompt has a structure unsupported by LiteLLM’s conversion path. |
| **500** | `Unsupported response_format type - json` | A client uses the unsupported format name `json`; the conversion layer raises an internal exception. |
| **500** | `'async_generator' object has no attribute 'get'` | A streaming completion reaches code expecting a completed dictionary response. |
| **Internal; status absent in stored record** | `KeyError: 'prompt'` | The completion-processing setup accesses a missing `prompt` field. |
| **Internal; status absent in stored record** | `files_settings is not set` | A file operation is attempted without configuring its storage/provider backend. |
| **500** | `InternalServerError … Connection error` / `ServerDisconnectedError` | LiteLLM’s upstream connection closes or fails before a usable response is obtained. |
| **502** | nginx `connect() failed (111: Connection refused)` | LiteLLM—or another configured upstream—is not listening, commonly during a restart. |
| **502** | `engine unreachable: … Connection refused` | The gateway cannot connect to vLLM. |
| **502** | `engine unreachable: timed out` | The engine does not answer within the gateway’s **600-second** upstream timeout. Two such timeout records were found today; their specific engine-side cause remains unresolved. |
| **500**, or an SSE error after **200** | `EngineGenerateError` / `GenerationError` | A request fails inside generation or related engine processing. |
| **500**, then potentially connection failures | `EngineDeadError` | The engine’s background execution fails globally, rather than only one request failing. |
| **500 in proxy records**, possibly after edge **200** | `Response payload is not completed` / `TransferEncodingError` | A chunked upstream response ends before its declared transfer completes. Restarts and interrupted streams can cause this. |
| **Broken stream**, possibly after **200** | `upstream stream failed` / prematurely closed connection | An upstream read fails after headers have already been sent. The gateway closes the incomplete stream. |
| **499 / connection closed** | `client_disconnected`, `BrokenPipeError`, `ConnectionResetError` | The downstream client cancels, times out, or closes its connection. A 499 is generally an internal/log status rather than a response the disconnected client receives. |
| **503**, health endpoint | `{"engine":"down"}` | The gateway’s engine-health probe fails or returns an unsuccessful status. |
| **503** | Authentication database unavailable | LiteLLM cannot access the database required to authenticate a virtual key. |
| **400**, configuration failure | `No connected db` | Virtual-key authentication is attempted with no database client configured. |
| **504** | `Gateway Timeout` | nginx waits too long for its upstream. The inference location has a **900-second** read timeout. |
| **444 / no HTTP response** | Default-server rejection | The request does not match an accepted HTTP host. nginx closes the connection. |
| **TLS failure / no HTTP status** | Handshake or certificate error | Unsupported TLS negotiation, rejected SNI, expired/untrusted certificate, or hostname mismatch prevents HTTP from starting. |

## 6. Earlier errors still present in logs

These explain many of today’s recorded failures, but their old admission rules are no longer active in throughput mode.

| Earlier error | Why it happened | Current behavior |
|---|---|---|
| `workload budget: long_output_slots` — **429/503** | A fixed count of simultaneous long-output requests was exhausted. | The fixed long-output count gate is disabled. |
| `workload budget: long_output_budget` — **429/503** | Predicted output reservations exceeded the former base/burst output budget. | That separate output-budget gate is disabled in throughput mode. |
| `workload budget: large_context_slots` — **429/503** | A fixed number of large-context requests was already active. | Large-context admission now uses shared capacity and physical reservations. |
| `P1-short-chat at its concurrency limit of …` — **429** | The class reached its static allocation despite spare capacity elsewhere. | Local classes share the adaptive execution pool. |
| `workload budget: engine_queue` — **429/503** | The earlier heavy-request guard rejected work when engine waiting exceeded its tolerance. | Engine queue pressure now affects adaptive execution starts; the old workload rejection branch is disabled. |
| nginx `limiting requests … k3_ip_req` — **503** | A shared portal IP exceeded the earlier **20 requests/second** bucket. | The throughput-mode inference location no longer applies that fixed request-rate limiter. |
| `Invalid tool choice … type: allowed_tools` — **500** | Responses-style tool selection reached a Chat conversion path that did not accept it. | The deployed normalizer converts this form into Chat-compatible tool selection. |
| Unsupported thinking-effort values — **500** | K3’s encoder received values outside its accepted effort names. | The normalizer maps common aliases and removes unsupported values. |

A saved NewAPI configuration also contains a **400 → 503 status mapping**. Its current external setting was not verified; the HTTP behavior described above refers to the direct `api.cflowx.in` service.

## 7. Dashboard alerts and degraded responses

| Error / alert / symptom | Why it appears |
|---|---|
| `Dashboard API unavailable: HTTP …` / fetch failure | The browser cannot fetch or decode `/api/state`. |
| Dashboard **404** `not found` | An unknown dashboard route was requested. |
| Dashboard **500** `<ExceptionType>: …` | Reading the UI file or generating the state response raises an exception. |
| Engine `unreachable` / `http_error` | The monitor cannot connect to `/metrics`, times out, or receives an HTTP error. |
| `no_metrics` | The endpoint returns no parseable metric samples. |
| `monitor_error` | The monitoring code itself throws an exception while scraping/processing data. |
| `Gateway metrics are unavailable` | Admission/control-plane metrics cannot be collected. |
| `Gateway tokenizer is quarantined` | A completed request’s actual prompt-token count disagrees with the inspection count. The gateway stops trusting exact inspection and falls back to conservative reservations. |
| Admission timeout alert | One or more admission deadlines expired in the observed window. |
| Engine-error alert | Gateway upstream failures were recorded in the observed window. |
| Requests queued for a decode slot | The engine has waiting requests. |
| KV cache nearly full | Observed KV utilization reaches **90%**. |
| Preemption alert | The engine discarded/recomputed work under memory pressure. |
| Prefix caching disabled | Every turn must process its shared prefix again. |
| 5xx / 429 / repeated 401 alerts | Recent access logs contain server errors, throttling, or at least **20 unauthorized requests in 60 seconds**. |
| Certificate-expiry alert | Fewer than **21 days** remain; severity increases below **7 days**. |
| Usage-log unreadable / attribution unavailable | The file is missing, inaccessible, or cannot be read. |
| `no GPU samples` | Both exporter/fallback GPU collection fail to produce usable samples. |
| `[attached media could not be decoded; continuing with the text]` | Media download, format parsing, image decoding, or video decoding fails or exceeds its limits. This marker is inserted into the model input; it is not necessarily an HTTP error. |
| `[additional video omitted; one video is allowed per request]` | More than one video was supplied. |
| Output ends with `finish_reason: length` | Generation reaches its granted output allowance. This normally returns **200**, rather than an HTTP error. Current class ceilings range from **1,536 tokens for agentic requests** to **32,768 for batch requests**. |

Source: `dashboard/server.py:1382`, `dashboard/index.html:171`, and `redesign/gateway/media.py`.

## 8. Deployment, startup, and operator-tool errors

| Error / failure | Why it happens |
|---|---|
| `repo-root deploy.sh is retired` | The old deployment entry point is intentionally disabled. |
| `ENGINE_IMAGE is unset` / `image ID mismatch` | The deployment lacks its required image setting, or the launch utility’s image does not match its pinned expectation. |
| `docker/python3/rocm-smi/nginx not found` | A required runtime or host dependency is missing. |
| `found … GPUs … requires …` | The host does not expose the required **8 GPUs**. |
| `weights missing at /scratch/hf` | The model files are absent or hidden by an incorrect mount. |
| Insufficient disk-space error | Available storage is below the deployment’s weight-download or operating-headroom requirement. |
| `no such profile` / image does not accept flags | A nonexistent or incompatible engine profile was selected. |
| Capacity-model hypothesis test failed | The deployment’s capacity assumptions fail their self-check. |
| SGLang `Hybrid state cache is too small to serve any requests` | In the retained SGLang deployment path, dividing admission capacity across DP ranks leaves too little per-rank capacity. |
| Gateway configuration `ValueError` | Invalid queue limits, wait duration, memory thresholds, adaptive thresholds, prediction settings, or inconsistent execution-capacity bounds. |
| `K3_RESPONSE_CACHE_MODE must be static, opt_in, or off` | An unsupported response-cache mode is configured. |
| `LiteLLM Prisma schema is missing` | LiteLLM’s required database schema file is absent from the installed environment. |
| Failed to write Bearer map / invalid nginx configuration | Generated configuration cannot be written or fails `nginx -t`. |
| Public IP unknown / DNS not pointing here / DNS wait expired | Certificate issuance cannot confirm that the hostname resolves to this machine. |
| `certbot FAILED` / certificate missing after success | ACME issuance fails, or the expected certificate file is not produced. |
| Engine did not become ready within 40 minutes | Deployment readiness polling exceeds its startup deadline. |
| `CORRECTNESS GATE FAILED` / `GATE FAIL` | Factual, reasoning, long-context, or regression checks fail. HTTP availability alone does not establish output correctness. |
| `start request repeated too quickly` | Repeated service failures exhaust systemd’s configured restart allowance. |
| Historical GPU OOM / eviction / `RuntimeError: cancelled` | Insufficient GPU workspace or a worker failure interrupts the engine. The documented high-memory-utilization profile caused this previously. |
| Historical fp8 profiling assertion, `requires batch_size=1, got …` | An incompatible fp8/AITER MLA kernel variant is selected. That profile was rejected. |
| Historical fluent-but-wrong output | An incompatible AITER activation/weight-layout combination produces incorrect results without an HTTP failure. |
| Backup passphrases differ / passphrase too short | Interactive backup passphrase validation fails. |
| Backup missing/stale/inconsistent files | Required credential files are absent, differ from the live copies, or the stored legacy credential pair disagrees. |
| `decryption failed - wrong passphrase or corrupt file` | The encrypted backup cannot be decrypted and unpacked. |
| `refusing a partial restore` | The decrypted archive is missing required files. |
| `run as root` / unknown option or stage | The operator tool receives insufficient privileges or invalid CLI arguments. |
| **Currently failed `k3-distill.service`** | Its journal records **SIGTERM on 20 September** and a retained failed state. This is separate from the currently running inference services. |

## Primary evidence

- `/var/log/k3/usage.log` and its retained cleared log.
- `/var/log/nginx/error.log`.
- `LiteLLM_SpendLogs.metadata.error_information`.
- Gateway diagnostics at the recorded inspection time.
- Deployed configuration and installed dependency source.
- Repository source under `/root/cflow-gpu`.
