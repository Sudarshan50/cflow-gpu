"""Gateway-owned context accounting through real handlers and tokenizer HTTP.

Both HTTP peers bind ephemeral loopback ports. The engine peer returns synthetic
counts/completions; installed-LiteLLM cases pass its actual SDK request body into
the same gateway through a mock client transport. No inference service is used.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import http.client
import importlib.util
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from redesign.gateway import media
from redesign.gateway.capture import CompletionRecord, MemorySink, TraceRecord, TraceRecorder
from redesign.gateway.classification import AGENTIC, LONG_CONTEXT, SHORT_CHAT
from redesign.gateway.engine_tokens import EngineTokenEstimator
from redesign.gateway.server import GatewayService, Handler, build_service
from redesign.gateway.tokens import HeuristicEstimator
from redesign.tenancy.policy import TenancyPolicy
from redesign.tests.test_cancellation import metric


PATH = "/v1/chat/completions"
MODEL = "FW-Kimi-K3"
PRIVATE_ERROR = b"PRIVATE_TOKENIZER_ERROR_BODY"
PRIVATE_OUTPUT = "PRIVATE_COMPLETION_TEXT"


def chat(content="synthetic prompt", **kwargs):
    return {"model": MODEL, "messages": [{"role": "user", "content": content}], **kwargs}


def image_bytes():
    from PIL import Image
    with Image.new("RGBA", (3, 2), (30, 60, 90, 120)) as image, io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def sse(document):
    return b"data: " + json.dumps(document).encode() + b"\n\n"


class _PeerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 2

    def log_message(self, *args):
        pass

    def do_GET(self):
        pipeline = self.server.pipeline
        pipeline.events.append(self.path)
        if self.path == "/metrics":
            self._reply(200, (
                b"vllm:kv_cache_usage_perc 0.1\n"
                b"vllm:num_requests_running 0\n"
                b"vllm:num_requests_waiting 0\n"
                b"vllm:num_preemptions_total 0\n"
                b'vllm:cache_config_info{kv_cache_size_tokens="1000000"} 1\n'
            ), "text/plain")
        elif self.path == "/image.png":
            self._reply(200, pipeline.image, "image/png")
        else:
            raise AssertionError(f"unexpected peer GET {self.path}")

    def do_POST(self):
        pipeline = self.server.pipeline
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        pipeline.events.append(self.path)
        snapshot = pipeline.service.policy.budget_snapshot()
        if self.path == "/tokenize":
            pipeline.tokenize_requests.append(payload)
            pipeline.tokenize_budgets.append(snapshot)
            body = json.dumps({
                "count": pipeline.prompt_count, "tokens": [17] * pipeline.prompt_count,
                "max_model_len": pipeline.engine_window,
            }).encode() if pipeline.tokenize_status == 200 else PRIVATE_ERROR
            self._reply(pipeline.tokenize_status, body)
        elif self.path == PATH:
            pipeline.inference_requests.append(payload)
            pipeline.inference_budgets.append(snapshot)
            body = pipeline.completion_body(payload)
            pipeline.inference_bodies.append(body)
            self._reply(200, body, "text/event-stream" if payload.get("stream") else "application/json",
                        chunked=bool(payload.get("stream")))
        else:
            raise AssertionError(f"unexpected peer POST {self.path}")

    def _reply(self, status, body, content_type="application/json", *, chunked=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding" if chunked else "Content-Length",
                         "chunked" if chunked else str(len(body)))
        self.end_headers()
        self.close_connection = True
        try:
            if chunked:
                for offset in range(0, len(body), 41):
                    chunk = body[offset:offset + 41]
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A rejected tokenizer status is closed without reading its body.
            if self.path != "/tokenize" or status == 200:
                raise


class _Server(ThreadingHTTPServer):
    daemon_threads = False

    def handle_error(self, request, client_address):
        import sys
        self.pipeline.errors.append(sys.exc_info()[1])


@contextmanager
def serve(handler, pipeline):
    server = _Server(("127.0.0.1", 0), handler)
    server.pipeline = pipeline
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        if thread.is_alive():
            raise AssertionError("pipeline HTTP server thread leaked")


class _Pipeline:
    def __init__(self, *, prompt_count=37, engine_window=262_144,
                 gateway_window=262_144, tokenize_status=200, trace_directory=None,
                 context_token_budget=None):
        self.prompt_count = prompt_count
        self.engine_window = engine_window
        self.gateway_window = gateway_window
        self.tokenize_status = tokenize_status
        self.trace_directory = Path(trace_directory) if trace_directory else None
        self.context_token_budget = context_token_budget
        self.finish_reason = "stop"
        self.done = True
        self.reported_prompt_tokens = None
        self.image = b""
        self.events, self.errors = [], []
        self.tokenize_requests, self.tokenize_budgets = [], []
        self.inference_requests, self.inference_budgets, self.inference_bodies = [], [], []
        self.responses = []
        self.finished = queue.Queue()

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.media_cache = media._ValidationCache()
            self.stack.enter_context(patch.object(media, "_CACHE", self.media_cache))
            peer = self.stack.enter_context(serve(_PeerHandler, self))
            self.peer_url = f"http://127.0.0.1:{peer.server_port}"
            self.admission_path = self.trace_directory / "admissions.jsonl" if self.trace_directory else None
            self.completion_path = self.trace_directory / "completions.jsonl" if self.trace_directory else None
            self.service = build_service(
                self.peer_url, self.gateway_window, 64, self.admission_path, None, True,
                token_estimator="engine", tokenize_timeout=1.0, request_timeout=5.0,
                completion_trace_path=self.completion_path, context_token_budget=self.context_token_budget,
            )
            self.stack.callback(self.service.close)
            if self.trace_directory is None:
                self.admissions, self.completions = MemorySink(), MemorySink()
                self.service.recorder = TraceRecorder(self.admissions, completion_sink=self.completions)

            class GatewayHandler(Handler):
                def do_POST(handler):
                    try:
                        super().do_POST()
                    finally:
                        # HTTP EOF can precede terminal observation and release.
                        self.finished.put(None)

            GatewayHandler.service = self.service
            self.gateway = self.stack.enter_context(serve(GatewayHandler, self))
            self.api_base = f"http://127.0.0.1:{self.gateway.server_port}/v1"
            self.count = self.stack.enter_context(patch.object(
                self.service.estimator, "count", wraps=self.service.estimator.count,
            ))
            self.decide = self.stack.enter_context(patch.object(
                self.service.policy, "decide", wraps=self.service.policy.decide,
            ))
            self.acquire = self.stack.enter_context(patch.object(
                self.service.policy._budget, "try_acquire_lease",
                wraps=self.service.policy._budget.try_acquire_lease,
            ))
            self.release = self.stack.enter_context(patch.object(
                self.service.policy, "release", wraps=self.service.policy.release,
            ))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        try:
            return self.stack.__exit__(*exc)
        finally:
            if any(thread.is_alive() for thread in self.service.estimator._executor._threads):
                raise AssertionError("pipeline estimator worker leaked")
            if self.errors and exc[0] is None:
                raise AssertionError(self.errors)

    def completion_body(self, payload):
        prompt = self.prompt_count if self.reported_prompt_tokens is None else self.reported_prompt_tokens
        completion = min(4, payload["max_tokens"])
        usage = {
            "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": min(16, prompt)},
            "completion_tokens_details": {"reasoning_tokens": min(2, completion)},
        }
        base = {"id": "private-completion-id", "created": 1, "model": MODEL}
        if not payload.get("stream"):
            return json.dumps({
                **base, "object": "chat.completion", "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": PRIVATE_OUTPUT},
                    "finish_reason": self.finish_reason,
                }], "usage": usage,
            }).encode()
        base["object"] = "chat.completion.chunk"
        wire = b": heartbeat\n\n"
        for delta, finish in (
            ({"role": "assistant"}, None),
            ({"reasoning_content": "PRIVATE_REASONING_TEXT"}, None),
            ({"content": PRIVATE_OUTPUT}, None),
            ({}, self.finish_reason),
        ):
            wire += sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        wire += sse({**base, "choices": [], "usage": usage})
        return wire + (b"data: [DONE]\n\n" if self.done else b"")

    def post(self, payload):
        connection = http.client.HTTPConnection(*self.gateway.server_address, timeout=5)
        try:
            connection.request("POST", PATH, json.dumps(payload).encode(), {
                "Content-Type": "application/json", "Connection": "close", "x-k3-customer": "private-customer",
            })
            response = connection.getresponse()
            result = SimpleNamespace(status=response.status, headers=dict(response.getheaders()), body=response.read())
            self.finished.get(timeout=2)
            self.responses.append(result)
            return result
        finally:
            connection.close()


def assert_no_admission(test, pipeline):
    pipeline.decide.assert_not_called()
    pipeline.acquire.assert_not_called()
    pipeline.release.assert_not_called()
    test.assertEqual(pipeline.inference_requests, [])
    test.assertNotIn("/metrics", pipeline.events)
    test.assertEqual(pipeline.admissions.records, [])
    test.assertEqual(pipeline.completions.records, [])
    snapshot = pipeline.service.policy.budget_snapshot()
    test.assertEqual((snapshot.in_flight, snapshot.context_tokens_in_flight), (0, 0))
    test.assertEqual(metric(pipeline.service.registry, "requests_total"), 0)
    test.assertEqual(metric(pipeline.service.registry, "relay_terminal_total"), 0)


class GatewayContextPipelineTest(unittest.TestCase):
    def test_exact_count_drives_classification_clamp_and_one_context_lease(self):
        payload = chat(max_completion_tokens=8_192)
        self.assertLess(HeuristicEstimator().estimate(payload), 100)
        cost = 50_000 + 8_192
        with _Pipeline(prompt_count=50_000, engine_window=65_536, context_token_budget=cost) as pipeline:
            self.assertIsInstance(pipeline.service, GatewayService)
            self.assertIsInstance(pipeline.service.estimator, EngineTokenEstimator)
            with patch.object(HeuristicEstimator, "estimate", side_effect=AssertionError("heuristic fallback")):
                response = pipeline.post(payload)
            self.assertEqual(response.status, 200, response.body)
            pipeline.count.assert_called_once()
            pipeline.decide.assert_called_once()
            pipeline.acquire.assert_called_once()
            pipeline.release.assert_called_once()
            self.assertEqual(pipeline.events, ["/tokenize", "/metrics", PATH])
            self.assertEqual(pipeline.tokenize_budgets[0].in_flight, 0)
            during = pipeline.inference_budgets[0]
            self.assertEqual((during.in_flight, during.context_tokens_in_flight), (1, cost))
            self.assertEqual(during.by_class[LONG_CONTEXT.name], 1)
            after = pipeline.service.policy.budget_snapshot()
            self.assertEqual((after.in_flight, after.context_tokens_in_flight), (0, 0))
            forwarded = pipeline.inference_requests[0]
            self.assertEqual((forwarded["max_tokens"], forwarded["max_completion_tokens"]), (8_192, 8_192))
            self.assertEqual(forwarded["priority"], int(LONG_CONTEXT.priority))
            self.assertEqual(pipeline.admissions.records[0].prompt_tokens, 50_000)
            self.assertEqual(response.headers["x-k3-class"], LONG_CONTEXT.name)
            self.assertEqual(response.headers["x-k3-prompt-tokens"], "50000")
            self.assertEqual(response.headers["x-k3-prompt-count-source"], "engine_rendered")
            self.assertEqual(json.loads(response.body)["usage"]["prompt_tokens"], 50_000)
            completion = pipeline.completions.records[0]
            self.assertEqual((completion.counted_prompt_tokens, completion.prompt_tokens), (50_000, 50_000))
            self.assertEqual(completion.model_context_limit, 65_536)
            self.assertEqual(metric(pipeline.service.registry, "token_count_comparisons_total"), 1)
            self.assertEqual(metric(pipeline.service.registry, "token_count_mismatches_total"), 0)
        self.assertEqual(pipeline.service.estimator.snapshot()["closed"], 1)
        self.assertTrue(all(not thread.is_alive() for thread in pipeline.service.estimator._executor._threads))

    def test_exact_context_has_zero_reserve_and_honors_small_explicit_limits(self):
        for requested in (1, 16):
            for limiting_window in ("gateway", "engine"):
                with self.subTest(requested=requested, limiting_window=limiting_window):
                    windows = {"gateway_window": 1_024, "engine_window": 1_024}
                    windows[limiting_window + "_window"] = 31 + requested
                    with _Pipeline(prompt_count=31, **windows) as pipeline:
                        payload = chat(max_tokens=1_024, max_completion_tokens=requested)
                        response = pipeline.post(payload)
                        self.assertEqual(response.status, 200, response.body)
                        self.assertEqual(response.headers["x-k3-max-tokens-requested"], str(requested))
                        self.assertEqual(response.headers["x-k3-max-tokens-granted"], str(requested))
                        self.assertEqual(response.headers["x-k3-clamp-reason"], "unchanged")
                        forwarded = pipeline.inference_requests[0]
                        self.assertEqual((forwarded["max_tokens"], forwarded["max_completion_tokens"]),
                                         (requested, requested))
                        self.assertEqual(pipeline.inference_budgets[0].context_tokens_in_flight, 31 + requested)
                        # One more requested token cannot fit; a cached exact
                        # count still has to enforce the smaller window.
                        refused = pipeline.post({**payload, "max_completion_tokens": requested + 1})
                        self.assertEqual(refused.status, 400)
                        self.assertEqual(refused.headers["x-k3-clamp-reason"], "prompt_exceeds_window")
                        self.assertEqual(refused.headers["x-k3-prompt-count-cache-hit"], "true")
                        self.assertEqual(len(pipeline.tokenize_requests), 1)
                        self.assertEqual(len(pipeline.inference_requests), 1)
                        pipeline.acquire.assert_called_once()
                        self.assertEqual(pipeline.service.policy.budget_snapshot().in_flight, 0)

    def test_tokenizer_dependency_failures_return_503_before_admission(self):
        for status in (401, 403, 408, 429, 500, 502, 503, 504):
            with self.subTest(status=status), _Pipeline(tokenize_status=status) as pipeline:
                response = pipeline.post(chat())
                self.assertEqual(response.status, 503)
                self.assertEqual(response.headers["Retry-After"], "2")
                self.assertEqual(json.loads(response.body)["error"]["type"], "service_unavailable")
                self.assertNotIn(PRIVATE_ERROR, response.body)
                assert_no_admission(self, pipeline)
                pipeline.count.assert_called_once()
                self.assertEqual(len(pipeline.tokenize_requests), 1)
                stats = pipeline.service.estimator.snapshot()
                self.assertEqual((stats["unavailable_total"], stats["input_errors_total"]), (1, 0))
                self.assertEqual((stats["cache_entries"], stats["response_bytes_total"]), (0, 0))
                self.assertEqual(metric(pipeline.service.registry, "preprocessing_rejections_total", reason="http_status"), 1)
                self.assertNotIn(PRIVATE_ERROR.decode(), pipeline.service.registry.render())

    def test_tokenizer_input_failures_return_400_before_admission(self):
        for status in (400, 404, 413, 422):
            with self.subTest(status=status), _Pipeline(tokenize_status=status) as pipeline:
                response = pipeline.post(chat())
                self.assertEqual(response.status, 400)
                self.assertEqual(json.loads(response.body)["error"]["type"], "invalid_request_error")
                self.assertNotIn(PRIVATE_ERROR, response.body)
                assert_no_admission(self, pipeline)
                self.assertEqual(len(pipeline.tokenize_requests), 1)
                self.assertEqual(pipeline.service.estimator.snapshot()["input_errors_total"], 1)
                self.assertEqual(metric(pipeline.service.registry, "input_rejections_total", reason="TokenCountInputError"), 1)

    def test_invalid_media_is_400_before_tokenization_or_admission(self):
        url = "data:image/png;base64," + base64.b64encode(b"PRIVATE_INVALID_IMAGE").decode()
        payload = chat([{"type": "text", "text": "private prompt"},
                        {"type": "image_url", "image_url": {"url": url}}])
        with _Pipeline() as pipeline:
            response = pipeline.post(payload)
            self.assertEqual(response.status, 400)
            error = json.loads(response.body)["error"]
            self.assertEqual(error["type"], "invalid_request_error")
            self.assertEqual(error["param"], "messages[0].content[1]")
            self.assertNotIn(url.encode(), response.body)
            self.assertNotIn(b"PRIVATE_INVALID_IMAGE", response.body)
            assert_no_admission(self, pipeline)
            pipeline.count.assert_not_called()
            self.assertEqual(pipeline.tokenize_requests, [])
            self.assertEqual(pipeline.media_cache.stats()["entries"], 0)
            self.assertEqual(metric(pipeline.service.registry, "input_rejections_total", reason="MediaValidationError"), 1)

    def test_sse_completion_trace_is_separate_and_records_usage_and_finish_stats(self):
        with tempfile.TemporaryDirectory() as directory, _Pipeline(trace_directory=directory) as pipeline:
            pipeline.finish_reason = "length"
            response = pipeline.post(chat("PRIVATE_PROMPT_TEXT", stream=True, max_tokens=16))
            self.assertEqual(response.status, 200)
            self.assertEqual(response.body, pipeline.inference_bodies[0])
            admissions = [json.loads(line) for line in pipeline.admission_path.read_text().splitlines()]
            completions = [json.loads(line) for line in pipeline.completion_path.read_text().splitlines()]
            self.assertEqual((len(admissions), len(completions)), (1, 1))
            self.assertEqual(admissions[0]["outcome"], "ADMIT")
            self.assertNotIn("protocol_complete", admissions[0])
            completion = completions[0]
            self.assertNotIn("outcome", completion)
            self.assertEqual((completion["counted_prompt_tokens"], completion["prompt_tokens"]), (37, 37))
            self.assertEqual((completion["completion_tokens"], completion["cached_tokens"], completion["reasoning_tokens"]),
                             (4, 16, 2))
            self.assertEqual(completion["finish_reasons"], ["length"])
            self.assertIs(completion["protocol_complete"], True)
            self.assertFalse(completion["protocol_error"])
            self.assertEqual((completion["relay_outcome"], completion["upstream_status"]), ("relayed", 200))
            self.assertGreaterEqual(completion["first_token_seconds"], 0)
            self.assertLessEqual(completion["first_token_seconds"], completion["elapsed_seconds"])
            registry = pipeline.service.registry
            self.assertEqual(metric(registry, "protocol_completions_total"), 1)
            self.assertEqual(metric(registry, "finish_reasons_total", reason="length"), 1)
            self.assertEqual(metric(registry, "protocol_incomplete_total"), 0)
            self.assertEqual(metric(registry, "ttft_seconds_count"), 1)
            self.assertEqual(metric(registry, "token_count_comparisons_total"), 1)
            self.assertEqual(metric(registry, "token_count_mismatches_total"), 0)
            diagnostics = json.dumps([admissions, completions, pipeline.service.estimator.snapshot()]) + registry.render()
            for private in ("PRIVATE_PROMPT_TEXT", PRIVATE_OUTPUT, "PRIVATE_REASONING_TEXT", "private-completion-id", "private-customer"):
                self.assertNotIn(private, diagnostics)

    def test_done_without_finish_and_finish_without_done_are_only_transport_relayed(self):
        for finish, done in ((None, True), ("stop", False)):
            with self.subTest(finish=finish, done=done), _Pipeline() as pipeline:
                pipeline.finish_reason, pipeline.done = finish, done
                response = pipeline.post(chat(stream=True))
                self.assertEqual(response.status, 200)
                self.assertEqual(response.body, pipeline.inference_bodies[0])
                completion = pipeline.completions.records[0]
                self.assertIs(completion.protocol_complete, False)
                self.assertFalse(completion.protocol_error)
                self.assertEqual(completion.relay_outcome, "relayed")
                self.assertEqual(completion.prompt_tokens, 37)
                self.assertEqual(metric(pipeline.service.registry, "relay_terminal_total", outcome="relayed"), 1)
                self.assertEqual(metric(pipeline.service.registry, "protocol_incomplete_total"), 1)
                self.assertEqual(metric(pipeline.service.registry, "protocol_completions_total"), 0)
                self.assertEqual(metric(pipeline.service.registry, "finish_reasons_total"), 0)

    def test_reported_usage_mismatch_does_not_rewrite_admission_accounting(self):
        with _Pipeline() as pipeline:
            pipeline.reported_prompt_tokens = 39
            response = pipeline.post(chat(max_tokens=16))
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.body)["usage"]["prompt_tokens"], 39)
            admission, completion = pipeline.admissions.records[0], pipeline.completions.records[0]
            self.assertIsInstance(admission, TraceRecord)
            self.assertIsInstance(completion, CompletionRecord)
            self.assertEqual(admission.prompt_tokens, 37)
            self.assertEqual((completion.counted_prompt_tokens, completion.prompt_tokens), (37, 39))
            self.assertIsNone(completion.first_token_seconds, "nonstream JSON is not a TTFT sample")
            self.assertEqual(pipeline.inference_budgets[0].context_tokens_in_flight, 37 + 16)
            self.assertEqual(metric(pipeline.service.registry, "token_count_comparisons_total"), 1)
            self.assertEqual(metric(pipeline.service.registry, "token_count_mismatches_total"), 1)
            self.assertEqual(metric(pipeline.service.registry, "ttft_seconds_count"), 0)


@unittest.skipUnless(importlib.util.find_spec("litellm"), "requires installed proxy LiteLLM")
class CallbackContextPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import litellm
        from openai import AsyncOpenAI
        from redesign.tenancy.callback import K3TenancyCallback

        self.pipeline = self.enterContext(_Pipeline(gateway_window=8_192, engine_window=8_192))
        self.enterContext(patch.dict(os.environ, {"K3_GATEWAY_POLICY_URL": self.pipeline.api_base}))
        self.callback = K3TenancyCallback(
            policy=TenancyPolicy(max_model_len=8_192), cache_mode="off", team_pool_ids=frozenset(),
        )
        self.litellm = litellm
        for name in ("input_callback", "success_callback", "failure_callback", "service_callback",
                     "_async_input_callback", "_async_success_callback", "_async_failure_callback"):
            self.enterContext(patch.object(litellm, name, []))
        self.enterContext(patch.object(litellm, "callbacks", [self.callback]))
        self.enterContext(patch.object(litellm, "cache", None))
        self.ingress, self.before, self.after = [], [], []

        def transport(request):
            self.assertEqual(str(request.url), self.pipeline.api_base + "/chat/completions")
            body = json.loads(request.content)
            self.ingress.append(body)
            result = self.pipeline.post(body)
            return httpx.Response(result.status, headers=result.headers, content=result.body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.client = AsyncOpenAI(api_key="offline-provider", base_url=self.pipeline.api_base,
                                  http_client=client, max_retries=0)
        self.addAsyncCleanup(self.client.close)
        original = self.callback.async_pre_call_deployment_hook

        async def deployment(kwargs, call_type):
            self.before.append(self.shape(kwargs))
            result = await original(kwargs, call_type)
            self.after.append(self.shape(result))
            return result

        self.enterContext(patch.object(self.callback, "async_pre_call_deployment_hook", side_effect=deployment))

    @staticmethod
    def shape(data):
        # Detached synthetic request fields, excluding logging/client objects.
        return copy.deepcopy({key: data[key] for key in (
            "messages", "model", "api_base", "max_tokens", "max_completion_tokens", "max_output_tokens",
            "metadata", "litellm_metadata", "extra_body", "chat_template_kwargs", "stream_options",
        ) if key in data})

    async def complete(self, payload, **defaults):
        data = copy.deepcopy(payload)
        await self.callback.async_pre_call_hook({"api_key": "offline-auth"}, None, data, "acompletion")
        router = self.litellm.Router(model_list=[{
            "model_name": MODEL, "litellm_params": {
                "model": "openai/" + MODEL, "api_base": self.pipeline.api_base,
                "api_key": "offline-provider", **defaults,
            },
        }], num_retries=0)
        with patch.object(router, "_get_async_openai_model_client", return_value=self.client):
            result = await router.acompletion(**data)
            if payload.get("stream"):
                result = [chunk async for chunk in result]
        for _ in range(10):
            await asyncio.sleep(0)
        return result

    async def test_callback_preserves_caller_and_deployment_budgets_until_exact_gateway_policy(self):
        text = "PRIVATE_PROMPT_TEXT" + "x" * 35_000
        self.assertGreater(HeuristicEstimator().estimate(chat(text)), 8_192)
        cases = (
            ({}, {}, None, 1_024, "default"),
            ({"max_tokens": 8_192}, {}, 8_192, 4_096, "class_ceiling"),
            ({}, {"max_tokens": 8_192}, 8_192, 4_096, "class_ceiling"),
            ({"max_tokens": 1_024, "max_completion_tokens": 16}, {}, 16, 16, "unchanged"),
            ({"max_output_tokens": 8_192}, {}, 8_192, 4_096, "class_ceiling"),
        )
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch.object(self.callback.policy.estimator, "estimate", side_effect=AssertionError("early heuristic")) as estimate, \
             patch("redesign.tenancy.callback.normalize_payload", side_effect=AssertionError("early media decode")) as normalize:
            for aliases, defaults, requested, granted, reason in cases:
                with self.subTest(aliases=aliases, defaults=defaults):
                    result = await self.complete(chat(text, **aliases), **defaults)
                    pending = self.after[-1]["metadata"]["k3_output_policy"]
                    self.assertEqual(pending, {"stage": "gateway_pending", "requested_max_tokens": requested})
                    self.assertNotIn("max_output_tokens", self.after[-1])
                    if requested is None:
                        self.assertIsNone(self.after[-1].get("max_tokens"))
                        self.assertIsNone(self.ingress[-1].get("max_tokens"))
                    elif "max_completion_tokens" in aliases:
                        self.assertEqual(self.after[-1]["max_completion_tokens"], requested)
                        self.assertEqual(self.after[-1]["max_tokens"], aliases["max_tokens"])
                    else:
                        self.assertEqual(self.after[-1]["max_tokens"], requested)
                        self.assertEqual(self.ingress[-1]["max_tokens"], requested)
                    record = self.pipeline.completions.records[-1]
                    self.assertEqual((record.requested_max_tokens, record.granted_max_tokens), (requested, granted))
                    self.assertEqual(record.default_applied, requested is None)
                    self.assertEqual(record.default_output_tokens, SHORT_CHAT.default_output_tokens)
                    self.assertEqual(record.clamp_reason, reason)
                    self.assertEqual(record.traffic_class, SHORT_CHAT.name)
                    self.assertEqual((record.counted_prompt_tokens, record.prompt_tokens), (37, 37))
                    self.assertEqual(result.usage.prompt_tokens, 37)
                    self.assertEqual(self.pipeline.inference_requests[-1]["max_tokens"], granted)
                    self.assertEqual(self.pipeline.inference_budgets[-1].context_tokens_in_flight, 37 + granted)
                    headers = self.pipeline.responses[-1].headers
                    self.assertEqual(headers["x-k3-max-tokens-requested"], "none" if requested is None else str(requested))
                    self.assertEqual(headers["x-k3-output-policy-stage"], "gateway")
            apply.assert_not_called()
            estimate.assert_not_called()
            normalize.assert_not_called()
        self.assertEqual(len(self.before), len(cases))
        self.assertEqual(self.pipeline.count.call_count, len(cases))
        self.assertEqual(self.pipeline.decide.call_count, len(cases))
        self.assertEqual(self.pipeline.acquire.call_count, len(cases))
        self.assertEqual(len(self.pipeline.tokenize_requests), 1, "only output settings changed")
        self.assertEqual(self.pipeline.service.estimator.snapshot()["cache_hits_total"], len(cases) - 1)
        self.assertEqual(metric(self.pipeline.service.registry, "token_count_comparisons_total"), len(cases))
        self.assertEqual(metric(self.pipeline.service.registry, "token_count_mismatches_total"), 0)
        self.assertEqual(self.pipeline.service.policy.budget_snapshot().in_flight, 0)

    async def test_callback_defers_image_download_and_count_and_inference_receive_identical_media(self):
        self.pipeline.image = image_bytes()
        self.pipeline.prompt_count = 600
        self.pipeline.engine_window = 4_096
        url = self.pipeline.peer_url + "/image.png"
        tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                 for name in ("z_lookup", "a_lookup")]
        payload = chat([
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
        ], tools=tools, reasoning_effort="medium", max_completion_tokens=8_192,
            cache_salt="private-salt", extra_body={"kv_cache_salt": "private-salt", "priority": 99}, stream=True)
        with patch("redesign.tenancy.callback.normalize_payload", side_effect=AssertionError("callback decoded media")) as normalize, \
             patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch.object(media, "_validate_image", wraps=media._validate_image) as decode:
            chunks = await self.complete(payload)
            normalize.assert_not_called()
            apply.assert_not_called()
            decode.assert_called_once()
        self.assertEqual(self.after[0]["messages"], payload["messages"])
        self.assertEqual(self.ingress[0]["messages"], payload["messages"])
        self.assertEqual(self.pipeline.events, ["/image.png", "/tokenize", "/metrics", PATH])
        counted, forwarded = self.pipeline.tokenize_requests[0], self.pipeline.inference_requests[0]
        self.assertEqual(counted["messages"], forwarded["messages"])
        normalized = forwarded["messages"][0]["content"][1]["image_url"]
        self.assertEqual(normalized["detail"], "auto")
        self.assertEqual(base64.b64decode(normalized["url"].split(",", 1)[1]), self.pipeline.image)
        self.assertEqual(counted["tools"], forwarded["tools"])
        self.assertEqual([tool["function"]["name"] for tool in counted["tools"]], ["a_lookup", "z_lookup"])
        self.assertEqual(counted["chat_template_kwargs"]["thinking_effort"], "high")
        self.assertEqual(forwarded["chat_template_kwargs"]["thinking_effort"], "high")
        self.assertEqual((forwarded["max_tokens"], forwarded["max_completion_tokens"]), (3_496, 3_496))
        self.assertEqual(forwarded["priority"], int(AGENTIC.priority))
        for body in (self.ingress[0], counted):
            for field in ("cache_salt", "kv_cache_salt", "priority"):
                self.assertNotIn(field, body)
        self.assertEqual(self.pipeline.media_cache.stats()["validations"], 1)
        self.pipeline.count.assert_called_once()
        self.pipeline.acquire.assert_called_once()
        record = self.pipeline.completions.records[0]
        self.assertEqual((record.counted_prompt_tokens, record.prompt_tokens), (600, 600))
        self.assertEqual(record.modalities, ("text", "image"))
        self.assertEqual(record.finish_reasons, ("stop",))
        self.assertTrue(record.protocol_complete)
        self.assertEqual(record.clamp_reason, "context_window")
        usage = [chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None]
        self.assertEqual(usage[-1].prompt_tokens, 600)
        self.assertEqual(usage[-1].prompt_tokens_details.cached_tokens, 16)
        self.assertNotIn(url, json.dumps(asdict(record)))

    async def test_callback_defers_invalid_media_to_gateway_400_without_inference(self):
        payload = chat([{"type": "image_url", "image_url": {"url": "bad-image"}}])
        with patch("redesign.tenancy.callback.normalize_payload", side_effect=AssertionError("early decode")) as normalize:
            with self.assertRaises(Exception) as error:
                await self.complete(payload)
            self.assertEqual(getattr(error.exception, "status_code", None), 400, repr(error.exception))
            normalize.assert_not_called()
        self.assertEqual(self.after[0]["messages"], payload["messages"])
        self.assertEqual(len(self.ingress), 1, "the gateway must receive and validate the media")
        self.assertEqual(self.pipeline.responses[0].status, 400)
        self.pipeline.count.assert_not_called()
        assert_no_admission(self, self.pipeline)

    async def test_gateway_policy_url_requires_exact_deployment_match(self):
        from redesign.tenancy.callback import K3TenancyCallback
        base = self.pipeline.api_base
        cases = (
            (base, base, True), (base + "/", base + "/", True),
            (base, self.pipeline.peer_url + "/v1", False),
            (base, base + "/other", False), (base, base + "?tenant=local", False),
            (base, base.replace("127.0.0.1", "localhost"), False),
            (base, base.replace("http:", "https:"), False),
            (base, None, False), ("", base, False),
        )
        for configured, api_base, deferred in cases:
            with self.subTest(configured=configured, api_base=api_base):
                with patch.dict(os.environ, {"K3_GATEWAY_POLICY_URL": configured}):
                    callback = K3TenancyCallback(policy=TenancyPolicy(), cache_mode="off", team_pool_ids=frozenset())
                data = chat(api_base=api_base, max_tokens=8_192,
                            metadata={"k3_output_policy": {"stage": "gateway_pending"}})
                with patch.object(callback.policy, "apply", wraps=callback.policy.apply) as apply, \
                     patch("redesign.tenancy.callback.normalize_payload", wraps=media.normalize_payload) as normalize:
                    await callback.async_pre_call_deployment_hook(data, "acompletion")
                self.assertEqual(apply.call_count, int(not deferred))
                self.assertEqual(normalize.call_count, int(not deferred))
                self.assertEqual(data["max_tokens"], 8_192 if deferred else 4_096)
                self.assertEqual(data["metadata"]["k3_output_policy"]["stage"], "gateway_pending" if deferred else "tenancy")
        self.assertEqual(self.pipeline.events, [])


if __name__ == "__main__":
    unittest.main()
