"""Offline tokenizer HTTP peers: protocol parity, bounded work and cancellation.

All HTTP uses ephemeral loopback stubs. Nothing contacts the live engine or
generates tokens. Gates and peer EOF observations exercise real EngineClient
transport cleanup rather than mocks of close/cancel.
"""

from __future__ import annotations

import copy
import json
import select
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from redesign.gateway import engine_tokens
from redesign.gateway.cancellation import (
    CancelReason, RequestCancellation, RequestCancelled, current_cancellation,
)
from redesign.gateway.engine_tokens import (
    EngineTokenEstimator, TokenCountInputError, TokenCountResult, TokenCountUnavailable,
)
from redesign.gateway.prompt_protocol import build_tokenize_request, normalize_engine_prompt


CHAT = "/v1/chat/completions"
COMPLETION = "/v1/completions"


def chat(text="synthetic prompt", **kwargs):
    return {"model": "FW-Kimi-K3", "messages": [{"role": "user", "content": text}], **kwargs}


def tool(name="lookup"):
    return {"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }}


def response_format():
    return {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}},
        "required": ["answer"], "additionalProperties": False,
    }}}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


@dataclass
class Reply:
    body: object = field(default_factory=lambda: {"count": 3, "max_model_len": 262144, "tokens": [17, 29, 31], "token_strs": None})
    status: int = 200
    hold: bool = False
    chunked: bool = False
    drip: float = 0.0
    truncate: bool = False


@dataclass
class Seen:
    path: str
    payload: dict
    raw: bytes
    reply: Reply
    release: threading.Event = field(default_factory=threading.Event)
    peer_closed: threading.Event = field(default_factory=threading.Event)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 2

    def log_message(self, *args):
        pass

    def do_POST(self):
        stub = self.server.stub
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        seen = Seen(self.path, json.loads(raw), raw, copy.copy(stub.reply))
        with stub.changed:
            stub.requests.append(seen)
            stub.changed.notify_all()
        try:
            if seen.reply.hold and not self._pause(seen):
                return
            body = seen.reply.body if isinstance(seen.reply.body, bytes) else encoded(seen.reply.body)
            self.send_response(seen.reply.status)
            self.send_header("Content-Type", "application/json")
            if seen.reply.chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if seen.reply.truncate:
                self.wfile.write(body[:max(1, len(body) // 2)])
                self.close_connection = True
                return
            step = 1 if seen.reply.drip else 1024
            for offset in range(0, len(body), step):
                chunk = body[offset:offset + step]
                if seen.reply.chunked:
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                if seen.reply.drip and self._closed(seen, seen.reply.drip):
                    return
            if seen.reply.chunked:
                self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            seen.peer_closed.set()
            self.close_connection = True

    def _closed(self, seen, timeout):
        ready, _, _ = select.select([self.connection], [], [], timeout)
        if ready:
            try:
                closed = not self.connection.recv(1)
            except ConnectionResetError:
                closed = True
            if closed:
                seen.peer_closed.set()
                self.close_connection = True
                return True
        return self.server.stub.stop.is_set()

    def _pause(self, seen):
        while not seen.release.is_set() and not self.server.stub.stop.is_set():
            if self._closed(seen, 0.01):
                return False
        return not self.server.stub.stop.is_set()


class _Server(ThreadingHTTPServer):
    daemon_threads = False

    def handle_error(self, request, client_address):
        import sys
        self.stub.errors.append(sys.exc_info()[1])


class Stub:
    def __init__(self, reply=None):
        self.reply = reply or Reply()
        self.changed = threading.Condition()
        self.requests = []
        self.errors = []
        self.stop = threading.Event()
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.stub = self
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})

    def __enter__(self):
        self.thread.start()
        return self

    def wait(self, count, timeout=2):
        with self.changed:
            if not self.changed.wait_for(lambda: len(self.requests) >= count, timeout):
                raise AssertionError(f"expected {count} tokenizer requests, got {len(self.requests)}")
            return self.requests[:count]

    def release_all(self):
        self.reply.hold = False
        with self.changed:
            for request in self.requests:
                request.release.set()

    def __exit__(self, *exc):
        self.stop.set()
        self.release_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        if self.thread.is_alive():
            raise AssertionError("stub server thread leaked")
        if self.errors and exc[0] is None:
            raise AssertionError(self.errors)


def wait_stats(estimator, predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = estimator.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"estimator did not reach expected state: {estimator.snapshot()}")


def count_with_parent(estimator, payload, parent):
    with parent:
        return estimator.count(payload)


class PromptProtocolTest(unittest.TestCase):
    def test_shared_legacy_normalization_preserves_input_and_schema(self):
        function = tool()["function"]
        original = chat(
            functions=[function], function_call={"name": "lookup"},
            response_format=response_format(), temperature=0.2, max_completion_tokens=900,
        )
        before = copy.deepcopy(original)
        normalized = normalize_engine_prompt(original, CHAT)
        self.assertEqual(original, before)
        self.assertNotIn("functions", normalized)
        self.assertNotIn("function_call", normalized)
        self.assertEqual(normalized["tools"], [tool()])
        self.assertEqual(normalized["tool_choice"], {"type": "function", "function": {"name": "lookup"}})
        self.assertEqual(normalized["response_format"], original["response_format"])
        self.assertEqual(normalized["max_completion_tokens"], 900)
        self.assertEqual(normalized["temperature"], 0.2)
        self.assertEqual(normalize_engine_prompt(normalized, CHAT), normalized)
        normalized["tools"][0]["function"]["parameters"]["properties"].clear()
        self.assertEqual(original, before)

    def test_legacy_choices_and_conflicts(self):
        for old in ("none", "auto"):
            normalized = normalize_engine_prompt(chat(functions=[tool()["function"]], function_call=old), CHAT)
            self.assertEqual(normalized["tool_choice"], old)
        for extras in (
            {"functions": [tool()["function"]], "tools": [tool()]},
            {"function_call": "none", "tool_choice": "auto", "tools": [tool()]},
            {"function_call": {"name": "missing"}, "functions": [tool()["function"]]},
            {"function_call": "required", "functions": [tool()["function"]]},
        ):
            with self.subTest(extras=extras), self.assertRaises(TokenCountInputError):
                normalize_engine_prompt(chat(**extras), CHAT)

    def test_legacy_history_is_converted_without_changing_arguments_or_content(self):
        arguments = '{"query": "x", "options": {"z": 1, "a": 2}}'
        body = {"messages": [
            {"role": "assistant", "content": None, "function_call": {"name": "lookup", "arguments": arguments}},
            {"role": "function", "name": "lookup", "content": [{"type": "text", "text": "result"}]},
            {"role": "user", "content": "continue"},
        ]}
        normalized = normalize_engine_prompt(body, CHAT)
        call = normalized["messages"][0]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], arguments)
        self.assertEqual(normalized["messages"][1], {
            "role": "tool", "name": "lookup", "tool_call_id": call["id"],
            "content": [{"type": "text", "text": "result"}],
        })
        self.assertEqual(normalize_engine_prompt(normalized, CHAT), normalized)
        self.assertIn("function_call", body["messages"][0])
        body["messages"][1]["name"] = "different"
        with self.assertRaises(TokenCountInputError):
            normalize_engine_prompt(body, CHAT)

    def test_reasoning_alias_precedence_including_empty_string(self):
        for message, expected in (
            ({"reasoning_content": "old"}, "old"),
            ({"reasoning_content": "old", "reasoning": None}, "old"),
            ({"reasoning_content": "old", "reasoning": "new"}, "new"),
            ({"reasoning_content": "old", "reasoning": ""}, ""),
        ):
            with self.subTest(message=message):
                body = {"messages": [{"role": "assistant", "content": "answer", **message}]}
                normalized = normalize_engine_prompt(body, CHAT)
                self.assertEqual(normalized["messages"][0]["reasoning"], expected)
                self.assertNotIn("reasoning_content", normalized["messages"][0])
                self.assertIn("reasoning_content", body["messages"][0])

    def test_unsupported_semantics_are_explicit_errors(self):
        examples = [
            chat(truncate_prompt_tokens=10), chat(truncation_side="left"),
            chat(continue_final_message=True), chat(chat_template="custom"),
            chat(chat_template_kwargs={"chat_template": "custom"}),
            chat(chat_template_kwargs={"continue_final_message": True}),
            chat(chat_template_kwargs={"truncation": True}),
            chat(chat_template_kwargs={"image_prompts": ["stub"]}),
            chat(extra_body={"reasoning_effort": "high"}),
            {"messages": [{"role": "system", "content": "tools", "tools": [tool()]}]},
            {"messages": [{"role": "developer", "content": "important instruction"}]},
            {"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_data": "x"}}]}]},
            {"prompt": [1, 2]}, {"prompt": ["first", "second"]}, {"prompt": [[1], [2]]},
        ]
        for body in examples:
            with self.subTest(body=body), self.assertRaises(TokenCountInputError) as error:
                build_tokenize_request(body)
            self.assertEqual(error.exception.status_code, 400)

    def test_malformed_shapes_are_typed_input_errors(self):
        examples = [
            None, [], {"messages": []}, {"messages": ["text"]},
            {"messages": [{"role": [], "content": "x"}]},
            {"messages": [{"role": "assistant", "tool_calls": 12, "content": "x"}]},
            chat(tools={}), chat(tool_choice=[]), chat(response_format={"type": []}),
            chat(chat_template_kwargs=[]), chat(add_special_tokens="false"),
            chat(reasoning_effort={}), chat(media_io_kwargs=[]),
            chat(response_format={"type": "json_schema", "json_schema": {"name": "x"}}),
            chat(prompt="ambiguous"),
        ]
        for body in examples:
            with self.subTest(body=body), self.assertRaises(TokenCountInputError):
                build_tokenize_request(body)

    def test_plain_completion_contract_and_route_validation(self):
        body = {"model": "FW-Kimi-K3", "prompt": "hello", "max_tokens": 99, "stream": True}
        self.assertEqual(normalize_engine_prompt(body, COMPLETION), body)
        self.assertEqual(build_tokenize_request(body), {
            "model": "FW-Kimi-K3", "prompt": "hello", "add_special_tokens": True, "return_token_strs": False,
        })
        self.assertFalse(build_tokenize_request({**body, "add_special_tokens": False})["add_special_tokens"])
        for path in ("/v1/responses", "/tokenize", "/v1/chat/completions/render"):
            with self.subTest(path=path), self.assertRaises(TokenCountInputError):
                normalize_engine_prompt(body, path)
        with self.assertRaises(TokenCountInputError):
            normalize_engine_prompt(body, CHAT)
        with self.assertRaises(TokenCountInputError):
            normalize_engine_prompt(chat(), COMPLETION)

    def test_native_root_controls_must_already_be_bound_by_the_control_normalizer(self):
        body = chat(thinking_effort="low", chat_template_kwargs={"thinking_effort": "high"})
        # Explicit template settings take precedence over the retained root
        # alias, just as in the existing media/control normalization stage.
        self.assertEqual(normalize_engine_prompt(body, CHAT), body)
        projection = build_tokenize_request(body)
        self.assertEqual(projection["chat_template_kwargs"]["thinking_effort"], "high")
        self.assertNotIn("thinking_effort", projection)
        with self.assertRaises(TokenCountInputError):
            normalize_engine_prompt(chat(thinking_effort="low"), CHAT)


class EngineTokenHTTPTest(unittest.TestCase):
    def test_projection_preserves_full_supported_prompt_fields(self):
        schema = response_format()
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,U1lOVEhFVElD", "detail": "auto"}}
        payload = {
            "model": "FW-Kimi-K3", "messages": [
                {"role": "assistant", "content": "previous", "reasoning_content": "reasoning history", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"query":"x"}'}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "tool result"},
                {"role": "user", "content": [{"type": "text", "text": "latest"}, image]},
            ],
            "tools": [tool()], "tool_choice": "none", "response_format": schema,
            "documents": [{"title": "doc", "text": "context"}], "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": True, "thinking_effort": "low", "response_schema": {"type": "object"}},
            "media_io_kwargs": {"image": {"image_mode": None}}, "mm_processor_kwargs": {"custom": {"z": 2, "a": 1}},
            "max_completion_tokens": 12345, "stream": True, "stream_options": {"include_usage": True},
            "priority": -1, "cache_salt": "ignored-for-count", "user": "unused", "temperature": 0,
        }
        before = copy.deepcopy(payload)
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            result = estimator.count(payload)
            sent = stub.wait(1)[0]
            self.assertEqual(sent.path, "/tokenize")
            self.assertEqual(result, TokenCountResult(count=3, max_model_len=262144))
            self.assertEqual(sent.payload["messages"][0]["reasoning"], "reasoning history")
            self.assertNotIn("reasoning_content", sent.payload["messages"][0])
            self.assertEqual(sent.payload["messages"][1:], payload["messages"][1:])
            self.assertEqual(sent.payload["tools"], payload["tools"])
            self.assertEqual(sent.payload["chat_template_kwargs"], {
                "thinking": True, "thinking_effort": "low", "response_schema": {"type": "object"},
                "add_generation_prompt": True, "continue_final_message": False,
                "documents": payload["documents"], "reasoning_effort": "high", "tool_choice": "none",
                "response_format": schema, "enable_thinking": True,
            })
            self.assertEqual(sent.payload["media_io_kwargs"], payload["media_io_kwargs"])
            self.assertEqual(sent.payload["mm_processor_kwargs"], payload["mm_processor_kwargs"])
            self.assertEqual(set(sent.payload), {
                "model", "messages", "tools", "add_generation_prompt", "continue_final_message",
                "add_special_tokens", "chat_template_kwargs", "media_io_kwargs", "mm_processor_kwargs", "return_token_strs",
            })
            self.assertFalse(sent.payload["return_token_strs"])
            self.assertFalse(sent.payload["add_special_tokens"])
        self.assertEqual(payload, before)

    def test_chat_default_and_unset_merge_matches_installed_protocol(self):
        with Stub() as stub, EngineTokenEstimator(stub.url, cache_entries=0) as estimator:
            estimator.count(chat())
            kwargs = stub.wait(1)[0].payload["chat_template_kwargs"]
            self.assertEqual(kwargs, {"add_generation_prompt": True, "continue_final_message": False})
            payload = chat(
                tools=[tool()], tool_choice="auto", response_format=None, reasoning_effort=None,
                chat_template_kwargs={"tool_choice": "required", "response_format": "json_object", "reasoning_effort": "low", "add_generation_prompt": False},
            )
            estimator.count(payload)
            kwargs = stub.wait(2)[1].payload["chat_template_kwargs"]
            self.assertEqual(kwargs["tool_choice"], "required")
            self.assertEqual(kwargs["response_format"], "json_object")
            self.assertEqual(kwargs["reasoning_effort"], "low")
            self.assertTrue(kwargs["add_generation_prompt"])
            estimator.count(chat(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False}))
            kwargs = stub.wait(3)[2].payload["chat_template_kwargs"]
            self.assertEqual(kwargs["reasoning_effort"], "none")
            self.assertFalse(kwargs["enable_thinking"])
            estimator.count(chat(reasoning_effort="high", chat_template_kwargs={"enable_thinking": False}))
            self.assertFalse(stub.wait(4)[3].payload["chat_template_kwargs"]["enable_thinking"])

    def test_root_fields_override_same_template_keys_except_unset_values(self):
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            estimator.count(chat(
                tools=[tool()], tool_choice="none", response_format=response_format(), reasoning_effort="high",
                documents=[{"text": "root"}], add_generation_prompt=False,
                chat_template_kwargs={
                    "tool_choice": "required", "response_format": "json_object", "reasoning_effort": "low",
                    "documents": [{"text": "template"}], "add_generation_prompt": True,
                },
            ))
            kwargs = stub.wait(1)[0].payload["chat_template_kwargs"]
            self.assertEqual(kwargs["tool_choice"], "none")
            self.assertEqual(kwargs["response_format"], response_format())
            self.assertEqual(kwargs["reasoning_effort"], "high")
            self.assertEqual(kwargs["documents"], [{"text": "root"}])
            self.assertFalse(kwargs["add_generation_prompt"])

    def test_count_result_cache_and_numeric_detached_metrics(self):
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            result = estimator.count(chat("PRIVATE_SYNTHETIC_MARKER"))
            self.assertEqual(result.source, "engine_rendered")
            self.assertFalse(result.cache_hit)
            with self.assertRaises(FrozenInstanceError):
                result.count = 7
            cached = estimator.count(chat("PRIVATE_SYNTHETIC_MARKER"))
            self.assertTrue(cached.cache_hit)
            self.assertEqual(estimator.estimate(chat("PRIVATE_SYNTHETIC_MARKER")), 3)
            self.assertEqual(len(stub.requests), 1)
            metrics = estimator.snapshot()
            self.assertEqual(metrics["requests_total"], 3)
            self.assertEqual(metrics["successes_total"], 3)
            self.assertEqual(metrics["cache_hits_total"], 2)
            self.assertEqual(metrics["leaders_total"], 1)
            self.assertEqual(metrics["jobs_succeeded_total"], 1)
            for gauge in ("pending", "running", "queued", "waiters", "calls_in_progress"):
                self.assertEqual(metrics[gauge], 0)
            self.assertTrue(all(type(value) in (int, float) for value in metrics.values()))
            self.assertNotIn("PRIVATE_SYNTHETIC_MARKER", json.dumps(metrics))
            metrics["leaders_total"] = 99
            self.assertEqual(estimator.snapshot()["leaders_total"], 1)
            self.assertTrue(all(type(key) is bytes and len(key) == 32 for key in estimator._cache))
            for entry in estimator._cache.values():
                self.assertEqual(set(vars(entry)), {"result", "expires"})
                self.assertEqual(set(vars(entry.result)), {"count", "max_model_len", "cache_hit"})
            self.assertFalse(estimator._jobs)
            self.assertFalse(estimator._pending)

    def test_output_only_fields_do_not_bust_count_cache(self):
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            estimator.count(chat())
            result = estimator.count(chat(
                max_tokens=5000, max_completion_tokens=5001, stream=True, stream_options={"include_usage": True},
                temperature=0.4, top_p=0.8, priority=2, request_id="new", cache_salt="new", user="other",
            ))
            self.assertTrue(result.cache_hit)
            self.assertEqual(len(stub.requests), 1)

    def test_prompt_affecting_fields_and_nested_insertion_order_partition_cache(self):
        base = chat(tools=[tool("a"), tool("b")], response_format=response_format())
        variants = []
        for mutate in (
            lambda p: p.update(model="other-alias"),
            lambda p: p["tools"].reverse(),
            lambda p: p.update(tool_choice="none"),
            lambda p: p.update(reasoning_effort="low"),
            lambda p: p.update(chat_template_kwargs={"thinking": False}),
            lambda p: p.update(add_generation_prompt=False),
            lambda p: p.update(add_special_tokens=True),
            lambda p: p.update(media_io_kwargs={"image": {"image_mode": None}}),
            lambda p: p.update(mm_processor_kwargs={"options": {"z": 1, "a": 2}}),
            lambda p: p.update(mm_processor_kwargs={"options": {"a": 2, "z": 1}}),
            lambda p: p["response_format"]["json_schema"]["schema"]["properties"].update(other={"type": "integer"}),
            lambda p: p["messages"].insert(0, {"role": "assistant", "content": "old", "reasoning": "history"}),
            lambda p: p.update(documents=[{"text": "retrieval"}]),
        ):
            variant = copy.deepcopy(base)
            mutate(variant)
            variants.append(variant)
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            for payload in [base, *variants]:
                self.assertFalse(estimator.count(payload).cache_hit)
            self.assertEqual(len(stub.requests), len(variants) + 1)
            self.assertEqual(list(stub.requests[9].payload["mm_processor_kwargs"]["options"]), ["z", "a"])
            self.assertEqual(list(stub.requests[10].payload["mm_processor_kwargs"]["options"]), ["a", "z"])

    def test_completion_strings_use_the_same_bounded_client(self):
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            self.assertEqual(estimator.estimate({"prompt": "plain", "max_tokens": 999}), 3)
            self.assertEqual(stub.wait(1)[0].payload, {
                "prompt": "plain", "add_special_tokens": True, "return_token_strs": False,
            })
            self.assertTrue(estimator.count({"prompt": "plain"}).cache_hit)
            with self.assertRaises(TokenCountInputError):
                estimator.count({"prompt": [1, 2, 3]})
            self.assertEqual(len(stub.requests), 1)

    def test_lru_eviction_ttl_and_revision(self):
        with Stub() as stub, EngineTokenEstimator(stub.url, cache_entries=2, cache_ttl=0.15, revision="v1") as estimator:
            estimator.count(chat("a"))
            first_key = next(iter(estimator._cache))
            estimator.count(chat("b"))
            self.assertTrue(estimator.count(chat("a")).cache_hit)
            estimator.count(chat("c"))
            self.assertFalse(estimator.count(chat("b")).cache_hit)
            self.assertEqual(estimator.snapshot()["cache_evictions_total"], 2)
            time.sleep(0.17)
            self.assertEqual(estimator.snapshot()["cache_entries"], 0)
            self.assertEqual(estimator.snapshot()["cache_expirations_total"], 2)
            with EngineTokenEstimator(stub.url, revision="v2") as other:
                other.count(chat("a"))
                self.assertNotEqual(first_key, next(iter(other._cache)))

    def test_request_limit_is_encoded_bytes_including_exact_boundary(self):
        payload = chat("漢字" * 30)
        with Stub() as stub, EngineTokenEstimator(stub.url, cache_entries=0) as estimator:
            estimator.count(payload)
            size = len(stub.wait(1)[0].raw)
            with patch.object(engine_tokens, "MAX_REQUEST_BYTES", size):
                estimator.count(payload)
            with patch.object(engine_tokens, "MAX_REQUEST_BYTES", size - 1):
                with self.assertRaises(TokenCountInputError) as error:
                    estimator.count(payload)
                self.assertEqual(error.exception.reason, "request_too_large")
            self.assertEqual(len(stub.requests), 2)
            self.assertEqual(estimator.snapshot()["input_errors_total"], 1)
        self.assertEqual(engine_tokens.MAX_REQUEST_BYTES, 64 * 1024 * 1024)

    def test_oversized_response_is_bounded_even_with_chunked_transport(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked), Stub(Reply(body=b" " * 512, chunked=chunked)) as stub:
                with EngineTokenEstimator(stub.url) as estimator, patch.object(engine_tokens, "MAX_RESPONSE_BYTES", 128):
                    with self.assertRaises(TokenCountUnavailable) as error:
                        estimator.count(chat())
                    self.assertEqual(error.exception.reason, "response_too_large")
                    self.assertLessEqual(estimator.snapshot()["response_bytes_total"], 128 + 8192)
                    self.assertEqual(estimator.snapshot()["cache_entries"], 0)
        self.assertEqual(engine_tokens.MAX_RESPONSE_BYTES, 8 * 1024 * 1024)

    def test_prompt_http_statuses_are_typed_input_errors_without_upstream_body(self):
        for status in (400, 404, 413, 422):
            with self.subTest(status=status), Stub(Reply(status=status, body={"error": "DO_NOT_ECHO_BODY"})) as stub:
                with EngineTokenEstimator(stub.url) as estimator:
                    with self.assertRaises(TokenCountInputError) as error:
                        estimator.count(chat())
                    self.assertEqual(error.exception.upstream_status, status)
                    self.assertEqual(error.exception.status_code, 400)
                    self.assertEqual(error.exception.reason, "http_4xx")
                    self.assertNotIn("DO_NOT_ECHO_BODY", str(error.exception))
                    self.assertNotIn("DO_NOT_ECHO_BODY", repr(vars(error.exception)))
                    self.assertEqual(estimator.snapshot()["jobs_input_errors_total"], 1)
                    self.assertEqual(estimator.snapshot()["response_bytes_total"], 0)
                    self.assertEqual(estimator.snapshot()["cache_entries"], 0)

    def test_auth_capacity_and_timeout_statuses_are_unavailable_without_upstream_body(self):
        for status in (401, 403, 408, 429, 500, 502, 503, 504):
            with self.subTest(status=status), Stub(Reply(status=status, body=b"DO_NOT_ECHO_BODY")) as stub:
                with EngineTokenEstimator(stub.url) as estimator:
                    with self.assertRaises(TokenCountUnavailable) as error:
                        estimator.count(chat())
                    self.assertEqual(error.exception.upstream_status, status)
                    self.assertEqual(error.exception.status_code, 503)
                    self.assertEqual(error.exception.reason, "http_status")
                    self.assertNotIn("DO_NOT_ECHO_BODY", str(error.exception))
                    self.assertNotIn("DO_NOT_ECHO_BODY", repr(vars(error.exception)))
                    self.assertEqual(estimator.snapshot()["jobs_unavailable_total"], 1)
                    self.assertEqual(estimator.snapshot()["input_errors_total"], 0)
                    self.assertEqual(estimator.snapshot()["response_bytes_total"], 0)
                    self.assertEqual(estimator.snapshot()["cache_entries"], 0)

    def test_transient_http_and_malformed_response_fail_without_fallback(self):
        replies = [Reply(status=500), Reply(status=503), Reply(status=301), Reply(truncate=True)]
        replies += [Reply(body=body) for body in (
            b"invalid JSON", [], {},
            {"count": 1, "tokens": [], "max_model_len": 10},
            {"count": -1, "tokens": [], "max_model_len": 10},
            {"count": True, "tokens": [1], "max_model_len": 10},
            {"count": 1, "tokens": [False], "max_model_len": 10},
            {"count": 1, "tokens": [-1], "max_model_len": 10},
            {"count": 1, "tokens": [1], "max_model_len": 0},
            {"count": 1, "tokens": [1], "max_model_len": True},
            {"count": 1, "tokens": [1], "max_model_len": "10"},
        )]
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            for reply in replies:
                with self.subTest(reply=reply):
                    stub.reply = reply
                    with self.assertRaises(TokenCountUnavailable):
                        estimator.count(chat())
            metrics = estimator.snapshot()
            self.assertEqual(metrics["unavailable_total"], len(replies))
            self.assertEqual(metrics["jobs_unavailable_total"], len(replies))
            self.assertEqual(metrics["cache_entries"], 0)
            self.assertEqual(len(stub.requests), len(replies))

    def test_zero_count_and_over_context_count_are_valid_results(self):
        with Stub() as stub, EngineTokenEstimator(stub.url, cache_entries=0) as estimator:
            stub.reply = Reply(body={"count": 0, "tokens": [], "max_model_len": 2})
            self.assertEqual(estimator.count(chat()).count, 0)
            stub.reply = Reply(body={"count": 3, "tokens": [1, 2, 3], "max_model_len": 2})
            self.assertEqual(estimator.count(chat()).count, 3)

    def test_invalid_json_input_never_reaches_http(self):
        cyclic = {}
        cyclic["loop"] = cyclic
        for kwargs in ({"arbitrary": float("nan")}, {"arbitrary": cyclic}, {"arbitrary": object()}, {"arbitrary": "\ud800"}):
            with self.subTest(kwargs_type=type(kwargs["arbitrary"])), Stub() as stub:
                with EngineTokenEstimator(stub.url) as estimator:
                    with self.assertRaises(TokenCountInputError):
                        estimator.count(chat(chat_template_kwargs=kwargs))
                    self.assertFalse(stub.requests)

    def test_constructor_rejects_invalid_limits_and_unimplemented_origins(self):
        for kwargs in (
            {"timeout": 0}, {"timeout": float("nan")}, {"timeout": True},
            {"max_workers": 0}, {"max_workers": True}, {"max_pending": 0},
            {"cache_entries": -1}, {"cache_ttl": -1}, {"revision": {}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EngineTokenEstimator("http://127.0.0.1:1", **kwargs)
        for url in ("https://127.0.0.1", "http://user:pass@127.0.0.1", "http://127.0.0.1/v1", "http://127.0.0.1?q=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                EngineTokenEstimator(url)


class EngineTokenConcurrencyTest(unittest.TestCase):
    def test_singleflight_and_saturation_bound_distinct_jobs_not_waiters(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url, max_workers=2, max_pending=3) as estimator:
            with ThreadPoolExecutor(max_workers=4) as callers:
                first = callers.submit(estimator.count, chat("first"))
                first_request = stub.wait(1)[0]
                second = callers.submit(estimator.count, chat("second"))
                stub.wait(2)
                third = callers.submit(estimator.count, chat("third"))
                wait_stats(estimator, lambda s: s["pending"] == 3 and s["queued"] == 1)
                with self.assertRaises(TokenCountUnavailable) as error:
                    estimator.count(chat("fourth"))
                self.assertEqual(error.exception.reason, "overloaded")
                duplicate = callers.submit(estimator.count, chat("first"))
                metrics = wait_stats(estimator, lambda s: s["singleflight_joins_total"] == 1)
                self.assertEqual(metrics["pending"], 3)
                self.assertEqual(metrics["running"], 2)
                self.assertEqual(metrics["waiters"], 4)
                self.assertEqual(len(stub.requests), 2)
                first_request.release.set()
                stub.wait(3)
                stub.release_all()
                for future in (first, second, third, duplicate):
                    self.assertEqual(future.result(2).count, 3)
                    self.assertFalse(future.result().cache_hit)
            self.assertEqual(estimator.snapshot()["leaders_total"], 3)
            self.assertEqual(estimator.snapshot()["overloaded_total"], 1)
            self.assertEqual(estimator.snapshot()["pending"], 0)

    def test_cancel_one_waiter_keeps_shared_job_for_the_other(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url) as estimator:
            first_parent, second_parent = RequestCancellation(600), RequestCancellation(600)
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(count_with_parent, estimator, chat(), first_parent)
                request = stub.wait(1)[0]
                second = callers.submit(count_with_parent, estimator, chat(), second_parent)
                wait_stats(estimator, lambda s: s["waiters"] == 2)
                with estimator._condition:
                    job = next(iter(estimator._pending))
                self.assertIsNot(job.cancellation, first_parent)
                self.assertIsNot(job.cancellation, second_parent)
                first_parent.cancel(CancelReason.CLIENT_DISCONNECT)
                with self.assertRaises(RequestCancelled) as error:
                    first.result(1)
                self.assertEqual(error.exception.reason, CancelReason.CLIENT_DISCONNECT)
                self.assertIsNone(job.cancellation.reason)
                self.assertFalse(request.peer_closed.wait(0.06))
                request.release.set()
                self.assertEqual(second.result(1).count, 3)
                self.assertTrue(estimator.count(chat()).cache_hit)
                self.assertEqual(len(stub.requests), 1)
                self.assertIsNone(second_parent.reason)
                self.assertGreater(second_parent.remaining, 590)
            self.assertIsNone(job.body)
            self.assertFalse(job.cancellation._thread.is_alive())
            self.assertEqual(estimator.snapshot()["waiter_cancellations_total"], 1)
            self.assertEqual(estimator.snapshot()["jobs_aborted_total"], 0)

    def test_cancel_all_waiters_aborts_socket_and_allows_a_fresh_job(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url) as estimator:
            parents = [RequestCancellation(600), RequestCancellation(600)]
            with ThreadPoolExecutor(max_workers=2) as callers:
                futures = [callers.submit(count_with_parent, estimator, chat(), parent) for parent in parents]
                request = stub.wait(1)[0]
                wait_stats(estimator, lambda s: s["waiters"] == 2)
                with estimator._condition:
                    job = next(iter(estimator._pending))
                for parent in parents:
                    parent.cancel(CancelReason.CLIENT_DISCONNECT)
                for future in futures:
                    with self.assertRaises(RequestCancelled):
                        future.result(1)
                self.assertTrue(request.peer_closed.wait(1))
                metrics = wait_stats(estimator, lambda s: s["pending"] == 0)
                self.assertEqual(metrics["cache_entries"], 0)
                self.assertEqual(metrics["jobs_aborted_total"], 1)
                self.assertIsNone(job.body)
                self.assertFalse(job.cancellation._thread.is_alive())
                stub.release_all()
                self.assertFalse(estimator.count(chat()).cache_hit)
                self.assertEqual(len(stub.requests), 2)

    def test_repeated_queued_cancellations_remove_work_not_just_cancel_futures(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url, timeout=5, max_workers=1, max_pending=2) as estimator:
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(estimator.count, chat("running"))
                request = stub.wait(1)[0]
                for index in range(12):
                    parent = RequestCancellation(600)
                    queued = callers.submit(count_with_parent, estimator, chat(f"queued {index}"), parent)
                    wait_stats(estimator, lambda s: s["queued"] == 1)
                    with estimator._condition:
                        job = estimator._queue[0]
                    parent.cancel(CancelReason.CLIENT_DISCONNECT)
                    with self.assertRaises(RequestCancelled):
                        queued.result(1)
                    self.assertTrue(job.future.cancelled())
                    self.assertIsNone(job.body)
                    self.assertIsNone(job.cancellation._thread)
                    self.assertEqual(estimator.snapshot()["queued"], 0)
                    self.assertEqual(estimator.snapshot()["pending"], 1)
                    self.assertEqual(len(stub.requests), 1)
                request.release.set()
                self.assertEqual(first.result(1).count, 3)
            self.assertEqual(estimator.snapshot()["jobs_cancelled_queued_total"], 12)

    def test_job_deadline_does_not_shorten_parent_and_drips_cannot_renew_it(self):
        for reply in (Reply(hold=True), Reply(drip=0.015), Reply(drip=0.015, chunked=True)):
            with self.subTest(reply=reply), Stub(reply) as stub, EngineTokenEstimator(stub.url, timeout=0.18) as estimator:
                with RequestCancellation(600) as parent:
                    started = time.monotonic()
                    with self.assertRaises(TokenCountUnavailable) as error:
                        estimator.count(chat())
                    self.assertEqual(error.exception.reason, "timeout")
                    self.assertLess(time.monotonic() - started, 0.8)
                    self.assertIsNone(parent.reason)
                    self.assertIs(current_cancellation(), parent)
                    self.assertGreater(parent.remaining, 599)
                    parent.check()
                self.assertTrue(stub.wait(1)[0].peer_closed.wait(1))
                metrics = wait_stats(estimator, lambda s: s["pending"] == 0)
                self.assertEqual(metrics["jobs_timed_out_total"], 1)
                self.assertEqual(metrics["count_timeouts_total"], 1)

    def test_queue_time_is_charged_to_the_shared_job_deadline(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url, timeout=0.4, max_workers=1) as estimator:
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(estimator.count, chat("first"))
                request = stub.wait(1)[0]
                second = callers.submit(estimator.count, chat("second"))
                wait_stats(estimator, lambda s: s["queued"] == 1)
                with estimator._condition:
                    queued_job = estimator._queue[0]
                time.sleep(0.16)
                request.release.set()
                self.assertEqual(first.result(1).count, 3)
                second_request = stub.wait(2)[1]
                self.assertLess(queued_job.cancellation.remaining, 0.3)
                with self.assertRaises(TokenCountUnavailable) as error:
                    second.result(1)
                self.assertEqual(error.exception.reason, "timeout")
                self.assertLess(time.monotonic() - queued_job.cancellation.started, 0.65)
                self.assertTrue(second_request.peer_closed.wait(1))

    def test_parent_deadline_propagates_request_cancelled_not_count_timeout(self):
        with Stub(Reply(hold=True)) as stub, EngineTokenEstimator(stub.url) as estimator:
            with RequestCancellation(0.12) as parent:
                with self.assertRaises(RequestCancelled) as error:
                    estimator.count(chat())
                self.assertEqual(error.exception.reason, CancelReason.DEADLINE)
                self.assertEqual(parent.reason, CancelReason.DEADLINE)
            self.assertTrue(stub.wait(1)[0].peer_closed.wait(1))
            metrics = wait_stats(estimator, lambda s: s["pending"] == 0)
            self.assertEqual(metrics["waiter_cancellations_total"], 1)
            self.assertEqual(metrics["count_timeouts_total"], 0)

    def test_already_cancelled_waiter_does_not_start_work_even_on_cache_hit(self):
        with Stub() as stub, EngineTokenEstimator(stub.url) as estimator:
            estimator.count(chat())
            with RequestCancellation(600) as parent:
                parent.cancel(CancelReason.CLIENT_DISCONNECT)
                with self.assertRaises(RequestCancelled):
                    estimator.count(chat())
            self.assertEqual(len(stub.requests), 1)
            self.assertEqual(estimator.snapshot()["cache_hits_total"], 0)

    def test_shared_failure_is_one_job_and_is_not_cached(self):
        with Stub(Reply(status=503, hold=True)) as stub, EngineTokenEstimator(stub.url) as estimator:
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(estimator.count, chat())
                request = stub.wait(1)[0]
                second = callers.submit(estimator.count, chat())
                wait_stats(estimator, lambda s: s["waiters"] == 2)
                request.release.set()
                for future in (first, second):
                    with self.assertRaises(TokenCountUnavailable):
                        future.result(1)
            self.assertEqual(estimator.snapshot()["jobs_unavailable_total"], 1)
            self.assertEqual(estimator.snapshot()["unavailable_total"], 2)
            self.assertEqual(estimator.snapshot()["cache_entries"], 0)
            stub.reply = Reply()
            estimator.count(chat())
            self.assertEqual(len(stub.requests), 2)

    def test_close_aborts_running_and_queued_jobs_and_joins_owned_threads(self):
        with Stub(Reply(hold=True)) as stub:
            estimator = EngineTokenEstimator(stub.url, timeout=5, max_workers=2, max_pending=3)
            try:
                with ThreadPoolExecutor(max_workers=3) as callers:
                    futures = [callers.submit(estimator.count, chat(str(i))) for i in range(3)]
                    requests = stub.wait(2)
                    wait_stats(estimator, lambda s: s["queued"] == 1)
                    with estimator._condition:
                        jobs = list(estimator._pending)
                    estimator.close()
                    for future in futures:
                        with self.assertRaises(TokenCountUnavailable) as error:
                            future.result(1)
                        self.assertEqual(error.exception.reason, "closed")
                    for request in requests:
                        self.assertTrue(request.peer_closed.wait(1))
                    self.assertEqual(len(stub.requests), 2)
                self.assertTrue(all(not thread.is_alive() for thread in estimator._executor._threads))
                self.assertTrue(all(job.body is None for job in jobs))
                self.assertTrue(all(job.cancellation._thread is None or not job.cancellation._thread.is_alive() for job in jobs))
                metrics = estimator.snapshot()
                for key in ("pending", "queued", "running", "waiters", "cache_entries", "calls_in_progress"):
                    self.assertEqual(metrics[key], 0)
                self.assertEqual(metrics["closed"], 1)
                with self.assertRaises(TokenCountUnavailable):
                    estimator.count(chat())
            finally:
                estimator.close()


if __name__ == "__main__":
    unittest.main()
