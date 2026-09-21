"""Offline checks for content-free, bounded completion observation and traces."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tracemalloc
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from redesign.gateway.capture import (
    CompletionRecord,
    JsonlSink,
    MemorySink,
    TraceRecord,
    TraceRecorder,
)
from redesign.gateway.completions import (
    FINISH_REASONS,
    CompletionObserver,
    CompletionSummary,
)
from redesign.gateway.models import ClampResult, Decision, Outcome, Priority, RequestEnvelope


def _sse(document, newline=b"\n") -> bytes:
    data = json.dumps(document, ensure_ascii=False).encode("utf-8")
    return b"data: " + data + newline * 2


def _choice(delta=None, finish=None) -> dict:
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def _json_completion(finish="stop", **overrides) -> bytes:
    document = {
        "choices": [{"message": {"content": "private completion"}, "finish_reason": finish}],
        **overrides,
    }
    return json.dumps(document, ensure_ascii=False).encode("utf-8")


def _decision(**overrides) -> Decision:
    envelope = RequestEnvelope(
        customer="private-customer", prompt_tokens=101,
        requested_max_tokens=32_768, streaming=True, has_tools=True, has_images=True,
    )
    values = dict(
        envelope=envelope, traffic_class="p2-agentic", priority=Priority.LONG_CONTEXT,
        clamp=ClampResult(
            requested=32_768, granted=16_384, reason="class_ceiling",
            default_output_tokens=4_096,
        ),
        outcome=Outcome.ADMIT, reason="admitted", admission_id="private-admission-id",
        metadata={"borrowed": True, "untrusted": "private-metadata"},
    )
    return Decision(**{**values, **overrides})


class CompletionObserverTest(unittest.TestCase):
    def test_unicode_multiline_sse_at_every_chunk_boundary_and_line_ending(self):
        document = _choice({"content": "答案🌋"})
        document["id"] = "private-upstream-id"
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 123, "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
        expected = CompletionSummary(
            first_token_seconds=2.5, prompt_tokens=123, completion_tokens=8,
            cached_tokens=64, reasoning_tokens=3, finish_reasons=("length",),
            protocol_complete=True,
        )
        with patch("redesign.gateway.completions.time.monotonic", return_value=102.5):
            for newline in (b"\n", b"\r", b"\r\n"):
                multiline = newline.join(
                    b"data: " + line.encode("utf-8")
                    for line in json.dumps(document, indent=2, ensure_ascii=False).split("\n")
                )
                wire = (
                    b"\xef\xbb\xbf: heartbeat" + newline * 2
                    + b"id: private-sse-id" + newline + b"event: message" + newline
                    + multiline + newline * 2
                    + _sse(_choice(finish="length"), newline)
                    + _sse(usage, newline) + b"data: [DONE]" + newline * 2
                )
                for boundary in range(len(wire) + 1):
                    with self.subTest(newline=newline, boundary=boundary):
                        observer = CompletionObserver(True, 100.0)
                        observer.feed(wire[:boundary])
                        observer.feed(wire[boundary:])
                        self.assertEqual(observer.finish(), expected)
                observer = CompletionObserver(True, 100.0)
                for byte in wire:
                    observer.feed(bytes([byte]))
                self.assertEqual(observer.finish(), expected)

    def test_ttft_waits_for_meaningful_delta_and_is_recorded_once(self):
        meaningful = (
            {"content": " "}, {"reasoning": "thinking"}, {"reasoning_content": "thinking"},
            {"tool_calls": [{"index": 0, "function": {"name": "lookup", "arguments": ""}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]},
        )
        for delta in meaningful:
            with self.subTest(delta=delta), patch(
                "redesign.gateway.completions.time.monotonic", return_value=104.0,
            ) as clock:
                observer = CompletionObserver(True, 100.0)
                observer.feed(b": keepalive\n\n")
                observer.feed(_sse(_choice({"role": "assistant", "content": ""})))
                observer.feed(_sse({"choices": [], "usage": {"prompt_tokens": 20}}))
                observer.feed(_sse(_choice({"content": None, "reasoning": "", "tool_calls": []})))
                observer.feed(_sse(_choice({"tool_calls": [{"id": "private-id", "index": 0}]})))
                observer.feed(_sse(_choice({"tool_calls": [{"function": {"arguments": ""}}]})))
                clock.assert_not_called()
                event = _sse(_choice(delta))
                observer.feed(event[:-1])
                clock.assert_not_called()
                observer.feed(event[-1:])
                observer.feed(_sse(_choice({"content": "later"}, finish="stop")))
                observer.feed(b"data: [DONE]\n\n")
                summary = observer.finish()
                self.assertEqual(summary.first_token_seconds, 4.0)
                self.assertTrue(summary.protocol_complete)
                clock.assert_called_once_with()

    def test_role_usage_and_finish_only_stream_has_no_ttft(self):
        observer = CompletionObserver(True, 100.0)
        with patch("redesign.gateway.completions.time.monotonic") as clock:
            observer.feed(_sse(_choice({"role": "assistant"})))
            observer.feed(_sse({"usage": {"completion_tokens": 0}}))
            observer.feed(_sse(_choice(finish="content_filter")))
            observer.feed(b"data: [DONE]\n\n")
            summary = observer.finish()
            clock.assert_not_called()
        self.assertIsNone(summary.first_token_seconds)
        self.assertTrue(summary.protocol_complete)
        self.assertFalse(summary.protocol_error)

    def test_length_is_terminal_even_when_all_output_is_reasoning(self):
        observer = CompletionObserver(True, 0.0)
        with patch("redesign.gateway.completions.time.monotonic", return_value=1.0):
            observer.feed(_sse(_choice({"reasoning_content": "private thinking"})))
        observer.feed(_sse(_choice(finish="length")))
        observer.feed(_sse({"choices": [], "usage": {
            "completion_tokens": 16, "completion_tokens_details": {"reasoning_tokens": 16},
        }}))
        observer.feed(b"data: [DONE]\n\n")
        summary = observer.finish()
        self.assertTrue(summary.protocol_complete)
        self.assertFalse(summary.protocol_error)
        self.assertEqual(summary.finish_reasons, ("length",))
        self.assertEqual(summary.reasoning_tokens, 16)

    def test_finish_reason_without_done_and_truncated_done_are_incomplete(self):
        for suffix in (b"", b"data: [DONE]", b"data: [DONE]\n", b'data: {"choices":['):
            with self.subTest(suffix=suffix):
                observer = CompletionObserver(True, 0.0)
                observer.feed(_sse(_choice(finish="length")) + suffix)
                summary = observer.finish()
                self.assertIs(summary.protocol_complete, False)
                self.assertFalse(summary.protocol_error)
                self.assertEqual(summary.finish_reasons, ("length",))
                self.assertEqual(observer.buffered_bytes, 0)

    def test_done_does_not_hide_a_truncated_or_malformed_later_event(self):
        for suffix in (b"data: {", b"data: {}\n\n", b"event: error\n\n"):
            with self.subTest(suffix=suffix):
                observer = CompletionObserver(True, 0.0)
                observer.feed(_sse(_choice(finish="stop")) + b"data: [DONE]\n\n" + suffix)
                self.assertIs(observer.finish().protocol_complete, False)

    def test_error_events_and_documents_remain_errors_after_done(self):
        errors = (
            b"event: error\n\n",
            b"event: error\ndata: private error message\n\n",
            _sse({"error": {"message": "private error message", "request_id": "private-id"}}),
            _sse({"type": "error", "message": "private error message"}),
            _sse({**_choice(finish="stop"), "error": {"message": "private error message"}}),
        )
        for event in errors:
            with self.subTest(event=event):
                observer = CompletionObserver(True, 0.0)
                observer.feed(event + b"data: [DONE]\n\n")
                summary = observer.finish()
                self.assertTrue(summary.protocol_error)
                self.assertIs(summary.protocol_complete, False)
                self.assertIsNone(summary.first_token_seconds)
                self.assertNotIn("private", repr(vars(observer)))

    def test_malformed_sse_is_observed_without_interrupting_byte_relay(self):
        bad_payloads = (
            b"{", b"\xff", b"null", b"[]", b"{}",
            b'{"choices":[null]}', b'{"choices":[{"delta":42}]}',
            b'{"choices":[{"finish_reason":[]}]}', b"[" * 2_000 + b"]" * 2_000,
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload[:50]):
                observer = CompletionObserver(True, 0.0)
                chunks = (b"data: " + payload + b"\n\n", b"data: [DONE]\n\n")
                relayed = []
                for chunk in chunks:
                    observer.feed(chunk)
                    relayed.append(chunk)
                self.assertEqual(b"".join(relayed), b"".join(chunks))
                summary = observer.finish()
                self.assertTrue(summary.protocol_error)
                self.assertIs(summary.protocol_complete, False)

    def test_nonstream_json_is_complete_but_never_ttft(self):
        for reason in FINISH_REASONS:
            with self.subTest(reason=reason), patch(
                "redesign.gateway.completions.time.monotonic",
            ) as clock:
                body = _json_completion(reason, usage={
                    "prompt_tokens": 12, "completion_tokens": 6,
                    "prompt_tokens_details": {"cached_tokens": 8.0},
                    "completion_tokens_details": {"reasoning_tokens": 4},
                })
                observer = CompletionObserver(False, 0.0)
                for byte in body:
                    observer.feed(bytes([byte]))
                summary = observer.finish()
                self.assertTrue(summary.protocol_complete)
                self.assertFalse(summary.protocol_error)
                self.assertIsNone(summary.first_token_seconds)
                self.assertEqual(summary.finish_reasons, (reason,))
                self.assertEqual((summary.prompt_tokens, summary.completion_tokens), (12, 6))
                self.assertEqual((summary.cached_tokens, summary.reasoning_tokens), (8.0, 4))
                clock.assert_not_called()

    def test_nonstream_choices_and_all_finish_reasons_must_be_valid(self):
        for choices in (
            None, {}, [], [None], [{}], [{"finish_reason": None}],
            [{"finish_reason": ""}], [{"finish_reason": "error"}],
            [{"finish_reason": {}}], [{"finish_reason": "stop"}, {"finish_reason": None}],
        ):
            with self.subTest(choices=choices):
                observer = CompletionObserver(False, 0.0)
                observer.feed(_json_completion(choices=choices))
                summary = observer.finish()
                self.assertIs(summary.protocol_complete, False)
                self.assertTrue(summary.protocol_error)

    def test_nonstream_invalid_json_truncation_and_error_document(self):
        bodies = (
            b"{", b"\xff", b"null", b"[]", b'"private raw body"', b"{}",
            _json_completion()[:-1],
            _json_completion(error={"message": "private error message"}),
        )
        for body in bodies:
            with self.subTest(body=body):
                observer = CompletionObserver(False, 0.0)
                observer.feed(body)
                summary = observer.finish()
                self.assertIs(summary.protocol_complete, False)
                self.assertTrue(summary.protocol_error)
                self.assertEqual(observer.buffered_bytes, 0)
                self.assertNotIn("private", repr(vars(observer)))

    def test_missing_counters_are_unknown_while_reported_zero_is_known(self):
        for usage in (None, {}, {"prompt_tokens_details": {}, "completion_tokens_details": None}):
            with self.subTest(usage=usage):
                observer = CompletionObserver(False, 0.0)
                observer.feed(_json_completion(usage=usage))
                summary = observer.finish()
                self.assertTrue(summary.protocol_complete)
                self.assertEqual(
                    (summary.prompt_tokens, summary.completion_tokens, summary.cached_tokens,
                     summary.reasoning_tokens), (None, None, None, None),
                )
        observer = CompletionObserver(False, 0.0)
        observer.feed(_json_completion(usage={
            "prompt_tokens": 0, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }))
        summary = observer.finish()
        self.assertEqual(
            (summary.prompt_tokens, summary.completion_tokens, summary.cached_tokens,
             summary.reasoning_tokens), (0, 0, 0, 0),
        )

    def test_invalid_counters_are_not_coerced_into_numeric_observations(self):
        fields = (
            ("prompt_tokens", None), ("completion_tokens", None),
            ("cached_tokens", "prompt_tokens_details"),
            ("reasoning_tokens", "completion_tokens_details"),
        )
        for field, details in fields:
            for value in (True, False, "12", -1, -0.5, float("nan"), float("inf"), [], {}):
                with self.subTest(field=field, value=value):
                    usage = {details: {field: value}} if details else {field: value}
                    observer = CompletionObserver(False, 0.0)
                    observer.feed(_json_completion(usage=usage))
                    summary = observer.finish()
                    self.assertIsNone(getattr(summary, field))
                    self.assertTrue(summary.protocol_error)
                    self.assertIs(summary.protocol_complete, False)

    def test_usage_container_shapes_are_validated(self):
        for usage in (
            [], "private usage text", True,
            {"prompt_tokens_details": []}, {"completion_tokens_details": "private details"},
        ):
            with self.subTest(usage=usage):
                observer = CompletionObserver(False, 0.0)
                observer.feed(_json_completion(usage=usage))
                summary = observer.finish()
                self.assertTrue(summary.protocol_error)
                self.assertIs(summary.protocol_complete, False)
                self.assertNotIn("private", repr(vars(observer)))

    def test_repeated_usage_snapshots_replace_counts_and_preserve_absent_details(self):
        observer = CompletionObserver(True, 0.0)
        for usage in (
            {"prompt_tokens": 100, "completion_tokens": 1,
             "prompt_tokens_details": {"cached_tokens": 64}},
            {"prompt_tokens": 100, "completion_tokens": 4},
            {"completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 0}},
        ):
            observer.feed(_sse({"choices": [], "usage": usage}))
        observer.feed(_sse(_choice(finish="stop")))
        observer.feed(b"data: [DONE]\n\n")
        summary = observer.finish()
        self.assertEqual((summary.prompt_tokens, summary.completion_tokens), (100, 4))
        self.assertEqual((summary.cached_tokens, summary.reasoning_tokens), (64, 0))
        self.assertEqual(summary.finish_reasons, ("stop",))
        self.assertTrue(summary.protocol_complete)

    def test_usage_and_done_without_a_finish_reason_are_incomplete(self):
        observer = CompletionObserver(True, 0.0)
        observer.feed(_sse({"choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }}))
        observer.feed(b"data: [DONE]\n\n")
        summary = observer.finish()
        self.assertEqual((summary.prompt_tokens, summary.completion_tokens), (100, 4))
        self.assertEqual((summary.cached_tokens, summary.reasoning_tokens), (64, 0))
        self.assertEqual(summary.finish_reasons, ())
        self.assertIs(summary.protocol_complete, False)
        self.assertFalse(summary.protocol_error)
        self.assertIsNone(summary.first_token_seconds)
        self.assertEqual(observer.buffered_bytes, 0)

    def test_reasons_are_deduplicated_bounded_and_never_arbitrary_text(self):
        observer = CompletionObserver(True, 0.0)
        for _ in range(100):
            for reason in FINISH_REASONS:
                observer.feed(_sse(_choice(finish=reason)))
        observer.feed(_sse(_choice(finish="private-unrecognized-finish-reason")))
        observer.feed(b"data: [DONE]\n\n")
        summary = observer.finish()
        self.assertEqual(summary.finish_reasons, tuple(sorted(FINISH_REASONS)))
        self.assertTrue(summary.protocol_error)
        self.assertNotIn("private", repr(vars(observer)))

    def test_summary_is_frozen_idempotent_and_clears_all_raw_state(self):
        private_values = (
            "private-content-秘密", "private-thinking", "private-tool-name",
            "private-arguments", "private-completion-id", "private-model-id",
        )
        delta = {
            "content": private_values[0], "reasoning_content": private_values[1],
            "tool_calls": [{"id": private_values[4], "function": {
                "name": private_values[2], "arguments": private_values[3],
            }}],
        }
        document = {**_choice(delta, "tool_calls"), "id": private_values[4], "model": private_values[5]}
        for stream in (False, True):
            with self.subTest(stream=stream):
                observer = CompletionObserver(stream, 0.0)
                raw = _sse(document) + b"data: [DONE]\n\n" if stream else json.dumps(document).encode()
                observer.feed(raw)
                summary = observer.finish()
                with self.assertRaises(FrozenInstanceError):
                    summary.prompt_tokens = 123
                observer.feed(raw)
                self.assertIs(observer.finish(), summary)
                self.assertEqual(observer.buffered_bytes, 0)
                for private in private_values:
                    self.assertNotIn(private, repr(vars(observer)))
                    self.assertNotIn(private, json.dumps(asdict(summary), ensure_ascii=False))
                for name, value in asdict(summary).items():
                    if name != "finish_reasons":
                        self.assertTrue(value is None or type(value) in (int, float, bool))

    def test_finishing_an_unstarted_observation_is_unknown(self):
        for stream in (False, True):
            observer = CompletionObserver(stream, 0.0)
            observer.feed(b"")
            self.assertEqual(observer.finish(), CompletionSummary())

    def test_nonstream_limit_accepts_exact_size_and_drops_oversize(self):
        body = _json_completion()
        for limit, expected in ((len(body), True), (len(body) - 1, None), (0, None)):
            with self.subTest(limit=limit):
                observer = CompletionObserver(False, 0.0, max_bytes=limit)
                for byte in body:
                    observer.feed(bytes([byte]))
                    self.assertLessEqual(observer.buffered_bytes, limit)
                summary = observer.finish()
                self.assertIs(summary.protocol_complete, expected)
                self.assertFalse(summary.protocol_error)
                self.assertEqual(observer.buffered_bytes, 0)

    def test_sse_overflow_drops_pending_content_and_cannot_be_completed_later(self):
        for oversize in (b"data: " + b"x" * 300, b"data: x\n" * 50):
            with self.subTest(oversize=oversize[:20]):
                observer = CompletionObserver(True, 100.0, max_bytes=128)
                with patch("redesign.gateway.completions.time.monotonic", return_value=101.0):
                    observer.feed(_sse(_choice({"content": "observed"})))
                for byte in oversize:
                    observer.feed(bytes([byte]))
                    self.assertLessEqual(observer.buffered_bytes, 128)
                self.assertEqual(observer.buffered_bytes, 0)
                observer.feed(_sse({"usage": {"completion_tokens": 12}}) + b"data: [DONE]\n\n")
                summary = observer.finish()
                self.assertIsNone(summary.protocol_complete)
                self.assertFalse(summary.protocol_error)
                self.assertIsNone(summary.completion_tokens)
                self.assertEqual(summary.first_token_seconds, 1.0)

    def test_large_single_chunk_is_rejected_before_copying_into_the_buffer(self):
        body = b"x" * (4 * 1024 * 1024)
        for stream in (False, True):
            with self.subTest(stream=stream):
                observer = CompletionObserver(stream, 0.0, max_bytes=128)
                tracemalloc.start()
                try:
                    observer.feed(body)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 512 * 1024)
                self.assertEqual(observer.buffered_bytes, 0)
                self.assertIsNone(observer.finish().protocol_complete)

    def test_stream_limit_is_per_pending_event_not_total_bytes_or_relay_chunk(self):
        event = _sse(_choice({"content": "token"}))
        wire = event * 200 + _sse(_choice(finish="stop")) + b"data: [DONE]\n\n"
        observer = CompletionObserver(True, 0.0, max_bytes=128)
        self.assertGreater(len(wire), 128)
        observer.feed(wire)
        self.assertEqual(observer.buffered_bytes, 0)
        summary = observer.finish()
        self.assertTrue(summary.protocol_complete)
        self.assertFalse(summary.protocol_error)


class CompletionRecorderTest(unittest.TestCase):
    def test_completion_records_use_only_the_separate_optional_sink(self):
        admission_sink, completion_sink = MemorySink(), MemorySink()
        recorder = TraceRecorder(admission_sink, completion_sink=completion_sink)
        decision = _decision()
        admission = recorder.record(decision, timestamp=10.0)
        completion = recorder.record_completion(
            decision, CompletionSummary(protocol_complete=True), 2.0, "relayed", 200,
            timestamp=12.0,
        )
        self.assertEqual(admission_sink.records, [admission])
        self.assertEqual(completion_sink.records, [completion])
        self.assertIsInstance(admission, TraceRecord)
        self.assertIsInstance(completion, CompletionRecord)
        self.assertEqual((admission.timestamp, completion.timestamp), (10.0, 12.0))
        recorder = TraceRecorder(admission_sink)
        unsaved = recorder.record_completion(decision, CompletionSummary(), 0.0, "upstream_error", None)
        self.assertIsInstance(unsaved, CompletionRecord)
        self.assertEqual(admission_sink.records, [admission])

    def test_record_distinguishes_gateway_accounting_and_reported_usage(self):
        decision = _decision()
        envelope = SimpleNamespace(**{
            **asdict(decision.envelope), "prompt_count_source": "engine_rendered",
            "prompt_count_cached": True, "model_context_limit": 262_144,
        })
        decision = replace(decision, envelope=envelope)
        summary = CompletionSummary(
            first_token_seconds=1.5, prompt_tokens=111, completion_tokens=16,
            cached_tokens=64, reasoning_tokens=8, finish_reasons=("length",),
            protocol_complete=True,
        )
        recorder = TraceRecorder(MemorySink())
        with patch("redesign.gateway.capture.time.time", return_value=1234.0):
            record = recorder.record_completion(decision, summary, 3.5, "relayed", 200)
        self.assertEqual(record.timestamp, 1234.0)
        self.assertEqual(record.counted_prompt_tokens, 101)
        self.assertEqual((record.prompt_tokens, record.completion_tokens), (111, 16))
        self.assertEqual((record.cached_tokens, record.reasoning_tokens), (64, 8))
        self.assertEqual(record.prompt_count_source, "engine_rendered")
        self.assertTrue(record.prompt_count_cached)
        self.assertEqual(record.model_context_limit, 262_144)
        self.assertEqual(record.traffic_class, decision.traffic_class)
        self.assertEqual(record.priority, int(decision.priority))
        self.assertEqual(record.requested_max_tokens, 32_768)
        self.assertEqual(record.granted_max_tokens, 16_384)
        self.assertEqual(record.default_output_tokens, 4_096)
        self.assertFalse(record.default_applied)
        self.assertEqual(record.clamp_reason, "class_ceiling")
        self.assertEqual(record.finish_reasons, ("length",))
        self.assertTrue(record.protocol_complete)
        self.assertFalse(record.protocol_error)
        self.assertEqual((record.first_token_seconds, record.elapsed_seconds), (1.5, 3.5))
        self.assertEqual((record.relay_outcome, record.upstream_status), ("relayed", 200))
        self.assertTrue(record.borrowed)
        self.assertTrue(record.streaming)
        self.assertTrue(record.has_tools)
        self.assertTrue(record.has_images)
        self.assertEqual(record.modalities, ("text", "image"))

    def test_default_budget_at_this_hop_stays_distinct_from_an_explicit_request(self):
        decision = _decision()
        decision = replace(
            decision, envelope=replace(decision.envelope, requested_max_tokens=None),
            clamp=ClampResult(
                requested=None, granted=1_000, reason="context_window", default_output_tokens=4_096,
            ),
        )
        record = TraceRecorder(MemorySink()).record_completion(
            decision, CompletionSummary(), 0.1, "upstream_error", None,
        )
        self.assertIsNone(record.requested_max_tokens)
        self.assertEqual(record.granted_max_tokens, 1_000)
        self.assertEqual(record.default_output_tokens, 4_096)
        self.assertTrue(record.default_applied)
        self.assertEqual(record.clamp_reason, "context_window")
        self.assertIsNone(record.prompt_tokens)

    def test_legacy_decisions_have_backwards_compatible_defaults(self):
        envelope = SimpleNamespace(
            customer="legacy", prompt_tokens=12, requested_max_tokens=None,
            streaming=False, has_tools=False, path="/v1/chat/completions",
        )
        decision = SimpleNamespace(
            envelope=envelope, clamp=SimpleNamespace(granted=16, reason="default"),
            traffic_class="short-chat", priority=1,
        )
        record = TraceRecorder(MemorySink()).record_completion(
            decision, CompletionSummary(), 1.0, "client_cancel", None,
        )
        self.assertEqual(record.prompt_count_source, "heuristic")
        self.assertFalse(record.prompt_count_cached)
        self.assertIsNone(record.model_context_limit)
        self.assertIsNone(record.default_output_tokens)
        self.assertTrue(record.default_applied)
        self.assertFalse(record.borrowed)
        self.assertFalse(record.has_images)
        self.assertEqual(record.modalities, ("text",))
        self.assertIsNone(record.upstream_status)
        self.assertIsNone(record.protocol_complete)
        self.assertIsNone(record.prompt_tokens)
        self.assertIsNone(record.first_token_seconds)

    def test_record_has_hashed_customer_and_no_raw_metadata_or_response_data(self):
        observer = CompletionObserver(False, 0.0)
        observer.feed(_json_completion(
            id="private-response-id", model="private-model-id",
            error={"message": "private-error-message"},
        ))
        record = TraceRecorder(MemorySink()).record_completion(
            _decision(), observer.finish(), 1.0, "upstream_http_5xx", 500,
        )
        self.assertEqual(
            record.customer, hashlib.sha256(b"private-customer").hexdigest()[:12],
        )
        self.assertNotIn("private", json.dumps(asdict(record)))
        self.assertTrue(record.protocol_error)
        self.assertIs(record.protocol_complete, False)
        self.assertEqual(record.upstream_status, 500)
        with self.assertRaises(FrozenInstanceError):
            record.customer = "changed"

    def test_customer_hash_opt_out_keeps_existing_positional_api(self):
        admission_sink, completion_sink = MemorySink(), MemorySink()
        recorder = TraceRecorder(admission_sink, False, completion_sink=completion_sink)
        admission = recorder.record(_decision())
        completion = recorder.record_completion(_decision(), CompletionSummary(), 1.0, "deadline", None)
        self.assertEqual(admission.customer, "private-customer")
        self.assertEqual(completion.customer, "private-customer")

    def test_jsonl_sinks_keep_admission_and_completion_files_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            admission_path = Path(directory) / "admission.jsonl"
            completion_path = Path(directory) / "completion.jsonl"
            recorder = TraceRecorder(
                JsonlSink(admission_path), completion_sink=JsonlSink(completion_path),
            )
            recorder.record(_decision())
            recorder.record_completion(
                _decision(), CompletionSummary(finish_reasons=("stop",), protocol_complete=True),
                2.0, "relayed", 200,
            )
            admissions = [json.loads(line) for line in admission_path.read_text().splitlines()]
            completions = [json.loads(line) for line in completion_path.read_text().splitlines()]
            self.assertEqual(len(admissions), 1)
            self.assertEqual(len(completions), 1)
            self.assertEqual(admissions[0]["outcome"], "ADMIT")
            self.assertNotIn("protocol_complete", admissions[0])
            self.assertEqual(completions[0]["finish_reasons"], ["stop"])
            self.assertTrue(completions[0]["protocol_complete"])
            self.assertIsNone(completions[0]["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
