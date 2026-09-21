"""Prefix probe correctness with fake HTTP transports only; never runs inference."""

from __future__ import annotations

import http.client
import io
import json
import unittest
from unittest.mock import Mock, patch

from redesign.gateway.engine import EngineClient, ProxyResponse
from redesign.probe.cache_salt import audit, compare_counters, main


def completion(cached=0, finish_reason="stop"):
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 350,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def fake_transport(bodies=None, statuses=(200, 200)):
    """Use the real proxy/drain/close code with fake HTTP connections."""
    if bodies is None:
        bodies = (completion(), completion(128))
    connections, responses = [], []
    for body, status in zip(bodies, statuses):
        response = Mock(spec=http.client.HTTPResponse)
        response.status = status
        response.getheaders.return_value = [("Content-Type", "application/json")]
        response.read.return_value = body if isinstance(body, bytes) else json.dumps(body).encode()
        connection = Mock(spec=http.client.HTTPConnection)
        connection.getresponse.return_value = response
        responses.append(response)
        connections.append(connection)
    engine = EngineClient("http://offline.invalid")
    engine._connect = Mock(side_effect=connections)
    return engine, connections, responses


def counters(*values):
    return Mock(side_effect=[f"vllm:prefix_cache_hits_total {value}\n" for value in values])


class PrefixProbeTest(unittest.TestCase):
    def assert_closed_and_drained(self, connections, responses):
        for connection, response in zip(connections, responses):
            response.read.assert_called_once_with()
            response.close.assert_called_once_with()
            connection.close.assert_called_once_with()

    def test_http_500_with_unrelated_live_hits_cannot_pass(self):
        for statuses in ((500, 200), (200, 500), (500, 500)):
            with self.subTest(statuses=statuses):
                engine, connections, responses = fake_transport(statuses=statuses)
                verdict = audit(engine, counters(0, 100, 300))
                self.assertFalse(verdict.passed)
                self.assertIn("HTTP 500", verdict.detail)
                self.assertEqual(verdict.second_hits, 200)
                self.assert_closed_and_drained(connections, responses)

    def test_positive_second_response_usage_passes_without_global_delta(self):
        engine, connections, responses = fake_transport()
        verdict = audit(engine, counters(900, 900, 900))
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.first_cached_tokens, 0)
        self.assertEqual(verdict.second_cached_tokens, 128)
        self.assertEqual(verdict.second_hits, 0)
        self.assertIn("second response reports cached_tokens=128", verdict.detail)
        self.assert_closed_and_drained(connections, responses)

    def test_only_second_request_usage_decides_reuse(self):
        engine, connections, responses = fake_transport((completion(256), completion(0)))
        verdict = audit(engine, counters(0, 100, 500))
        self.assertFalse(verdict.passed)
        self.assertIn("cached_tokens=0", verdict.detail)
        self.assertEqual(verdict.second_hits, 400)
        self.assert_closed_and_drained(connections, responses)

    def test_global_diagnostics_can_be_absent_broken_or_reset(self):
        for scrape in (
            Mock(return_value=""),
            Mock(side_effect=OSError("metrics unavailable")),
            counters(100, 0, 0),
            counters("NaN", "NaN", "NaN"),
        ):
            with self.subTest(scrape=scrape):
                engine, connections, responses = fake_transport()
                verdict = audit(engine, scrape)
                self.assertTrue(verdict.passed)
                self.assertIn("diagnostic only", verdict.detail)
                self.assert_closed_and_drained(connections, responses)

    def test_missing_usage_fails_with_an_actionable_explanation(self):
        for usage in (None, {}, [], {"prompt_tokens": 350}, {"prompt_tokens_details": {}}):
            for failed_request in (0, 1):
                with self.subTest(usage=usage, failed_request=failed_request):
                    bodies = [completion(), completion(128)]
                    bodies[failed_request]["usage"] = usage
                    engine, connections, responses = fake_transport(bodies)
                    verdict = audit(engine, counters(0, 100, 500))
                    self.assertFalse(verdict.passed)
                    self.assertIn("usage.prompt_tokens_details.cached_tokens", verdict.detail)
                    self.assertIn("numeric per-response cache count", verdict.detail)
                    self.assert_closed_and_drained(connections, responses)

    def test_cached_tokens_must_be_a_finite_nonnegative_number(self):
        for value in (None, "128", True, False, -1, float("nan"), float("inf"), [], {}):
            with self.subTest(value=value):
                engine, connections, responses = fake_transport((completion(), completion(value)))
                verdict = audit(engine, counters(0, 100, 500))
                self.assertFalse(verdict.passed)
                self.assertIn("cached_tokens", verdict.detail)
                self.assert_closed_and_drained(connections, responses)

    def test_float_cache_count_is_numeric(self):
        engine, _, _ = fake_transport((completion(), completion(128.0)))
        self.assertTrue(audit(engine, counters(0, 0, 0)).passed)

    def test_malformed_body_is_not_a_cache_hit(self):
        for body in (b"", b"{", b"\xff", b"[]", b"null", b'"text"', b'data: {}\n\n'):
            with self.subTest(body=body):
                engine, connections, responses = fake_transport((completion(), body))
                verdict = audit(engine, counters(0, 100, 500))
                self.assertFalse(verdict.passed)
                self.assertIn("second request failed", verdict.detail)
                self.assertIn("response", verdict.detail)
                self.assert_closed_and_drained(connections, responses)

    def test_choices_and_finish_reason_must_describe_a_completed_response(self):
        valid_choice = completion()["choices"][0]
        invalid_choices = (
            None, [], {}, [None], [{}],
            [{"finish_reason": "stop"}],
            [{**valid_choice, "message": None}],
            [{**valid_choice, "finish_reason": None}],
            [{**valid_choice, "finish_reason": ""}],
            [{**valid_choice, "finish_reason": "error"}],
            [valid_choice, {**valid_choice, "finish_reason": None}],
        )
        for choices in invalid_choices:
            with self.subTest(choices=choices):
                body = completion(128)
                body["choices"] = choices
                engine, connections, responses = fake_transport((completion(), body))
                verdict = audit(engine, counters(0, 100, 500))
                self.assertFalse(verdict.passed)
                self.assert_closed_and_drained(connections, responses)

    def test_error_document_is_rejected_even_with_choices_and_usage(self):
        body = {**completion(128), "error": {"message": "failed"}}
        engine, connections, responses = fake_transport((completion(), body))
        verdict = audit(engine, counters(0, 100, 500))
        self.assertFalse(verdict.passed)
        self.assertIn("invalid completion response", verdict.detail)
        self.assert_closed_and_drained(connections, responses)

    def test_length_finish_is_valid_even_when_budget_is_spent_thinking(self):
        bodies = [completion(0, "length"), completion(128, "length")]
        for body in bodies:
            body["choices"][0]["message"] = {
                "role": "assistant", "content": None, "reasoning_content": "Let me think",
            }
        engine, connections, responses = fake_transport(bodies)
        self.assertTrue(audit(engine, counters(0, 0, 0)).passed)
        self.assert_closed_and_drained(connections, responses)

    def test_incomplete_transport_fails_even_if_partial_body_contains_valid_json(self):
        for failed_request in (0, 1):
            with self.subTest(failed_request=failed_request):
                engine, connections, responses = fake_transport()
                responses[failed_request].read.side_effect = http.client.IncompleteRead(
                    json.dumps(completion(128)).encode(), 100,
                )
                verdict = audit(engine, counters(0, 100, 500))
                self.assertFalse(verdict.passed)
                self.assertIn("IncompleteRead", verdict.detail)
                self.assert_closed_and_drained(connections, responses)

    def test_proxy_failure_closes_connection_and_cannot_pass(self):
        for stage in ("request", "getresponse"):
            with self.subTest(stage=stage):
                engine, connections, responses = fake_transport()
                getattr(connections[0], stage).side_effect = OSError("transport failed")
                verdict = audit(engine, counters(0, 100, 500))
                self.assertFalse(verdict.passed)
                self.assertIn("first request failed", verdict.detail)
                connections[0].close.assert_called_once_with()
                responses[0].read.assert_not_called()
                self.assert_closed_and_drained(connections[1:], responses[1:])

    def test_response_cleanup_failure_cannot_pass_and_connection_still_closes(self):
        engine, connections, responses = fake_transport()
        responses[1].close.side_effect = OSError("close failed")
        verdict = audit(engine, counters(0, 100, 500))
        self.assertFalse(verdict.passed)
        self.assertIn("close failed", verdict.detail)
        self.assert_closed_and_drained(connections, responses)

    def test_body_is_fully_exhausted_and_closed_by_the_probe(self):
        # This fake does not clean itself up on exhaustion: audit must close it.
        responses, finished = [], []

        def chunks(index):
            raw = json.dumps(completion(index * 128)).encode()
            yield raw[:20]
            yield raw[20:]
            finished.append(index)

        for index in range(2):
            response = Mock(spec=ProxyResponse)
            response.status = 200
            response.body = chunks(index)
            responses.append(response)
        engine = Mock(proxy=Mock(side_effect=responses))
        verdict = audit(engine, counters(0, 0, 0))
        self.assertTrue(verdict.passed)
        self.assertEqual(finished, [0, 1])
        for response in responses:
            response.close.assert_called_once_with()

    def test_requests_use_identical_bounded_nonstream_json_with_requested_model(self):
        engine, connections, _ = fake_transport()
        audit(engine, counters(0, 0, 0), model="my-model")
        payloads = []
        for connection in connections:
            args, kwargs = connection.request.call_args
            self.assertEqual(args, ("POST", "/v1/chat/completions"))
            payloads.append(json.loads(kwargs["body"]))
        self.assertEqual(payloads[0], payloads[1])
        payload = payloads[0]
        self.assertEqual(payload["model"], "my-model")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["temperature"], 0)
        prompt = payload["messages"][0]["content"]
        self.assertEqual(prompt.splitlines()[-1].split(), [f"{n:03d}" for n in range(160)])
        self.assertLess(len(prompt), 700)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_legacy_counter_helper_is_explicitly_diagnostic(self):
        verdict = compare_counters(0, 10)
        self.assertTrue(verdict.passed)  # Backward compatibility only.
        self.assertIn("diagnostic only", verdict.detail)
        self.assertIn("cannot attribute", verdict.detail)


class PrefixProbeCliTest(unittest.TestCase):
    def test_cli_preserves_url_model_json_and_exit_status_without_network(self):
        for cached, expected_code in ((128, 0), (0, 1)):
            with self.subTest(cached=cached):
                engine, connections, _ = fake_transport((completion(), completion(cached)))
                engine.metrics_text = counters(0, 0, 0)
                with (
                    patch("redesign.probe.cache_salt.EngineClient", return_value=engine) as client,
                    patch("sys.stdout", new_callable=io.StringIO) as output,
                ):
                    code = main(["--url", "http://offline.invalid:1234", "--model", "test-model", "--json"])
                self.assertEqual(code, expected_code)
                client.assert_called_once_with("http://offline.invalid:1234")
                report = json.loads(output.getvalue())
                self.assertEqual(report["second_cached_tokens"], cached)
                self.assertEqual(report["sharing"], cached > 0)
                for connection in connections:
                    sent = json.loads(connection.request.call_args.kwargs["body"])
                    self.assertEqual(sent["model"], "test-model")

    def test_text_cli_handles_unavailable_diagnostics(self):
        engine, _, _ = fake_transport()
        engine.metrics_text = Mock(side_effect=OSError("no metrics"))
        with (
            patch("redesign.probe.cache_salt.EngineClient", return_value=engine),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            self.assertEqual(main([]), 0)
        self.assertIn("PASS", output.getvalue())
        self.assertIn("diagnostic only", output.getvalue())
        self.assertIn("unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
