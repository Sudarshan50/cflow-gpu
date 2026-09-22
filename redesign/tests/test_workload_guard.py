"""Resource isolation and content-free diagnostics, including real HTTP wiring."""
import concurrent.futures
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from redesign.gateway.backpressure import StaticHealthSource
from redesign.gateway.inspection import InspectionError, PromptInspector, UsageObserver, inspectable, tokenize_projection
from redesign.gateway.models import EngineSnapshot, RequestEnvelope
from redesign.gateway.policy import build_default
from redesign.gateway.server import Handler, build_service, overcommitted_replicas
from redesign.gateway.workload import WorkloadBudget, WorkloadLimits
from redesign.tenancy.policy import TenancyPolicy


class ReservationTests(unittest.TestCase):
    def budget(self, **values):
        config = dict(large_context_tokens=100, large_context_requests=1,
                      long_output_tokens=20, long_output_requests=1, reserved_tokens=1000)
        config.update(values)
        return WorkloadBudget(WorkloadLimits(**config), StaticHealthSource(EngineSnapshot.healthy()))

    def test_large_and_long_output_budgets_are_independent(self):
        budget = self.budget()
        a, _ = budget.acquire(200, 10)
        self.assertIsNotNone(a)
        self.assertEqual(budget.acquire(200, 10)[1], "large_context_slots")
        b, _ = budget.acquire(10, 30)
        self.assertIsNotNone(b)
        self.assertEqual(budget.acquire(10, 30)[1], "long_output_slots")
        c, _ = budget.acquire(10, 10)
        self.assertIsNotNone(c)
        for lease in (a, b, c):
            self.assertTrue(budget.release(lease))
        self.assertEqual(budget.state()["reserved_tokens"], 0)

    def test_double_release_cannot_free_another_reservation(self):
        budget = self.budget()
        a, _ = budget.acquire(10, 10)
        b, _ = budget.acquire(10, 10)
        self.assertTrue(budget.release(a))
        self.assertFalse(budget.release(a))
        self.assertEqual(budget.state()["active_requests"], 1)
        self.assertEqual(budget.state()["reserved_tokens"], 20)
        budget.release(b)

    def test_projected_usage_covers_untracked_engine_work(self):
        budget = self.budget()
        budget.health_source = StaticHealthSource(EngineSnapshot(.8, 0, 0, 0, 1000))
        self.assertEqual(budget.acquire(100, 10)[1], "kv_headroom")
        self.assertEqual(budget.state()["active_requests"], 0)

    def test_unowned_engine_work_blocks_new_heavy_reservations(self):
        budget = self.budget()
        budget.health_source = StaticHealthSource(EngineSnapshot(.3, 2, 0, 0, 1000))
        self.assertEqual(budget.acquire(100, 10)[1], "untracked_engine_work")

    def test_queue_blocks_heavy_work_but_small_requests_can_fit(self):
        budget = self.budget()
        budget.health_source = StaticHealthSource(EngineSnapshot(.1, 2, 1, 0, 1000))
        self.assertEqual(budget.acquire(100, 10)[1], "engine_queue")
        self.assertIsNotNone(budget.acquire(10, 10)[0])

    def test_small_short_work_keeps_headroom_above_heavy_soft_limit(self):
        budget = self.budget()
        budget.health_source = StaticHealthSource(EngineSnapshot(.9, 4, 1, 0, 1000))
        self.assertIsNotNone(budget.acquire(10, 10)[0])

    def test_concurrent_admission_does_not_oversubscribe(self):
        budget = self.budget(large_context_requests=4, reserved_tokens=10000)
        barrier = threading.Barrier(32)
        def acquire(_):
            barrier.wait()
            return budget.acquire(100, 10)[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            leases = list(pool.map(acquire, range(32)))
        self.assertEqual(sum(lease is not None for lease in leases), 4)
        for lease in leases:
            if lease is not None:
                budget.release(lease)
        self.assertEqual(budget.state()["reserved_tokens"], 0)

    def test_reservation_rejection_and_double_release_preserve_class_slots(self):
        policy = build_default(262144, 64)
        policy.workload_budget = self.budget(large_context_tokens=10000, long_output_tokens=10000)
        first = policy.decide(RequestEnvelope("a", 600, 100))
        rejected = policy.decide(RequestEnvelope("b", 600, 100))
        second = policy.decide(RequestEnvelope("c", 100, 100))
        self.assertTrue(first.admitted)
        self.assertFalse(rejected.admitted)
        self.assertTrue(second.admitted)
        policy.release(first)
        policy.release(first)
        self.assertEqual(sum(policy._budget._in_flight.values()), 1)
        policy.release(second)
        self.assertEqual(sum(policy._budget._in_flight.values()), 0)


class FingerprintTests(unittest.TestCase):
    def test_history_means_completed_prefix_not_guaranteed_cache_hit(self):
        inspector = PromptInspector("http://127.0.0.1:1")
        first = inspector._fingerprint(list(range(2000)), "test", 1)
        inspector.record(first, http_status=200, usage={"prompt_tokens": 2000, "completion_tokens": 1,
                         "prompt_tokens_details": {"cached_tokens": 0}}, complete=True, outcome="ADMIT", granted=1)
        second = inspector._fingerprint(list(range(2001)), "test", 1)
        self.assertEqual(second.prior_completed_prefix_tokens, 1536)
        self.assertEqual(first.prefix_768, second.prefix_768)
        self.assertIsInstance(first.replay_fingerprint, str)

    def test_changed_early_token_changes_prefix_fingerprint(self):
        inspector = PromptInspector("http://127.0.0.1:1")
        tokens = list(range(800))
        first = inspector._fingerprint(tokens, "test", 1)
        tokens[0] = 9999
        second = inspector._fingerprint(tokens, "test", 1)
        self.assertNotEqual(first.prefix_768, second.prefix_768)

    def test_engine_epoch_change_discards_prior_history_and_late_old_results(self):
        inspector = PromptInspector("http://127.0.0.1:1")
        old = inspector._fingerprint(list(range(800)), "test", 1)
        new = inspector._fingerprint(list(range(800)), "test", 2)
        inspector.record(old, http_status=200, usage={"prompt_tokens": 800, "prompt_tokens_details": {"cached_tokens": 0}},
                         complete=True, outcome="ADMIT", granted=1)
        self.assertNotEqual(old.scope, new.scope)
        self.assertEqual(inspector._fingerprint(list(range(800)), "test", 2).prior_completed_prefix_tokens, 0)

    def test_usage_mismatch_or_incomplete_response_does_not_teach_history(self):
        inspector = PromptInspector("http://127.0.0.1:1")
        first = inspector._fingerprint(list(range(800)), "test", 1)
        inspector.record(first, http_status=200, usage={"prompt_tokens": 801, "prompt_tokens_details": {"cached_tokens": 0}},
                         complete=True, outcome="ADMIT", granted=1)
        self.assertEqual(inspector.recent()["history_entries"], 0)
        self.assertTrue(inspector.recent()["counting_quarantined"])
        self.assertIsNone(inspector.inspect({"prompt": [1, 2]}))

    def test_projection_preserves_reasoning_alias_without_mutating_the_request(self):
        message = {"role": "assistant", "content": "answer", "reasoning_content": "preserved thinking"}
        payload = {"messages": [message], "reasoning_effort": "low"}
        projected = tokenize_projection(payload)
        self.assertEqual(projected["messages"][0]["reasoning"], "preserved thinking")
        self.assertNotIn("reasoning_content", projected["messages"][0])
        self.assertEqual(message["reasoning_content"], "preserved thinking")
        self.assertNotIn("reasoning", message)
        self.assertEqual(projected["chat_template_kwargs"]["reasoning_effort"], "low")
        self.assertTrue(projected["chat_template_kwargs"]["enable_thinking"])

    def test_projection_uses_the_same_reasoning_precedence_as_inference(self):
        payload = {"messages": [{"role": "assistant", "content": "x", "reasoning": "primary", "reasoning_content": "legacy"}],
                   "reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}
        projected = tokenize_projection(payload)
        self.assertEqual(projected["messages"][0]["reasoning"], "primary")
        self.assertFalse(projected["chat_template_kwargs"]["enable_thinking"])

    def test_multimodal_and_truncation_use_conservative_path(self):
        self.assertFalse(inspectable({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}))
        self.assertFalse(inspectable({"messages": [], "truncate_prompt_tokens": 100}))
        self.assertTrue(inspectable({"messages": [{"role": "assistant", "content": None, "reasoning_content": "retained"}]}))
        self.assertFalse(inspectable({"messages": [], "prompt": "a different completion input"}))

    def test_replica_limits_cannot_be_bypassed_with_coercible_values(self):
        for value in ("2", "1", True, 0, -1, 2):
            self.assertTrue(overcommitted_replicas({"n": value}))
        self.assertFalse(overcommitted_replicas({"n": 1, "best_of": None}))

    def test_observer_discards_text_and_requires_complete_stream(self):
        observer = UsageObserver(True)
        payload = b'data: {"choices":[{"delta":{"content":"PRIVATE_SENTINEL"},"finish_reason":"stop"}]}\n\n'
        payload += b'data: {"choices":[],"usage":{"prompt_tokens":800,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":768},"private":"PRIVATE_SENTINEL"}}\n\n'
        for byte in payload:
            observer.feed(bytes([byte]))
        self.assertFalse(observer.finish())
        self.assertNotIn("PRIVATE_SENTINEL", repr(observer.__dict__))
        observer.feed(b"data: [DONE]\n\n")
        self.assertTrue(observer.finish())

    def test_deferred_proxy_context_check_still_clamps_output(self):
        payload = {"messages": [{"role": "user", "content": " " * 10000}], "max_tokens": 10000}
        normal = TenancyPolicy(max_model_len=1000)
        deferred = TenancyPolicy(max_model_len=1000, defer_local_context_check=True)
        self.assertTrue(normal.decide(payload, "FW-Kimi-K3").reject)
        result = deferred.decide(payload, "FW-Kimi-K3")
        self.assertFalse(result.reject)
        self.assertLessEqual(result.granted_max_tokens, 4096)


class HttpGuardTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.started = threading.Event()
        self.release = threading.Event()
        self.inferences = []
        self.tokenizations = []
        class Engine(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/metrics":
                    data = (b'vllm:kv_cache_usage_perc 0\nvllm:num_requests_running 0\n'
                            b'vllm:num_requests_waiting 0\nvllm:num_preemptions_total 0\n'
                            b'vllm:cache_config_info{kv_cache_size_tokens="5000"} 1\n')
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send(200, {"healthy": True})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][0]["content"]
                count = 1000 if content == "large" else 10
                if self.path == "/tokenize":
                    owner.tokenizations.append(body)
                    self.send(200, {"tokens": list(range(count)), "count": count})
                    return
                owner.inferences.append(body)
                if content == "large":
                    owner.started.set()
                    owner.release.wait(5)
                self.send(200, {"choices": [{"message": {"content": "PRIVATE_ANSWER"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": count, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}})
        self.engine = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
        self.engine.daemon_threads = True
        threading.Thread(target=self.engine.serve_forever, daemon=True).start()
        service = build_service(f"http://127.0.0.1:{self.engine.server_port}", 262144, 64, None, None, True,
            workload_limits=WorkloadLimits(large_context_tokens=500, large_context_requests=1,
                long_output_tokens=128, long_output_requests=1, reserved_tokens=2000))
        class Gateway(Handler):
            pass
        Gateway.service = service
        self.service = service
        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        self.gateway.daemon_threads = True
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()

    def tearDown(self):
        self.release.set()
        self.gateway.shutdown()
        self.gateway.server_close()
        self.engine.shutdown()
        self.engine.server_close()

    def post(self, text):
        request = urllib.request.Request(f"http://127.0.0.1:{self.gateway.server_port}/v1/chat/completions",
            json.dumps({"model": "FW-Kimi-K3", "messages": [{"role": "user", "content": text}],
                        "max_tokens": 16, "chat_template_kwargs": {"thinking": False}}).encode(),
            {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return response.status, json.load(response), dict(response.headers)
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.load(exc), dict(exc.headers)

    def test_large_limit_allows_small_work_and_releases_after_completion(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.post, "large")
            self.assertTrue(self.started.wait(3))
            code, body, headers = self.post("large")
            self.assertEqual(code, 503)
            self.assertIn("large_context_slots", body["error"]["message"])
            self.assertIn("Retry-After", headers)
            self.assertEqual(self.post("small")[0], 200)
            self.release.set()
            self.assertEqual(pending.result()[0], 200)
        self.assertEqual(len(self.inferences), 2)
        self.assertEqual(self.service.policy.workload_budget.state()["active_requests"], 0)
        events = self.service.inspector.recent()["events"]
        self.assertTrue(any(row["verified_usage"] for row in events))
        self.assertNotIn("PRIVATE_ANSWER", json.dumps(events))
        self.assertEqual(self.tokenizations[0]["chat_template_kwargs"], {"thinking": False})

    def test_batched_token_prompts_reserve_each_context(self):
        self.service.policy.workload_budget.limits = WorkloadLimits(reserved_tokens=600000)
        request = urllib.request.Request(f"http://127.0.0.1:{self.gateway.server_port}/v1/completions",
            json.dumps({"model": "FW-Kimi-K3", "prompt": [[1, 2], [1, 3], [1, 4]], "max_tokens": 16}).encode(),
            {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request, timeout=5)
        with context.exception as response:
            body = json.load(response)
        self.assertEqual(context.exception.code, 503)
        self.assertIn("reserved_tokens", body["error"]["message"])
        self.assertEqual(self.inferences, [])


if __name__ == "__main__":
    unittest.main()
