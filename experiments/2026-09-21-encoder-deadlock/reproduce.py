#!/usr/bin/env python3
"""Reproduce K3 encoder/alignment deadlocks using installed source and CPU metadata.

Only selected AST definitions are executed; vLLM, torch, transformers, image
processors, and model modules are never imported. Run with host Python 3.12 -B.
"""

import argparse
import ast
import bisect
import hashlib
import json
import math
import subprocess
import sys
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace as N


IMAGE_ID = "sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8"
VLLM_PATH = "usr/local/lib/python3.12/dist-packages/vllm"
SNAPSHOT = (
    "hf/hub/models--moonshotai--Kimi-K3/snapshots/"
    "f831ab66814297da540d832a5235f8e904f29d06"
)
BATCH_TOKENS = 4096
ENCODER_TOKENS = 16817
BLOCK_TOKENS = 768
PROPOSED_CACHE_TOKENS = 262144
METHODS = (
    "_try_schedule_encoder_inputs",
    "_mamba_block_aligned_split",
    "_free_encoder_inputs",
)


def emit(label, **fields):
    print(label, json.dumps(fields, sort_keys=True))


def inspect_source(container):
    # Read-only container metadata; never docker exec or an inference endpoint.
    result = subprocess.run(
        [
            "docker", "inspect", "--format",
            "{{.State.Pid}} {{.State.Running}} {{.Image}}", container,
        ],
        check=True, capture_output=True, text=True, timeout=10,
    )
    pid_text, running, image_id = result.stdout.strip().split()
    if running != "true" or int(pid_text) <= 0:
        raise RuntimeError(f"Container {container!r} must already be running")
    if image_id != IMAGE_ID:
        raise RuntimeError(f"Source pin mismatch: expected {IMAGE_ID}, got {image_id}")
    container_root = Path("/proc") / pid_text / "root"
    root = container_root / VLLM_PATH
    emit("SOURCE", container=container, pid=int(pid_text), image_id=image_id,
         root=str(root))
    return root, container_root / SNAPSHOT / "preprocessor_config.json"


def load_source(root):
    namespace = {
        "__name__": __name__, "bisect": bisect, "math": math,
        "OrderedDict": OrderedDict, "dataclass": dataclass,
        "cached_property": cached_property, "ImageSize": N,
    }
    fingerprints = {}

    def load(relative_path, names, class_name=None):
        path = root / relative_path
        raw = path.read_bytes()
        fingerprints[relative_path] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw, filename=str(path))
        body = tree.body
        if class_name is not None:
            body = next(
                node.body for node in body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        selected = [
            node for node in body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        assert len(selected) == len(names), (relative_path, names)
        # Preserve function bodies and source line numbers. Deferred annotations
        # avoid resolving torch/vLLM types. No installed module imports execute.
        module = ast.Module(
            body=[ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0,
            )] + selected,
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        return tree

    load("v1/core/encoder_cache_manager.py",
         ("EncoderCacheManager", "compute_mm_encoder_budget"))
    load("multimodal/inputs.py", ("PlaceholderRange",))
    load("multimodal/utils.py", ("get_mm_features_in_window",))
    load("v1/core/sched/scheduler.py", METHODS, "Scheduler")
    load("models/kimi_k3/common/mm_preprocess.py", ("navit_resize_image",))
    load("models/kimi_k3/common/mm_preprocess.py",
         ("get_max_image_size",), "KimiK3ProcessingInfo")
    budget_methods = ("get_encoder_budget", "_get_max_items")
    load("multimodal/encoder_budget.py", budget_methods, "MultiModalBudget")
    runner_methods = (
        "_cache_encoder_output", "_get_encoder_output_from_cache",
        "_process_encoder_cache_scheduler_output",
    )
    runner_tree = load("v1/worker/gpu_model_runner.py", runner_methods, "GPUModelRunner")
    cache_initializers = [
        node for node in ast.walk(runner_tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute) and node.target.attr == "encoder_cache"
    ]
    assert len(cache_initializers) == 1
    assert isinstance(cache_initializers[0].value, ast.Dict)
    assert not cache_initializers[0].value.keys
    namespace["Budget"] = type(
        "InstalledBudgetMethods", (), {name: namespace[name] for name in budget_methods},
    )
    namespace["Runner"] = type(
        "InstalledRunnerCacheMethods", (), {name: namespace[name] for name in runner_methods},
    )
    namespace["Scheduler"] = type(
        "InstalledSchedulerMethods", (), {name: namespace[name] for name in METHODS},
    )
    namespace["ProcessingInfo"] = type(
        "InstalledProcessingInfoMethod", (),
        {"get_max_image_size": namespace["get_max_image_size"]},
    )
    emit("SOURCE_SHA256", **fingerprints)
    return namespace


def scheduler_config():
    # Startup-pinned base fields; compute_mm_encoder_budget raises both to 16817.
    return N(
        max_num_encoder_input_tokens=BATCH_TOKENS,
        encoder_cache_size=BATCH_TOKENS,
        max_num_batched_tokens=BATCH_TOKENS,
        disable_chunked_mm_input=False,
        long_prefill_token_threshold=0,
    )


def check_profile(source, preprocessor_path):
    config = json.loads(preprocessor_path.read_text())["media_proc_cfg"]
    keys = ("patch_size", "merge_kernel_size", "in_patch_limit",
            "patch_limit_on_one_side", "fixed_output_tokens")
    parameters = tuple(config[key] for key in keys)
    assert parameters == (14, 2, 65536, 512, None), parameters
    resize = source["navit_resize_image"]
    size = source["ProcessingInfo"].get_max_image_size(*parameters)
    max_embeds = resize(size.width, size.height, *parameters)["num_tokens"]
    image_embeds = resize(2800, 2800, *parameters)["num_tokens"]
    budget, capacity = source["compute_mm_encoder_budget"](
        scheduler_config(), {"image": max_embeds},
    )
    assert (max_embeds, budget, capacity, image_embeds) == (16817, 16817, 16817, 10000)
    emit("PROFILE_METADATA", max_image_size=[size.width, size.height],
         max_image_embeds=max_embeds, encoder_compute_budget=budget,
         encoder_cache_size=capacity, image_2800x2800_embeds=image_embeds,
         decoder_batch_tokens=BATCH_TOKENS, mamba_block_tokens=BLOCK_TOKENS)


class MetadataRequest(N):
    def get_num_encoder_embeds(self, input_id):
        return self.mm_features[input_id].mm_position.get_num_embeds()


def make_case(source, items, capacity=ENCODER_TOKENS):
    scheduler = source["Scheduler"]()
    scheduler.scheduler_config = scheduler_config()
    scheduler.cache_config = N(block_size=BLOCK_TOKENS, mamba_cache_mode="align")
    scheduler.max_num_scheduled_tokens = BATCH_TOKENS
    scheduler.hash_block_size = BLOCK_TOKENS
    scheduler.mamba_partial_cache_hit = False
    scheduler.use_eagle = False
    scheduler.is_encoder_decoder = False
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = source["EncoderCacheManager"](capacity)
    features = []
    for index, (offset, embeds) in enumerate(items):
        position = source["PlaceholderRange"](offset=offset, length=embeds + 10)
        # Synthetic wrapper: nine non-embedding slots, N embedding slots, one
        # trailing slot. Supply the cumulative mask directly, without tensors.
        object.__setattr__(position, "embeds_cumsum",
                           [0] * 9 + list(range(1, embeds + 1)) + [embeds])
        features.append(N(identifier=f"image{index}", mm_position=position))
    last = features[-1].mm_position
    prompt_end = last.offset + last.length + 1000
    request = MetadataRequest(
        request_id="metadata-request", mm_features=features, has_encoder_inputs=True,
        num_computed_tokens=0, num_output_placeholders=0,
        num_prompt_tokens=prompt_end, num_tokens=prompt_end, shared_prefix_boundary=0,
    )
    return scheduler, request


def run_running_case(source, label, items, capacity=ENCODER_TOKENS,
                     aligned=True, trace=False):
    scheduler, request = make_case(source, items, capacity)
    manager = scheduler.encoder_cache_manager
    stalls = 0
    for step in range(24):
        start = request.num_computed_tokens
        ids, before, _, external = scheduler._try_schedule_encoder_inputs(
            request, start, min(BATCH_TOKENS, request.num_tokens - start), ENCODER_TOKENS,
        )
        assert not external
        after = scheduler._mamba_block_aligned_split(request, before) if aligned else before
        # Mirror the installed running flow: encoder gate -> alignment ->
        # commit allocations/progress only for a nonzero step -> consumed-input free.
        if after:
            for input_id in ids:
                manager.allocate(request, input_id)
            request.num_computed_tokens += after
            scheduler._free_encoder_inputs(request)
        if trace:
            emit("RUNNING_STEP", step=step, start=start, after_encoder_gate=before,
                 after_align=after, encoder_ids=ids, freeable=manager.num_freeable_slots,
                 held_inputs=sorted(manager.get_cached_input_ids(request)))
        stalls = stalls + 1 if after == 0 else 0
        if request.num_computed_tokens == request.num_tokens or stalls == 3:
            break
    completed = request.num_computed_tokens == request.num_tokens
    stalled = stalls == 3
    assert completed or stalled, "Bounded metadata run did not reach an outcome"
    emit("RESULT", case=label, computed=request.num_computed_tokens,
         prompt_tokens=request.num_tokens, completed=completed, stalled=stalled,
         freeable=manager.num_freeable_slots)
    return scheduler, request, stalled


def run_waiting_case(source):
    scheduler, request = make_case(source, [(20, 10000), (10040, 10000)])
    request.request_id = "waiting-request"
    request.mm_features[0].identifier = "shared-image"
    request.mm_features[1].identifier = "new-image"
    manager = scheduler.encoder_cache_manager
    prior = MetadataRequest(request_id="prior-finished-request", mm_features=request.mm_features)
    manager.allocate(prior, 0)
    manager.free(prior)
    assert manager.num_freeable_slots == ENCODER_TOKENS
    assert not manager.cached["shared-image"]
    emit("BEFORE_WAITING_ADMISSION", running_requests=0,
         freeable=manager.num_freeable_slots, cached_image_refs=0)
    for attempt in range(3):
        ids, before, _, external = scheduler._try_schedule_encoder_inputs(
            request, 9984, BATCH_TOKENS, ENCODER_TOKENS,
        )
        after = scheduler._mamba_block_aligned_split(
            request, before, num_new_local_computed_tokens=9984,
        )
        # Installed waiting flow breaks on zero, before queue removal/KV
        # allocation and its failure-cleanup hook. Leave the touched ref intact.
        assert before == 56 and after == 0 and not ids and not external
        assert manager.num_freeable_slots == 6817
        assert manager.cached["shared-image"] == {request.request_id}
        assert request.num_computed_tokens == 0
        emit("WAITING_ATTEMPT", attempt=attempt, prefix_cache_hit_tokens=9984,
             after_encoder_gate=before, after_mamba_alignment=after,
             cache_freeable=manager.num_freeable_slots,
             cached_image_refs=sorted(manager.cached["shared-image"]),
             request_computed_tokens=request.num_computed_tokens)
    return scheduler, request


def check_abort_cleanup(source, scheduler, request, label):
    manager = scheduler.encoder_cache_manager
    # Exact hook invoked by Scheduler._free_request after a delivered abort.
    # This checks cache cleanup, not cancellation transport/API delivery.
    manager.free(request)
    freeable_after_free = manager.num_freeable_slots
    assert freeable_after_free == ENCODER_TOKENS
    assert not manager.request_cached_ids
    _, fresh = make_case(source, [(20, ENCODER_TOKENS)])
    fresh.request_id = "next-metadata-request"
    fresh.mm_features[0].identifier = "fresh-image"
    can_allocate = manager.can_allocate(fresh, 0, ENCODER_TOKENS, 0)
    assert can_allocate and manager.num_free_slots == ENCODER_TOKENS
    evicted = manager.get_freed_mm_hashes()
    assert evicted == [request.mm_features[0].identifier]
    emit("ABORT_CLEANUP", case=label, freeable_after_free=freeable_after_free,
         fresh_max_image_can_allocate=can_allocate, evicted_hashes=evicted,
         remaining_request_refs=len(manager.request_cached_ids))


def check_capacity_profile_and_memory(source, preprocessor_path):
    base = scheduler_config()
    enlarged = copy(base)
    enlarged.encoder_cache_size = PROPOSED_CACHE_TOKENS
    assert base.encoder_cache_size == BATCH_TOKENS
    assert enlarged.max_num_encoder_input_tokens == base.max_num_encoder_input_tokens
    for config, expected_capacity in ((base, ENCODER_TOKENS),
                                      (enlarged, PROPOSED_CACHE_TOKENS)):
        compute, capacity = source["compute_mm_encoder_budget"](
            config, {"image": ENCODER_TOKENS},
        )
        assert compute == ENCODER_TOKENS and capacity == expected_capacity
        budget = source["Budget"]()
        budget.encoder_compute_budget = compute
        budget.encoder_cache_size = capacity
        budget.max_model_len = PROPOSED_CACHE_TOKENS
        budget.max_num_reqs = 64
        budget.mm_limits = {"image": 999}
        budget.scheduler_config = copy(config)
        budget.scheduler_config.enable_chunked_prefill = True
        assert budget.get_encoder_budget() == ENCODER_TOKENS
        assert budget._get_max_items("image", ENCODER_TOKENS)[1] == 1
    emit("CAPACITY_PROFILE_CONTROL", encoder_compute_budget=compute,
         encoder_cache_size=capacity, profile_budget=budget.get_encoder_budget(),
         profile_max_image_items=1, decoder_batch_tokens=BATCH_TOKENS,
         original_config_unchanged=base.encoder_cache_size == BATCH_TOKENS)

    runner = source["Runner"]()
    runner.encoder_cache = {}
    runner.maybe_save_ec_to_connector = lambda cache, key: None
    marker = object()
    runner._cache_encoder_output("synthetic-output", marker, None, [])
    assert runner._get_encoder_output_from_cache("synthetic-output") is marker
    runner._process_encoder_cache_scheduler_output(
        N(free_encoder_mm_hashes=["synthetic-output"]),
    )
    assert not runner.encoder_cache
    emit("RUNNER_CACHE_CONTROL", initializer="empty dict", stores_output_by_reference=True,
         eviction_passed=True)

    model = json.loads(preprocessor_path.with_name("config.json").read_text())
    hidden = model["text_config"]["hidden_size"]
    assert hidden == model["vision_config"]["text_hidden_size"] == 7168
    assert model["dtype"] == model["text_config"]["dtype"] == "bfloat16"
    dtype_bytes = 2
    old_bytes = ENCODER_TOKENS * hidden * dtype_bytes
    full_bytes = PROPOSED_CACHE_TOKENS * hidden * dtype_bytes
    assert full_bytes == 3758096384 and full_bytes < 4 * 1024**3
    emit("MEMORY_PAYLOAD_ESTIMATE", hidden_size=hidden, dtype_bytes=dtype_bytes,
         total_logical_bytes_per_rank=full_bytes,
         total_logical_gib_per_rank=full_bytes / 1024**3,
         additional_logical_gib_per_rank=round((full_bytes - old_bytes) / 1024**3, 6),
         physical_allocator_bound=False)


def run_reserved_concurrent_control(source):
    """Model the parent's reservation contract, not its gateway implementation."""
    scheduler, _ = make_case(source, [(20, 10000), (10040, 10000)], PROPOSED_CACHE_TOKENS)
    manager = scheduler.encoder_cache_manager
    reservations = {}

    def reserve(request, rendered_tokens):
        # The fixture knows the complete rendered length. An understated count
        # or an over-capacity admission cannot authorize an engine request.
        if rendered_tokens != request.num_prompt_tokens:
            return False
        if sum(reservations.values()) + rendered_tokens > PROPOSED_CACHE_TOKENS:
            return False
        assert sum(request.get_num_encoder_embeds(i)
                   for i in range(len(request.mm_features))) <= rendered_tokens
        reservations[request.request_id] = rendered_tokens
        return True

    running, waiting, requests = [], [], []
    for index in range(12):
        _, request = make_case(source, [(20, 10000), (10040, 10000)], PROPOSED_CACHE_TOKENS)
        request.request_id = f"vision-{index}"
        for item, feature in enumerate(request.mm_features):
            feature.identifier = f"vision-{index}-image-{item}"
        assert reserve(request, request.num_prompt_tokens)
        requests.append(request)
        if index < 6:
            running.append(request)
        else:
            # Warm first-image encoder entry and a separate decoder prefix hit.
            prior = MetadataRequest(request_id=f"prior-{index}", mm_features=request.mm_features)
            manager.allocate(prior, 0)
            manager.free(prior)
            waiting.append((request, 9984))
    reserved_at_admission = sum(reservations.values())
    assert reserved_at_admission == 252600 <= PROPOSED_CACHE_TOKENS
    _, overflow = make_case(source, [(20, 10000), (10040, 10000)], PROPOSED_CACHE_TOKENS)
    overflow.request_id = "not-admitted"
    assert not reserve(overflow, overflow.num_prompt_tokens - 1)
    assert not reserve(overflow, overflow.num_prompt_tokens)
    assert overflow.request_id not in reservations
    emit("RESERVATION_CONTROL", admitted_requests=12, full_prompt_tokens=reserved_at_admission,
         full_image_embeddings=240000, limit=PROPOSED_CACHE_TOKENS,
         thirteenth_request_rejected=True, understated_reservation_rejected=True)

    max_step_compute = max_step_decoder = peak_cache_slots = 0
    for step in range(512):
        token_budget, compute_budget = BATCH_TOKENS, ENCODER_TOKENS
        scheduled = []
        for request in list(running):
            if token_budget == 0:
                break
            start = request.num_computed_tokens
            ids, before, remaining, external = scheduler._try_schedule_encoder_inputs(
                request, start, min(token_budget, request.num_tokens - start), compute_budget,
            )
            assert not external
            after = scheduler._mamba_block_aligned_split(request, before)
            if after == 0:
                continue
            for input_id in ids:
                manager.allocate(request, input_id)
            if ids:
                compute_budget = remaining
            token_budget -= after
            request.num_computed_tokens += after
            scheduled.append(request)
        while waiting and token_budget > 0:
            request, prefix_hit = waiting[0]
            ids, before, remaining, external = scheduler._try_schedule_encoder_inputs(
                request, prefix_hit, min(token_budget, request.num_tokens - prefix_hit),
                compute_budget,
            )
            assert not external
            after = scheduler._mamba_block_aligned_split(
                request, before, num_new_local_computed_tokens=prefix_hit,
            )
            if after == 0:
                break
            for input_id in ids:
                manager.allocate(request, input_id)
            if ids:
                compute_budget = remaining
            token_budget -= after
            request.num_computed_tokens = prefix_hit + after
            waiting.pop(0)
            running.append(request)
            scheduled.append(request)
        assert scheduled, "Warm-prefix concurrent positive control made no progress"
        assert 0 <= compute_budget <= ENCODER_TOKENS
        assert 0 <= token_budget <= BATCH_TOKENS
        max_step_compute = max(max_step_compute, ENCODER_TOKENS - compute_budget)
        max_step_decoder = max(max_step_decoder, BATCH_TOKENS - token_budget)
        peak_cache_slots = max(peak_cache_slots, PROPOSED_CACHE_TOKENS - manager.num_free_slots)
        for request in scheduled:
            scheduler._free_encoder_inputs(request)
            if request.num_computed_tokens == request.num_tokens:
                assert not manager.get_cached_input_ids(request)
                # Model the first-output/end boundary after consumed-input free.
                reservations.pop(request.request_id)
        running = [request for request in running
                   if request.num_computed_tokens < request.num_tokens]
        assert sum(reservations.values()) <= PROPOSED_CACHE_TOKENS
        if not running and not waiting:
            break
    assert all(request.num_computed_tokens == request.num_tokens for request in requests)
    assert not reservations and manager.num_freeable_slots == PROPOSED_CACHE_TOKENS
    emit("CONCURRENT_CONTROL", completed=12, initial_running=6, initial_waiting=6,
         steps=step + 1, max_step_encoder_embeds=max_step_compute,
         max_step_decoder_tokens=max_step_decoder, peak_cache_slots=peak_cache_slots,
         final_freeable=manager.num_freeable_slots)


def run_large_cache_compute_blocker(source):
    scheduler, request = make_case(
        source, [(20, 10000), (10040, 10000)], PROPOSED_CACHE_TOKENS,
    )
    request.request_id = "uncached-images-prefix-hit"
    assert request.num_prompt_tokens == 21050 <= PROPOSED_CACHE_TOKENS
    assert sum(request.get_num_encoder_embeds(i) for i in range(2)) == 20000
    for attempt in range(3):
        ids, before, remaining, external = scheduler._try_schedule_encoder_inputs(
            request, 9984, BATCH_TOKENS, ENCODER_TOKENS,
        )
        after = scheduler._mamba_block_aligned_split(
            request, before, num_new_local_computed_tokens=9984,
        )
        # Both encoder entries are absent, even though the decoder prefix hits.
        # Image 1 is planned; image 2 exceeds *compute*, not cache, capacity.
        # Alignment returns zero before image 1 can be committed to the cache.
        assert ids == [0] and before == 56 and after == 0 and remaining == 6817
        assert not external and request.num_computed_tokens == 0
        assert not scheduler.encoder_cache_manager.cached
        assert scheduler.encoder_cache_manager.num_freeable_slots == PROPOSED_CACHE_TOKENS
        emit("LARGE_CACHE_COMPUTE_BLOCKER", attempt=attempt,
             reserved_full_prompt_tokens=request.num_prompt_tokens,
             cache_freeable=scheduler.encoder_cache_manager.num_freeable_slots,
             encoder_inputs_planned=ids, encoder_compute_remaining=remaining,
             after_encoder_gate=before, after_mamba_alignment=after,
             allocated_encoder_entries=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="k3", help="already-running pinned container")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Run without -O: assertions are the reproducibility checks")
    root, preprocessor_path = inspect_source(args.container)
    source = load_source(root)
    check_profile(source, preprocessor_path)
    running, request, stalled = run_running_case(
        source, "two_images_align768_cache16817", [(20, 10000), (10040, 10000)], trace=True,
    )
    assert stalled and request.num_computed_tokens == 9984
    assert running.encoder_cache_manager.num_freeable_slots == 6817
    assert running.encoder_cache_manager.get_cached_input_ids(request) == {0}
    _, _, stalled = run_running_case(
        source, "single_16817_embed_image", [(20, ENCODER_TOKENS)],
    )
    assert not stalled
    _, _, stalled = run_running_case(
        source, "two_images_without_alignment", [(20, 10000), (10040, 10000)], aligned=False,
    )
    assert not stalled
    _, _, stalled = run_running_case(
        source, "two_images_cache20000", [(20, 10000), (10040, 10000)], capacity=20000,
    )
    assert not stalled
    waiting, waiting_request = run_waiting_case(source)
    check_abort_cleanup(source, running, request, "running")
    check_abort_cleanup(source, waiting, waiting_request, "waiting")
    check_capacity_profile_and_memory(source, preprocessor_path)
    _, _, stalled = run_running_case(
        source, "two_images_cache262144", [(20, 10000), (10040, 10000)],
        capacity=PROPOSED_CACHE_TOKENS,
    )
    assert not stalled
    run_reserved_concurrent_control(source)
    run_large_cache_compute_blocker(source)
    forbidden = {"torch", "vllm", "transformers"}
    assert not {name.split(".")[0] for name in sys.modules} & forbidden
    print("PASS: running + waiting deadlocks; single-image + isolation controls; "
          "abort cleanup; no vLLM/torch/transformers imports")
    print("REVIEW BLOCKED: capacity-only growth plus full-prompt reservations "
          "does not resolve the cold-encoder-prefix compute/alignment deadlock")


if __name__ == "__main__":
    main()
