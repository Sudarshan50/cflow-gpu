"""Concrete engine probes. Adding an engine means adding a subclass here."""

from __future__ import annotations

import argparse

from .base import EngineProbe, FlagSpec


class VllmProbe(EngineProbe):
    name = "vllm"

    flags = (
        FlagSpec("decode_context_parallel_size", "A1  DCP, shards the MLA latent by token"),
        FlagSpec("tensor_parallel_size", "    baseline parallelism"),
        FlagSpec("enable_expert_parallel", "    MoE expert parallelism"),
        FlagSpec("kv_cache_dtype", "A2  fp8 KV"),
        FlagSpec("kv_transfer_config", "A3  CPU/NVMe offload tier"),
        FlagSpec("kv_offloading_size", "A3  host DRAM L2 size (GiB)"),
        FlagSpec("kv_offloading_backend", "A3  host DRAM L2 backend"),
        FlagSpec("max_num_seqs", "B1  admission ceiling"),
        FlagSpec("scheduling_policy", "B2  priority scheduling"),
        FlagSpec("long_prefill_token_threshold", "B3  long-prefill cap"),
    )

    a1_flags = ("decode_context_parallel_size",)

    scan_modules = (
        "vllm.v1.attention.backends.mla.common",
        "vllm.distributed.parallel_state",
        "vllm.config.parallel",
    )

    def cli_surface(self) -> set[str]:
        from vllm.engine.arg_utils import EngineArgs

        parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
        return {action.dest for action in parser._actions}


class SglangProbe(EngineProbe):
    name = "sglang"

    flags = (
        FlagSpec("enable_dp_attention", "A1  DP attention, partitions KV by sequence"),
        FlagSpec("dp_size", "A1  attention DP width"),
        FlagSpec("enable_prefill_cp", "A1  prefill context parallelism"),
        FlagSpec("attn_cp_size", "A1  CP width"),
        FlagSpec("cp_strategy", "A1  CP layout; interleave asserts dp_size == 1"),
        FlagSpec("tp_size", "    baseline parallelism"),
        FlagSpec("kv_cache_dtype", "A2  fp8 KV"),
        FlagSpec("mamba_full_memory_ratio", "    hybrid dual-pool split, static"),
        FlagSpec("enable_hierarchical_cache", "A3  HiCache L2"),
        FlagSpec("hicache_ratio", "A3  host pool multiplier"),
        FlagSpec("hicache_storage_backend", "A3  L3 backend"),
        FlagSpec("max_running_requests", "B1  ceiling, floor-divided by dp_size"),
        FlagSpec("context_length", "    context window"),
        FlagSpec("chunked_prefill_size", "C2  prefill chunk"),
        FlagSpec("schedule_policy", "B2  scheduling policy"),
        FlagSpec("enable_priority_scheduling", "B2  honour request priority"),
        FlagSpec("schedule_low_priority_values_first", "B2  0=P0, matching gateway"),
        FlagSpec("retraction_policy", "B2  retract lower priority first"),
        FlagSpec("attention_backend", "C1  AITER on MI35x"),
        FlagSpec("mem_fraction_static", "    HBM split; auto 0.67 is below this model's weights"),
        FlagSpec("language_model_only", "    drop the vision tower so KV/mamba fit"),
        FlagSpec("cuda_graph_max_bs_decode", "    graph cap; default 512 does not fit"),
        FlagSpec("mamba_ssm_dtype", "    bf16 halves the 428 MB hybrid slot"),
        FlagSpec("disable_decode_cuda_graph", "    skip 10 GB pre-capture reserve"),
        FlagSpec("disable_prefill_cuda_graph", "    skip 10 GB pre-capture reserve"),
        FlagSpec("enable_metrics", "    /metrics for gateway health scrape"),
    )

    a1_flags = ("enable_dp_attention", "enable_prefill_cp")

    scan_modules = (
        "sglang.srt.layers.dp_attention",
        "sglang.srt.managers.schedule_batch",
    )

    def cli_surface(self) -> set[str]:
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return {action.dest for action in parser._actions}


REGISTRY: dict[str, type[EngineProbe]] = {
    VllmProbe.name: VllmProbe,
    SglangProbe.name: SglangProbe,
}
