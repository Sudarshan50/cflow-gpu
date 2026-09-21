"""Scrape GPU prefix vs host-offload counters.

    python3 -m redesign.probe.offload --url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


def scrape(url: str) -> dict[str, float]:
    text = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10).read().decode()
    out: dict[str, float] = {}
    for name in (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:num_preemptions_total",
        "vllm:kv_cache_usage_perc",
    ):
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*}}\s+([0-9.eE+-]+)", text, re.M)
        if m:
            out[name] = float(m.group(1))
    for line in text.splitlines():
        if line.startswith("vllm:kv_offload_total_bytes_total"):
            value = float(line.split()[-1])
            if "CPU_to_GPU" in line:
                out["cpu_to_gpu_bytes"] = value
            elif "GPU_to_CPU" in line:
                out["gpu_to_cpu_bytes"] = value
        if line.startswith("vllm:cache_config_info") and "kv_offloading_size" in line:
            m = re.search(r'kv_offloading_size="([^"]+)"', line)
            if m and m.group(1) not in ("None", ""):
                out["kv_offloading_size"] = float(m.group(1))
            elif m:
                out["kv_offloading_size"] = 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.probe.offload")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args(argv)
    print(json.dumps(scrape(args.url), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
