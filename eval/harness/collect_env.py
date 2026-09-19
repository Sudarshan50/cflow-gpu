#!/usr/bin/env python3
"""Capture the facts a reader needs to trust the health-check numbers.

The reference report opens with a platform/gateway/model/channel table. This is
the equivalent for a self-hosted deployment: what hardware, what quantisation,
what serve arguments, what the edge policy is. Everything here is read from the
running system rather than from documentation, so a stale doc cannot make the
report wrong.

No credential is emitted: vllm-k3.env is copied with VLLM_API_KEY masked, and
customer keys are reduced to names only.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

OUT_KEYS = ("hardware", "os", "gpu", "model", "engine", "edge", "services",
            "storage", "customers", "versions")


def sh(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout)
        return p.stdout.strip()
    except Exception as e:
        return "ERROR %s: %s" % (type(e).__name__, e)


def first(pattern, text, cast=str, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


BANNER_PATTERNS = (
    "Initializing a V1 LLM engine|quantization=|GPU KV cache size|"
    "Maximum concurrency for|Available KV cache memory|Loading weights took|"
    "AITER_MXFP4|non-default args|enable_chunked_prefill"
)


def docker_logs():
    """Grep the whole container log, not its tail.

    The startup banner carries quantisation, KV sizing and backend selection,
    and it scrolls out of any fixed tail within minutes of serving traffic.
    """
    return sh("docker logs k3 2>&1 | grep -E %s" % json.dumps(BANNER_PATTERNS),
              timeout=120)


def collect():
    env = {"collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    lscpu = sh("lscpu")
    env["hardware"] = {
        "cpu_model": first(r"Model name:\s+(.+)", lscpu),
        "vcpus": first(r"^CPU\(s\):\s+(\d+)", lscpu, int),
        "numa_nodes": first(r"NUMA node\(s\):\s+(\d+)", lscpu, int),
        "mem_total": first(r"Mem:\s+(\S+)", sh("free -h")),
        "hostname": sh("hostname"),
        "virtualisation": first(r"Hypervisor vendor:\s+(.+)", lscpu),
    }

    env["os"] = {
        "pretty_name": first(r'PRETTY_NAME="([^"]+)"', sh("cat /etc/os-release")),
        "kernel": sh("uname -r"),
        "rocm_packages": sh("dpkg -l | grep -c amdrocm"),
        "rocm_version": first(r"(\d+\.\d+\.\d+)", sh("dpkg -l amdrocm 2>/dev/null | tail -1")),
    }

    smi = sh("rocm-smi --showid")
    ids = sorted(set(re.findall(r"^GPU\[(\d+)\]", smi, re.M)))
    hw = sh("rocm-smi --showhw")
    vram = sh("rocm-smi --showmeminfo vram")
    totals = re.findall(r"VRAM Total Memory \(B\): (\d+)", vram)
    used = re.findall(r"VRAM Total Used Memory \(B\): (\d+)", vram)
    env["gpu"] = {
        "count": len(ids),
        "device_name": first(r"Device Name:\s+(.+)", smi),
        "device_ids": sorted(set(re.findall(r"Device ID:\s+(0x[0-9a-f]+)", smi))),
        "gfx_arch": first(r"(gfx\d+\w*)", hw),
        "vbios": first(r"(\d{3}-M\S+)", hw),
        "vram_total_gib_per_card": round(int(totals[0]) / 2**30, 1) if totals else None,
        "vram_used_gib_per_card": [round(int(u) / 2**30, 1) for u in used] or None,
    }

    logs = docker_logs()
    env["engine"] = {
        "vllm_version": first(r"Initializing a V1 LLM engine \(v([^)]+)\)", logs),
        "quantization": first(r"quantization=(\w+)", logs),
        "kv_cache_dtype": first(r"kv_cache_dtype=(\w+)", logs),
        "dtype": first(r"dtype=torch\.(\w+)", logs),
        "tensor_parallel_size": first(r"tensor_parallel_size=(\d+)", logs, int),
        "enable_expert_parallel": "enable_expert_parallel" in logs,
        "max_model_len": first(r"max_seq_len=(\d+)", logs, int),
        "kv_cache_tokens": first(r"GPU KV cache size: ([\d,]+) tokens", logs,
                                 lambda v: int(v.replace(",", ""))),
        "kv_cache_memory_gib": first(r"Available KV cache memory: ([\d.]+) GiB",
                                     logs, float),
        "max_concurrency_x": first(r"Maximum concurrency for [\d,]+ tokens per "
                                   r"request: ([\d.]+)x", logs, float),
        "weight_load_seconds": first(r"Loading weights took ([\d.]+) seconds",
                                     logs, float),
        "moe_backend": first(r"Using (AITER_MXFP4_\w+) for", logs),
        "enable_chunked_prefill": first(r"enable_chunked_prefill=(\w+)", logs),
        # Appears as max_num_batched_tokens=8192 in the engine config banner and
        # as 'max_num_batched_tokens': 8192 in the non-default-args line.
        "max_num_batched_tokens": first(
            r"max_num_batched_tokens'?[=:]\s*(\d+)", logs, int),
        "image_digest": first(r"sha256:([0-9a-f]{64})",
                              sh("docker inspect k3 --format '{{.Image}}'")),
        "aiter_flags": {},
    }
    denv = sh("docker inspect k3 --format '{{range .Config.Env}}{{println .}}{{end}}'")
    for line in denv.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.startswith(("AITER", "VLLM_ROCM", "SAFETENSORS", "HF_")):
            env["engine"]["aiter_flags"][k] = v

    served = []
    try:
        with open("/scratch/deploy/api-key.txt") as f:
            key = f.read().strip()
        req = urllib.request.Request("http://127.0.0.1:8001/v1/models",
                                     headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        served = [{"id": m.get("id"), "max_model_len": m.get("max_model_len"),
                   "root": m.get("root")} for m in d.get("data", [])]
    except Exception as e:
        served = [{"error": "%s: %s" % (type(e).__name__, e)}]
    env["model"] = {
        "repo": "moonshotai/Kimi-K3",
        "served_names": served,
        "weights_bytes": sh("du -sbL /scratch/hf/hub/models--moonshotai--Kimi-K3 "
                            "2>/dev/null | cut -f1"),
        "weights_human": sh("du -shL /scratch/hf/hub/models--moonshotai--Kimi-K3 "
                            "2>/dev/null | cut -f1"),
        "safetensors_files": sh("find /scratch/hf/hub/models--moonshotai--Kimi-K3 "
                                "-name '*.safetensors' | wc -l"),
        "revision": sh("cat /scratch/hf/hub/models--moonshotai--Kimi-K3/refs/main "
                       "2>/dev/null"),
    }

    env["edge"] = {
        "public_url": "https://cflox.store/v1",
        "nginx_version": sh("nginx -v 2>&1"),
        "cert_not_after": first(r"notAfter=(.+)", sh(
            "openssl x509 -enddate -noout -in "
            "/etc/letsencrypt/live/cflox.store/fullchain.pem 2>/dev/null")),
        "cert_issuer": first(r"issuer=(.+)", sh(
            "openssl x509 -issuer -noout -in "
            "/etc/letsencrypt/live/cflox.store/fullchain.pem 2>/dev/null")),
        "certbot_timer": sh("systemctl is-enabled certbot.timer 2>/dev/null"),
        "per_customer_conn_cap": first(r"limit_conn k3_percust (\d+)", sh(
            "cat /etc/nginx/conf.d/k3-limits.inc 2>/dev/null"), int),
        "global_conn_cap": first(r"limit_conn k3_global\s+(\d+)", sh(
            "cat /etc/nginx/conf.d/k3-limits.inc 2>/dev/null"), int),
        "per_customer_rate": first(r"rate=(\S+);", sh(
            "cat /etc/nginx/conf.d/00-k3-keys.conf /etc/nginx/conf.d/k3-limits.inc "
            "2>/dev/null")),
        "vllm_bind": sh("ss -lntH 'sport = :8001' | awk '{print $4}'"),
    }

    env["services"] = {}
    for unit in ("k3.service", "k3dash.service", "nginx.service", "docker.service",
                 "certbot.timer"):
        env["services"][unit] = {
            "active": sh("systemctl is-active %s 2>/dev/null" % unit),
            "enabled": sh("systemctl is-enabled %s 2>/dev/null" % unit),
            "since": sh("systemctl show %s -p ActiveEnterTimestamp --value "
                        "2>/dev/null" % unit),
        }

    env["storage"] = {
        "root_fs": sh("df -h / | tail -1"),
        "scratch_on": "boot disk" if sh("mountpoint -q /scratch && echo m")
                      != "m" else "separate volume",
        "bulk_volume": sh("df -h /mnt/bulk 2>/dev/null | tail -1"),
    }

    names = []
    try:
        with open("/scratch/deploy/customers.tsv") as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#") \
                        and "\t" in line:
                    names.append(line.split("\t")[0].strip())
    except OSError:
        pass
    env["customers"] = {"count": len(names), "names": names,
                        "note": "key values deliberately omitted"}

    env["versions"] = {
        "python": sys.version.split()[0],
        "docker": sh("docker --version"),
        "repo_commit": sh("git -C /scratch/deploy rev-parse --short HEAD"),
        "repo_dirty": bool(sh("git -C /scratch/deploy status --porcelain")),
    }
    return env


def masked_env_file(path="/scratch/deploy/vllm-k3.env"):
    out = []
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("VLLM_API_KEY="):
                    out.append("VLLM_API_KEY=sk-k3-<redacted>\n")
                else:
                    out.append(line)
    except OSError as e:
        out = ["unreadable: %s\n" % e]
    return "".join(out)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    env = collect()
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(out_dir, "vllm-k3.env.redacted"), "w") as f:
        f.write(masked_env_file())

    lines = []
    for key in OUT_KEYS:
        lines.append("== %s ==" % key)
        lines.append(json.dumps(env.get(key), indent=2))
        lines.append("")
    with open(os.path.join(out_dir, "env.txt"), "w") as f:
        f.write("\n".join(lines))
    print("environment captured: %d GPUs, quantization=%s, kv=%s, kv_tokens=%s"
          % (env["gpu"]["count"], env["engine"]["quantization"],
             env["engine"]["kv_cache_dtype"], env["engine"]["kv_cache_tokens"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
