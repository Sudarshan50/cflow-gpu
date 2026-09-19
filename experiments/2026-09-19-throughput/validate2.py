"""Version-tolerant config validator. Works on both the kimi-k3 dev build and
v0.29.0, where entrypoints.openai.cli_args moved to entrypoints.launchers.
"""
import sys, glob, os

try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
except ModuleNotFoundError:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
import vllm

print("vllm", vllm.__version__)

KEYS = ("max_num_batched_tokens", "max_num_seqs", "max_model_len",
        "kv_offloading_size", "kv_offloading_backend", "async_scheduling",
        "long_prefill_token_threshold", "gpu_memory_utilization",
        "enable_prefix_caching", "prefix_match_unit", "tensor_parallel_size",
        "enable_expert_parallel", "load_format", "reasoning_parser",
        "tool_call_parser")

rc = 0
for cfg in sorted(glob.glob("/results/abcfg/*.yaml")):
    name = os.path.basename(cfg)
    parser = make_arg_parser(FlexibleArgumentParser())
    try:
        args = parser.parse_args(["moonshotai/Kimi-K3", "--config", cfg])
    except SystemExit as e:
        print(f"FAIL  {name}: parser exited {e.code}"); rc = 1; continue
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}"); rc = 1; continue
    got = {k: getattr(args, k) for k in KEYS if hasattr(args, k)}
    missing = [k for k in KEYS if not hasattr(args, k)]
    print(f"OK    {name}")
    for k, v in got.items():
        print(f"        {k} = {v}")
    if missing:
        print(f"      MISSING ATTRS: {missing}")
sys.exit(rc)
