"""Parse each candidate config through vLLM's real CLI path.

Catches unknown/renamed flags in seconds instead of after a 6-minute boot.
Argv-level only: no GPU, no weights, no engine.
"""
import sys, glob, os
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

rc = 0
for cfg in sorted(glob.glob("/scratch/ab/configs/*.yaml")):
    name = os.path.basename(cfg)
    parser = make_arg_parser(FlexibleArgumentParser())
    argv = ["serve", "moonshotai/Kimi-K3", "--config", cfg]
    try:
        args = parser.parse_args(argv[1:] if argv[0] == "serve" else argv)
    except SystemExit as e:
        print(f"FAIL  {name}: parser exited {e.code}")
        rc = 1
        continue
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        rc = 1
        continue
    interesting = {
        k: getattr(args, k)
        for k in ("max_num_batched_tokens", "max_num_seqs", "max_model_len",
                  "kv_offloading_size", "kv_offloading_backend",
                  "async_scheduling", "long_prefill_token_threshold",
                  "gpu_memory_utilization")
        if hasattr(args, k)
    }
    print(f"OK    {name}: {interesting}")

sys.exit(rc)
