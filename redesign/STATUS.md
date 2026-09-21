# Production status — 2026-09-20

```
NewAPI → nginx :443 → LiteLLM :4000 → gateway :8002 → vLLM :8001
```

| Component | Recorded configuration |
|---|---|
| Engine | AMD optimized image, digest `5f3007aff1bc…` |
| Parallelism | TP8, expert parallelism disabled |
| Limits | 262,144-token context, 64 sequences, 4,096 batched tokens |
| Memory/cache | Utilization 0.88; auto/BF16 KV; GPU prefix caching enabled |
| Speculation | DSpark tested and disabled for the concurrent production workload |
| Gateway | Global ceiling 64; class caps still apply |
| LiteLLM | 32 workers; scoped response caching and virtual-key authentication |
| Public edge | `https://api.cflowx.in/v1`; no direct-gateway authentication bypass |

Active engine configuration is
`/scratch/deploy-state/amd-optimized/config-base.yaml`, selected by the
`60-amd-optimized.conf` systemd override. Repository-root `config.yaml` and
`/scratch/hf/config.yaml` are stock/reference profiles, not the active optimized
configuration.

See [measured rollout results](../experiments/2026-09-20-amd-optimized/RESULTS.md)
for the exact image, profiles, tests, cache behavior and rollback details.

Known remaining work includes the static agentic admission/output limits,
Responses compatibility in the deployed control plane, template-aware token
counting and cancellation. Source changes are not deployed merely by committing
or pushing them.

The SGLang profiles, earlier capacity studies and dated audit reports are retained
as experiments/history. Host KV offload and external fallback are not active.
