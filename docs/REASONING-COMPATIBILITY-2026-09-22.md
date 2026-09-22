# Kimi K3 reasoning compatibility

## Deployment status

The public Chat and Responses paths support K3 reasoning controls and usage:

- Chat accepts `reasoning_effort`, `thinking_effort`, `reasoning` objects,
  `thinking` objects, and `enable_thinking`.
- Chat responses expose both `reasoning` and `reasoning_content`.
- Non-streaming and streaming Chat usage exposes
  `completion_tokens_details.reasoning_tokens`.
- Responses usage exposes `output_tokens_details.reasoning_tokens`.

The vLLM Chat serving patch counts generated token IDs inside K3's XTML
`think` channel. It does not estimate from output text.

## Live verification

On 22 September 2026:

- Direct vLLM reported 25 reasoning tokens in a 39-token completion.
- Public Chat reasoning objects produced both reasoning response aliases.
- Disabling reasoning through either `reasoning.enabled`,
  `thinking.type`, or `enable_thinking` produced no reasoning and reported
  zero reasoning tokens.
- Streaming Chat included both reasoning aliases and terminal reasoning usage.
- Responses returned a reasoning output item and non-null reasoning usage.
- 337 redesign tests passed.

The model, gateway, and LiteLLM services were restarted to activate the fixes.
