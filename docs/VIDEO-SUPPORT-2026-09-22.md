# Kimi K3 video support

## Deployment status

Video input is enabled on the customer serving path:

`nginx → LiteLLM :4000 → gateway :8002 → vLLM :8001`

The deployed vLLM Kimi K3 adapter is image-only even though the checkpoint is
video-native. Direct requests to vLLM `:8001` therefore still reject
`video_url`. The gateway provides bounded compatibility support by decoding
video into ordered, timestamped image frames before forwarding the request.

## Production limits

- At most 1 video per request and 8 uniformly sampled frames per video.
- Longest frame edge is 768 pixels; aspect ratio is preserved.
- Source frame rate is never increased, so low-FPS videos are not duplicated.
- At most 2 video decodes run concurrently.
- Decode timeout is 20 seconds.
- Remote media downloads are limited to 15 MiB.
- Inline video bytes are limited to 48 MiB; the gateway request-body limit is
  64 MiB.
- Decoder input is first written to a local temporary file. ffmpeg receives no
  remote URL and runs as the unprivileged gateway user.
- Video requests retain the existing conservative 262,144-token admission
  reservation.

Accepted OpenAI-compatible content types include `video_url`, `video`, and
`input_video`. Base64 data URLs and HTTP(S) URLs are normalized by the gateway.

## Verification

On 22 September 2026:

- 307 redesign tests passed (16 skipped).
- A 4-second 1280×720, 24 FPS clip produced 8 frames at 768×432 with timestamps
  from 0.0 through 3.5 seconds.
- A red-to-blue live video request through the gateway returned the correct
  ordered colors in 2.8 seconds. It used 2,638 prompt tokens, including 2,456
  multimodal image tokens.
- A 1-second, 4 FPS green clip through LiteLLM returned `Green`.
- After preventing low-FPS duplication, that request fell from 909 to 469
  prompt tokens and completed in 0.4 seconds.

The engine process and model weights were not restarted during rollout.
