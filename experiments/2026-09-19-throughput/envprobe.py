"""Compare the AITER / ROCm env-var surface between images.

Section 11.1 of K3-DEPLOYMENT.md warns the AITER flag contract changed between
builds and that carrying the old vars forward verbatim is the most likely way
to recreate the silent-garbage failure of 7.3. This enumerates what each build
actually reads so the comparison is evidence, not assumption.
"""
import vllm, vllm.envs as envs

print("vllm", vllm.__version__)

names = sorted(n for n in dir(envs) if n.isupper())
rocm = [n for n in names if "ROCM" in n or "AITER" in n or "MOE" in n
        or "CUDAGRAPH" in n]
print(f"--- {len(rocm)} ROCm/AITER/MoE env vars recognised by vllm.envs ---")
for n in rocm:
    try:
        v = getattr(envs, n)
    except Exception as e:
        v = f"<error {e}>"
    print(f"  {n} = {v!r}")

WE_SET = ["VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE",
          "AITER_SITUV2_A8W4", "AITER_BF16_FP8_MOE_BOUND",
          "VLLM_USE_BREAKABLE_CUDAGRAPH", "SAFETENSORS_FAST_GPU",
          "VLLM_ROCM_USE_AITER_MOE_SITUV2"]
print("--- vars our env-file sets, vs whether this build knows them ---")
for n in WE_SET:
    known = hasattr(envs, n)
    print(f"  {n:34s} known_to_vllm_envs={known}")
