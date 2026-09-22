ARG BASE_IMAGE=johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8
FROM ${BASE_IMAGE}

ARG AITER_COMMIT=5337bf2cdb88
ENV GPU_ARCHS=gfx950 \
    MAX_JOBS=64 \
    PREBUILD_KERNELS=0

# PR 5702 changes only the two Kimi-K3 BF16 tables. The production image has
# newer Kimi-specific FlyDSL modules that upstream 5337bf2c does not contain,
# so retain that compatible runtime and backport the pinned table payload.
# Keep the full source tree because the official tuner is not in the wheel.
RUN git clone --recursive https://github.com/ROCm/aiter.git /opt/aiter-candidate \
 && cd /opt/aiter-candidate \
 && git checkout "${AITER_COMMIT}" \
 && git submodule update --init --recursive \
 && install -m 0644 aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv \
      /usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/ \
 && install -m 0644 aiter/configs/model_configs/kimik3_bf16_untuned_gemm.csv \
      /usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/

LABEL org.cflowx.k3.aiter-candidate="5337bf2cdb88" \
      org.cflowx.k3.aiter-integration="pr5702-table-backport" \
      org.cflowx.k3.base-image="sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8"
