#!/usr/bin/env bash

export ENV_NAME="${ENV_NAME:-llava}"
if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    echo "INFO: Using HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\"" >&2
    mkdir -p "${HUGGINGFACE_HUB_CACHE}" || { echo "ERROR: Could not create HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE}\" (exit: $?)"; exit 1; }
    echo "INFO: HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\" (avail: $(df -h --output=avail -h "${HUGGINGFACE_HUB_CACHE}" | tail -n 1 || true))" >&2
else
    lev=WARNING
    [ "${IGNORE_UNSET_HUGGINGFACE_HUB_CACHE:-0}" = 0 ] && lev=ERROR 
    echo "${lev}: HUGGINGFACE_HUB_CACHE not set!" >&2
    echo "${lev}: Please set the environment variable HUGGINGFACE_HUB_CACHE explicity or set IGNORE_UNSET_HUGGINGFACE_HUB_CACHE=1. This is because it takes up a lot of space!" >&2
    [ "${IGNORE_UNSET_HUGGINGFACE_HUB_CACHE:-0}" = 0 ] && exit 1
fi

ngpu="$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo 0)"
if [ "${ngpu}" -gt 0 ]; then
    echo "INFO: Found ${ngpu} GPU(s) on this machine." >&2
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "INFO: Using CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\"" >&2
else
    echo "WARNING: No GPU found on this machine." >&2
fi

printf "Started at " && date -Is >&2
exec micromamba run -n "${ENV_NAME}" "$@"
