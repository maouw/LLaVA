#!/usr/bin/env bash

set -Eu -o pipefail
[[ "${XTRACE:-0}" =~ ^[1yYtT] ]] && set -x
eval "$("${MAMBA_EXE}" shell hook --shell=bash)"
micromamba activate "${ENV_NAME}"
python -m llava.eval.run_llava "$@"
