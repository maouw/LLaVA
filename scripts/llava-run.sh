#!/usr/bin/env bash

set -Eu -o pipefail
[[ "${XTRACE:-0}" =~ ^[1yYtT] ]] && set -x

python -m llava.eval.run_llava "$@"
