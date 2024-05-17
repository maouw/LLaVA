#!/usr/bin/env bash
set -eE -o pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRCFILE="${1:-${SCRIPT_DIR}/../docs/MODEL_ZOO.md}"
DSTFILE="${2:-${SCRIPT_DIR}/../llava/suggested_models.py}"
[ "${DSTFILE:-}" = '-' ] && DSTFILE=/dev/stdout
found_models=($(grep -Po '(?<=huggingface.co/)liuhaotian/llava-v[0-9]+\.[5-9][^[:space:]/]+(?=[)])' "${SRCFILE}" | grep -v pretrain || true))
if [ "${#found_models[@]}" -eq 0 ]; then
	echo "No models found in ${SRCFILE}. Exiting."
	exit 1
fi

python3 -c 'import sys; print(f"SUGGESTED_MODELS = {sys.argv[1:]}")' "${found_models[@]}" | tee "${DSTFILE}"
