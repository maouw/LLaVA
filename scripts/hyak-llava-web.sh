#!/usr/bin/env bash

set -Eu -o pipefail
[[ "${XTRACE:-0}" =~ ^[1yYtT] ]] && set -x

# This script is used to start the LLaVA gradio web server on Hyak.

# Set the following variables to your own values:

# MODEL_PATHS: a list of model paths, separated by space (default: liuhaotian/llava-v1.5-7b)
#   Available models include, but are not limited to:
#     - liuhaotian/llava-v1.5-7b
#     - liuhaotian/llava-v1.5-13b
#     - liuhaotian/llava-v1.5-7b-lora
#     - liuhaotian/llava-v1.5-13b-lora
#   See https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md for more details.
MODEL_PATHS="${MODEL_PATHS:-liuhaotian/llava-v1.5-7b}"
# LOCAL_HTTP_PORT: the port number to print for the local HTTP server SSH tunnel command (default: 8000)
LOCAL_HTTP_PORT="${LOCAL_HTTP_PORT:-8000}"

# GRADIO_CONTROLLER_PORT: the port number for the gradio controller (or leave it empty to use a random port)
if [[ -z "${GRADIO_CONTROLLER_PORT:-}" ]]; then
    GRADIO_CONTROLLER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')" || { echo "Failed to get a random port. Exiting."; exit 1; }
fi

# GRADIO_WEB_PORT: the port number for the gradio controller (or leave it empty to use a random port)
if [[ -z "${GRADIO_WEB_PORT:-}" ]]; then
    GRADIO_WEB_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')" || { echo "Failed to get a random port. Exiting."; exit 1; }
fi

# Launch the controller:
python -m llava.serve.controller --host 0.0.0.0 --port "${GRADIO_CONTROLLER_PORT}"  2> >(grep -v 'INFO' || true >&2) 1> >(grep -v 'INFO' || true >&1) &
sleep 5

# Launch the gradio web server:
python -m llava.serve.gradio_web_server --controller "http://localhost:${GRADIO_CONTROLLER_PORT}" --port "${GRADIO_WEB_PORT}" --model-list-mode reload 2> >(grep -v 'INFO' || true >&2) 1> >(grep -v 'INFO' || true >&1) &
sleep 5

# Launch the model workers:

declare -a _model_paths
readarray -t _model_paths <<<"${MODEL_PATHS:-}"
declare -a _model_ports=()

for model_path in "${_model_paths[@]}"; do
    model_port="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')" || { echo "Failed to get a random port. Exiting."; exit 1; }
    _model_ports+=("${model_port}")
    python -m llava.serve.model_worker --host 0.0.0.0 --controller "http://localhost:${GRADIO_CONTROLLER_PORT}" --port "${model_port}" --worker "http://localhost:${model_port}" --model-path "${model_path}" 2> >(grep -v 'INFO' || true >&2) 1> >(grep -v 'INFO' || true >&1) &
    sleep 5
done

printf "To access the gradio web server, run the following command on your local machine:\n"
printf 'ssh -o StrictHostKeyChecking=no -N -L %d:localhost:%d -J %s@klone.hyak.uw.edu %s@%s\n' "${LOCAL_HTTP_PORT}" "${GRADIO_WEB_PORT}" "${USER}" "${USER}" "$(uname -n)"

# Keep the container running:
while true; do
	sleep 1
done
