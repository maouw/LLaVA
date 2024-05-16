#!/usr/bin/env bash
set -ef -o pipefail

# If the file /etc/arg_mamba_user exists and its contents don't match $MAMBA_USER...
if [[ -f /etc/arg_mamba_user && "${MAMBA_USER}" != "$(cat "/etc/arg_mamba_user")" ]]; then
    echo "ERROR: This micromamba-docker image was built with" \
    "'ARG MAMBA_USER=$(cat "/etc/arg_mamba_user")', but the corresponding" \
    "environment variable has been modified to 'MAMBA_USER=${MAMBA_USER}'." \
    "For instructions on how to properly change the username, please refer" \
    "to the documentation at <https://github.com/mamba-org/micromamba-docker>." >&2
    exit 1
fi

# if USER is not set and not root
if [[ ! -v USER && $(id -u) -gt 0 ]]; then
  # should get here if 'docker run...' was passed -u with a numeric UID
  export USER="$MAMBA_USER"
  export HOME="/home/$USER"
fi

source _activate_current_env.sh

if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
    echo "Using HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\"" >&2
else
    echo "HUGGINGFACE_HUB_CACHE not set!" >&2
    if [ "${IGNORE_UNSET_HUGGINGFACE_HUB_CACHE:-0}" = 0 ]; then
        echo "Please set the environment variable HUGGINGFACE_HUB_CACHE explicity or set IGNORE_UNSET_HUGGINGFACE_HUB_CACHE=1. This is because it takes up a lot of space!" >&2
        exit 1
    fi

fi
printf "Started at " && date -Is >&2
exec "$@"
