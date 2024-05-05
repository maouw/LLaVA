Bootstrap: docker
From: mambaorg/micromamba:jammy

%arguments
    CUDA_VERSION=11.7
    CUDA_ARCHITECTURES=sm_86 sm_89
    ENV_NAME=llava
    INSTALL_TRAINING_TOOLS=0

%setup
    mkdir -p "${APPTAINER_ROOTFS}/.setup"

%files
    ./ /opt/.setup

%post
    set -exu
    export DEBIAN_FRONTEND=noninteractive
    export ENV_NAME="{{ ENV_NAME }}"
    INSTALL_TRAINING_TOOLS="{{ INSTALL_TRAINING_TOOLS }}"
    CUDA_VERSION="{{ CUDA_VERSION }}"
    CUDA_ARCHITECTURES="{{ CUDA_ARCHITECTURES }}"
    export TORCH_ARCH_CUDA_LIST="${CUDA_ARCHITECTURES}"
    export CI=1

    cd /opt/.setup

    export LANG=C.UTF-8 LC_ALL=C.UTF-8

    # Install LLaVA dependencies:
    micromamba create --verbose -y -n "${ENV_NAME}" -f environment.yml && rm environment.yml

    # Show Python version:
    micromamba run -n "${ENV_NAME}" python --version >&2

    if [ "${INSTALL_TRAINING_TOOLS:-0}" != 0 ]; then
        micromamba install --verbose -y -n "${ENV_NAME}" nvidia/label/cuda-12.1.1::cuda-nvcc conda-forge::deepspeed conda-forge::ninja
        micromamba install --verbose -y -n "${ENV_NAME}" "${ENV_NAME}" python -m pip install --no-cache-dir flash-attn --no-build-isolation
    fi

    # Install LLaVA:
    micromamba run -n "${ENV_NAME}" python -m pip install --no-deps --no-cache-dir --config-settings="--install-data=$PWD/llava" .

    # Copy executables:
    cp scripts/hyak-llava-web.sh "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/bin/hyak-llava-web"
    cp scripts/llava-run.sh "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/bin/llava-run"

	# Clean up:
	micromamba run -n "${ENV_NAME}" python -m pip cache purge
	micromamba clean --all --yes
    cd /opt
	rm -rf /opt/.setup

%environment
    export ENV_NAME="{{ ENV_NAME }}"
    export SHELL="/bin/bash"

%runscript
    if [ "${VERBOSE:-0}" != 0 ]; then
        if [ -n "${HUGGINGFACE_HUB_CACHE:-}" ]; then
            echo "INFO: Using HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\"" >&2
            mkdir -p "${HUGGINGFACE_HUB_CACHE}" || { echo "ERROR: Could not create HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE}\" (exit: $?)"; exit 1; }
            echo "INFO: HUGGINGFACE_HUB_CACHE=\"${HUGGINGFACE_HUB_CACHE:-}\" (avail: $(df -h --output=avail -h "${HUGGINGFACE_HUB_CACHE}" | tail -n 1 || true))" >&2
        else
             echo "INFO: HUGGINGFACE_HUB_CACHE is not set!" >&2
        fi
    fi
	exec micromamba run -n "${ENV_NAME}" "$@"

