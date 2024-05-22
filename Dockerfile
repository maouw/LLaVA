# syntax=docker/dockerfile:1
ARG BASE_IMAGE=mambaorg/micromamba:jammy

FROM ${BASE_IMAGE} AS build-llava-prereqs-python

USER root

ENV LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
WORKDIR /tmp
COPY envs/environment-python.yaml .
RUN --mount=type=cache,sharing=locked,target=/opt/conda/pkgs  \
<<-EOF

    # Create environment dependencies:
    export CI=1
    
    # Add channels:

    for channel in intel conda-forge anaconda defaults; do
        micromamba config append channels "${channel}"
    done

    micromamba install -v -y -n base -f environment-python.yaml

EOF

FROM build-llava-prereqs-python AS build-llava-prereqs-pytorch
ARG TORCH_CUDA_ARCH_LIST="sm_86 sm_89"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
COPY envs/environment-pytorch-cuda.yaml .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN --mount=type=cache,sharing=locked,target=/opt/conda/pkgs \
<<-EOF

    # Install PyTorch dependencies:
    export CI=1
    for channel in nvidia pytorch; do
        micromamba config prepend channels "${channel}"
    done

    micromamba install -v -y -n base -f environment-pytorch-cuda.yaml

EOF


FROM build-llava-prereqs-pytorch as build-llava-prereqs
COPY envs/environment-pkgs.yaml .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN --mount=type=cache,sharing=locked,target=/opt/conda/pkgs \
<<-EOF

    # Install LLaVA dependencies:
    export CI=1
    micromamba install -v -y --freeze-installed -n base -f environment-pkgs.yaml

    # Clean up:
    micromamba run -n base python -m pip cache purge
EOF

# Final image
FROM build-llava-prereqs as build-llava
WORKDIR /opt/build/llava

COPY _entrypoint.sh pyproject.toml README.md .

COPY scripts ./scripts

# Script to show web server launch
COPY --chmod=755 scripts/hyak-llava-web.sh /usr/local/bin/hyak-llava-web

# Script which launches commands passed to "docker run"
COPY --chmod=755 _entrypoint.sh /usr/local/bin/_entrypoint.sh

COPY docs ./docs
COPY envs ./envs
COPY images ./images
COPY playground ./playground

COPY llava ./llava
COPY pyproject.toml .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN --mount=type=cache,sharing=locked,target=/opt/conda/pkgs \
<<-EOF
 
    ls -1

    # Install LLaVA:
    micromamba run -n base -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip install --no-deps --no-cache-dir --config-settings="--install-data=$PWD/llava" .

    # Clean up:
    micromamba run -n base python -m pip cache purge

EOF

ENV SHELL=/bin/bash
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
