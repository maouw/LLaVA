ARG BASE_IMAGE=mambaorg/micromamba:jammy

FROM ${BASE_IMAGE} AS base
USER root
RUN <<-EOF
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-insatll-recommends \
        bzip2 \
        ca-certificates \
        curl \
        git \
        less \
        nano
    apt-get clean --yes --quiet
    rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log

EOF

FROM base AS build-llava
USER root
ARG LLAVA_URL=https://github.com/haotian-liu/LLaVA.git#c121f0432da27facab705978f83c4ada465e46fd
ARG ENV_NAME=llava
ENV ENV_NAME=${ENV_NAME}
ENV LC_ALL=C.UTF-8
ARG CUDA_ARCHITECTURES="sm_86 sm_89"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
WORKDIR /opt/build/llava
ADD --link --keep-git-dir=true ${LLAVA_URL} .
COPY environment.yaml .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN <<-EOF

    # Update pyproject.toml to include the package data in examples (web server breaks without):
    grep -qF '[tool.setuptools.package-data]' pyproject.toml || printf '[tool.setuptools.package-data]\nllava = ["serve/examples/*.jpg"]' >>pyproject.toml

    # Install LLaVA dependencies:
    export CI=1
    micromamba create -v -y -n "${ENV_NAME}" -f environment.yaml
    # Install LLaVA:
    micromamba run -n "${ENV_NAME}" -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip install --no-deps --no-cache-dir --config-settings="--install-data=$PWD/llava" .

    # Clean up:
    micromamba run -n "${ENV_NAME}" python -m pip cache purge
    micromamba clean -y -all
EOF

COPY --chmod=755 llava-run.py /usr/local/bin/llava-run
COPY --chmod=755 hyak-llava-web /usr/local/bin/hyak-llava-web
COPY --chmod=755 _entrypoint.sh /usr/local/bin/_entrypoint.sh

WORKDIR /data
