ARG BASE_IMAGE=mambaorg/micromamba:jammy

FROM ${BASE_IMAGE} AS build-llava-prereqs
ARG ENV_NAME=llava
ENV ENV_NAME=${ENV_NAME}
ENV LC_ALL=C.UTF-8
WORKDIR /opt/build/llava
COPY environment.yaml .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN <<-EOF

    # Install LLaVA dependencies:
    export CI=1
    micromamba create -v -y -n "${ENV_NAME}" -f environment.yaml
    micromamba clean -y -all

EOF

FROM build-llava-prereqs as build-llava
ENV LC_ALL=C.UTF-8
WORKDIR /opt/build/llava
COPY ./ .
SHELL ["/bin/bash", "-eEx", "-o", "pipefail", "-c"]
RUN <<-EOF

    # Install LLaVA:
    micromamba run -n "${ENV_NAME}" -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip install --no-deps --no-cache-dir --config-settings="--install-data=$PWD/llava" .

    # Clean up:
    micromamba run -n "${ENV_NAME}" python -m pip cache purge
    micromamba clean -y -all

EOF

# Script to show web server launch
COPY --chmod=755 scripts/hyak-llava-web.sh /usr/local/bin/hyak-llava-web

# Script which launches commands passed to "docker run"
COPY --chmod=755 _entrypoint.sh /usr/local/bin/_entrypoint.sh

WORKDIR /data
ENV SHELL=/bin/bash
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
