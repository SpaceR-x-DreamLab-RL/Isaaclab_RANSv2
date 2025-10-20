## Base <https://hub.docker.com/_/ubuntu>
ARG BASE_IMAGE_NAME_ARG
ARG BASE_IMAGE_TAG_ARG

## Continue with the base image
FROM ${BASE_IMAGE_NAME_ARG}:${BASE_IMAGE_TAG_ARG}

## Use bash as the default shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## Create a barebones entrypoint that is conditionally updated throughout the Dockerfile
RUN echo "#!/usr/bin/env bash" >> /entrypoint.bash && \
    chmod +x /entrypoint.bash

ARG PYTHON_VERSION_ARG

## Install system dependencies
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    # Common
    bash-completion \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    mold \
    unzip \
    xz-utils \
    # Graphics
    libgl1 \
    libglu1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libxt-dev \
    # Python
    python3.11 \
    python3-pip \
    python3-dev \
    python3.11-venv \
    # Video recording/processing
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

## Python environment
# ENV VIRTUAL_ENV=/opt/isaaclab_venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN python3.11 -m venv ${VIRTUAL_ENV} && \
#     pip install --no-input --no-cache-dir --upgrade pip

## Upgrade pip
RUN python3.11 -m pip install --no-input --no-cache-dir --upgrade pip

## Isaac Sim
ARG TORCH_VERSION_ARG
ARG TORCHVISION_VERSION_ARG
ARG TORCH_INDEX_URL_ARG
RUN python3.11 -m pip install torch==${TORCH_VERSION_ARG} torchvision==${TORCHVISION_VERSION_ARG} --index-url ${TORCH_INDEX_URL_ARG}

ARG ISAAC_SIM_VERSION_ARG
ARG ISAAC_SIM_EXTRA_INDEX_URL_ARG
RUN python3.11 -m pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION_ARG}" --extra-index-url ${ISAAC_SIM_EXTRA_INDEX_URL_ARG}

## Isaac Lab
ARG ISAACLAB_PATH="/root/isaaclab"
ARG ISAACLAB_REMOTE="https://github.com/SpaceR-x-DreamLab-RL/Isaaclab.git"
ARG ISAACLAB_BRANCH="main"
ARG ISAACLAB_COMMIT_SHA="3cea8df06c13b7e39b6d5f49a9fde36a46454ede" # Oct 6, 2025

RUN git clone --branch ${ISAACLAB_BRANCH} ${ISAACLAB_REMOTE} ${ISAACLAB_PATH}
WORKDIR ${ISAACLAB_PATH}
RUN git checkout ${ISAACLAB_COMMIT_SHA} && \
    for extension in $(find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d); do \
    if [ -f "${extension}/pyproject.toml" ] || [ -f "${extension}/setup.py" ]; then \
    python3.11 -m pip install --no-input --no-cache-dir --editable "${extension}" ; \
    fi ; \
    done 

##################
### Entrypoint ###
##################

## Define the workspace of the project
ARG Isaaclab_RANSv2_PATH="/root/ws"
RUN echo -e "\n# Isaaclab RANSv2" >> /entrypoint.bash && \
    echo "export Isaaclab_RANSv2_PATH=\"${Isaaclab_RANSv2_PATH}\"" >> /entrypoint.bash
WORKDIR "${Isaaclab_RANSv2_PATH}"

## Finalize the entrypoint
# hadolint ignore=SC2016
RUN echo -e "\n# Execute command" >> /entrypoint.bash && \
    echo -en 'exec "${@}"\n' >> /entrypoint.bash && \
    sed -i '$a source /entrypoint.bash --' ~/.bashrc
ENTRYPOINT ["/entrypoint.bash"]

####################
### Dependencies ###
####################
## Install Python dependencies
# hadolint ignore=DL3013,SC2046
# RUN --mount=type=bind,source="source/Isaaclab_RANSv2/pyproject.toml",target="${Isaaclab_RANSv2_PATH}/pyproject.toml" \
#     "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --ignore-installed toml~=0.10 
    #&& \
    # "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir $("${ISAAC_SIM_PYTHON}" -c "f='${Isaaclab_RANSv2_PATH}/pyproject.toml'; from toml import load; print(' '.join(filter(lambda d: not d.startswith(p['name'] + '['), (*p.get('dependencies', ()), *(d for ds in p.get('optional-dependencies', {}).values() for d in ds)))) if (p := load(f).get('project', None)) else '')")
