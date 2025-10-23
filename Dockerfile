## Base <https://hub.docker.com/_/ubuntu>
ARG BASE_IMAGE_NAME="ubuntu"
ARG BASE_IMAGE_TAG="24.04"

## Isaac Sim <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim>
## Label as isaac-sim for copying into the final image
ARG ISAAC_SIM_IMAGE_NAME="nvcr.io/nvidia/isaac-sim"
ARG ISAAC_SIM_IMAGE_TAG="5.0.0"
FROM ${ISAAC_SIM_IMAGE_NAME}:${ISAAC_SIM_IMAGE_TAG} AS isaac-sim

## Continue with the base image
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

## Use bash as the default shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

## Create a barebones entrypoint that is conditionally updated throughout the Dockerfile
RUN echo "#!/usr/bin/env bash" >> /entrypoint.bash && \
    chmod +x /entrypoint.bash

###########################
### System Dependencies ###
###########################

## Copy Isaac Sim into the base image
ARG ISAAC_SIM_PATH="/root/isaac-sim"
ENV ISAAC_SIM_PYTHON="${ISAAC_SIM_PATH}/python.sh"
COPY --from=isaac-sim /isaac-sim "${ISAAC_SIM_PATH}"
COPY --from=isaac-sim /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
RUN ISAAC_SIM_VERSION="$(cut -d'-' -f1 < "${ISAAC_SIM_PATH}/VERSION")" && \
    echo -e "\n# Isaac Sim ${ISAAC_SIM_VERSION}" >> /entrypoint.bash && \
    echo "export ISAAC_SIM_PATH=\"${ISAAC_SIM_PATH}\"" >> /entrypoint.bash && \
    echo "export OMNI_KIT_ALLOW_ROOT=\"1\"" >> /entrypoint.bash
## Fix cosmetic issues in `isaac-sim/setup_python_env.sh` that append nonsense paths to `PYTHONPATH` and `LD_LIBRARY_PATH`
# hadolint ignore=SC2016
RUN sed -i 's|$SCRIPT_DIR/../../../$LD_LIBRARY_PATH:||' "${ISAAC_SIM_PATH}/setup_python_env.sh" && \
    sed -i 's|$SCRIPT_DIR/../../../$PYTHONPATH:||' "${ISAAC_SIM_PATH}/setup_python_env.sh"

## Build Python with enabled optimizations to improve the runtime training performance
ARG PYTHON_VERSION="3.11.13"
ARG PYTHON_PREFIX="/usr/local"
ENV PYTHONEXE="${PYTHON_PREFIX}/bin/python${PYTHON_VERSION%.*}"
# hadolint ignore=DL3003,DL3008
RUN PYTHON_DL_PATH="/tmp/Python-${PYTHON_VERSION}.tar.xz" && \
    PYTHON_SRC_DIR="/tmp/python${PYTHON_VERSION}" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    build-essential \
    ca-certificates \
    checkinstall \
    curl \
    gdb \
    git \
    inetutils-inetd \
    lcov \
    libbz2-dev \
    libc6-dev \
    libedit-dev \
    libffi-dev \
    libgdbm-compat-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libnss3-dev  \
    libreadline-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    libzstd-dev \
    llvm \
    lzma-dev \
    pkg-config \
    python3-openssl \
    tk-dev \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto "=https" --tlsv1.2 -sSfL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" -o "${PYTHON_DL_PATH}" && \
    mkdir -p "${PYTHON_SRC_DIR}" && \
    tar xf "${PYTHON_DL_PATH}" -C "${PYTHON_SRC_DIR}" --strip-components=1 && \
    rm "${PYTHON_DL_PATH}" && \
    cd "${PYTHON_SRC_DIR}" && \
    "${PYTHON_SRC_DIR}/configure" --enable-shared --enable-optimizations --with-lto --prefix="${PYTHON_PREFIX}" && \
    make -j "$(nproc)" && \
    make install && \
    cd - && \
    rm -rf "${PYTHON_SRC_DIR}"
## Create a 'python' symlink for convenience
RUN ln -sr "${PYTHONEXE}" "${PYTHON_PREFIX}/bin/python"
## Fix `PYTHONEXE` by disabling the append of "isaac-sim/kit/kernel/plugins" to `LD_LIBRARY_PATH` inside `isaac-sim/setup_python_env.sh`
# hadolint ignore=SC2016
RUN sed -i 's|$SCRIPT_DIR/kit/kernel/plugins:||' "${ISAAC_SIM_PATH}/setup_python_env.sh"
## Make the system Python identical with Isaac Sim Python
# hadolint ignore=SC2016
RUN mv "${PYTHONEXE}" "${PYTHON_PREFIX}/bin/python${PYTHON_VERSION}" && \
    echo -e '#!/bin/bash\n${ISAAC_SIM_PYTHON} "${@}"' > "${PYTHONEXE}" && \
    chmod +x "${PYTHONEXE}"
ENV PYTHONEXE="${PYTHON_PREFIX}/bin/python${PYTHON_VERSION}"
## Fake that Python was installed via apt
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    equivs && \
    rm -rf /var/lib/apt/lists/* && \
    for pkg in "libpython${PYTHON_VERSION%.*}" "libpython${PYTHON_VERSION%.*}-dev" "python${PYTHON_VERSION%.*}-dev"; do \
    equivs-control "${pkg}" && \
    echo -e "Package: ${pkg}\nProvides: ${pkg}\nVersion: ${PYTHON_VERSION}\nArchitecture: all" > "${pkg}" && \
    equivs-build "${pkg}" && \
    dpkg -i "${pkg}_${PYTHON_VERSION}_all.deb" && \
    apt-mark hold "${pkg}" && \
    rm "${pkg}" "${pkg}_${PYTHON_VERSION}_all.deb" ; \
    done

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
    git-lfs \
    mold \
    unzip \
    xz-utils \
    # Graphics
    libgl1 \
    libglu1 \
    libxi6 \
    libxkbcommon-x11-0 \
    libxt-dev \
    # Video recording/processing
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

## Upgrade pip
RUN "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --upgrade pip

# ## Install ROS | Note: Unsuitable because Isaac Sim 5.0 requires Python 3.11 that is not supported by any non-EoL ROS 2 distribution
# ARG ROS_DISTRO="jazzy"
# # hadolint ignore=SC1091,DL3008
# RUN curl --proto "=https" --tlsv1.2 -sSfL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o /usr/share/keyrings/ros-archive-keyring.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo "${UBUNTU_CODENAME}") main" > /etc/apt/sources.list.d/ros2.list && \
#     apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
#     ros-dev-tools \
#     "ros-${ROS_DISTRO}-ros-base" \
#     "ros-${ROS_DISTRO}-rmw-fastrtps-cpp" \
#     "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" && \
#     rm -rf /var/lib/apt/lists/* && \
#     "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir catkin_pkg && \
#     rosdep init --rosdistro "${ROS_DISTRO}" && \
#     echo -e "\n# ROS ${ROS_DISTRO^}" >> /entrypoint.bash && \
#     echo "source \"/opt/ros/${ROS_DISTRO}/setup.bash\" --" >> /entrypoint.bash

## Build ROS
ARG ROS_DISTRO="jazzy"
ENV ROS_UNDERLAY_PATH="/root/ros/${ROS_DISTRO}"
ARG ROS_PACKAGES_SKIP="python_orocos_kdl_vendor qt_gui_cpp"
WORKDIR "${ROS_UNDERLAY_PATH}"
# hadolint ignore=SC2046,SC2086,DL3008
RUN curl --proto "=https" --tlsv1.2 -sSfL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    ros-dev-tools && \
    rosdep init --rosdistro "${ROS_DISTRO}" && \
    rosdep update && \
    "${ISAAC_SIM_PYTHON}" -m pip install build --no-input --no-cache-dir setuptools-scm && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir catkin_pkg lark pytest~=8.0 empy==3.3.4 && \
    mkdir -p "${ROS_UNDERLAY_PATH}/src" && \
    curl --proto "=https" --tlsv1.2 -sSfL "https://raw.githubusercontent.com/ros2/ros2/${ROS_DISTRO}/ros2.repos" -o "${ROS_UNDERLAY_PATH}/src/ros2.repos" && \
    sed -i -e 's|  \([^/]*\)/\([^:]*\):|  \2:|g' "${ROS_UNDERLAY_PATH}/src/ros2.repos" && \
    vcs import "${ROS_UNDERLAY_PATH}/src" < "${ROS_UNDERLAY_PATH}/src/ros2.repos" && \
    rm -rf $(find "${ROS_UNDERLAY_PATH}/src" -type d -name .git) && \
    DEBIAN_FRONTEND=noninteractive rosdep install --default-yes --ignore-src --rosdistro "${ROS_DISTRO}" --from-paths "${ROS_UNDERLAY_PATH}/src" --skip-keys "$(rosdep db 2>/dev/null | grep -i "example\|demo\|tutorial" | awk '{print $1}' | tr '\n' ' ')" --skip-keys rti-connext-dds-6.0.1 --skip-keys python3-pybind11 && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="${ISAAC_SIM_PYTHON}" --build-base "${ROS_UNDERLAY_PATH}/build" --install-base "${ROS_UNDERLAY_PATH}/install" --packages-skip ${ROS_PACKAGES_SKIP} --packages-skip-by-dep ${ROS_PACKAGES_SKIP} --packages-skip-regex ".*examples.*" ".*tutorial.*" && \
    echo -e "\n# ROS ${ROS_DISTRO^} (built from source)" >> /entrypoint.bash && \
    echo "source \"${ROS_UNDERLAY_PATH}/install/setup.bash\" --" >> /entrypoint.bash && \
    rm -rf "${ROS_UNDERLAY_PATH}/src/ros2.repos" /var/lib/apt/lists/* /root/.ros/rosdep/sources.cache ./log

###################
### Development ###
###################
ARG DEV=true

## Simulation
ARG ISAACLAB_DEV=true
ARG ISAACLAB_PATH="/root/isaaclab"
ARG ISAACLAB_REMOTE="https://github.com/SpaceR-x-DreamLab-RL/Isaaclab.git"
ARG ISAACLAB_BRANCH="main"
ARG ISAACLAB_COMMIT_SHA="429ff008e2a726bd7d35c7f3342596e17a9c044f" # 2025-09-19
# hadolint ignore=SC2044
ENV TERM=xterm-256color
RUN if [[ "${DEV,,}" = true && "${ISAACLAB_DEV,,}" = true ]]; then \
    echo -e "\n# Isaac Lab ${ISAACLAB_COMMIT_SHA}" >> /entrypoint.bash && \
    echo "export ISAACLAB_PATH=\"${ISAACLAB_PATH}\"" >> /entrypoint.bash && \
    git clone "${ISAACLAB_REMOTE}" "${ISAACLAB_PATH}" --branch "${ISAACLAB_BRANCH}" && \
    git -C "${ISAACLAB_PATH}" reset --hard "${ISAACLAB_COMMIT_SHA}" && \
    git -C "${ISAACLAB_PATH}" lfs pull && \
    for extension in $(find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d); do \
    if [ -f "${extension}/pyproject.toml" ] || [ -f "${extension}/setup.py" ]; then \
        "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --editable "${extension}" ; \
    fi ; \
    done && \
    ln -sf "${ISAAC_SIM_PATH}" "${ISAACLAB_PATH}/_isaac_sim" ; \
    ${ISAACLAB_PATH}/isaaclab.sh --install ; \
    fi

## Reinforcement Learning
ARG DREAMER_DEV=true
ARG DREAMER_PATH="/root/dreamerv3"
ARG DREAMER_REMOTE="https://github.com/AndrejOrsula/dreamerv3.git"
ARG DREAMER_BRANCH="main"
ARG DREAMER_COMMIT_SHA="4049794d4135e41c691f18da38a9af7541b01553" # 2025-07-16
RUN if [[ "${DEV,,}" = true && "${DREAMER_DEV,,}" = true ]]; then \
    git clone "${DREAMER_REMOTE}" "${DREAMER_PATH}" --branch "${DREAMER_BRANCH}" && \
    git -C "${DREAMER_PATH}" reset --hard "${DREAMER_COMMIT_SHA}" && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --editable "${DREAMER_PATH}" ; \
    fi

##################
### Entrypoint ###
##################

## Define the workspace of the project
ARG ISAACLAB_RANSV2_PATH="/root/ws"
RUN echo -e "\n# Isaac lab RANSv2" >> /entrypoint.bash && \
    echo "export ISAACLAB_RANSV2_PATH=\"${ISAACLAB_RANSV2_PATH}\"" >> /entrypoint.bash
WORKDIR "${ISAACLAB_RANSV2_PATH}"

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
RUN --mount=type=bind,source="source/Isaaclab_RANSv2/pyproject.toml",target="${ISAACLAB_RANSV2_PATH}/source/Isaaclab_RANSv2/pyproject.toml" \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --ignore-installed toml~=0.10 && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir $("${ISAAC_SIM_PYTHON}" -c "f='${ISAACLAB_RANSV2_PATH}/source/Isaaclab_RANSv2/pyproject.toml'; from toml import load; print(' '.join(filter(lambda d: not d.startswith(p['name'] + '['), (*p.get('dependencies', ()), *(d for ds in p.get('optional-dependencies', {}).values() for d in ds)))) if (p := load(f).get('project', None)) else '')")

###############
### Project ###
###############

## Copy the source code into the image
COPY . "${ISAACLAB_RANSV2_PATH}"

## Install project as editable Python package
# hadolint ignore=SC1091
RUN source /entrypoint.bash -- && \
    "${ISAAC_SIM_PYTHON}" -m pip install --no-input --no-cache-dir --no-deps --editable "${ISAACLAB_RANSV2_PATH}/source/Isaaclab_RANSv2[all]"


## Configure argcomplete
RUN echo "source /etc/bash_completion" >> "/etc/bash.bashrc" && \
    for exe in ${EXECUTABLES}; do \
    register-python-argcomplete "${exe}" > "/etc/bash_completion.d/${exe}" ; \
    done

## Set the default command
CMD ["bash"]

############
### Misc ###
############

## Skip writing Python bytecode to the disk to avoid polluting mounted host volume with `__pycache__` directories
ENV PYTHONDONTWRITEBYTECODE=1

## Enable full error backtrace with Hydra
ENV HYDRA_FULL_ERROR=1