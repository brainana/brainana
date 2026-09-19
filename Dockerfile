ARG DEBIAN_FRONTEND=noninteractive
# All tools (FSL, AFNI, FreeSurfer, ANTs) ship linux/amd64 binaries only.
# Override with --build-arg PLATFORM=linux/amd64 if needed (default is correct).
ARG PLATFORM=linux/amd64

##############
# ANTs build #
##############
# Use pre-compiled release binaries (built with generic x86-64 flags) instead of
# compiling from source. Source builds auto-detect AVX2/AVX-512 from the build
# host, which then crash under QEMU/Rosetta2 emulation on Apple Silicon Macs.
FROM --platform=${PLATFORM} debian:bookworm-slim AS ants-builder

ARG DEBIAN_FRONTEND
ARG ANTS_VERSION=2.5.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      unzip \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN set -eu && \
    mkdir -p /opt/ants/bin /opt/ants/lib && \
    wget --no-verbose -O /tmp/ants.zip \
      "https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-ubuntu-22.04-X64-gcc.zip" && \
    unzip -q /tmp/ants.zip -d /tmp/ants-extract && \
    ANTS_BIN="$(find /tmp/ants-extract -name 'antsRegistration' -type f -exec dirname {} \; | head -1)" && \
    [ -n "${ANTS_BIN}" ] || (echo "ERROR: Cannot locate antsRegistration in extracted zip" && exit 1) && \
    cp -r "${ANTS_BIN}/." /opt/ants/bin/ && \
    ANTS_PARENT="$(dirname "${ANTS_BIN}")" && \
    { [ -d "${ANTS_PARENT}/lib" ] && cp -r "${ANTS_PARENT}/lib/." /opt/ants/lib/ || true; } && \
    rm -rf /tmp/ants.zip /tmp/ants-extract && \
    test -x /opt/ants/bin/antsRegistration || (echo "ERROR: antsRegistration not found after install" && exit 1)

###########################
# Python env builder      #
# (build-essential only  #
#  here, not in final)   #
###########################
FROM --platform=${PLATFORM} debian:bookworm-slim AS python-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      libsuitesparse-dev \
      python3 \
      python3-dev \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

WORKDIR /opt/brainana

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Dependencies first, resolved from the lock alone. Keeping the source tree out of
# this layer is what makes it cacheable: with `COPY . .` above the sync, any edit in
# the repo invalidated it and, through /opt/venv, the FireANTs CUDA compile in the
# next stage -- the slowest layer in the build.
#
# There is deliberately no `|| uv pip install -e '.[full]'` fallback. It turned any
# transient sync failure into a silent unpinned resolution at build time, which also
# drops [tool.uv] override-dependencies and with it the simpleitk>=2.3.0 override
# that exists because fireants==1.5.0 hard-pins simpleitk==2.2.1.
COPY pyproject.toml uv.lock setup.cfg README.md ./
RUN mkdir -p /opt/brainana/tmp && \
    uv venv /opt/venv && \
    TMPDIR=/opt/brainana/tmp uv sync --python /opt/venv/bin/python --extra full \
        --frozen --no-cache --no-install-project && \
    rm -rf /opt/brainana/tmp

# Then the project itself, on top of the cached dependency layer.
COPY . .
RUN uv sync --python /opt/venv/bin/python --extra full --frozen --no-cache

# Clean venv: remove caches and test dirs (~1-2 GB savings)
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -name "*.pyo" -delete 2>/dev/null; \
    true

###########################
# FireANTs fused_ops (CUDA) — build into venv for GreedyRegistration/syn
###########################
FROM --platform=${PLATFORM} debian:bookworm-slim AS fireants-fused-ops-builder

ARG DEBIAN_FRONTEND=noninteractive

# NVIDIA CUDA repo for Debian 12 (repo serves 12.3; so venv Python from python-builder stays valid)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cuda-toolkit-12-3 \
      git \
      ninja-build \
      python3 \
      python3-dev \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

# System compiler first (avoid toolchain conflicts when loading .so at runtime)
ENV PATH=/usr/bin:/usr/local/cuda/bin:${PATH}

# No GPU at build time: set arch list so PyTorch cpp_extension does not get an empty list (IndexError).
# 7.0–9.0 cover Volta/Turing/Ampere/Ada/Hopper; 9.0+PTX allows JIT for newer/future GPUs.
# If the user's GPU is not in this list, fused_ops may fail at runtime; ants_register() then
# falls back to ANTs (CPU) automatically (rigid/affine still use FireANTs without fused_ops).
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

COPY --from=python-builder /opt/venv /opt/venv

# Pinned to the tag matching fireants==1.5.0 in uv.lock. Unpinned, this cloned
# whatever the default branch pointed at on build day, so the compiled fused_ops
# kernels could drift out of ABI agreement with the wheel in /opt/venv.
RUN git clone --depth 1 --branch v1.5.0 https://github.com/rohitrango/FireANTs.git /tmp/FireANTs && \
    cd /tmp/FireANTs/fused_ops && \
    /opt/venv/bin/python setup.py build_ext && \
    /opt/venv/bin/python setup.py install && \
    cd / && rm -rf /tmp/FireANTs

###########################
# FreeSurfer (exclude-list extract)
###########################
FROM --platform=${PLATFORM} debian:bookworm-slim AS freesurfer-download
ARG DEBIAN_FRONTEND=noninteractive
ARG FREESURFER_VERSION=7.4.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY docker/files/freesurfer7.4.1-exclude.txt /tmp/freesurfer-exclude.txt
RUN curl -fsSL "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FREESURFER_VERSION}/freesurfer-linux-ubuntu22_amd64-${FREESURFER_VERSION}.tar.gz" \
    | tar --no-same-owner -xzf - -C /usr/local --exclude-from=/tmp/freesurfer-exclude.txt

#########################
# Runtime with all libs #
#########################
FROM --platform=${PLATFORM} debian:bookworm-slim

ARG DEBIAN_FRONTEND
ARG FSL_VERSION=6.0.5.1
ARG AFNI_TARBALL=linux_rocky_8.tgz
# Image version baked in at build time (--build-arg BRAINANA_VERSION=<tag>).
# Exposed as BRAINANA_IMAGE_TAG so nhp_mri_prep.version.get_version() can
# surface the exact Docker image tag in QC reports and configs.
ARG BRAINANA_VERSION=unknown
ENV BRAINANA_IMAGE_TAG=${BRAINANA_VERSION}
# get_version() returns BRAINANA_IMAGE_TAG first and "unknown" is non-empty, so it
# never falls back to pyproject.toml -- a forgotten build-arg would stamp "unknown"
# into every sidecar, dataset_description.json and QC report. Fail loudly instead.
RUN if [ "${BRAINANA_VERSION}" = "unknown" ]; then \
        echo "ERROR: pass --build-arg BRAINANA_VERSION=<X.Y.Z> (or =dev for a throwaway build)." >&2; \
        exit 1; \
    fi

# Install uv (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TERM=xterm-256color

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bc \
      bzip2 \
      ca-certificates \
      curl \
      dc \
      git \
      gosu \
      locales \
      netpbm \
      perl \
      python3 \
      python3-venv \
      tar \
      tcsh \
      wget \
      openjdk-17-jre-headless \
      # graphviz provides 'dot', required by Nextflow to render DAG as SVG \
      graphviz \
      # procps provides 'ps', required by Nextflow to collect task metrics \
      procps \
      # CHOLMOD runtime for scikit-sparse (no build-essential in final image) \
      libcholmod3 \
      # graphics/openGL + X11 runtime for AFNI/FSL/FreeSurfer \
      # libglut3.12 is the bookworm runtime package for freeglut (AFNI binaries link it); \
      # other GL runtime libs (libgl1, libglu1-mesa) are installed separately below. \
      libglut3.12 \
      libfontconfig1 \
      libfreetype6 \
      libgl1 \
      libglw1-mesa \
      libglu1-mesa \
      libgomp1 \
      libgsl27 \
      libgslcblas0 \
      libice6 \
      libjpeg62-turbo \
      libmotif-common \
      libopenblas0 \
      libpng16-16 \
      libquadmath0 \
      libsm6 \
      libx11-6 \
      libxau6 \
      libxcb1 \
      libxcomposite1 \
      libxcursor1 \
      libxdamage1 \
      libxdmcp6 \
      libxext6 \
      libxfixes3 \
      libxi6 \
      libxinerama1 \
      libxmu6 \
      libxm4 \
      libxpm4 \
      libxrandr2 \
      libxrender1 \
      libxshmfence1 \
      libxt6 \
      libxxf86vm1 \
      zlib1g \
      # Connectome Workbench for QC surface snapshots (wb_command) and xvfb for headless rendering \
      connectome-workbench \
      xauth \
      xvfb && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    # Set OpenBLAS as the default BLAS/LAPACK so CHOLMOD (libcholmod3) and
    # scikit-sparse find the threaded symbols (e.g. sgemv_thread_n) at runtime.
    update-alternatives --set libblas.so.3-x86_64-linux-gnu \
        /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 && \
    update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
        /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3 && \
    rm -rf /var/lib/apt/lists/*

######################
# Install ANTs build #
######################
COPY --from=ants-builder /opt/ants /opt/ants

# Add neuro user to handle permissions
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} neuro && \
    useradd -l -u ${USER_ID} -g neuro -m neuro && \
    usermod -aG sudo neuro || true

###########
# Install FSL
###########
ENV FSLDIR=/usr/local/fsl
RUN curl -fsSL "https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-${FSL_VERSION}-centos7_64.tar.gz" \
    | tar xz -C /usr/local && \
    # Prune FSL (~2.75 GB) -- pipeline only uses flirt, mcflirt, fslmaths,
    # fslstats, fslroi, convert_xfm; none reference $FSLDIR/data/
    rm -rf ${FSLDIR}/data \
           ${FSLDIR}/src \
           ${FSLDIR}/doc \
           ${FSLDIR}/refdoc \
           ${FSLDIR}/tcl
ENV FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE

# Remove FSL's bundled OpenBLAS and old gfortran — they shadow the system
# openblas-pthread and gfortran-12 via LD_LIBRARY_PATH, causing
# "undefined symbol: sgemv_thread_n" when CHOLMOD / scikit-sparse loads.
RUN rm -f /usr/local/fsl/lib/libopenblas.so.0 \
         /usr/local/fsl/lib/libgfortran.so.3

############
# Install AFNI
############
ENV AFNI_HOME=/usr/local/afni
RUN mkdir -p "${AFNI_HOME}" && \
    curl -fsSL "https://afni.nimh.nih.gov/pub/dist/tgz/${AFNI_TARBALL}" \
      | tar xz -C "${AFNI_HOME}" --strip-components=1

################
# FreeSurfer (slim install via exclude list; subjects/ excluded for NHP)
################
COPY --link --from=freesurfer-download /usr/local/freesurfer /usr/local/freesurfer
# Empty subjects dir so SUBJECTS_DIR exists (pipeline uses custom template only)
RUN mkdir -p /usr/local/freesurfer/subjects
ENV FREESURFER_HOME=/usr/local/freesurfer

##############
# Environment
##############
ENV ANTSPATH=/opt/ants/bin/ \
    FSLDIR=/usr/local/fsl \
    AFNI_HOME=/usr/local/afni \
    AFNIPATH=/usr/local/afni \
    FREESURFER_HOME=/usr/local/freesurfer \
    FS_LICENSE=/fs_license.txt \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

ENV PATH=${FSLDIR}/bin:${AFNI_HOME}:${ANTSPATH}:${FREESURFER_HOME}/bin:${JAVA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/ants/lib:${FSLDIR}/lib
# Note: FSFAST_HOME, FMRI_ANALYSIS_DIR, FUNCTIONALS_DIR intentionally omitted
# because fsfast/ and sessions/ are excluded from the slim FreeSurfer install.
ENV SUBJECTS_DIR=${FREESURFER_HOME}/subjects \
    LOCAL_DIR=${FREESURFER_HOME}/local \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT=nii.gz \
    OS=Linux \
    FREESURFER=${FREESURFER_HOME} \
    MINC_BIN_DIR=${FREESURFER_HOME}/mni/bin \
    MINC_LIB_DIR=${FREESURFER_HOME}/mni/lib \
    MNI_DIR=${FREESURFER_HOME}/mni \
    MNI_DATAPATH=${FREESURFER_HOME}/mni/data \
    MNI_PERL5LIB=${FREESURFER_HOME}/mni/share/perl5 \
    PERL5LIB=${FREESURFER_HOME}/mni/share/perl5 \
    PYTHONPATH=/opt/brainana/src

# Create a shell script to source environments
RUN printf '#!/bin/bash\n\
# Neuroimaging tools environment\n\
export FSLDIR=%s\n\
export AFNI_HOME=%s\n\
export AFNIPATH=%s\n\
export ANTSPATH=%s\n\
export FREESURFER_HOME=%s\n\
export FS_LICENSE=%s\n\
export FSLOUTPUTTYPE=NIFTI_GZ\n\
export PATH=$FSLDIR/bin:$AFNI_HOME:$ANTSPATH:$FREESURFER_HOME/bin:${JAVA_HOME}/bin:/usr/local/bin:$PATH\n\
\n\
# License check\n\
if [ ! -f "$FS_LICENSE" ]; then\n\
    echo "--------------------------------------------------------------------------------"\n\
    echo "WARNING: FreeSurfer license not found at $FS_LICENSE"\n\
    echo "To run FreeSurfer tools, please mount your license file:"\n\
    echo "  docker run ... -v /path/to/license.txt:/fs_license.txt ..."\n\
    echo "--------------------------------------------------------------------------------"\n\
fi\n\
\n\
# Welcome Message\n\
if [ "$PS1" ]; then\n\
    echo "================================================================================"\n\
    echo "Welcome to brainana Interactive Environment"\n\
    echo "--------------------------------------------------------------------------------"\n\
    echo "Installed Tools:"\n\
    echo "  - FSL:        $(flirt -version 2>&1 | head -n 1 || echo "Installed")"\n\
    echo "  - ANTs:       $(antsRegistration --version 2>&1 | grep -i version | head -n 1 || echo "Installed")"\n\
    echo "  - FireANTs:   $(python3 -c "import fireants; print(fireants.__version__)" 2>/dev/null || echo "Installed")"\n\
    echo "  - AFNI:       $(afni -version | head -n 1 || echo "Installed")"\n\
    echo "  - FreeSurfer: $(cat $FREESURFER_HOME/build-stamp.txt 2>/dev/null || echo "Installed")"\n\
    echo "  - Python:     $(python3 --version)"\n\
    echo "  - Java:       $(java -version 2>&1 | head -n 1)"\n\
    echo "  - uv:         $(uv --version)"\n\
    echo "  - Nextflow:   $(nextflow -version 2>/dev/null | head -n 1 || echo "Installed")"\n\
    echo "  - Workbench:  $(wb_command -version 2>/dev/null | head -n 1 || echo "Installed")"\n\
    echo "--------------------------------------------------------------------------------"\n\
    echo "Usage Examples:"\n\
    echo "  ./run_brainana.sh run main.nf --bids_dir /data --output_dir /output"\n\
    echo "  (Config generator: https://brainana.readthedocs.io/en/stable/_static/config_generator.html)"\n\
    echo "================================================================================"\n\
fi\n' \
    "${FSLDIR}" "${AFNI_HOME}" "${AFNIPATH}" "${ANTSPATH}" "${FREESURFER_HOME}" "${FS_LICENSE}" \
    > /etc/profile.d/neuroenv.sh && \
    chmod +x /etc/profile.d/neuroenv.sh

# Setup Bash for interactive use
RUN echo "source /etc/profile.d/neuroenv.sh" >> /etc/bash.bashrc && \
    echo "alias ls='ls --color=auto'" >> /etc/bash.bashrc && \
    echo "alias ll='ls -alF'" >> /etc/bash.bashrc

##############
# Nextflow
##############
ARG NEXTFLOW_VERSION=25.10.2
# Pin version at runtime to prevent auto-update and version drift
ENV NXF_VER=${NEXTFLOW_VERSION}
# Pre-cache framework JAR so no network is needed at runtime
ENV NXF_HOME=/opt/nextflow

WORKDIR /tmp
RUN curl -s https://get.nextflow.io | bash && \
    chmod +x nextflow && \
    mv nextflow /usr/local/bin/ && \
    nextflow -version && \
    chmod -R a+rX /opt/nextflow

#################
# Project Install (from builder – no gcc/build-essential in final image)
#################
WORKDIR /opt/brainana

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Redirect caches to writable locations to support arbitrary UIDs
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPYCACHEPREFIX=/tmp/pycache

COPY --chmod=755 --from=fireants-fused-ops-builder /opt/venv /opt/venv
COPY --chmod=755 --from=python-builder /opt/brainana /opt/brainana

# FireANTs fused_ops: use venv CUDA/torch libs so fused_ops load with --gpus all
RUN FER="$(find /opt/venv -path '*nvidia/cuda_runtime/lib' -type d 2>/dev/null | head -1)" && \
    TORCH_LIB="$(find /opt/venv -path '*torch/lib' -type d 2>/dev/null | head -1)" && \
    if [ -n "$FER" ]; then \
      echo '' >> /etc/profile.d/neuroenv.sh && \
      echo '# FireANTs fused_ops: venv CUDA runtime' >> /etc/profile.d/neuroenv.sh && \
      { [ -n "$TORCH_LIB" ] && echo "export LD_LIBRARY_PATH=\"$FER:$TORCH_LIB:\${LD_LIBRARY_PATH}\"" || echo "export LD_LIBRARY_PATH=\"$FER:\${LD_LIBRARY_PATH}\""; } >> /etc/profile.d/neuroenv.sh && \
      [ -f "$FER/libcudart.so.12" ] && echo "export LD_PRELOAD=\"$FER/libcudart.so.12\"" >> /etc/profile.d/neuroenv.sh; \
    fi

# World-writable temp dirs so any UID (via -u or gosu) can use them
RUN mkdir -p /tmp/matplotlib /tmp/pycache /tmp/.X11-unix /tmp/home && \
    chmod 1777 /tmp/matplotlib /tmp/pycache /tmp/.X11-unix /tmp/home && \
    chmod -R 755 /home/neuro && \
    chmod +x /opt/brainana/entrypoint.sh

# Pre-generate matplotlib font cache at build time so any UID can read it.
# Without this, the HEALTHCHECK (root) or first runtime import creates it as
# root-owned, and non-root pipeline users get "Permission denied".
RUN python3 -c "import matplotlib.font_manager" 2>/dev/null || true && \
    chmod -R a+rw /tmp/matplotlib 2>/dev/null || true

# NOTE: No "USER neuro" here. The entrypoint starts as root,
# detects the UID of /output, and drops to that user via gosu.
WORKDIR /opt/brainana

# Health check: lightweight probe that doesn't import matplotlib (which would
# create a root-owned font cache that blocks non-root pipeline users).
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "print('OK')" || exit 1

# Labels for image metadata
LABEL org.opencontainers.image.title="brainana" \
      org.opencontainers.image.description="Macaque MRI preprocessing pipeline" \
      org.opencontainers.image.version="${BRAINANA_VERSION}" \
      org.opencontainers.image.source="https://github.com/brainana/brainana"

ENTRYPOINT ["/opt/brainana/entrypoint.sh"]
CMD ["/input", "/output"]
