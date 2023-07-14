FROM mambaorg/micromamba:1.4.1

LABEL org.opencontainers.image.source=https://github.com/choderalab/espaloma
LABEL org.opencontainers.image.description="Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm"
LABEL org.opencontainers.image.licenses=MIT
# OpenFE Version we want to build
ARG VERSION

# Don't buffer stdout & stderr streams, so if there is a crash no partial buffer output is lost
# https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED=1

RUN micromamba install -y -n base -c conda-forge -c dglteam dgl==0.9.0 git "espaloma==$VERSION" && \
    micromamba clean --all --yes

# Ensure that conda environment is automatically activated
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
