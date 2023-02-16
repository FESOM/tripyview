FROM mambaorg/micromamba:1.3.0
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -c conda-forge python=3.7
RUN micromamba install -c conda-forge cartopy
RUN micromamba install -c conda-forge pickle5
RUN pip install .
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/app/entrypoint.sh"]
