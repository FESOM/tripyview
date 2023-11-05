FROM mambaorg/micromamba:1.3.0
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install necessary development tools, including gcc
# RUN apt-get update && apt-get install -y build-essential

RUN micromamba install -c conda-forge python=3.8
RUN micromamba install -c conda-forge cartopy

# Ensure that gcc is available for package installations
RUN micromamba install -c conda-forge gcc

# RUN micromamba install -c conda-forge pickle5
# RUN micromamba install -c conda-forge jupyterhub
RUN micromamba install -c conda-forge netcdf4=1.6.0
RUN pip install .
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/app/entrypoint.sh"]
