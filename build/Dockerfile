FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1
RUN apt-get update && \
apt-get install -y software-properties-common && \
add-apt-repository ppa:ubuntuhandbook1/ffmpeg7 -y

RUN apt-get update && apt-get install -y git wget jq ffmpeg curl python3-venv && apt-get clean

ENV ROOT=/comfy
ENV DATA=/data
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
## Verify installation
RUN node --version && npm --version

RUN pip install --upgrade pip

# Clone and setup ComfyUI with venv
RUN --mount=type=cache,target=/root/.cache/pip \
  git clone --depth 1 --branch v0.3.34 https://github.com/comfyanonymous/ComfyUI.git ${ROOT} && \
  cd ${ROOT} && \
  ## Use latest torch and torchvision
  sed -i 's/^torch$/torch~=2.7.0/' requirements.txt && \
  sed -i 's/^torchvision$/torchvision~=0.22.0/' requirements.txt && \
  sed -i 's/^torchaudio$/torchaudio~=2.7.0/' requirements.txt && \
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# HACK, we want to include apiKey in the prompt call if it is present
# /opt/conda/lib/python3.11/site-packages
RUN sed -i 's/ = this.requestHeader;/ = this.requestHeader; if (localStorage.apikey) fetchOptions.headers.Authorization=`Bearer ${localStorage.apikey}`;/' $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/comfyui_frontend_package/static/assets/index-0j2RaeNn.js

RUN rm -rf ${ROOT}/custom_nodes ${ROOT}/input ${ROOT}/models ${ROOT}/notebooks ${ROOT}/output && \
    mkdir -p ${DATA}/.cache ${DATA}/user ${DATA}/custom_nodes ${DATA}/input ${DATA}/models ${DATA}/notebooks ${DATA}/output

# Create symlinks
RUN ln -sT ${DATA}/.cache ${ROOT}/.cache && \
    ln -sT ${DATA}/custom_nodes ${ROOT}/custom_nodes && \
    ln -sT ${DATA}/input ${ROOT}/input && \
    ln -sT ${DATA}/models ${ROOT}/models && \
    ln -sT ${DATA}/notebooks ${ROOT}/notebooks && \
    ln -sT ${DATA}/output ${ROOT}/output && \
    ln -sT ${DATA}/user ${ROOT}/user

WORKDIR ${ROOT}
COPY . /docker/

RUN chmod u+x /docker/entrypoint.sh ${ROOT}

WORKDIR ${ROOT}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH="/usr/bin:${PATH}"
ENV CLI_ARGS=""

EXPOSE 7860
ENTRYPOINT ["/docker/entrypoint.sh"]
CMD ["sh", "-c", "python -u main.py --listen --port 7860 ${CLI_ARGS}"]