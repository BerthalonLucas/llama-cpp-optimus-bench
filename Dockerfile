FROM ghcr.io/ggml-org/llama.cpp:full-cuda

ENV HF_HOME=/models \
    HF_HUB_CACHE=/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    LLAMA_BIN=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      procps \
      zsh \
      python3 \
      python3-venv \
      python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir llama-optimus huggingface_hub

SHELL ["/bin/zsh", "-c"]

ENTRYPOINT ["zsh"]
CMD ["-i"]
