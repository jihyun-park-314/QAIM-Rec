# QAIM-Rec training/embedding environment.
# Target hardware: 4x NVIDIA RTX 2080 Ti (Turing, sm75), driver 535.274.02, CUDA 12.2.
# Base image ships a prebuilt torch+cu121 stack compatible with this driver.
# LLM inference (gemma4) runs separately via the Ollama server
# (jhpark-gemma4-server container, http://localhost:11434/api/generate) —
# this container only needs torch/transformers for embeddings and SASRec.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /qaim-rec

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code is bind-mounted via docker-compose (live editing), not copied here.

CMD ["bash"]
