# ============================================================
# BUILDER
# ============================================================
FROM python:3.12-slim-bookworm AS builder

ARG EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/querypilot/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_MODEL_PATH=/opt/models/mxbai-embed-large-v1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    libmagic-dev \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch (only needed because sentence-transformers imports it).
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.0

# Install application dependencies.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Verify Python dependencies and CPU-only PyTorch.
RUN python -m pip check && \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); assert torch.version.cuda is None"

# ------------------------------------------------------------
# Download the embedding model as a plain folder, ONNX only:
# config files, tokenizer and the one quantized ONNX file.
# No PyTorch weights are downloaded.
# ------------------------------------------------------------
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['EMBEDDING_MODEL'], local_dir=os.environ['EMBEDDING_MODEL_PATH'], allow_patterns=['*.json', '*.txt', 'onnx/model_quantized.onnx'])" && \
    rm -rf "${EMBEDDING_MODEL_PATH}/.cache" && \
    test -f "${EMBEDDING_MODEL_PATH}/onnx/model_quantized.onnx" && \
    ls -laR "${EMBEDDING_MODEL_PATH}"

# ------------------------------------------------------------
# Verify the model loads from that folder with Hugging Face
# offline, using the same ONNX settings as production.
# The build fails here if anything is missing or needs the network.
# ------------------------------------------------------------
RUN HF_HUB_OFFLINE=1 python -c "import os, onnxruntime as ort; from sentence_transformers import SentenceTransformer; m = SentenceTransformer(os.environ['EMBEDDING_MODEL_PATH'], backend='onnx', model_kwargs={'file_name': 'onnx/model_quantized.onnx', 'provider': 'CPUExecutionProvider', 'session_options': ort.SessionOptions()}, local_files_only=True, trust_remote_code=False); v = m.encode(['QueryPilot build-time offline verification'], normalize_embeddings=True); assert v.shape == (1, 1024), v.shape; print('Offline ONNX load OK', v.shape)"


# ============================================================
# RUNTIME
# ============================================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/querypilot \
    HF_HOME=/home/querypilot/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    EMBEDDING_MODEL_PATH=/opt/models/mxbai-embed-large-v1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libmariadb3 \
    libmagic1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin querypilot \
    && mkdir -p /home/querypilot/.cache/huggingface \
    && chown -R querypilot:querypilot /home/querypilot

WORKDIR /app

# Copy installed Python dependencies.
COPY --from=builder /usr/local /usr/local

# Copy the verified ONNX model folder.
COPY --from=builder --chown=querypilot:querypilot \
    /opt/models/mxbai-embed-large-v1 \
    /opt/models/mxbai-embed-large-v1

# Copy application source.
COPY --chown=querypilot:querypilot . .

USER querypilot

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "config.asgi:application", "--host", "0.0.0.0", "--port", "8000"]