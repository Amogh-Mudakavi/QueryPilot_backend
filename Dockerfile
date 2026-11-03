# ============================================================
# BUILDER
# ============================================================
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/querypilot/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

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

# Install CPU-only PyTorch.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.0

# Install application dependencies.
RUN python -m pip install --no-cache-dir -r requirements.txt

# Verify Python dependencies and CPU-only PyTorch.
RUN python -m pip check && \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); assert torch.version.cuda is None"

# ------------------------------------------------------------
# Download and verify the embedding model during image build.
# ------------------------------------------------------------

ARG EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1

RUN mkdir -p "${HF_HOME}" && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('${EMBEDDING_MODEL}', device='cpu'); vectors = model.encode(['QueryPilot build-time model verification']); assert len(vectors) == 1; print('Embedding model downloaded and verified: ${EMBEDDING_MODEL}')"


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
    TOKENIZERS_PARALLELISM=false

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

# Copy the model cache downloaded during the build.
COPY --from=builder --chown=querypilot:querypilot \
    /home/querypilot/.cache/huggingface \
    /home/querypilot/.cache/huggingface

# Copy application source.
COPY --chown=querypilot:querypilot . .

USER querypilot

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "config.asgi:application", "--host", "0.0.0.0", "--port", "8000"]