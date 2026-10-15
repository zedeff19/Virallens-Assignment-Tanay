# Bake the embedding model and the built index at image build time.
#
#   docker build -t virallens-rag .
#   docker run --rm --env-file .env virallens-rag --demo
#   docker run --rm --env-file .env virallens-rag "How far did Macquarie's stock price fall?" --trace
#   docker run --rm --entrypoint pytest virallens-rag tests -q               # no API keys needed
#
# --env-file needs plain KEY=value lines (no spaces around "="). Otherwise mount the file instead:
#   docker run --rm -v "$(pwd)/.env:/app/.env:ro" virallens-rag --demo
#
# Needs linux/amd64 (requirements.txt pins torch==2.14.1+cpu); on an ARM host add --platform linux/amd64.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/opt/hf

# The app user owns /app (SQLite LLM cache, Chroma) and /opt/hf (embedding model cache).
RUN useradd --create-home app && mkdir -p /app /opt/hf && chown app:app /app /opt/hf
WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
# The --extra-index-url line inside requirements.txt selects the CPU-only torch wheel.
COPY requirements.txt .
RUN pip install -r requirements.txt

USER app

# Embedding model (also gives src/ingest/chunk.py its tokenizer). Separate layer so a code edit
# does not trigger a re-download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=app:app src ./src
COPY --chown=app:app data/*.pdf ./data/
COPY --chown=app:app tests ./tests
COPY --chown=app:app main.py .

# PDFs -> data/processed/ (exits 1 on validation problems, failing the build), then the Chroma + BM25 index.
RUN python -m src.ingest && python -m src.ingest.index

# From here on the container must use only what was baked in.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# API keys are never baked in: pass them with --env-file .env or -e GROQ_API_KEY=...
ENTRYPOINT ["python", "main.py"]
CMD ["--demo"]
