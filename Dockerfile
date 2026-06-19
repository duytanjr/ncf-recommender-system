FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Debug: show what files exist in src
RUN echo "=== Files in /workspace/src ===" && \
    find /workspace/src -type f -name "*.py" | head -30 && \
    echo "=== sys.path ===" && \
    python -c "import sys; print('\n'.join(sys.path))" && \
    echo "=== Try import ===" && \
    python -c "import ncf_recommender; print('ncf_recommender found at:', ncf_recommender.__file__)" || true

# Install ONLY third-party dependencies
RUN pip install --upgrade pip && \
    pip install "torch>=2.6.0" "lightning>=2.5.0" "hydra-core>=1.3.2" "omegaconf>=2.3.0" \
    "numpy>=1.26.4" "pandas>=2.2.3" "scipy>=1.13.1" "scikit-learn>=1.6.1" \
    "pyyaml>=6.0.2" "tqdm>=4.67.1" "rich>=14.0.0" "typer>=0.16.0" "streamlit>=1.45.0"

# Debug after pip install
RUN echo "=== sys.path after pip ===" && \
    python -c "import sys; print('\n'.join(sys.path))" && \
    echo "=== Try import after pip ===" && \
    python -c "from ncf_recommender.data.datasets import load_legacy_implicit_dataset; print('Import OK')" || \
    (echo "=== FAILED. Checking files ===" && \
     ls -la /workspace/src/ncf_recommender/ && \
     ls -la /workspace/src/ncf_recommender/data/ && \
     python -c "import ncf_recommender; print(dir(ncf_recommender)); print(ncf_recommender.__path__)" && \
     echo "=== DONE DEBUG ===" && false)

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
