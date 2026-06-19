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

# Install ONLY third-party dependencies (skip hatchling package build entirely)
RUN pip install --upgrade pip && \
    pip install "torch>=2.6.0" "lightning>=2.5.0" "hydra-core>=1.3.2" "omegaconf>=2.3.0" \
    "numpy>=1.26.4" "pandas>=2.2.3" "scipy>=1.13.1" "scikit-learn>=1.6.1" \
    "pyyaml>=6.0.2" "tqdm>=4.67.1" "rich>=14.0.0" "typer>=0.16.0" "streamlit>=1.45.0"

# Verify imports work during build — fail fast if broken
RUN python -c "from ncf_recommender.data.datasets import load_legacy_implicit_dataset; print('Import OK')"

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
