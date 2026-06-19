FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy build files first (for caching)
COPY pyproject.toml README.md ./
COPY src ./src

# Install package (non-editable so hatchling properly installs all sub-packages)
RUN pip install --upgrade pip && pip install .[demo]

# Copy runtime files
COPY configs ./configs
COPY app ./app
COPY Data ./Data
COPY artifacts ./artifacts

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
