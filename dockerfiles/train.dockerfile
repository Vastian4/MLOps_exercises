# Base image
# FROM python:3.11-slim AS base
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# COPY src src/
# COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
# COPY README.md README.md
# COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/


# RUN pip install -r requirements.txt --no-cache-dir --verbose
# RUN pip install . --no-deps --no-cache-dir --verbose
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir


ENTRYPOINT ["python", "-u", "src/code_structures_project/train.py"]
