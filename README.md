# Geospatial Classification with CNN-ViT Hybrids

A modular, production-ready deep-learning pipeline for geospatial land classification (agricultural vs. non-agricultural) in PyTorch. The project trains a CNN backbone from scratch and then constructs a CNN–Vision Transformer (ViT) hybrid that uses the pre-trained CNN as a feature extractor, training the full model end-to-end.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Inference API](#inference-api)
- [Docker](#docker)
- [Development](#development)
- [Testing](#testing)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Notebooks](#notebooks)
- [License](#license)

---

## Project Overview

This repository refactors five educational Jupyter notebooks into a maintainable Python package with:

- **Modular architecture** (`src/geospatial_classification/`)
- **YAML-driven configuration** for reproducible experiments
- **CLI entry points** for headless training and evaluation
- **FastAPI inference service** (`api/`) for point and polygon predictions
- **Comprehensive tests** with `pytest`
- **Docker support** for portable deployments
- **Code quality tools**: `black`, `isort`, `flake8`, `mypy`, `pre-commit`

The original notebooks remain in `notebooks/` for reference and learning, while all reusable code has been extracted into the package.

---

## Repository Structure

```
geospatial-classification/
├── src/
│   └── geospatial_classification/     # Main Python package
│       ├── data/                      # Dataset helpers, transforms, auto-extract, splits
│       ├── models/                    # ConvNet, ViT, CNN_ViT_Hybrid
│       ├── training/                  # Train/val loops, reproducibility, checkpointing
│       ├── evaluation/                # Metrics (accuracy, F1, ROC-AUC, etc.) & plots
│       ├── utils/                     # Logging setup, visualization helpers
│       └── scripts/                   # CLI scripts (train_cnn, train_hybrid, evaluate)
├── api/                               # FastAPI inference service
│   └── app/
│       ├── main.py                    # Uvicorn entry point
│       ├── api/predict.py             # /predict/point and /predict/area endpoints
│       ├── schemas/prediction.py      # Request/response Pydantic models
│       └── services/                  # Model loading, tiling, satellite imagery
├── data/                              # Data storage (raw / processed / external)
│   ├── raw/                           # Extracted satellite images
│   ├── processed/                     # Cached train/val split indices
│   └── external/                      # Third-party datasets
├── checkpoints/                       # Saved model weights (.pth files)
├── configs/                           # YAML experiment configurations
├── scripts/                           # Stand-alone shell/Python runner scripts
├── tests/                             # pytest unit tests
├── notebooks/                         # Demo notebook
├── logs/                              # Training/evaluation logs (auto-created)
├── plots/                             # Evaluation plots (auto-created)
├── Dockerfile                         # Container image definition
├── docker-compose.yml                 # Multi-service orchestration
├── pyproject.toml                     # Project metadata, dependencies, tool configs
├── .pre-commit-config.yaml            # Git hooks for code quality
├── .gitignore
├── README.md
└── SUMMARY.md
```

---

## Installation

### Local (editable) install

This project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management and virtual environment creation.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment
# Optional: add --prompt geospatial-classification to show (geospatial-classification) in your terminal prompt
uv venv .venv --prompt geospatial-classification
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with all development tools
uv pip install -e ".[dev]"
```

> **Tip:** The `--prompt` flag is optional. If you omit it, your prompt will show `(.venv)` instead of `(geospatial-classification)` - both work fine.

### System install (not recommended for development)

```bash
pip install .
```

---

## Quick Start

### 1. Dataset

The dataset (~20 MB) is handled automatically:

- If `data/raw/images_dataSAT/` already exists → scripts use it immediately.
- If the extracted folder is missing but `images-dataSAT.tar` is present at the repo root → scripts **auto-extract** it.
- If **both** are missing → scripts **auto-download** the tar from IBM Cloud Object Storage, then extract it.

If you prefer to prep the data manually before training:

```bash
python scripts/setup_data.py
```

**Data layout:**

```
data/
├── raw/               # Extracted satellite images (auto-created)
│   └── images_dataSAT/
│       ├── class_0_non_agri/
│       └── class_1_agri/
├── processed/         # Cached train/val split indices (auto-created)
│   └── split_indices_train0.8_seed42.json
└── external/          # Reserved for third-party datasets
```

### 2. Train the CNN backbone

```bash
gsc-train-cnn --config configs/train_cnn.yaml
```

Or via the module:

```bash
python -m geospatial_classification.scripts.train_cnn --config configs/train_cnn.yaml
```

### 3. Train the CNN-ViT hybrid

```bash
gsc-train-hybrid --config configs/train_hybrid.yaml
```

### 4. Evaluate the PyTorch model

```bash
gsc-evaluate --config configs/evaluate.yaml
```

---

## Configuration

All hyperparameters, paths, and settings live in **YAML files** under `configs/`:

| Config | Purpose |
|--------|---------|
| `configs/train_cnn.yaml` | Train the 6-block CNN backbone |
| `configs/train_hybrid.yaml` | Train the CNN-ViT hybrid (warm-start CNN from Stage 1) |
| `configs/evaluate.yaml` | Evaluate the PyTorch CNN-ViT hybrid |

Example (`configs/train_cnn.yaml`):

```yaml
data:
  dataset_path: "data/raw/images_dataSAT"
  img_size: 64
  batch_size: 128
  train_split: 0.8

model:
  num_classes: 2
  checkpoint_name: "checkpoints/cnn_best.pth"

training:
  epochs: 20
  lr: 0.001
  seed: 42
  device: "auto"
  early_stopping:
    patience: 3
```

You can override any value via environment variables prefixed with `GSC_` (handled by `config.merge_with_env`).

---

## CLI Usage

After installation, three CLI commands are available:

| Command | Description |
|---------|-------------|
| `gsc-train-cnn` | Train the pure CNN classifier |
| `gsc-train-hybrid` | Train the CNN-ViT hybrid |
| `gsc-evaluate` | Evaluate the PyTorch model |

All commands accept `--config <path>`.

---

## Inference API

A FastAPI service in `api/` serves the trained CNN-ViT hybrid for interactive predictions. It is **not** started by Docker Compose; run it separately.

### Install API dependencies

The API deps are in an optional group:

```bash
uv pip install -e ".[api]"
```

### Environment variables

Create `api/.env` and set the following variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `MAPBOX_ACCESS_TOKEN` | Mapbox token for satellite tile fetching | `pk.ey...` |
| `MODEL_PATH` | Path to the trained hybrid checkpoint | `checkpoints/hybrid_best.pth` |
| `DEVICE` | Torch device: `auto`, `cuda`, or `cpu` | `auto` |
| `MAX_AREA_PATCHES` | Max 64×64 tiles allowed for `/predict/area` | `200` |

### Run the server

```bash
uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status, GPU availability, and model load state |
| `POST` | `/api/v1/predict/point` | Classify a single lat/lng point |
| `POST` | `/api/v1/predict/area` | Classify all grid points inside a polygon |

### Example requests

**Point prediction:**

```bash
curl -X POST http://localhost:8000/api/v1/predict/point \
  -H "Content-Type: application/json" \
  -d '{"lat": 41.878, "lng": -93.0977, "zoom": 16}'
```

**Area prediction:**

```bash
curl -X POST http://localhost:8000/api/v1/predict/area \
  -H "Content-Type: application/json" \
  -d '{
    "polygon": [
      [-93.10, 41.87],
      [-93.09, 41.87],
      [-93.09, 41.88],
      [-93.10, 41.88],
      [-93.10, 41.87]
    ],
    "zoom": 16
  }'
```

Polygon coordinates are `[longitude, latitude]` and the ring must be closed. Requests exceeding `MAX_AREA_PATCHES` return `400 Bad Request`.

---

## Docker

### Prerequisites

- **Docker & Docker Compose** installed
- **NVIDIA GPU + drivers** installed on the host
- **nvidia-docker runtime** enabled (Docker Desktop: *Settings → Resources → WSL Integration*; or install `nvidia-docker2` on Linux)

### Quick Run (recommended)

A helper script builds the image and runs the full pipeline sequentially:

```bash
# Default: stable PyTorch CUDA 12.8 (RTX 30/40 series)
./run-docker.sh

# RTX 5060 / Blackwell: nightly PyTorch CUDA 12.8
PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 USE_PRE=1 ./run-docker.sh
```

Or step-by-step:

```bash
# Build the image (default = GPU with stable PyTorch CUDA 12.8)
docker-compose build

# Run individual services
docker-compose run --rm train-cnn
docker-compose run --rm train-hybrid
docker-compose run --rm evaluate
```

### GPU Variant Selection

The Dockerfile supports three build targets via `--build-arg`. The helper scripts (`run-docker.sh`, `docker-test.sh`) support the same selection via environment variables:

| Target | `docker-compose` Build Command | Helper Script Environment Variables |
|--------|-------------------------------|-------------------------------------|
| **GPU (default)** — Stable PyTorch CUDA 12.8, works for most NVIDIA GPUs (RTX 30/40 series) | `docker-compose build` | (none needed) |
| **GPU (RTX 5060 / Blackwell)** — Requires PyTorch nightly with CUDA 12.8 | See command below | `PYTORCH_INDEX=... USE_PRE=1 ./run-docker.sh` |
| **CPU-only** — No GPU needed, smaller image | `docker-compose build --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu` | `PYTORCH_INDEX=https://download.pytorch.org/whl/cpu ./run-docker.sh` |

**RTX 5060 / Blackwell (sm_100/101) — PyTorch Nightly Required:**

The standard PyTorch stable wheels do not yet include native support for the Blackwell architecture. Build with the nightly CUDA 12.8 index and `USE_PRE=1`:

```bash
# Using docker-compose directly
docker-compose build \
  --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 \
  --build-arg USE_PRE=1

# Or using the helper script
PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 USE_PRE=1 ./run-docker.sh
```

> **Why `USE_PRE=1`?** This tells `pip install` to accept pre-release wheels. It is only needed when using a nightly index (stable wheels are always release builds).

### Run tests inside Docker

```bash
# Default: stable PyTorch CUDA 12.8
./docker-test.sh

# RTX 5060 / Blackwell: nightly PyTorch CUDA 12.8
PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 USE_PRE=1 ./docker-test.sh

# Or directly with docker-compose:
docker-compose run --rm test
```

### Notes

- The Dockerfile installs the standard PyTorch wheel, which **bundles CUDA 12.x runtime libraries** — no separate CUDA toolkit is needed inside the container.
- `docker-compose.yml` exposes the GPU to each service via `deploy.resources.reservations.devices`.
- Checkpoints, logs, and plots are written to local `./checkpoints/`, `./logs/`, and `./plots/` via bind mounts.

---

## Development

### Install dev dependencies

```bash
uv pip install -e ".[dev]"
```

### Code formatting

```bash
black src tests scripts
isort src tests scripts
```

### Linting

```bash
flake8 src tests scripts
mypy src
```

---

## Testing

Run the full test suite:

```bash
pytest
```

Run only fast unit tests (skip integration tests):

```bash
pytest -m "not integration"
```

Generate coverage report:

```bash
pytest --cov=geospatial_classification --cov-report=html
```

---

## Pre-commit Hooks

Install hooks once:

```bash
pre-commit install
```

Run manually on all files:

```bash
pre-commit run --all-files
```

Configured hooks:
- `trailing-whitespace`
- `end-of-file-fixer`
- `check-yaml`, `check-json`, `check-toml`
- `black`
- `isort`
- `flake8`

---

## Notebooks

- **`notebooks/`** — Demo notebook that imports from the `geospatial_classification` package.

---

## License

MIT License — see `pyproject.toml` for details.

---

> **Happy hacking!** If you extend this project (e.g., new datasets, deeper ViT configs, or new evaluation metrics), please consider updating the tests and configs accordingly.
