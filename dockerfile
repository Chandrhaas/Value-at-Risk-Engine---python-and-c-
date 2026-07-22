# syntax=docker/dockerfile:1
#
# Multi-stage build:
#   1. "builder" -- has the full C++ toolchain, compiles the pybind11
#      riskengine extension. Nothing from this stage's toolchain ships
#      in the final image.
#   2. "runtime" -- slim image, just the compiled extension + Python deps
#      + application code. This is what actually gets deployed.
#
# IMPORTANT: both stages pin the exact same Python base image tag. A
# pybind11 extension is compiled against a specific Python ABI -- if the
# builder and runtime stages used different Python versions, `import
# riskengine` would fail at container start, not at build time.

# ---------------------------------------------------------------------------
# Stage 1: build the C++ Monte Carlo engine
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pybind11 provides the headers/CMake config the C++ side compiles
# against. This is a build-time-only tool, kept separate from the
# runtime requirements.txt install in stage 2.
RUN pip install --no-cache-dir pybind11

COPY Makefile .
COPY src_cpp/ src_cpp/

# Assumption to verify against your actual Makefile: that a bare `make`
# (default/first target) builds the riskengine extension into build/,
# matching what api/main.py expects:
#   sys.path.append(os.path.join(..., 'build')); import riskengine
RUN make


# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# libstdc++6: the compiled riskengine extension links against it. The
# builder stage has it as a side effect of build-essential; this slim
# runtime stage does NOT by default. Without this, the container builds
# fine and then fails at actual startup with a "cannot open shared
# object file" ImportError -- easy to miss until you actually run it,
# since `docker build` alone wouldn't catch it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Compiled extension from the build stage (source .cpp/.h files are not
# needed at runtime, so they're deliberately not copied here).
COPY --from=builder /app/build/ build/

# Application code
COPY api/ api/
COPY src_python/ src_python/
COPY frontend/ frontend/

# Trained model artifacts. NOT produced by this build -- train locally
# first (python src_python/train_var_model.py) so models/ already exists
# in the build context before running `docker build`. Baking in an
# already-trained model is deliberate: training needs live internet
# access to yfinance and real time, neither of which belongs in an image
# build, and re-training on every build would make the image
# non-reproducible.
COPY models/ models/

EXPOSE 8000

# No dedicated /health endpoint exists yet -- GET / (the served
# frontend's index.html) is used as a stand-in liveness check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]