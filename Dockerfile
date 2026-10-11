# Tek imaj, uc yerlesim. Ayni Dockerfile x86 (bulut VPS) ve aarch64
# (Raspberry Pi 4/5, 64-bit OS) icin derlenir:
#   docker buildx build --platform linux/amd64,linux/arm64 -t wesad-stress-server:v2 .
# ya da her makinede yerelde: docker compose build
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WESAD_CHECKPOINT_DIR=/app/checkpoints \
    WESAD_RESULTS_DIR=/app/results \
    WESAD_IMAGE_TAG=wesad-stress-server:v2

WORKDIR /app

COPY requirements.txt .
# torch CPU indeksinden: PyPI'nin x86 tekerlegi CUDA'li gelir (+cu130),
# CPU-only VPS'te yuzlerce MB fazladan bellek ve farkli kod yolu demek.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch; assert '+cu' not in torch.__version__, torch.__version__; print('torch', torch.__version__)"

COPY config.py .
COPY data ./data
COPY models ./models
COPY profiling ./profiling
COPY serving ./serving
COPY streaming ./streaming
COPY xai ./xai
COPY checkpoints ./checkpoints
COPY wesad_best_model_f32-128_k9.pt .

RUN mkdir -p /app/results

EXPOSE 9099

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 9099), 2); s.close()"

CMD ["python", "-m", "serving.server", "--host", "0.0.0.0", "--port", "9099", "--norm-mode", "running", "--threads", "1"]
