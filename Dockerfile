FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WESAD_CHECKPOINT_DIR=/app/checkpoints \
    WESAD_RESULTS_DIR=/app/results

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 9099), 2); s.close()"

CMD ["python", "-m", "serving.server", "--host", "0.0.0.0", "--port", "9099", "--norm-mode", "running", "--threads", "1"]