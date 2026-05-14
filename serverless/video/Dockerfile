# AI Gen API v2 — Video worker (LTX-2.3 22B i2v + t2v)
#
# Build:
#   docker build -f serverless/video/Dockerfile -t <user>/ai-gen-video:latest .
#   docker push   <user>/ai-gen-video:latest
#
# Build context = repo root.
#
# Models (~44 GB) live on the RunPod network volume — start.sh symlinks them
# into ComfyUI's models tree. Do NOT bake the weights into the image.

FROM runpod/worker-comfyui:5.4.1-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/comfyui \
    COMFYUI_URL=http://127.0.0.1:8188 \
    VOLUME_OUTPUTS=/runpod-volume/outputs

COPY serverless/video/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY workflows.py /app/workflows.py

COPY serverless/video/handler.py /app/handler.py
COPY serverless/video/start.sh   /start_serverless.sh
RUN chmod +x /start_serverless.sh

WORKDIR /app

CMD ["/start_serverless.sh"]
