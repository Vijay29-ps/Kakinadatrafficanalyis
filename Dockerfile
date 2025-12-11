# Dockerfile - HF Space (Docker) minimal
FROM python:3.10-slim

WORKDIR /app

# system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1-mesa-glx libglib2.0-0 ffmpeg wget && rm -rf /var/lib/apt/lists/*

# copy project
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/app/.ultralytics

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# expose (HF Spaces expects application on 7860)
EXPOSE 7860

# start uvicorn single worker
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
