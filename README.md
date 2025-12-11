---
title: Kakinadatrafficanalyis
emoji: 😻
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---
# Helmet + Vehicle Processor (YOLO) — Single-file FastAPI setup

This repo contains a single-file FastAPI application (`main.py`) that:
- runs YOLO detection (ultralytics)
- optionally loads a helmet model at ./models/best.pt if present
- processes images/videos and returns annotated outputs + deduped CSV/JSON/summary as a ZIP

## Files
- main.py            - FastAPI server (single file)
- requirements.txt
- Dockerfile         - CPU container
- Dockerfile.gpu     - optional GPU container (requires nvidia runtime)
- .gitignore
- outputs/           - runtime outputs (annotated, CSV, JSON)

## Local testing (no docker)
1. Create and activate a virtualenv (optional)
2. Install:
   pip install -r requirements.txt
3. Run:
   uvicorn main:app --host 0.0.0.0 --port 7860
4. Call /health or /process endpoints. Example using curl:
   curl -F "file=@/path/to/video.mp4" http://127.0.0.1:7860/process --output results.zip

## Docker (local)
1. Build:
   docker build -t helmet-processor:latest .
   # For GPU Dockerfile:
   # docker build -f Dockerfile.gpu -t helmet-processor:gpu .

2. Run:
   docker run --rm -p 7860:7860 -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs helmet-processor:latest

   # GPU (host must have nvidia-container-toolkit):
   # docker run --gpus all --rm -p 7860:7860 -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs helmet-processor:gpu

3. Upload a file via curl or your frontend:
   curl -F "file=@/path/to/test.mp4" http://127.0.0.1:7860/process --output results.zip

## Deploy to Hugging Face Spaces (Docker)
1. Create a new Space on Hugging Face: choose **Docker**.
2. Add this repo as the Space git remote (replace placeholders):
   git init
   git add .
   git commit -m "initial"
   git remote add hf https://huggingface.co/spaces/<hf-username>/<space-name>
   git push hf main

   The Space will build the Dockerfile automatically. The app must bind to port 7860.

Notes:
- For a helmet model, upload `best.pt` to `/app/models/` in the repo or mount it into the container.
- Free Spaces may not provide GPU; for fast video inference choose a paid Space with GPU or host elsewhere.

## Integrating with your Vercel frontend
From your Vercel site (https://video-ten-silk.vercel.app/) make an HTTP POST to:
  https://<your-space>.hf.space/process
Example JS snippet:

```javascript
async function uploadFile(file) {
  const processUrl = "https://<your-space>.hf.space/process";
  const form = new FormData();
  form.append("file", file, file.name);
  const resp = await fetch(processUrl, { method: "POST", body: form });
  if (!resp.ok) throw new Error("Server error: " + resp.status);
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  // download or preview - example download:
  const a = document.createElement("a");
  a.href = url;
  a.download = "results.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
