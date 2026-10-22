# CPU-only image. NOTE: not built or run in the authoring environment (see BUILD_REPORT.md).
# For Jetson (aarch64) build on the device itself; do not assume desktop CUDA/PyTorch wheels.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-optional.txt
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8001
# Models are LOADED, never trained, at startup: mount or bake in ./artifacts first.
CMD ["uvicorn", "plant_ml.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
