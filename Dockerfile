# Backend API only. The React frontend is deployed on Vercel.
# Bookworm has OpenSSL 3.0. The newer image's OpenSSL 3.5 times out reaching PyPI from the Azure VM.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# scikit-learn needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements-deploy.txt .
RUN pip install --no-cache-dir --timeout 60 --retries 10 -r requirements-deploy.txt
COPY backend/ ./backend/
RUN useradd --no-create-home sigma
USER sigma
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
