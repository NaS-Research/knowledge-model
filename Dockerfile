FROM python:3.11-slim

# ---------- system-level deps ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Strip build tool‑chain to shrink final image
RUN apt-get purge -y build-essential git curl && \
    apt-get autoremove -y --purge && \
    rm -rf /var/lib/apt/lists/*

# ---------- project code ----------
COPY . .

# ---------- runtime ----------
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "knowledge_model.main:app", "--host", "0.0.0.0", "--port", "8080"]