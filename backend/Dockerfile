FROM python:3.11-slim

# libgomp1 is LightGBM's OpenMP runtime dependency -- python:3.11-slim
# doesn't ship it, and without it `import lightgbm` fails at the C-extension
# load step (not an ImportError, so app.train's fallback wouldn't catch it).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
