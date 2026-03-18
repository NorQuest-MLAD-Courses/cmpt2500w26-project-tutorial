# ---- Base image ----
FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Install Python dependencies ----
# Copy only requirements first so Docker caches this layer.
# Re-running `docker build` after a code change skips the install
# unless requirements.txt itself changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code and models ----
COPY src/ src/
COPY models/ models/

# ---- Expose the API port ----
EXPOSE 5000

# ---- Run with Gunicorn (production server) ----
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "src", "app:app"]
