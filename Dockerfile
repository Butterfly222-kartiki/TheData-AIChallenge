FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary application files and directories
COPY app.py .
COPY engine/ ./engine/
COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY rank.py .
COPY precompute/ ./precompute/

# Expose Gradio's default port
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app.py"]
