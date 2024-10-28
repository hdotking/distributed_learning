# Base image with CUDA, cuDNN, and Python setup
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libexpat1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install flash-attn from source as root
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    pip install . && \
    cd .. && \
    rm -rf flash-attention

# Copy application code. Optionally change user for enhanced security.
COPY . .

# Clean up unnecessary files to reduce image size
RUN rm -rf /tmp/* /root/.cache /usr/share/man /usr/share/doc /usr/share/locale /usr/share/info && \
    find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -r {} + && \
    rm -rf /var/lib/apt/lists/*


# Expose JupyterLab port for access
EXPOSE 8888

# Run JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.disable_check_xsrf=True", \
     "--NotebookApp.allow_origin='*'", "--ServerApp.allow_origin='*'", \
     "--ServerApp.allow_credentials=True"]
