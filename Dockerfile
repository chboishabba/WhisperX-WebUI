# 1. Use official NVIDIA CUDA 12.6 image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# 2. Install system dependencies + Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-setuptools python3-dev git ffmpeg libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Set working directory
WORKDIR /app

# 4. Copy the repository files
COPY . /app

# 5. Upgrade pip, setuptools, and wheel FIRST so git repos compile correctly
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 6. Install dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# 7. Set the default startup command
ENTRYPOINT ["/bin/bash", "-c", "python app.py ${COMMANDLINE_ARGS}"]
