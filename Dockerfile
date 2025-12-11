# Use Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some python packages
# git is often needed for pip installs from git repos
# build-essential for compiling C extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Copy the installation and patching script
COPY install_and_patch.py .

# Copy the patches directory
COPY patches ./patches

# Install dependencies and apply patches
# We use the script to ensure the same patching logic is applied
RUN python install_and_patch.py

# Copy the rest of the application
COPY . .

# Create directories for data mapping
RUN mkdir -p nnUNet_raw nnUNet_preprocessed nnUNet_results plots

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
