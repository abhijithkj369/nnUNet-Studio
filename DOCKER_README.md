# Running with Docker

This guide explains how to run the nnUNetv2 Training UI using Docker. This is the easiest way to run the application on any system (Windows, Linux, macOS) without worrying about Python environments or dependencies.

## Prerequisites

1.  **Docker Desktop** (or Docker Engine on Linux) installed.
2.  **NVIDIA Container Toolkit** (Optional, but highly recommended for GPU training).
    *   Without this, training will run on CPU, which is very slow.

## Quick Start

1.  Open a terminal in the project folder.
2.  Build and start the container:
    ```bash
    docker-compose up --build
    ```
3.  Open your browser and go to: `http://localhost:7860`

## Data Persistence

The `docker-compose.yml` file is configured to save your data to the host machine so it persists even if you delete the container.

The following folders in your project directory are mapped to the container:
*   `nnUNet_raw`: Raw dataset files.
*   `nnUNet_preprocessed`: Preprocessed data.
*   `nnUNet_results`: Trained models and checkpoints.
*   `plots`: Training visualization plots.

## GPU Support

The configuration is set up to use all available NVIDIA GPUs.
If you don't have a GPU or haven't set up the NVIDIA Container Toolkit, you may need to comment out the `deploy` section in `docker-compose.yml` to run on CPU (not recommended for training).

## Moving to Another System

To run this on a different computer:

1.  **Copy the Project Folder**: Copy the entire `nnUNet_Tool` folder to the new machine.
    *   **Tip**: You can exclude the `venv` folder and large data folders (`nnUNet_raw`, `nnUNet_results`, etc.) if you want to transfer them separately or start fresh.
    *   **Essential Files**: You MUST include `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `install_and_patch.py`, `patches/`, and `app.py` (plus the `backend` folder).

2.  **Install Docker**: Install Docker Desktop (Windows/Mac) or Docker Engine (Linux) on the new machine.

3.  **Run**: Open a terminal in the folder on the new machine and run:
    ```bash
    docker-compose up --build
    ```

## Troubleshooting

*   **"driver: nvidia not found"**: This means Docker can't see your GPU. Ensure you have installed the NVIDIA Container Toolkit and restarted Docker.
    *   To run on CPU only, remove the `deploy` section from `docker-compose.yml`.
