# Migration Guide

This guide explains how to move this project to another Windows system and set it up correctly.

## Prerequisites
- **Python 3.9 or higher** installed on the new system.
- **Git** (optional, but recommended).

## Steps

### 1. Copy Project Files
Copy the entire project folder to the new system. Ensure the `patches` folder is included.

### 2. Open Terminal
Open a terminal (Command Prompt or PowerShell) and navigate to the project folder:
```bash
cd path\to\nnUNet_Tool
```

### 3. Create Virtual Environment (Recommended)
It's best to use a virtual environment to avoid conflicts:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Run Installation Script
**IMPORTANT:** Do not just run `pip install`. Use the provided script to install dependencies AND apply necessary fixes for Windows:

```bash
python install_and_patch.py
```

This script will:
1. Install all required packages from `requirements.txt`.
2. Automatically patch the `nnunetv2` library to fix known Windows issues (TypeError and small dataset errors).

### 5. Run the Application
Once the installation is complete, start the application:

```bash
python app.py
```

## Troubleshooting
- **"Module not found"**: Make sure you activated the virtual environment before running the script.
- **WinError 2**: If you see this during preprocessing, ensure you are running the patched version. Run `python install_and_patch.py` again to re-apply patches.
