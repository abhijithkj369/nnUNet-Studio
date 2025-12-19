"""
Configuration settings for nnUNetv2 Training UI
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Default paths - Force local project folders to avoid conflicts with system env vars
DEFAULT_NNUNET_RAW = str(PROJECT_ROOT / "nnUNet_raw")
DEFAULT_NNUNET_PREPROCESSED = str(PROJECT_ROOT / "nnUNet_preprocessed")
DEFAULT_NNUNET_RESULTS = str(PROJECT_ROOT / "nnUNet_results")

# Supported file formats
SUPPORTED_FILE_ENDING = ".nii.gz"

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_DATASET_ID = 1
DEFAULT_TRAINER = "nnUNetTrainer"
DEFAULT_PLANS = "nnUNetPlans"
DEFAULT_NUM_WORKERS = 2 if os.name == 'nt' else 8

# UI Settings
UI_TITLE = "nnUNetv2 Training Interface"
UI_DESCRIPTION = """
Train nnUNetv2 models for medical image segmentation. 
Upload your dataset in nnUNet format (imagesTr, labelsTr folders) and start training.
"""

# Plot settings
PLOT_WIDTH = 10
PLOT_HEIGHT = 6
PLOT_DPI = 100

# Log settings
MAX_LOG_LINES = 1000
LOG_REFRESH_INTERVAL = 1  # seconds
