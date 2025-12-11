"""
Configuration settings for nnUNetv2 Training UI
"""
import os
from pathlib import Path

# Default paths
DEFAULT_NNUNET_RAW = os.environ.get("nnUNet_raw", "./nnUNet_raw")
DEFAULT_NNUNET_PREPROCESSED = os.environ.get("nnUNet_preprocessed", "./nnUNet_preprocessed")
DEFAULT_NNUNET_RESULTS = os.environ.get("nnUNet_results", "./nnUNet_results")

# Supported file formats
SUPPORTED_FILE_ENDING = ".nii.gz"

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_DATASET_ID = 1
DEFAULT_TRAINER = "nnUNetTrainer"
DEFAULT_PLANS = "nnUNetPlans"

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
