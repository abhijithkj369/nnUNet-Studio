import os
import shutil
from pathlib import Path

def setup_patches():
    print("Setting up patches directory...")
    
    # Source paths
    venv_site_packages = Path(r"D:\CDAC\nnUNet_Tool\venv\Lib\site-packages")
    src_run_training = venv_site_packages / "nnunetv2/run/run_training.py"
    src_trainer = venv_site_packages / "nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"
    
    # Target paths
    base_dir = Path(r"D:\CDAC\nnUNet_Tool")
    patches_dir = base_dir / "patches"
    target_run_dir = patches_dir / "nnunetv2/run"
    target_trainer_dir = patches_dir / "nnunetv2/training/nnUNetTrainer"
    
    # Create directories
    target_run_dir.mkdir(parents=True, exist_ok=True)
    target_trainer_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"Copying {src_run_training} to {target_run_dir}")
    shutil.copy2(src_run_training, target_run_dir / "run_training.py")
    
    print(f"Copying {src_trainer} to {target_trainer_dir}")
    shutil.copy2(src_trainer, target_trainer_dir / "nnUNetTrainer.py")
    
    print("✅ Patches setup complete")

if __name__ == "__main__":
    setup_patches()
