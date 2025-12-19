"""
Trainer for nnUNetv2
Manages training process execution and monitoring
"""
import os
import subprocess
import threading
import queue
from pathlib import Path
from typing import Callable, Optional, Dict
import time


class nnUNetTrainer:
    """Manages nnUNet training process"""
    
    def __init__(self, 
                 nnunet_raw: str,
                 nnunet_preprocessed: str,
                 nnunet_results: str):
        """
        Initialize trainer
        
        Args:
            nnunet_raw: Path to nnUNet_raw directory
            nnunet_preprocessed: Path to nnUNet_preprocessed directory
            nnunet_results: Path to nnUNet_results directory
        """
        self.nnunet_raw = Path(nnunet_raw)
        self.nnunet_preprocessed = Path(nnunet_preprocessed)
        self.nnunet_results = Path(nnunet_results)
        
        # Create directories
        self.nnunet_raw.mkdir(exist_ok=True, parents=True)
        self.nnunet_preprocessed.mkdir(exist_ok=True, parents=True)
        self.nnunet_results.mkdir(exist_ok=True, parents=True)
        
        # Process tracking
        self.process = None
        self.is_running = False
        self.log_queue = queue.Queue()
        
    def setup_environment(self):
        """Set up nnUNet environment variables"""
        os.environ['nnUNet_raw'] = str(self.nnunet_raw)
        os.environ['nnUNet_preprocessed'] = str(self.nnunet_preprocessed)
        os.environ['nnUNet_results'] = str(self.nnunet_results)
        
        return {
            'nnUNet_raw': str(self.nnunet_raw),
            'nnUNet_preprocessed': str(self.nnunet_preprocessed),
            'nnUNet_results': str(self.nnunet_results)
        }
    
    def prepare_dataset(self, dataset_folder: str, dataset_id: int) -> tuple[bool, str]:
        """
        Prepare dataset by copying to nnUNet_raw
        
        Args:
            dataset_folder: Source dataset folder with imagesTr, labelsTr, dataset.json
            dataset_id: Dataset ID (e.g., 001 for Dataset001)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            dataset_folder = Path(dataset_folder).resolve()  # Get absolute path
            
            # Check for dataset.json
            dataset_json = dataset_folder / "dataset.json"
            if not dataset_json.exists():
                return False, "dataset.json not found in dataset folder. Please generate it first."
            
            # Determine dataset name from dataset.json or use default
            import json
            with open(dataset_json, 'r') as f:
                dataset_info = json.load(f)
                dataset_name = dataset_info.get('name', 'CustomDataset')
            
            # Create target folder in nnUNet_raw
            target_folder = (self.nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}").resolve()
            
            # Check if source and target are the same
            if dataset_folder == target_folder:
                msg = f"Dataset is already in the correct location: {target_folder}\nNo copying needed."
                return True, msg
            
            # Create target folder
            target_folder.mkdir(exist_ok=True, parents=True)
            
            # Copy dataset.json
            import shutil
            target_json = target_folder / "dataset.json"
            if target_json.resolve() != dataset_json.resolve():
                shutil.copy(dataset_json, target_json)
            
            # Copy imagesTr and labelsTr
            for folder_name in ['imagesTr', 'labelsTr']:
                source = dataset_folder / folder_name
                target = target_folder / folder_name
            if log_callback:
                log_callback(f"Running preprocessing: {' '.join(cmd)}\n")
            
            # Run preprocessing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                if log_callback:
                    log_callback(line)
            
            process.wait()
            
            if process.returncode == 0:
                return True, "Preprocessing completed successfully"
            else:
                return False, f"Preprocessing failed with return code {process.returncode}"
                
        except Exception as e:
            return False, f"Error during preprocessing: {str(e)}"
    
    def start_training(self, 
                      dataset_id: int,
                      fold: int = 0,
                      trainer: str = "nnUNetTrainer",
                      plans: str = "nnUNetPlans",
                      configuration: str = "3d_fullres",
                      num_epochs: int = 100,
                      num_workers: int = 8,
                      log_callback: Optional[Callable] = None) -> tuple[bool, str]:
        """
        Start nnUNet training
        
        Args:
            dataset_id: Dataset ID
            fold: Fold to train (0-4, or 'all')
            trainer: Trainer class name
            plans: Plans identifier
            configuration: Configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
            num_epochs: Number of epochs to train
            num_workers: Number of data loading workers
            log_callback: Callback function for log messages
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.setup_environment()
            
            # Set number of workers
            os.environ['nnUNet_n_proc_DA'] = str(num_workers)
            
            # Build training command
            # Use python -m to call nnUNetv2 modules (works better on Windows)
            import sys
            cmd = [
                sys.executable,
                "-m", "nnunetv2.run.run_training",
                str(dataset_id),
                configuration,
                str(fold),
                "-tr", trainer,
                "-p", plans,
                "--npz"  # Save softmax predictions
            ]
            
            # Add epochs if specified
            if num_epochs and num_epochs != 1000:  # nnUNet default is 1000
                cmd.extend(["--c"])  # Continue from checkpoint if exists
            
            if log_callback:
                log_callback(f"Starting training: {' '.join(cmd)}\n")
                log_callback(f"Training for {num_epochs} epochs\n")
                log_callback("="*80 + "\n")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()
            )
            
            self.is_running = True
            
            # Start thread to read output
            def read_output():
                try:
                    for line in self.process.stdout:
                        if log_callback:
                            log_callback(line)
                        self.log_queue.put(line)
                except Exception as e:
                    if log_callback:
                        log_callback(f"Error reading output: {str(e)}\n")
                finally:
                    self.is_running = False
            
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            
            return True, "Training started successfully"
            
        except Exception as e:
            self.is_running = False
            return False, f"Error starting training: {str(e)}"
    
    def stop_training(self):
        """Stop the training process"""
        if self.process and self.is_running:
            self.process.terminate()
            self.is_running = False
            return True, "Training stopped"
        return False, "No training process is running"
    
    def is_training_running(self) -> bool:
        """Check if training is currently running"""
        return self.is_running
    
    def get_logs(self) -> list:
        """Get accumulated logs from queue"""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs
