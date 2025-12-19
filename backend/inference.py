
import os
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

class InferenceManager:
    """Manages nnUNet inference and evaluation"""
    
    def __init__(self, nnunet_results: str, nnunet_raw: str):
        self.nnunet_results = Path(nnunet_results)
        self.nnunet_raw = Path(nnunet_raw)
        
    def list_models(self) -> List[str]:
        """List available trained models (datasets)"""
        if not self.nnunet_results.exists():
            return []
            
        # Structure: nnUNet_results/DatasetXXX_Name/nnUNetTrainer__nnUNetPlans__3d_fullres
        models = []
        for dataset_dir in self.nnunet_results.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name.startswith('Dataset'):
                models.append(dataset_dir.name)
        return sorted(models)
        
    def list_folds(self, dataset_name: str, configuration: str = "3d_fullres", 
                  trainer: str = "nnUNetTrainerCustomEpochs", plans: str = "nnUNetPlans") -> List[str]:
        """List available folds for a model"""
        # Try custom trainer first, then default
        trainers_to_check = [trainer, "nnUNetTrainer"]
        
        for tr in trainers_to_check:
            model_dir = self.nnunet_results / dataset_name / f"{tr}__{plans}__{configuration}"
            if model_dir.exists():
                folds = []
                for fold_dir in model_dir.iterdir():
                    if fold_dir.is_dir() and fold_dir.name.startswith('fold_'):
                        # Check if checkpoint exists
                        if (fold_dir / "checkpoint_final.pth").exists() or (fold_dir / "checkpoint_best.pth").exists():
                            folds.append(fold_dir.name)
                if folds:
                    return sorted(folds)
        return []

    def run_inference(self, dataset_name: str, input_folder: str, output_folder: str, 
                     fold: str, configuration: str = "3d_fullres", 
                     trainer: str = "nnUNetTrainerCustomEpochs", plans: str = "nnUNetPlans",
                     chk: str = "checkpoint_best.pth") -> Tuple[bool, str]:
        """
        Run inference using nnUNetv2_predict
        """
        try:
            # Set environment variables to ensure correct paths are used
            import os
            env = os.environ.copy()
            
            # Use absolute paths to ensure we're using the correct directories
            nnunet_raw_abs = str(self.nnunet_raw.resolve())
            nnunet_preprocessed_abs = str(self.nnunet_raw.parent.resolve() / 'nnUNet_preprocessed')
            nnunet_results_abs = str(self.nnunet_results.resolve())
            
            env['nnUNet_raw'] = nnunet_raw_abs
            env['nnUNet_preprocessed'] = nnunet_preprocessed_abs  
            env['nnUNet_results'] = nnunet_results_abs
            
            print(f"DEBUG: Using paths:")
            print(f"  nnUNet_raw: {nnunet_raw_abs}")
            print(f"  nnUNet_preprocessed: {nnunet_preprocessed_abs}")
            print(f"  nnUNet_results: {nnunet_results_abs}")
            
            # Check if trainer exists, fallback to default if not
            model_dir = self.nnunet_results / dataset_name / f"{trainer}__{plans}__{configuration}"
            if not model_dir.exists():
                trainer = "nnUNetTrainer" # Fallback
            
            import sys
            cmd = [
                sys.executable, "run_inference_custom.py",
                "-i", input_folder,
                "-o", output_folder,
                "-d", dataset_name.replace("Dataset", "").split("_")[0], # Extract ID
                "-c", configuration,
                "-f", fold.replace("fold_", ""),
                "-tr", trainer,
                "-p", plans,
                "-chk", chk
            ]
            
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            
            # Run command with correct environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                env=env  # Use our custom environment
            )
            
            stdout, _ = process.communicate()
            
            if process.returncode == 0:
                return True, "Inference completed successfully!"
            else:
                return False, f"Inference failed:\n{stdout}"
                
        except Exception as e:
            return False, f"Error running inference: {str(e)}"

    def evaluate_predictions(self, gt_folder: str, pred_folder: str) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        Calculates Dice score.
        """
        try:
            import SimpleITK as sitk
            import numpy as np
            
            gt_path = Path(gt_folder)
            pred_path = Path(pred_folder)
            
            metrics = {
                'dice': [],
                'iou': []
            }
            
            # Iterate over prediction files
            for pred_file in pred_path.glob("*.nii.gz"):
                gt_file = gt_path / pred_file.name
                if not gt_file.exists():
                    # Try matching without _0000 suffix if it exists in pred but not GT
                    # nnUNet inputs have _0000, outputs usually don't, but let's check
                    continue
                    
                # Load images
                pred_img = sitk.ReadImage(str(pred_file))
                gt_img = sitk.ReadImage(str(gt_file))
                
                pred_arr = sitk.GetArrayFromImage(pred_img)
                gt_arr = sitk.GetArrayFromImage(gt_img)
                
                # Calculate metrics for each class (excluding background 0)
                # Assuming binary or multiclass
                classes = np.unique(gt_arr)
                classes = classes[classes != 0] # Remove background
                
                if len(classes) == 0:
                    continue
                    
                for c in classes:
                    pred_c = (pred_arr == c)
                    gt_c = (gt_arr == c)
                    
                    intersection = np.logical_and(pred_c, gt_c).sum()
                    union = np.logical_or(pred_c, gt_c).sum()
                    sum_pixels = pred_c.sum() + gt_c.sum()
                    
                    dice = (2.0 * intersection) / sum_pixels if sum_pixels > 0 else 1.0
                    iou = intersection / union if union > 0 else 1.0
                    
                    metrics['dice'].append(dice)
                    metrics['iou'].append(iou)
            
            if not metrics['dice']:
                return {'error': 'No matching files or classes found'}
                
            return {
                'Mean Dice': np.mean(metrics['dice']),
                'Mean IoU': np.mean(metrics['iou'])
            }
            
        except ImportError:
            return {'error': 'SimpleITK not installed'}
        except Exception as e:
            return {'error': f'Evaluation failed: {str(e)}'}
