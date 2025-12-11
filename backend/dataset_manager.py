"""
Dataset Manager for nnUNetv2
Handles dataset validation, scanning, and dataset.json generation
"""
import os
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from typing import Dict, List, Tuple, Optional


class DatasetManager:
    """Manages nnUNet dataset operations"""
    
    def __init__(self, dataset_folder: str):
        """
        Initialize dataset manager
        
        Args:
            dataset_folder: Path to dataset folder containing imagesTr and labelsTr
        """
        self.dataset_folder = Path(dataset_folder)
        self.images_folder = self.dataset_folder / "imagesTr"
        self.labels_folder = self.dataset_folder / "labelsTr"
        
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate that the dataset has proper nnUNet structure
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.dataset_folder.exists():
            return False, f"Dataset folder does not exist: {self.dataset_folder}"
        
        if not self.images_folder.exists():
            return False, f"imagesTr folder not found in {self.dataset_folder}"
        
        if not self.labels_folder.exists():
            return False, f"labelsTr folder not found in {self.dataset_folder}"
        
        # Check for .nii.gz files
        image_files = list(self.images_folder.glob("*.nii.gz"))
        label_files = list(self.labels_folder.glob("*.nii.gz"))
        
        if len(image_files) == 0:
            return False, "No .nii.gz files found in imagesTr folder"
        
        if len(label_files) == 0:
            return False, "No .nii.gz files found in labelsTr folder"
        
        return True, f"Valid dataset structure found with {len(image_files)} images and {len(label_files)} labels"
    
    def scan_dataset(self) -> Dict:
        """
        Scan dataset and extract metadata
        
        Returns:
            Dictionary with dataset information
        """
        image_files = sorted(self.images_folder.glob("*.nii.gz"))
        label_files = sorted(self.labels_folder.glob("*.nii.gz"))
        
        # Extract case identifiers and channel information
        cases = {}
        channels = set()
        
        for img_file in image_files:
            # Parse filename: case_XXXX_YYYY.nii.gz
            name = img_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
            
            # Extract case ID and channel (e.g., case_0000, case_0001)
            parts = name.split('_')
            if len(parts) >= 2:
                # Last part is channel (e.g., 0000)
                channel = parts[-1]
                case_id = '_'.join(parts[:-1])
                
                channels.add(channel)
                
                if case_id not in cases:
                    cases[case_id] = {'images': [], 'label': None}
                
                cases[case_id]['images'].append(str(img_file.name))
        
        # Match labels
        for label_file in label_files:
            name = label_file.stem.replace('.nii', '')
            # Label files typically don't have channel suffix
            case_id = name
            
            if case_id in cases:
                cases[case_id]['label'] = str(label_file.name)
        
        return {
            'cases': cases,
            'num_cases': len(cases),
            'channels': sorted(list(channels))
        }
    
    def extract_labels_from_segmentation(self, max_samples: int = 5) -> Dict[str, int]:
        """
        Extract unique labels from segmentation masks
        
        Args:
            max_samples: Maximum number of files to sample for label extraction
            
        Returns:
            Dictionary mapping label names to label values
        """
        label_files = sorted(self.labels_folder.glob("*.nii.gz"))
        unique_labels = set()
        
        # Sample files to extract labels
        sample_files = label_files[:min(max_samples, len(label_files))]
        
        print(f"Scanning {len(sample_files)} label files for unique labels...")
        
        for label_file in sample_files:
            try:
                img = nib.load(str(label_file))
                data = img.get_fdata()
                labels_in_file = np.unique(data)
                unique_labels.update(labels_in_file.astype(int))
            except Exception as e:
                print(f"Warning: Could not read {label_file.name}: {e}")
        
        # Sort labels
        sorted_labels = sorted(list(unique_labels))
        
        # Create label dictionary
        label_dict = {"background": 0}
        
        for label_value in sorted_labels:
            if label_value == 0:
                continue  # Skip background, already added
            label_dict[f"label_{int(label_value)}"] = int(label_value)
        
        return label_dict
    
    def generate_dataset_json(self, 
                             dataset_name: str = "CustomDataset",
                             description: str = "Medical Image Segmentation Dataset",
                             modality_names: Optional[List[str]] = None,
                             output_path: Optional[str] = None) -> Tuple[bool, str, Dict]:
        """
        Generate dataset.json file for nnUNet
        
        Args:
            dataset_name: Name of the dataset
            description: Description of the dataset
            modality_names: List of modality names (e.g., ["CT", "MRI"]). If None, auto-detect
            output_path: Where to save dataset.json. If None, saves to dataset folder
            
        Returns:
            Tuple of (success, message, dataset_dict)
        """
        try:
            # Validate structure first
            is_valid, msg = self.validate_structure()
            if not is_valid:
                return False, msg, {}
            
            # Scan dataset
            scan_result = self.scan_dataset()
            cases = scan_result['cases']
            channels = scan_result['channels']
            
            # Extract labels
            labels_dict = self.extract_labels_from_segmentation()
            
            # Create channel names
            if modality_names and len(modality_names) == len(channels):
                channel_names = {ch: modality_names[i] for i, ch in enumerate(channels)}
            else:
                # Auto-generate channel names
                channel_names = {ch: f"modality_{ch}" for ch in channels}
            
            # Build training list
            training_cases = []
            for case_id, case_data in cases.items():
                if case_data['label']:  # Only include cases with labels
                    training_cases.append({
                        "image": f"./imagesTr/{case_data['images'][0]}",  # First image channel
                        "label": f"./labelsTr/{case_data['label']}"
                    })
            
            # Create dataset.json structure
            dataset_json = {
                "channel_names": channel_names,
                "labels": labels_dict,
                "numTraining": len(training_cases),
                "file_ending": ".nii.gz",
                "name": dataset_name,
                "description": description,
                "tensorImageSize": "3D",
                "training": training_cases
            }
            
            # Save to file
            if output_path is None:
                output_path = self.dataset_folder / "dataset.json"
            else:
                output_path = Path(output_path)
            
            with open(output_path, 'w') as f:
                json.dump(dataset_json, f, indent=2)
            
            summary = f"""
Dataset JSON generated successfully!
- Dataset: {dataset_name}
- Training cases: {len(training_cases)}
- Channels: {len(channels)} ({', '.join(channel_names.values())})
- Labels: {len(labels_dict)} ({', '.join(labels_dict.keys())})
- Saved to: {output_path}
"""
            
            return True, summary, dataset_json
            
        except Exception as e:
            return False, f"Error generating dataset.json: {str(e)}", {}
