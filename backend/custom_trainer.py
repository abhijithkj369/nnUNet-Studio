
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerCustomEpochs(nnUNetTrainer):
    """
    Custom trainer that allows setting the number of epochs via environment variable.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Override num_epochs from environment variable if present
        if 'nnUNet_max_epochs' in os.environ:
            try:
                self.num_epochs = int(os.environ['nnUNet_max_epochs'])
                print(f"Using custom number of epochs: {self.num_epochs}")
            except ValueError:
                print(f"Invalid value for nnUNet_max_epochs: {os.environ['nnUNet_max_epochs']}. Using default: {self.num_epochs}")
