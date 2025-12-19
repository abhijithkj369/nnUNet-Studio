
import sys
import os
import argparse
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from backend.custom_trainer import nnUNetTrainerCustomEpochs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Can be "all"')
    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used. Default: nnUNetTrainer',
                        required=False, default='nnUNetTrainer')
    parser.add_argument('-p', '--plans_identifier', help='Name of the Plans file used. Default: nnUNetPlans',
                        required=False, default='nnUNetPlans')
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (will only '
                             'be used when actually training. --val will ignore this)')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='if set then validation softmax probabilities will be exported')
    parser.add_argument('--val', action='store_true', required=False,
                        help='if set then we will only run validation')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='if set then we will not save checkpoints')
    
    args = parser.parse_args()
    
    # Handle fold
    if args.fold == 'all':
        fold = 'all'
    else:
        fold = int(args.fold)
        
    # Get dataset name
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, dataset_name)
    
    # Load plans
    plans_file = join(preprocessed_dataset_folder_base, args.plans_identifier + '.json')
    plans = load_json(plans_file)
    
    # Load dataset.json
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    
    # Determine device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"Instantiating custom trainer: nnUNetTrainerCustomEpochs")
    print(f"Dataset: {dataset_name}")
    print(f"Configuration: {args.configuration}")
    print(f"Fold: {fold}")
    
    trainer = nnUNetTrainerCustomEpochs(
        plans=plans,
        configuration=args.configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )
    
    # Set optional args
    if args.disable_checkpointing:
        trainer.disable_checkpointing = args.disable_checkpointing
        
    # Run training
    if args.val:
        trainer.perform_actual_validation(args.npz)
    else:
        trainer.run_training()

if __name__ == '__main__':
    main()
