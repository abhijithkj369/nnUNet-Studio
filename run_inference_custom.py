
import sys
import os

# Add current directory to path so backend can be imported
sys.path.append(os.getcwd())

# Import nnunetv2 modules
import nnunetv2
import nnunetv2.inference.predict_from_raw_data
from nnunetv2.inference.predict_from_raw_data import predict_entry_point

# Import the custom trainer
try:
    from backend.custom_trainer import nnUNetTrainerCustomEpochs
    print("Successfully imported nnUNetTrainerCustomEpochs")
except ImportError as e:
    print(f"Warning: Could not import nnUNetTrainerCustomEpochs: {e}")
    nnUNetTrainerCustomEpochs = None

# Monkey-patch recursive_find_python_class to find our custom trainer
if nnUNetTrainerCustomEpochs is not None:
    original_find_class = nnunetv2.inference.predict_from_raw_data.recursive_find_python_class

    def patched_find_class(folder, class_name, current_module):
        if class_name == "nnUNetTrainerCustomEpochs":
            print(f"DEBUG: Returning patched class for {class_name}")
            return nnUNetTrainerCustomEpochs
        return original_find_class(folder, class_name, current_module)

    # Apply the patch to the module where it's used
    nnunetv2.inference.predict_from_raw_data.recursive_find_python_class = patched_find_class
    print("Successfully patched recursive_find_python_class")

if __name__ == '__main__':
    # This will parse sys.argv just like the original nnUNetv2_predict command
    predict_entry_point()
