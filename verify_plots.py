
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from backend.metrics_parser import MetricsParser
from backend.plotter import MetricsPlotter

# Path to the specific log file we found earlier
log_file = Path("d:/AI_PROJECTS/Abhijith/nnUNet_Tool/nnUNet_results/Dataset035_KneeData/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_2025_12_11_16_44_11.txt")

if not log_file.exists():
    print(f"Error: Log file not found at {log_file}")
    sys.exit(1)

print(f"Reading log file: {log_file}")

parser = MetricsParser()
with open(log_file, 'r') as f:
    for line in f:
        parser.update_from_line(line)

metrics = parser.get_metrics()
print(f"Parsed {len(metrics.epochs)} epochs")
print(f"Latest Dice: {metrics.dice_scores[-1] if metrics.dice_scores else 'None'}")

if not metrics.epochs:
    print("No epochs parsed! The fix might not be working or the log file is empty.")
    sys.exit(1)

# Try to generate plots
print("Generating plots...")
plotter = MetricsPlotter(output_dir="./test_plots")
plot_data = parser.get_plot_data()

try:
    loss_plot, dice_plot, combined_plot = plotter.create_all_plots(plot_data)
    print(f"Success! Plots generated:")
    print(f"  Loss Plot: {loss_plot}")
    print(f"  Dice Plot: {dice_plot}")
    print(f"  Combined: {combined_plot}")
except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc()
