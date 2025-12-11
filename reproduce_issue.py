import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from backend.metrics_parser import MetricsParser

def test_multiline_parsing():
    parser = MetricsParser(max_epochs=100)
    
    # Simulate log lines
    logs = [
        "2025-12-11 17:07:02.573555: Epoch 0 ",
        "2025-12-11 17:07:50.431181: train_loss -0.2512 ",
        "2025-12-11 17:07:50.431692: val_loss -0.3853 ",
        "2025-12-11 17:07:50.439940: Pseudo dice [np.float32(0.582), np.float32(0.5658)] ",
        "2025-12-11 17:07:51.048797: Epoch 1 ",
        "2025-12-11 17:08:35.329361: train_loss -0.6552 ",
        "2025-12-11 17:08:35.329361: val_loss -0.6187 ",
        "2025-12-11 17:08:35.337475: Pseudo dice [np.float32(0.7855), np.float32(0.7561)] "
    ]
    
    print("Simulating log stream...")
    for line in logs:
        print(f"Processing: {line.strip()}")
        parser.update_from_line(line)
        
    metrics = parser.get_metrics()
    print("\nExtracted Metrics:")
    print(f"Epochs: {metrics.epochs}")
    print(f"Train Losses: {metrics.train_losses}")
    print(f"Val Losses: {metrics.val_losses}")
    print(f"Dice Scores: {metrics.dice_scores}")
    
    # Verification
    assert len(metrics.epochs) == 2, f"Expected 2 epochs, got {len(metrics.epochs)}"
    assert len(metrics.train_losses) == 2, "Missing train losses"
    assert len(metrics.dice_scores) == 2, "Missing dice scores"
    
    # Check values for Epoch 0
    assert metrics.epochs[0] == 0
    assert metrics.train_losses[0] == -0.2512
    assert abs(metrics.dice_scores[0] - 0.5739) < 0.001  # (0.582 + 0.5658)/2 = 0.5739
    
    # Check values for Epoch 1
    assert metrics.epochs[1] == 1
    assert metrics.train_losses[1] == -0.6552
    
    # Check progress
    curr, total, pct = parser.get_progress()
    print(f"\nProgress: Epoch {curr}/{total} ({pct}%)")
    assert curr == 2  # 0-indexed, so finished epoch 1 means current is 2 (or 1 depending on interpretation, let's check code)
    # Code: current = self.current_epoch + 1. current_epoch is 1. So returns 2.
    
    print("\nSUCCESS: Multi-line parsing verified!")

if __name__ == "__main__":
    test_multiline_parsing()
