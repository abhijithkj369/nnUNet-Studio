import sys
import os
import time
import matplotlib
import matplotlib.style
from backend.plotter import MetricsPlotter

def test_init():
    print("Testing initialization...")
    try:
        plotter = MetricsPlotter(output_dir="./test_plots")
        print("Initialization successful.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

def test_warnings():
    print("Testing for warnings...")
    plotter = MetricsPlotter(output_dir="./test_plots")
    
    # Test case 1: Empty data
    print("1. Testing empty data...")
    try:
        plotter.plot_losses([], [], [])
    except Exception as e:
        print(f"Error: {e}")

    # Test case 2: Mismatched data
    print("2. Testing mismatched data...")
    try:
        plotter.plot_losses([1], [], []) 
    except Exception as e:
        print(f"Error: {e}")

def test_memory_allocation():
    print("\nTesting memory allocation (looping 100 times)...")
    plotter = MetricsPlotter(output_dir="./test_plots")
    
    # Simulate data
    epochs = list(range(100))
    train_losses = [0.5] * 100
    val_losses = [0.6] * 100
    dice_scores = [0.8] * 100
    
    plot_data = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'dice_scores': dice_scores
    }
    
    start_time = time.time()
    for i in range(100):
        if i % 10 == 0:
            print(f"Iteration {i}...")
        try:
            plotter.create_all_plots(plot_data)
        except Exception as e:
            print(f"Failed at iteration {i}: {e}")
            break
            
    print(f"Finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_init()
    test_warnings()
    test_memory_allocation()
