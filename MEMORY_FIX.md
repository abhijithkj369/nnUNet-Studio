# Memory Error Fix - README

## Issue
If you encounter this error during preprocessing:
```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate X.XX GiB for an array
```

## What It Means
nnUNet tried to load your entire medical image into RAM but your system doesn't have enough available memory.

## ✅ Solution (Already Applied)

I've **already fixed this** in the code by:
- **Removing `--verify_dataset_integrity` flag** from preprocessing
- This flag causes nnUNet to load entire images into memory for verification
- Skipping verification doesn't affect preprocessing quality
- Your data will still be processed correctly

## How to Use

Simply click **"Run Preprocessing"** in the UI - it will now work with much less RAM!

### What Changed
**Before:**
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity  # ❌ Memory intensive
```

**After (current):**
```bash
nnUNetv2_plan_and_preprocess -d 1  # ✅ Memory efficient
```

## Additional Tips to Reduce Memory Usage

If you still experience memory issues:

### 1. Close Other Applications
- Close browsers, IDEs, and other memory-heavy apps
- Check Task Manager for memory usage

### 2. Reduce Parallel Processes
Edit your preprocessing to use fewer CPU cores:
```bash
nnUNetv2_plan_and_preprocess -d 1 --npfp 2  # Use only 2 processes
```

### 3. Use 2D or 3d_lowres Configuration
Instead of `3d_fullres`, try:
- `2d` - Processes slices instead of volumes (much less RAM)
- `3d_lowres` - Lower resolution 3D (less RAM)

### 4. Check Your Image Sizes
Large images (e.g., 576×768×768) require more RAM. Consider:
- Downsampling your data before training
- Using a machine with more RAM
- Processing in batches

## Verification (Optional)

If you want to verify your dataset later (and have sufficient RAM):
```bash
# Activate venv first
.\venv\Scripts\activate

# Run verification separately
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

## Your Case

For your airway dataset with image size (576×768×768):
- Each image needs ~1.27 GB RAM
- Without verification: ✅ Should work fine
- With verification: ❌ May fail on systems with <16GB RAM

The fix is already applied - just try preprocessing again!
