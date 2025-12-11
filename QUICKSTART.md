# Quick Start Guide - nnUNetv2 Training UI

## 🚀 Getting Started in 3 Steps

### Step 1: Start the Application

Open a terminal and run:
```bash
cd d:\CDAC\nnUNet_Tool
.\venv\Scripts\activate
python app.py
```

The application will open in your browser at `http://localhost:7860`

### Step 2: Prepare Your Dataset

#### Tab 1: Dataset Preparation

1. **Enter your dataset path** (e.g., `D:\MyData\LiverDataset`)
   - Your folder should contain `imagesTr` and `labelsTr` subdirectories

2. **Click "Validate Dataset Structure"**
   - Verify it finds your .nii.gz files

3. **Fill in dataset information:**
   - Dataset Name: e.g., "LiverSegmentation"
   - Description: e.g., "CT liver segmentation dataset"
   - Modality Names: e.g., "CT" (or leave blank)

4. **Click "Generate dataset.json"**
   - Review the generated JSON
   - Check that labels are correctly identified

### Step 3: Train Your Model

#### Tab 2: Training Configuration

1. **Configure paths** (defaults are fine for first run):
   - nnUNet_raw: `./nnUNet_raw`
   - nnUNet_preprocessed: `./nnUNet_preprocessed`
   - nnUNet_results: `./nnUNet_results`

2. **Set training parameters:**
   - Dataset ID: `1` (for Dataset001)
   - Number of Epochs: `100` (or fewer for testing)
   - Fold: `0`
   - Configuration: `3d_fullres`

3. **Click "Setup Training Environment"**
   - Wait for confirmation

4. **Click "Run Preprocessing"**
   - This may take 5-30 minutes depending on dataset size
   - Watch the logs for progress

#### Tab 3: Training & Monitoring

1. **Click "Start Training"**
   - Training logs will appear in real-time
   - Loss plots update automatically every 2 seconds
   - Dice score plots show model performance

2. **Monitor progress:**
   - Watch the current metrics display
   - Check loss curves for convergence
   - Observe dice scores improving

3. **Stop training** (if needed):
   - Click "Stop Training" button

## 📊 Understanding the Plots

### Loss Plot
- **Blue line**: Training loss (should decrease)
- **Red line**: Validation loss (should decrease without overfitting)
- Look for smooth convergence

### Dice Score Plot
- **Green line**: Segmentation accuracy (0 to 1)
- **Gray dashed line**: Target score of 0.9
- Higher is better (1.0 = perfect)

## 💡 Tips

- **First time?** Start with just 10-20 epochs to test the workflow
- **GPU Memory?** nnUNet auto-configures batch size based on available memory
- **Slow preprocessing?** Normal for first run, subsequent runs use cached data
- **Monitor resources:** Keep Task Manager open to watch GPU/RAM usage

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| nnUNetv2 not found | `pip install nnunetv2` |
| Invalid dataset structure | Check folder names: `imagesTr`, `labelsTr` |
| Files not detected | Ensure `.nii.gz` format |
| Preprocessing fails | Check dataset.json is valid |
| Training crashes | Try `3d_lowres` configuration |

## 📁 Where Are My Models?

After training completes, find your models in:
```
nnUNet_results/Dataset001_YourDataset/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/
```

The best checkpoint is saved as `checkpoint_best.pth`

## 🎯 Next Steps

1. Train multiple folds (0-4) for cross-validation
2. Use the trained model for prediction:
   ```bash
   nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 1 -c 3d_fullres
   ```
3. Evaluate on test data
4. Fine-tune with different configurations

---

**Need Help?** Check the full [README.md](README.md) for detailed documentation.
