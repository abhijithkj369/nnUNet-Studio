# 🔴 Preprocessing Error - Duplicate Dataset ID

## Problem
```
RuntimeError: More than one dataset name found for dataset id 1
```

## Root Cause
You have **multiple folders** with ID 001 in different locations:
- `nnUNet_raw/Dataset001_001_Knee`
- `nnUNet_preprocessed/Dataset001_Mandibularcanal`

nnUNet requires each dataset ID to be unique across all folders.

---

## ✅ Solution 1: Clean Up Old Datasets (EASIEST)

**Run the cleanup script:**

1. **Double-click**: `cleanup_datasets.bat`
   - This will delete the old Dataset001 folders
   - Your airway dataset will remain safe

2. **Then in the UI**: Click "Setup Training Environment" again with your airway dataset

---

## ✅ Solution 2: Use a Different Dataset ID

In the UI, change the **Dataset ID** from `1` to `2` (or higher):

1. Go to **Tab 2: Training Configuration**
2. Change **Dataset ID** from `1` to `2`
3. Click **"Setup Training Environment"** again
4. Your dataset will be saved as `Dataset002_airway`

---

## ✅ Solution 3: Manual Cleanup (Advanced)

**Delete the old folders manually:**

```powershell
# In PowerShell or File Explorer
Remove-Item "nnUNet_raw\Dataset001_001_Knee" -Recurse -Force
Remove-Item "nnUNet_preprocessed\Dataset001_Mandibularcanal" -Recurse -Force
```

---

## 🎯 Recommended: Use Cleanup Script

The **easiest and safest** option is:

1. **Stop the app** (Ctrl+C in terminal)
2. **Double-click** `cleanup_datasets.bat`
3. **Restart the app**: `python app.py`
4. **Try preprocessing again**

---

## Why This Happens

You likely ran preprocessing with different datasets before:
- First with "001_Knee" dataset
- Then with "Mandibularcanal" dataset
- Both used ID 001

nnUNet remembers both and gets confused.

---

## Files Created
- ✅ [cleanup_datasets.bat](file:///d:/CDAC/nnUNet_Tool/cleanup_datasets.bat) - One-click cleanup script
