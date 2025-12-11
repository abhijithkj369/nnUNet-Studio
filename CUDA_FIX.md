# 🔴 CUDA Not Available Error

## Problem
```
AssertionError: Torch not compiled with CUDA enabled
```

Your PyTorch installation doesn't have CUDA (GPU) support, but nnUNet is trying to use your GPU.

---

## ✅ Solution 1: Install PyTorch with CUDA (GPU Training) ⭐

**If you have an NVIDIA GPU:**

### Quick Install (Recommended)
1. **Double-click**: `install_pytorch_cuda.bat`
2. Wait for installation (~2-5 minutes)
3. Restart the app: `python app.py`
4. Try training again - it will use your GPU! 🚀

### Manual Install
```powershell
# Activate venv
.\venv\Scripts\activate

# Uninstall old PyTorch
pip uninstall -y torch torchvision torchaudio

# Install with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify CUDA is available:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Should show: `CUDA: True`

---

## ✅ Solution 2: Use CPU-Only Training (Slower)

**If you DON'T have an NVIDIA GPU or want to train on CPU:**

### Quick Setup
1. **Double-click**: `configure_cpu_only.bat`
2. **RESTART your terminal** (important!)
3. Start the app: `python app.py`
4. Training will use CPU (much slower)

### Manual Setup
```powershell
# Set environment variable
$env:CUDA_VISIBLE_DEVICES = "-1"

# Or in PowerShell permanently:
[Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "-1", "User")
```

---

## 🎯 Which Should You Choose?

### Choose **GPU Training** (Option 1) if:
- ✅ You have an NVIDIA GPU (GeForce, RTX, Quadro, etc.)
- ✅ You want faster training (10-100x faster than CPU)
- ✅ You're working with large 3D medical images

### Choose **CPU Training** (Option 2) if:
- ❌ You don't have an NVIDIA GPU
- ❌ You have AMD/Intel GPU (not supported by CUDA)
- ⚠️ You're okay with VERY slow training (hours to days)

---

## 📊 Training Speed Comparison

| Configuration | Speed per Epoch | 100 Epochs |
|---------------|----------------|------------|
| GPU (CUDA) | ~2-5 minutes | 3-8 hours |
| CPU Only | ~30-120 minutes | 50-200 hours |

**Recommendation**: Use GPU if available!

---

## 🔍 Check If You Have a GPU

Run this in PowerShell:
```powershell
nvidia-smi
```

**If it shows GPU info** → You have NVIDIA GPU, use Option 1
**If command not found** → No NVIDIA GPU, use Option 2

---

## 📦 Files Created

- ✅ [install_pytorch_cuda.bat](file:///d:/CDAC/nnUNet_Tool/install_pytorch_cuda.bat) - Install GPU support
- ✅ [configure_cpu_only.bat](file:///d:/CDAC/nnUNet_Tool/configure_cpu_only.bat) - Use CPU only

---

## 🚀 Next Steps

1. **Check if you have NVIDIA GPU**: Run `nvidia-smi`
2. **Choose your option** above
3. **Run the script** (double-click .bat file)
4. **Restart the app**: `python app.py`
5. **Try training again** ✅

---

## ⚠️ Important Notes

- **AMD/Intel GPUs**: Not supported by PyTorch/CUDA (use CPU)
- **Laptop GPUs**: May work but check NVIDIA Control Panel
- **Memory**: GPU training needs ~4-8GB VRAM for medical images
- **CPU Training**: Feasible for testing, but very slow for full training
