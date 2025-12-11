import os
import sys
import shutil
import subprocess
import site
from pathlib import Path

def install_requirements():
    """Install requirements from requirements.txt"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def apply_patches():
    """Apply custom patches to nnunetv2"""
    print("\n🔧 Applying patches to nnunetv2...")
    
    # Find site-packages directory
    site_packages = site.getsitepackages()[0]
    if os.name == 'nt':
        # On Windows, it might be in Lib/site-packages
        site_packages = next((p for p in site.getsitepackages() if 'site-packages' in p), site_packages)
    
    nnunet_path = Path(site_packages) / "nnunetv2"
    
    if not nnunet_path.exists():
        print(f"❌ nnunetv2 not found at {nnunet_path}")
        print("   Make sure dependencies are installed correctly.")
        sys.exit(1)
        
    print(f"📍 Found nnunetv2 at: {nnunet_path}")
    
    # Define patches
    patches_dir = Path("patches/nnunetv2")
    
    if not patches_dir.exists():
        print("❌ 'patches' directory not found!")
        sys.exit(1)
        
    # Walk through patches directory and copy files
    patch_count = 0
    for patch_file in patches_dir.rglob("*.py"):
        # Calculate relative path from patches/nnunetv2
        rel_path = patch_file.relative_to(patches_dir)
        target_file = nnunet_path / rel_path
        
        print(f"   - Patching: {rel_path}")
        
        try:
            shutil.copy2(patch_file, target_file)
            patch_count += 1
        except Exception as e:
            print(f"❌ Error patching {rel_path}: {e}")
            
    if patch_count > 0:
        print(f"✅ Applied {patch_count} patches successfully")
    else:
        print("⚠️ No patches were applied")

def main():
    print("🚀 Starting Installation and Patching Process")
    print("============================================")
    
    install_requirements()
    apply_patches()
    
    print("\n✨ Setup Complete!")
    print("   You can now run the application using:")
    print("   python app.py")

if __name__ == "__main__":
    main()
