# verify_image_qa_setup.py
import sys
import importlib

def check_package(package_name):
    """Check if a package is installed"""
    try:
        module = importlib.import_module(package_name.replace('-', '_'))
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

print("=" * 60)
print("🔍 VERIFYING IMAGE QA SYSTEM SETUP")
print("=" * 60)

# Check Python
print(f"🐍 Python version: {sys.version}")
print()

# Check virtual environment
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"🔧 Virtual Env: {'✅ Active' if in_venv else '❌ Not active'}")
print()

# Check all required packages
packages = [
    'streamlit',
    'Pillow',
    'opencv-python',
    'torch',
    'torchvision',
    'transformers',
    'accelerate',
    'sentencepiece',
    'numpy',
    'pandas',
    'plotly',
    'protobuf'
]

print("📦 Package Status:")
print("-" * 40)

all_installed = True
for package in packages:
    installed, version = check_package(package)
    status = "✅" if installed else "❌"
    version_str = f"v{version}" if version else ""
    print(f"{status} {package:20} {version_str}")
    if not installed:
        all_installed = False

print("-" * 40)
if all_installed:
    print("✅ All packages installed successfully!")
else:
    print("⚠️ Some packages are missing. Check the ❌ marks above.")

print("=" * 60)

# Test PyTorch
print("\n🔧 Testing PyTorch:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Test tensor creation
    x = torch.tensor([1, 2, 3])
    print(f"   Tensor test: {x}")
    print("   ✅ PyTorch working!")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")

print("=" * 60)