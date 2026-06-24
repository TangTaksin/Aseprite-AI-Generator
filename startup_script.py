#!/usr/bin/env python3
"""
Local AI Generator for Aseprite
Dynamic Versioning
"""

import sys
import os
import subprocess
import shutil
import glob
from pathlib import Path

def get_version():
    try:
        version_file = Path(__file__).parent / "src" / "version.py"
        if version_file.exists():
            with open(version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "1.0.6"

__version__ = get_version()

def print_banner():
    print("\n" + "=" * 60)
    print(f"🎮 LOCAL AI GENERATOR FOR ASEPRITE v{__version__}")
    print("=" * 60)


def print_section(title):
    """Print section headers in a standardized format"""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))


def check_python_version():
    """Verify that Python version meets requirements (Target: >= 3.8)"""
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python version is too old for this system.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please use Python 3.10.x or newer (recommended 3.10+)")
        return False

    if version.major == 3 and version.minor > 14:
        print(
            f"⚠️ Warning: You are running Python {version.major}.{version.minor}.{version.micro}"
        )
        print("   If you encounter any runtime issues, consider downgrading to 3.14 or lower.")
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")

    return True


def install_dependencies():
    """Install required libraries verified to work with SDXL and Pony XL models"""
    print("\n📋 Managing internal library dependencies")
    print("-" * 30)

    # Upgrade pip
    print("📦 Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install PyTorch
    print("\n🔥 Installing PyTorch (CUDA 13.0)")
    torch_packages = [
        "torch",
        "torchvision",
    ]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *torch_packages,
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
    )

    # Install AI libraries (pinned versions to prevent compatibility conflicts)
    print("\n📚 Installing AI libraries and supplementary systems...")
    other_requirements = [
        "diffusers==0.38.0",
        "transformers==4.57.6",
        "safetensors==0.8.0rc0",
        "accelerate==1.13.0",
        "huggingface-hub==0.36.2",
        "flask",
        "flask-cors",
        "sentencepiece",
        "kornia",
        "einops",
        "pillow",
        "peft",
        "opencv-python",
        "timm",
        "numpy",
        "psutil",
    ]

    subprocess.check_call([sys.executable, "-m", "pip", "install"] + other_requirements)

    print("\n✅ All libraries installed successfully!")
    return True


def select_startup_model():
    """Interactively select the startup model"""
    print_section("Base Model Selection")

    # Fetch locally downloaded checkpoints from models directory
    local_models = []
    if os.path.exists("models"):
        local_models = [
            f for f in os.listdir("models") if f.endswith((".safetensors", ".ckpt"))
        ]

    models = [
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "description": "SDXL Base (Recommended) - High Quality",
            "size": "~7GB",
        },
        {
            "name": "runwayml/stable-diffusion-v1-5",
            "description": "SD 1.5 - Faster, Smaller size",
            "size": "~4GB",
        },
    ]

    print("Select the main model to load at startup:")
    print()

    idx = 1
    for model in models:
        print(f"   [{idx}] {model['description']} ({model['name']})")
        idx += 1

    # Print detected local models
    for l_model in local_models:
        print(f"   [{idx}] Local: {l_model}")
        idx += 1

    print(f"   [{idx}] None (Load manually later)")
    print()

    while True:
        try:
            choice_str = input(f"Enter choice [1-{idx}] (Default is 1): ").strip()
            choice = int(choice_str) if choice_str else 1

            if 1 <= choice <= len(models):
                selected = models[choice - 1]["name"]
                print(f"✅ Selected: {selected}")
                return selected
            elif len(models) < choice < idx:
                selected = local_models[choice - len(models) - 1]
                print(f"✅ Selected Local Model: {selected}")
                return selected
            elif choice == idx:
                print("✅ No model will be loaded at startup.")
                return "none"
            else:
                print(f"❌ Invalid choice. Please enter a value between 1 and {idx}.")
        except ValueError:
            print("❌ Please enter numbers only.")


def configure_offline_mode():
    print_section("Network Configuration")

    os.makedirs("models", exist_ok=True)
    local_files = glob.glob("models/*.safetensors") + glob.glob("models/*.ckpt")

    standard_models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]

    hf_cache_path = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )
    cached_standard_models = []

    if os.path.exists(hf_cache_path):
        for m_id in standard_models:
            folder_name = "models--" + m_id.replace("/", "--")
            full_path = os.path.join(hf_cache_path, folder_name)

            if os.path.exists(full_path) and os.path.exists(
                os.path.join(full_path, "snapshots")
            ):
                cached_standard_models.append(m_id)

    print(f"📦 Local resources status:")
    print(f"   • Local files (.safetensors): {len(local_files)} files")

    if cached_standard_models:
        print(f"   • Standard models ready offline:")
        for m in cached_standard_models:
            print(f"      ✅ {m.split('/')[-1]}")
    else:
        print(f"   • Standard models: ❌ Not found in cache (Online mode required to download)")

    print("-" * 45)

    if not local_files and not cached_standard_models:
        print("⚠️ No local model files found on this machine.")
        print("🌐 Enforcing 'Online Mode' for download preparation.")
        return False

    print("Select execution mode:")
    print("  [ 1 ] Online  : Download models from internet if needed")
    print("  [ 2 ] Offline : Use local cached models only, no internet connection")
    print()

    res = input("Select [1 or 2] (Default 2): ").strip()
    return res != "1"


def setup_directories():
    """Setup required directory structures"""
    print_section("Setting up directories")

    directories = ["loras", "models", "cache"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁  Checking/Creating: {directory}/")

    print("✅  Folder structure is ready for use.")


def check_system_requirements():
    """Check hardware specifications and output suggestions"""
    print_section("System Requirements Check")

    try:
        import psutil  # type: ignore

        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f}GB")
        disk_free = psutil.disk_usage(".").free / (1024**3)
        print(f"💽 Free Disk Space: {disk_free:.1f}GB")
    except ImportError:
        print("ℹ️  System recommends installing psutil: pip install psutil")

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"🔥 VRAM: {vram_gb:.1f}GB")
            if vram_gb >= 12:
                print("🚀 Your GPU is excellent!")
        else:
            print("⚠️ NVIDIA GPU (CUDA) not found - using CPU instead")
    except ImportError:
        print("📦 PyTorch is not installed")
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")


def main():
    """Main entry point for starting the system"""
    print_banner()

    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)

    setup_directories()

    chosen_model = select_startup_model()
    offline_mode = configure_offline_mode()

    if not offline_mode:
        print("\n🌐 Checking library updates (Online Mode)...")
        if not install_dependencies():
            print("\n❌ Library installation failed! (Check internet connection)")
            cont = input("Do you want to try running in Offline Mode anyway? [y/N]: ").lower()
            if cont not in ["y", "yes"]:
                sys.exit(1)
    else:
        print("\n🔌 Offline Mode: Skipping library check for faster startup")

    check_system_requirements()

    print_section("Starting Server")

    try:
        from sd_server import main as run_server

        print("\n" + "=" * 60)
        print("🎉 Server ready!")
        print("=" * 60)
        run_server(default_model_to_load=chosen_model, offline=offline_mode)

    except Exception as e:
        import traceback

        print(f"\n❌ An error occurred:")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
