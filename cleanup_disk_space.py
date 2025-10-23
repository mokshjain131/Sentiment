"""
Clean up Hugging Face and training cache to free disk space.

WARNING: This will delete cached models and checkpoints.
You'll need to re-download models if you run training again.
"""
import shutil
from pathlib import Path
import os

def get_dir_size(path):
    """Get size of directory in GB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total / (1024**3)  # Convert to GB

def clean_huggingface_cache():
    """Clean Hugging Face cache directory"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if cache_dir.exists():
        size_gb = get_dir_size(str(cache_dir))
        print(f"📦 Hugging Face cache: {size_gb:.2f} GB")
        
        response = input(f"   Delete {size_gb:.2f} GB? (y/n): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(cache_dir)
                print(f"   ✅ Deleted {size_gb:.2f} GB")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        print("📦 Hugging Face cache: Not found")

def clean_training_checkpoints():
    """Clean training checkpoint directories"""
    checkpoint_dirs = [
        Path("models/finetuned"),
        Path("models/finetuned_fpb"),
        Path("results")
    ]
    
    total_size = 0
    dirs_to_clean = []
    
    for base_dir in checkpoint_dirs:
        if base_dir.exists():
            # Find checkpoint subdirectories
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    size = get_dir_size(str(item))
                    total_size += size
                    dirs_to_clean.append((item, size))
    
    if dirs_to_clean:
        print(f"\n📁 Training checkpoints: {total_size:.2f} GB")
        for dir_path, size in dirs_to_clean:
            print(f"   - {dir_path}: {size:.2f} GB")
        
        response = input(f"\n   Delete all checkpoints ({total_size:.2f} GB)? (y/n): ")
        if response.lower() == 'y':
            for dir_path, _ in dirs_to_clean:
                try:
                    shutil.rmtree(dir_path)
                    print(f"   ✅ Deleted {dir_path.name}")
                except Exception as e:
                    print(f"   ❌ Error deleting {dir_path.name}: {e}")
    else:
        print("\n📁 Training checkpoints: None found")

def clean_datasets_cache():
    """Clean datasets cache"""
    datasets_cache = Path.home() / ".cache" / "datasets"
    
    if datasets_cache.exists():
        size_gb = get_dir_size(str(datasets_cache))
        print(f"\n📊 Datasets cache: {size_gb:.2f} GB")
        
        response = input(f"   Delete {size_gb:.2f} GB? (y/n): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(datasets_cache)
                print(f"   ✅ Deleted {size_gb:.2f} GB")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        print("\n📊 Datasets cache: Not found")

def check_workspace_size():
    """Check size of current workspace"""
    workspace_dirs = {
        "data/": Path("data"),
        "models/": Path("models"),
        ".venv/": Path(".venv")
    }
    
    print("\n📂 Workspace sizes:")
    for name, path in workspace_dirs.items():
        if path.exists():
            size = get_dir_size(str(path))
            print(f"   {name}: {size:.2f} GB")

def main():
    print("=" * 60)
    print("🧹 DISK SPACE CLEANUP")
    print("=" * 60)
    
    # Check what's using space
    clean_huggingface_cache()
    clean_training_checkpoints()
    clean_datasets_cache()
    check_workspace_size()
    
    print("\n" + "=" * 60)
    print("✅ Cleanup complete!")
    print("=" * 60)
    print("\n💡 Note: You'll need to re-download models if you train again.")
    print("   But this is normal - they'll be cached again.")

if __name__ == "__main__":
    main()
