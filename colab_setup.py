"""
Setup script for Google Colab
Run this first to install all required packages
"""

import subprocess
import sys


def install_packages():
    """Install required packages for Colab"""
    packages = [
        'torch',
        'torchaudio',
        'librosa',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tqdm'
    ]

    print("Installing required packages...")
    print("=" * 80)

    for package in packages:
        print(f"\nInstalling {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

    print("\n" + "=" * 80)
    print("All packages installed successfully!")


def check_gpu():
    """Check if GPU is available"""
    import torch

    print("\n" + "=" * 80)
    print("Checking GPU availability...")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"\n✓ GPU is available!")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("\n✗ GPU is NOT available")
        print("  Go to Runtime → Change runtime type → Select 'GPU'")

    print("=" * 80)


def check_data_directory(data_dir='/content/Data'):
    """Check if data directory exists"""
    from pathlib import Path

    print("\n" + "=" * 80)
    print("Checking data directory...")
    print("=" * 80)

    data_path = Path(data_dir)

    if data_path.exists():
        print(f"\n✓ Data directory found: {data_dir}")

        # List genres
        genres = [d.name for d in data_path.iterdir() if d.is_dir()]
        print(f"\nFound {len(genres)} genre folders:")

        for genre in sorted(genres):
            genre_dir = data_path / genre
            num_files = len(list(genre_dir.glob('*.wav')) + list(genre_dir.glob('*.mp3')))
            print(f"  - {genre}: {num_files} files")

    else:
        print(f"\n✗ Data directory NOT found: {data_dir}")
        print("\nPlease upload your Data folder to Colab:")
        print("  1. Click on folder icon on the left")
        print("  2. Upload your 'Data' folder")
        print("  3. Make sure it contains 10 genre folders")

    print("=" * 80)


if __name__ == '__main__':
    print("=" * 80)
    print("Google Colab Setup for Music Genre Classification")
    print("=" * 80)

    # Install packages
    install_packages()

    # Check GPU
    check_gpu()

    # Check data
    check_data_directory()

    print("\n" + "=" * 80)
    print("Setup complete! You're ready to train the model.")
    print("=" * 80)
