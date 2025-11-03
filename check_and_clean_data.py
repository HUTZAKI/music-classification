"""
Check and clean corrupted audio files from dataset
"""

import torch
import torchaudio
import librosa
from pathlib import Path
import argparse
from tqdm import tqdm


def check_audio_file(audio_path):
    """
    Check if audio file can be loaded

    Returns:
        (is_valid, error_message)
    """
    try:
        # Try torchaudio first
        waveform, sr = torchaudio.load(str(audio_path))

        # Check if waveform is valid
        if waveform.numel() == 0:
            return False, "Empty waveform"

        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            return False, "NaN or Inf in waveform"

        return True, None

    except Exception as e:
        # Try librosa as backup
        try:
            audio_data, sr = librosa.load(str(audio_path), sr=22050, mono=True)

            if len(audio_data) == 0:
                return False, "Empty audio"

            return True, None

        except Exception as e2:
            return False, f"Both torchaudio and librosa failed: {str(e)}, {str(e2)}"


def check_dataset(data_dir, remove_corrupted=False):
    """
    Check all audio files in dataset

    Args:
        data_dir: Path to data directory
        remove_corrupted: If True, remove corrupted files

    Returns:
        dict with statistics
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return None

    # Get all genre folders
    genres = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if len(genres) == 0:
        print(f"Error: No genre folders found in {data_dir}")
        return None

    print(f"Found {len(genres)} genres: {[g.name for g in genres]}")
    print("=" * 80)

    stats = {
        'total_files': 0,
        'valid_files': 0,
        'corrupted_files': 0,
        'corrupted_list': [],
        'per_genre': {}
    }

    for genre_dir in genres:
        genre_name = genre_dir.name
        audio_files = list(genre_dir.glob('*.wav')) + list(genre_dir.glob('*.mp3'))

        genre_stats = {
            'total': len(audio_files),
            'valid': 0,
            'corrupted': 0,
            'corrupted_files': []
        }

        print(f"\nChecking {genre_name} ({len(audio_files)} files)...")

        for audio_file in tqdm(audio_files, desc=f"  {genre_name}"):
            stats['total_files'] += 1
            is_valid, error_msg = check_audio_file(audio_file)

            if is_valid:
                stats['valid_files'] += 1
                genre_stats['valid'] += 1
            else:
                stats['corrupted_files'] += 1
                genre_stats['corrupted'] += 1
                stats['corrupted_list'].append((str(audio_file), error_msg))
                genre_stats['corrupted_files'].append(str(audio_file))

                print(f"  ✗ Corrupted: {audio_file.name} - {error_msg}")

                if remove_corrupted:
                    try:
                        audio_file.unlink()
                        print(f"    → Removed")
                    except Exception as e:
                        print(f"    → Failed to remove: {e}")

        stats['per_genre'][genre_name] = genre_stats

    return stats


def print_statistics(stats):
    """Print dataset statistics"""
    if stats is None:
        return

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    print(f"\nOverall:")
    print(f"  Total files:     {stats['total_files']}")
    print(f"  Valid files:     {stats['valid_files']} ({stats['valid_files']/stats['total_files']*100:.1f}%)")
    print(f"  Corrupted files: {stats['corrupted_files']} ({stats['corrupted_files']/stats['total_files']*100:.1f}%)")

    print(f"\nPer Genre:")
    print("-" * 80)
    print(f"{'Genre':<15} {'Total':<10} {'Valid':<10} {'Corrupted':<10}")
    print("-" * 80)

    for genre, genre_stats in stats['per_genre'].items():
        print(f"{genre:<15} {genre_stats['total']:<10} {genre_stats['valid']:<10} {genre_stats['corrupted']:<10}")

    if stats['corrupted_files'] > 0:
        print(f"\nCorrupted Files:")
        print("-" * 80)
        for file_path, error_msg in stats['corrupted_list']:
            print(f"  {file_path}")
            print(f"    Error: {error_msg}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Check and clean dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--remove', action='store_true',
                        help='Remove corrupted files')

    args = parser.parse_args()

    print("=" * 80)
    print("AUDIO DATASET CHECKER")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Remove corrupted: {args.remove}")
    print("=" * 80)

    if args.remove:
        confirm = input("\nAre you sure you want to REMOVE corrupted files? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return

    stats = check_dataset(args.data_dir, remove_corrupted=args.remove)
    print_statistics(stats)

    if args.remove and stats['corrupted_files'] > 0:
        print(f"\n✓ Removed {stats['corrupted_files']} corrupted files")


if __name__ == '__main__':
    main()
