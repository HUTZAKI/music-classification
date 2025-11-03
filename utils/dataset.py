import torch
from torch.utils.data import Dataset
import os
from pathlib import Path


class MusicGenreDataset(Dataset):
    """Dataset for music genre classification"""

    def __init__(self, data_dir, audio_processor, genre_to_idx=None, transform=None):
        """
        Args:
            data_dir: Path to directory containing genre folders
            audio_processor: AudioProcessor instance
            genre_to_idx: Dictionary mapping genre names to indices
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = audio_processor
        self.transform = transform

        # Get all genre folders
        self.genres = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        if genre_to_idx is None:
            self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        else:
            self.genre_to_idx = genre_to_idx

        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}

        # Collect all audio files
        self.audio_files = []
        self.labels = []

        for genre in self.genres:
            genre_dir = self.data_dir / genre
            audio_files = list(genre_dir.glob('*.wav')) + list(genre_dir.glob('*.mp3'))

            for audio_file in audio_files:
                self.audio_files.append(audio_file)
                self.labels.append(self.genre_to_idx[genre])

        print(f"Loaded {len(self.audio_files)} audio files from {len(self.genres)} genres")
        print(f"Genres: {self.genres}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Returns:
            mel_spec: Mel-spectrogram tensor [1, n_mels, time_steps]
            label: Genre label (integer)
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.audio_files)
                audio_path = self.audio_files[current_idx]
                label = self.labels[current_idx]

                # Load audio
                waveform = self.audio_processor.load_audio(str(audio_path))

                # Convert to mel-spectrogram
                mel_spec = self.audio_processor.audio_to_melspec(waveform)

                # Apply transforms if any
                if self.transform:
                    mel_spec = self.transform(mel_spec)

                # Verify shape before returning
                assert mel_spec.dim() == 3, f"Expected 3D tensor, got {mel_spec.dim()}D"
                assert mel_spec.shape[0] == 1, f"Expected 1 channel, got {mel_spec.shape[0]}"

                return mel_spec, label

            except Exception as e:
                if attempt == 0:  # Only print on first error
                    print(f"Error loading {audio_path}: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed, raise error
                    raise RuntimeError(f"Failed to load any valid sample after {max_retries} attempts")
                # Try next sample
                continue

        # Should never reach here
        raise RuntimeError("Unexpected error in dataset __getitem__")


def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets

    Args:
        dataset: MusicGenreDataset instance
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from torch.utils.data import random_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({test_ratio*100:.1f}%)")

    return train_dataset, val_dataset, test_dataset
