import torch
import torchaudio.transforms as T
import random


class AudioAugmentation:
    """
    Data augmentation for audio spectrograms
    Helps improve model generalization and handle small datasets
    """

    def __init__(self, time_mask_param=30, freq_mask_param=20, p=0.5):
        """
        Args:
            time_mask_param: Maximum time steps to mask
            freq_mask_param: Maximum frequency bins to mask
            p: Probability of applying augmentation
        """
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.p = p

    def __call__(self, mel_spec):
        """
        Apply augmentation to mel-spectrogram

        Args:
            mel_spec: Input mel-spectrogram [channels, freq, time]

        Returns:
            Augmented mel-spectrogram
        """
        if random.random() < self.p:
            # Apply time masking
            mel_spec = self.time_mask(mel_spec)

        if random.random() < self.p:
            # Apply frequency masking
            mel_spec = self.freq_mask(mel_spec)

        return mel_spec


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    More aggressive augmentation for very small datasets
    """

    def __init__(self, time_mask_param=40, freq_mask_param=30, num_masks=2, p=0.5):
        """
        Args:
            time_mask_param: Maximum time steps to mask
            freq_mask_param: Maximum frequency bins to mask
            num_masks: Number of masks to apply
            p: Probability of applying augmentation
        """
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.p = p

    def __call__(self, mel_spec):
        """Apply SpecAugment"""
        if random.random() > self.p:
            return mel_spec

        # Apply multiple time masks
        for _ in range(self.num_masks):
            mel_spec = T.TimeMasking(time_mask_param=self.time_mask_param)(mel_spec)

        # Apply multiple frequency masks
        for _ in range(self.num_masks):
            mel_spec = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(mel_spec)

        return mel_spec


class Compose:
    """Compose multiple augmentations"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, mel_spec):
        for transform in self.transforms:
            mel_spec = transform(mel_spec)
        return mel_spec
