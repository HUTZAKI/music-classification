import torch
import torchaudio
import librosa
import numpy as np


class AudioProcessor:
    """Audio preprocessing for music genre classification"""

    def __init__(self, sample_rate=22050, n_mels=128, duration=3.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            power=2.0
        )

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            waveform = torch.from_numpy(audio_data).unsqueeze(0).float()

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] > self.n_samples:
            waveform = waveform[:, :self.n_samples]
        elif waveform.shape[1] < self.n_samples:
            padding = self.n_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform

    def audio_to_melspec(self, waveform):
        """Convert audio to mel-spectrogram"""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)

        # Ensure consistent size [1, n_mels, time_steps]
        # Expected time_steps = n_samples / hop_length
        expected_time_steps = 1 + (self.n_samples // 512)  # hop_length = 512

        if mel_spec.shape[-1] != expected_time_steps:
            # Resize to expected size
            mel_spec = torch.nn.functional.interpolate(
                mel_spec.unsqueeze(0),  # Add batch dim
                size=(self.n_mels, expected_time_steps),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim

        return mel_spec
