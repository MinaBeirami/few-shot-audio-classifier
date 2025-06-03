import torch
import librosa
import numpy as np


def augment_audio(audio, sr=22050):
    """random augmentations to audio"""
    audio = audio.numpy()
    # Time stretching
    if np.random.random() > 0.5:
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    # Pitch shifting
    if np.random.random() > 0.5:
        n_steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    # Add subtle noise
    if np.random.random() > 0.7:
        noise_factor = np.random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_factor, audio.shape)
        audio = audio + noise

    return torch.from_numpy(audio)
