import librosa
import numpy as np
import warnings

def normalize_audio(audio):
    """Normalize audio using RMS normalization with peak limiting backup"""
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio / rms * 0.1  # Target RMS of 0.1
    
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    
    return audio

def standardize_duration(audio, sr=22050, target_duration=3.0):
    """Standardize audio duration - pad short clips, segment long clips"""
    target_length = int(target_duration * sr)
    
    if len(audio) < target_length:
        # Pad short clips with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        return [audio]
    else:
        # For long clips, extract multiple overlapping segments
        segments = []
        step_size = target_length // 2  # 50% overlap
        
        for i in range(0, len(audio) - target_length + 1, step_size):
            segments.append(audio[i:i + target_length])
        
        if not segments:
            segments = [audio[:target_length]]
            
        return segments

def augment_audio(audio, sr=22050):
    """Apply random augmentations to audio for minority class"""
    # Time stretching (±20%)
    if np.random.random() > 0.5:
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    
    # Pitch shifting (±2 semitones)
    if np.random.random() > 0.5:
        n_steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    # Add subtle noise
    if np.random.random() > 0.7:
        noise_factor = np.random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_factor, audio.shape)
        audio = audio + noise
    
    return audio

def preprocess_audio_file(file_path, target_duration=3.0, sr=22050, augment=False):
    """Preprocess a single audio file and return segments"""
    try:
        # Load audio with librosa
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        
        # Skip if audio is too short
        if len(audio) < sr * 0.1:  # Less than 0.1 seconds
            warnings.warn(f"Audio too short: {file_path}")
            return []
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Apply augmentation if requested
        if augment:
            audio = augment_audio(audio, sr)
        
        # Normalize energy
        audio = normalize_audio(audio)
        
        # Standardize duration
        audio_segments = standardize_duration(audio, sr, target_duration)
        
        return audio_segments
        
    except Exception as e:
        warnings.warn(f"Error processing {file_path}: {str(e)}")
        return []

# aud = preprocess_animal_sounds('/Users/mina/workspace/few-shot-audio-classifier/data/cat_dog_audio/cats_dogs/cat_1.wav')
# print(aud)