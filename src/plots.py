import matplotlib.pyplot as plt
import librosa
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
import torch


def audio_eda(data_dir):
    """
    Minimal EDA for cat vs dog audio dataset
    Args:
        data_dir: Path to data directory
    """
    
    # Gather all audio files
    wavs = sorted(Path(data_dir).rglob("*.wav"))
    labels = ['Cat' if p.parent.name.startswith("cat") else 'Dog' for p in wavs]
    
    # 1. Class Distribution (you already have this)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class counts
    class_counts = Counter(labels)
    axes[0,0].bar(class_counts.keys(), class_counts.values(), color=['skyblue', 'salmon'])
    axes[0,0].set_title('Class Distribution')
    axes[0,0].set_ylabel('Number of Files')
    
    # 2. Audio Duration Distribution
    durations = {'Cat': [], 'Dog': []}
    sample_files = wavs  # Process all files
    
    for wav_path in sample_files:
        try:
            duration = librosa.get_duration(filename=str(wav_path))
            label = 'Cat' if wav_path.parent.name.startswith("cat") else 'Dog'
            durations[label].append(duration)
        except:
            continue
    
    # Plot duration distributions
    all_durations = durations['Cat'] + durations['Dog']
    duration_labels = ['Cat'] * len(durations['Cat']) + ['Dog'] * len(durations['Dog'])
    
    axes[0,1].hist([durations['Cat'], durations['Dog']], bins=15, alpha=0.7, 
                   label=['Cat', 'Dog'], color=['skyblue', 'salmon'])
    axes[0,1].set_title('Audio Duration Distribution')
    axes[0,1].set_xlabel('Duration (seconds)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # 3. Sample Rate Distribution
    sample_rates = {'Cat': [], 'Dog': []}
    for wav_path in sample_files:
        try:
            _, sr = librosa.load(str(wav_path), sr=None)
            label = 'Cat' if wav_path.parent.name.startswith("cat") else 'Dog'
            sample_rates[label].append(sr)
        except:
            continue
    
    all_srs = sample_rates['Cat'] + sample_rates['Dog']
    sr_labels = ['Cat'] * len(sample_rates['Cat']) + ['Dog'] * len(sample_rates['Dog'])
    
    axes[1,0].hist([sample_rates['Cat'], sample_rates['Dog']], bins=10, alpha=0.7,
                   label=['Cat', 'Dog'], color=['skyblue', 'salmon'])
    axes[1,0].set_title('Sample Rate Distribution')
    axes[1,0].set_xlabel('Sample Rate (Hz)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    
    # 4. Audio Energy/RMS Distribution
    rms_values = {'Cat': [], 'Dog': []}
    for wav_path in sample_files:
        try:
            y, sr = librosa.load(str(wav_path))
            rms = librosa.feature.rms(y=y)[0].mean()
            label = 'Cat' if wav_path.parent.name.startswith("cat") else 'Dog'
            rms_values[label].append(rms)
        except:
            continue
    
    axes[1,1].boxplot([rms_values['Cat'], rms_values['Dog']], 
                      labels=['Cat', 'Dog'], patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1,1].set_title('Audio Energy Distribution (RMS)')
    axes[1,1].set_ylabel('RMS Energy')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=== Dataset Summary ===")
    print(f"Total files: {len(wavs)}")
    print(f"Cat files: {class_counts['Cat']} ({class_counts['Cat']/len(wavs)*100:.1f}%)")
    print(f"Dog files: {class_counts['Dog']} ({class_counts['Dog']/len(wavs)*100:.1f}%)")
    
    if all_durations:
        print(f"\nDuration stats (seconds):")
        print(f"  Mean: {np.mean(all_durations):.2f}")
        print(f"  Std: {np.std(all_durations):.2f}")
        print(f"  Min: {np.min(all_durations):.2f}")
        print(f"  Max: {np.max(all_durations):.2f}")
    
    if all_srs:
        print(f"\nSample rates found: {set(all_srs)}")

def plot_sample_waveforms(data_dir, n_samples=4):
    """Plot sample waveforms from each class"""
    wavs = sorted(Path(data_dir).rglob("*.wav"))
    
    cat_files = [w for w in wavs if w.parent.name.startswith("cat")][:n_samples]
    dog_files = [w for w in wavs if w.parent.name.startswith("dog")][:n_samples]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    
    for i, (cat_file, dog_file) in enumerate(zip(cat_files, dog_files)):
        # Cat waveforms
        try:
            y_cat, sr_cat = librosa.load(str(cat_file))
            time_cat = np.linspace(0, len(y_cat)/sr_cat, len(y_cat))
            axes[0,i].plot(time_cat, y_cat, color='skyblue', alpha=0.8)
            axes[0,i].set_title(f'Cat Sample {i+1}')
            axes[0,i].set_xlabel('Time (s)')
            if i == 0:
                axes[0,i].set_ylabel('Amplitude')
        except:
            axes[0,i].text(0.5, 0.5, 'Error loading', ha='center', transform=axes[0,i].transAxes)
        
        # Dog waveforms  
        try:
            y_dog, sr_dog = librosa.load(str(dog_file))
            time_dog = np.linspace(0, len(y_dog)/sr_dog, len(y_dog))
            axes[1,i].plot(time_dog, y_dog, color='salmon', alpha=0.8)
            axes[1,i].set_title(f'Dog Sample {i+1}')
            axes[1,i].set_xlabel('Time (s)')
            if i == 0:
                axes[1,i].set_ylabel('Amplitude')
        except:
            axes[1,i].text(0.5, 0.5, 'Error loading', ha='center', transform=axes[1,i].transAxes)
    
    plt.tight_layout()
    plt.show()



def plot_dataset_distribution(split_indices, all_labels):
    """Plot the distribution of cat/dog samples across train/val/test splits"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    splits = ['train', 'val', 'test']
    split_names = ['Train (Few-shot)', 'Validation', 'Test']
    colors = ['skyblue', 'lightcoral']
    
    for i, (split, split_name) in enumerate(zip(splits, split_names)):
        indices = split_indices[split]
        
        # Count samples per class using indices
        cat_count = sum(1 for idx in indices if all_labels[idx] == 0)
        dog_count = sum(1 for idx in indices if all_labels[idx] == 1)
        
        # Create bar plot
        classes = ['Cat', 'Dog']
        counts = [cat_count, dog_count]
        
        bars = axes[i].bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Formatting
        total = cat_count + dog_count
        axes[i].set_title(f'{split_name}\n(Total: {total})', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Number of Samples')
        axes[i].set_ylim(0, max(max(counts), 1) * 1.2)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        if total > 0:
            cat_pct = (cat_count / total) * 100
            dog_pct = (dog_count / total) * 100
            if cat_count > 0:
                axes[i].text(0, cat_count/2, f'{cat_pct:.1f}%', ha='center', va='center', 
                            fontweight='bold', color='white', fontsize=10)
            if dog_count > 0:
                axes[i].text(1, dog_count/2, f'{dog_pct:.1f}%', ha='center', va='center', 
                            fontweight='bold', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Dataset Distribution: Cat vs Dog Samples', y=1.02, fontsize=14, fontweight='bold')
    plt.show()
    
    # Print summary
    print("\nDataset Distribution Summary:")
    print("-" * 40)
    for split, split_name in zip(splits, split_names):
        indices = split_indices[split]
        cat_count = sum(1 for idx in indices if all_labels[idx] == 0)
        dog_count = sum(1 for idx in indices if all_labels[idx] == 1)
        total = cat_count + dog_count
        print(f"{split_name:15}: Cat={cat_count:2d}, Dog={dog_count:2d}, Total={total:2d}")


# Used for Contrastive Learning

def plot_tsne_embeddings(model, loader, device, epoch):
    """
    Run all training samples through `model.encode_audio`, collect their normalized-projection
    embeddings and labels, then run t-SNE with a dynamically-chosen perplexity (< n_samples).
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)
            audio_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in {"input_features", "attention_mask"}
            }
            emb = model.encode_audio(audio_inputs)  # shape: (batch_size, D)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    n_samples = all_embeddings.shape[0]

    # Choose perplexity < n_samples (e.g. min(30, n_samples - 1))
    perp = min(30, n_samples - 1)
    if perp < 1:
        perp = 1

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    emb_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(6, 6))
    for cls_id, color, name in zip([0, 1], ['#1f77b4', '#ff7f0e'], ['Cat', 'Dog']):
        mask = (all_labels == cls_id)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=15
        )

    plt.legend(fontsize=10)
    plt.title(f"t-SNE of Train Embeddings (Epoch {epoch}, perp={perp})", fontsize=12, fontweight='bold')
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.show()