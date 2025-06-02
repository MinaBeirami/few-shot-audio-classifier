
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ClapProcessor
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.plots import plot_dataset_distribution
from src.dataset import balance_dataset_explicit


SAMPLE_RATE = 48000

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with triplet sampling."""
    
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor
        
        # Precompute labels (0=cat, 1=dog) and class indices
        self.labels = [0 if p.parent.name.startswith("cat") else 1 for p in paths]
        self.class_to_indices = {0: [], 1: []}
        for i, label in enumerate(self.labels):
            self.class_to_indices[label].append(i)

    def __len__(self):
        return len(self.paths)

    def _load_audio(self, idx):
        """Load and preprocess audio file"""
        path = self.paths[idx]
        wav, sr = torchaudio.load(str(path))
        
        # Resample to 48kHz if needed
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
        
        # Convert to mono
        wav = wav.mean(0)
        
        # Process with CLAP processor
        inputs = self.processor(audios=wav, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, self.labels[idx]

    def __getitem__(self, idx):
        """Return triplet: anchor, positive, negative"""
        # Load anchor
        anchor_x, anchor_label = self._load_audio(idx)
        
        # Sample positive (same class, different sample)
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.class_to_indices[anchor_label])
        positive_x, _ = self._load_audio(pos_idx)
        
        # Sample negative (different class)
        neg_label = 1 - anchor_label
        neg_idx = random.choice(self.class_to_indices[neg_label])
        negative_x, _ = self._load_audio(neg_idx)

        return {
            'anchor': anchor_x,
            'positive': positive_x,
            'negative': negative_x
        }


class ClassificationDataset(Dataset):
    """Simple dataset that returns audio inputs and labels for classification."""
    
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor
        
        # Precompute labels (0=cat, 1=dog)
        self.labels =    [0 if p.parent.name.startswith("cat") else 1 for p in paths]

    def __len__(self):
        return len(self.paths)

    def _load_audio(self, idx):
        """Load and preprocess audio file"""
        path = self.paths[idx]
        wav, sr = torchaudio.load(str(path))
        
        # Resample to 48kHz if needed
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
        
        # Convert to mono
        wav = wav.mean(0)
        
        # Process with CLAP processor
        inputs = self.processor(audios=wav, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, self.labels[idx]

    def __getitem__(self, idx):
        """Return audio inputs and label"""
        inputs, label = self._load_audio(idx)
        inputs['labels'] = torch.tensor(label)
        return inputs

def contrastive_collate(batch):
    """Collate function for contrastive batches"""
    def stack_side(side):
        return {k: torch.stack([item[side][k] for item in batch])
                for k in batch[0]['anchor']}
    
    return {
        'anchor': stack_side('anchor'),
        'positive': stack_side('positive'),
        'negative': stack_side('negative')
    }


def classification_collate(batch):
    """Collate function for classification batches"""
    # batch is list of dicts with same keys
    out = {k: torch.stack([d[k] for d in batch]) 
           for k in batch[0] if k != 'labels'}
    out['labels'] = torch.tensor([d['labels'].item() for d in batch])
    return out

def build_contrastive_loader(config):
    """Build data loaders for contrastive learning with integrated preprocessing"""
    
    # Gather all audio files
    wavs = sorted(Path(config["data_dir"]).rglob("*.wav"))
    labels = [0 if p.parent.name.startswith("cat") else 1 for p in wavs]
    
    print(f"Found {len(wavs)} audio files: {sum(1 for l in labels if l == 0)} cats, {sum(1 for l in labels if l == 1)} dogs")
    
    # Apply dataset balancing method
    balance_method = config.get("balance_method", "none")  # Options: 'oversample', 'undersample', 'augment', 'none'
    
    if balance_method in ['oversample', 'undersample']:
        wavs, labels = balance_dataset_explicit(
            wavs, labels, 
            balance_method=balance_method,
            max_multiplier=config.get("oversample_multiplier", 3)
        )
    elif balance_method == 'augment':
        print("Using on-the-fly augmentation method for class balancing")
    else:
        print("No explicit class balancing applied")

    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(wavs))),
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    
    # Few-shot selection from training set
    few_shot_idx = []
    class_counts = {0: 0, 1: 0}
    for i in train_idx:
        if class_counts[labels[i]] < config["few_shot_per_class"]:
            few_shot_idx.append(i)
            class_counts[labels[i]] += 1

    # Create validation set from remaining training data
    remaining_train = [i for i in train_idx if i not in few_shot_idx]
    if len(remaining_train) > 0:
        val_idx, _ = train_test_split(
            remaining_train,
            test_size=0.5,
            stratify=[labels[i] for i in remaining_train],
            random_state=42
        )
    else:
        val_idx = []

    # Plot distribution before creating datasets
    split_indices = {
        'train': few_shot_idx,
        'val': val_idx, 
        'test': test_idx
    }
    plot_dataset_distribution(split_indices, labels)

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    # Create data loaders with balance method info
    loaders = {}
    for split, indices in [('train', few_shot_idx), ('val', val_idx), ('test', test_idx)]:
        dataset = ContrastiveDataset(
            [wavs[i] for i in indices], 
            processor,
            #balance_method=balance_method,
            #apply_augment=(split == 'train')  # Only augment during training
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == 'train'),
            collate_fn=contrastive_collate
        )
    
    return loaders, processor


def compute_class_centroids(model, train_loader, device):
    """Compute class centroids from training data"""
    model.eval()
    
    class_embeddings = {0: [], 1: []}
    
    with torch.no_grad():
        for batch in train_loader:
            # For classification loader, we have labels directly
            if 'labels' in batch:
                labels = batch['labels']
                audio_inputs = {k: v.to(device) for k, v in batch.items() 
                               if k in {"input_features", "attention_mask"}}
                
                # Get embeddings
                embeddings = model.encode_audio(audio_inputs)
                
                # Group by class
                for i, label in enumerate(labels):
                    class_embeddings[label.item()].append(embeddings[i])
            else:
                # For contrastive loader, we need to use the dataset directly
                train_dataset = train_loader.dataset
                for i in range(len(train_dataset.paths)):
                    inputs, label = train_dataset._load_audio(i)
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                    
                    embedding = model.encode_audio(inputs)
                    class_embeddings[label].append(embedding.squeeze(0))
                break  # Only need to do this once for the whole dataset
    
    # Compute centroids
    centroids = {}
    for class_id in [0, 1]:
        if class_embeddings[class_id]:
            centroids[class_id] = torch.stack(class_embeddings[class_id]).mean(dim=0)
        else:
            # Fallback to zero vector if no samples
            centroids[class_id] = torch.zeros(model.projection_head.out_features, device=device)
    
    return centroids


def build_classification_loader(config):
    """Build data loaders using ClassificationDataset instead of ContrastiveDataset"""
    # Gather all audio files
    wavs = sorted(Path(config["data_dir"]).rglob("*.wav"))
    labels = [0 if p.parent.name.startswith("cat") else 1 for p in wavs]

    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(wavs))),
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    
    # Few-shot selection from training set
    few_shot_idx = []
    class_counts = {0: 0, 1: 0}
    for i in train_idx:
        if class_counts[labels[i]] < config["few_shot_per_class"]:
            few_shot_idx.append(i)
            class_counts[labels[i]] += 1

    # Create validation set from remaining training data
    remaining_train = [i for i in train_idx if i not in few_shot_idx]
    val_idx, _ = train_test_split(
        remaining_train,
        test_size=0.5,
        stratify=[labels[i] for i in remaining_train],
        random_state=42
    )

    # Plot distribution before creating datasets
    split_indices = {
        'train': few_shot_idx,
        'val': val_idx, 
        'test': test_idx
    }
    plot_dataset_distribution(split_indices, labels)

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    # Create data loaders using ClassificationDataset
    loaders = {}
    for split, indices in [('train', few_shot_idx), ('val', val_idx), ('test', test_idx)]:
        dataset = ClassificationDataset([wavs[i] for i in indices], processor)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == 'train'),
            collate_fn=classification_collate
        )
    
    return loaders, processor
