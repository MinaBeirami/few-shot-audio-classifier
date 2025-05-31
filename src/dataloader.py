
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ClapProcessor, ClapModel
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.plots import plot_dataset_distribution, plot_tsne_embeddings
from src.dataset import balance_dataset_explicit
from src.losses import supervised_contrastive_loss, contrastive_loss


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

class ContrastiveModel(nn.Module):
    """Contrastive learning model with frozen CLAP + trainable projection head"""
    
    def __init__(self, device):
        super().__init__()
        # Load and freeze CLAP model
        self.clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        for param in self.clap.parameters():
            param.requires_grad = False
        
        # Trainable projection head (initialized as identity)
        proj_dim = self.clap.config.projection_dim
        self.projection_head = nn.Linear(proj_dim, 128, bias=False)
        
        self.to(device)
    
    def encode_audio(self, audio_inputs):
        """Encode audio and apply projection"""
        # Get CLAP audio features
        audio_features = self.clap.get_audio_features(**audio_inputs)
        audio_features = F.normalize(audio_features, dim=-1)
        
        # Apply projection head and normalize
        projected = self.projection_head(audio_features)
        projected = F.normalize(projected, dim=-1)
        
        return projected




def train_epoch_classification(model, loader, optimizer, device, temperature=0.07):
    """Train one epoch using supervised contrastive loss on classification dataset"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        labels = batch['labels'].to(device)
        audio_inputs = {k: v.to(device) for k, v in batch.items() 
                       if k in {"input_features", "attention_mask"}}
        
        # Debug info for first few batches
        if batch_idx < 3:
            print(f"  Batch {batch_idx}: size={len(labels)}, labels={labels.tolist()}")
        
        # Skip batches that are too small or have only one class
        unique_labels = torch.unique(labels)
        if len(labels) < 2 or len(unique_labels) < 2:
            print(f"  Skipping batch {batch_idx}: size={len(labels)}, unique_labels={len(unique_labels)}")
            continue
            
        try:
            # Get normalized embeddings
            embeddings = model.encode_audio(audio_inputs)  # Already normalized in encode_audio
            
            # Double-check normalization
            norms = torch.norm(embeddings, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                print(f"  Warning: embeddings not normalized. Norms: {norms}")
                embeddings = F.normalize(embeddings, dim=1)
            
            # Compute supervised contrastive loss
            loss = supervised_contrastive_loss(embeddings, labels, temperature)
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf loss in batch {batch_idx}, skipping...")
                continue
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.projection_head.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            valid_batches += 1
            
        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            continue
        
        num_batches += 1
    
    if valid_batches == 0:
        print("  Warning: No valid batches processed!")
        return 0.0
        
    avg_loss = total_loss / valid_batches
    print(f"  Processed {valid_batches}/{num_batches} batches successfully")
    return avg_loss

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

def evaluate_with_centroids(model, eval_loader, train_loader, device):
    """Evaluate using centroid-based classification"""
    model.eval()
    
    # Compute class centroids from training data
    centroids = compute_class_centroids(model, train_loader, device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        # Check if eval_loader is classification or contrastive type
        for batch in eval_loader:
            if 'labels' in batch:
                # Classification loader - we have labels and can batch process
                labels = batch['labels'].to(device)
                audio_inputs = {k: v.to(device) for k, v in batch.items() 
                               if k in {"input_features", "attention_mask"}}
                
                # Get embeddings for the batch
                embeddings = model.encode_audio(audio_inputs)
                
                # Compute similarities to centroids for each sample
                for i, embedding in enumerate(embeddings):
                    similarities = {}
                    for class_id, centroid in centroids.items():
                        similarities[class_id] = F.cosine_similarity(
                            embedding.unsqueeze(0), 
                            centroid.unsqueeze(0)
                        ).item()
                    
                    # Predict class with highest similarity
                    predicted_class = max(similarities.keys(), key=lambda k: similarities[k])
                    
                    all_predictions.append(predicted_class)
                    all_labels.append(labels[i].item())
            else:
                # Contrastive loader - process dataset directly
                eval_dataset = eval_loader.dataset
                for i in range(len(eval_dataset.paths)):
                    inputs, true_label = eval_dataset._load_audio(i)
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                    
                    embedding = model.encode_audio(inputs).squeeze(0)
                    
                    similarities = {}
                    for class_id, centroid in centroids.items():
                        similarities[class_id] = F.cosine_similarity(
                            embedding.unsqueeze(0), 
                            centroid.unsqueeze(0)
                        ).item()
                    
                    predicted_class = max(similarities.keys(), key=lambda k: similarities[k])
                    
                    all_predictions.append(predicted_class)
                    all_labels.append(true_label)
                break  # Only process once for the whole dataset
    
    # Compute accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    return accuracy, all_predictions, all_labels

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




def train_supervised_contrastive_model(config):
    """Train model using supervised contrastive loss with ClassificationDataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build classification data loaders
    loaders, processor = build_classification_loader(config)
    dataset_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Initialize model and optimizer
    model = ContrastiveModel(device)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=config["lr"])
    
    print(f"Training supervised contrastive model for {config['epochs']} epochs...")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, config["epochs"] + 1):
        # Train using supervised contrastive loss
        train_loss = train_epoch_classification(model, loaders['train'], optimizer, device)
        plot_tsne_embeddings(model, loaders['train'], device, epoch)
        # Validate using centroid-based classification
        val_acc, _, _ = evaluate_with_centroids(model, loaders['val'], loaders['train'], device)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        print(f"Epoch {epoch:2d}/{config['epochs']} | "
              f"Train SupCon Loss={train_loss:.3f} | "
              f"Val Centroid Acc={val_acc:.3f}")
        
        # Early stopping
        if epoch - best_epoch > 5 and epoch > 10:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break
    
    # Final evaluation on test set
    test_acc, test_preds, test_labels = evaluate_with_centroids(
        model, loaders['test'], loaders['train'], device
    )
    
    print(f"\nFinal Results (Supervised Contrastive):")
    print(f"Best Val Acc: {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"Test Centroid Acc: {test_acc:.3f}")
    
    # Print confusion matrix info
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print(f"Test Confusion Matrix:")
    print(f"  Cat predicted as: Cat={cm[0,0]}, Dog={cm[0,1]}")
    print(f"  Dog predicted as: Cat={cm[1,0]}, Dog={cm[1,1]}")
    
    return model, {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_confusion_matrix': cm
    }


