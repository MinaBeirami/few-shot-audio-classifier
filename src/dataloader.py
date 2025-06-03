import random
import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import ClapProcessor
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.plots import plot_dataset_distribution, plot_tsne_embeddings
from src.dataset import balance_dataset_explicit
from src.losses import supervised_contrastive_loss, contrastive_loss
from src.audio import augment_audio

SAMPLE_RATE = 48000


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with triplet loss."""

    def __init__(self, paths, processor, is_train):
        self.paths = paths
        self.processor = processor
        self.is_train = is_train

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

        # Resample to 48kHz
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)

        # Convert to mono : CLAP assume a single‐channel 1D waveform
        wav = wav.mean(0)

        if self.is_train:
            wav = augment_audio(wav)

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

        return {"anchor": anchor_x, "positive": positive_x, "negative": negative_x}


class ClassificationDataset(Dataset):
    """Simple dataset that returns audio inputs and labels for classification."""

    def __init__(self, paths, processor, is_train):
        self.paths = paths
        self.processor = processor
        self.is_train = is_train
        # Precompute labels (0=cat, 1=dog)
        self.labels = [0 if p.parent.name.startswith("cat") else 1 for p in paths]

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

        if self.is_train:
            wav = augment_audio(wav)

        inputs = self.processor(audios=wav, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, self.labels[idx]

    def __getitem__(self, idx):
        """
        Return audio inputs and label
        """
        inputs, label = self._load_audio(idx)
        inputs["labels"] = torch.tensor(label)
        return inputs


def oversample_indices(indices, labels, target_label=0, multiplier=3):
    """
    Duplicate indices of `target_label` to oversample class.
    """
    tgt = [i for i in indices if labels[i] == target_label]
    need = (multiplier - 1) * len(tgt)
    extra = [random.choice(tgt) for _ in range(need)]
    return indices + extra


def contrastive_collate(batch):
    """Collate function for contrastive batches"""

    def stack_side(side):
        return {
            k: torch.stack([item[side][k] for item in batch])
            for k in batch[0]["anchor"]
        }

    return {
        "anchor": stack_side("anchor"),
        "positive": stack_side("positive"),
        "negative": stack_side("negative"),
    }


def classification_collate(batch):
    """Collate function for classification batches"""
    # batch is list of dicts with same keys
    out = {k: torch.stack([d[k] for d in batch]) for k in batch[0] if k != "labels"}
    out["labels"] = torch.tensor([d["labels"].item() for d in batch])
    return out


def build_contrastive_loader(config):
    """Build data loaders for contrastive learning with integrated preprocessing"""

    # Gather all audio files
    wavs = sorted(Path(config["data_dir"]).rglob("*.wav"))
    labels = [0 if p.parent.name.startswith("cat") else 1 for p in wavs]
    print(
        f"Found {len(wavs)} audio files: {sum(1 for l in labels if l == 0)} cats, {sum(1 for l in labels if l == 1)} dogs"
    )

    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(wavs))), test_size=0.3, stratify=labels, random_state=42
    )

    # Few-shot selection from training set
    few_shot_idx = []
    class_counts = {0: 0, 1: 0}
    for i in train_idx:
        if class_counts[labels[i]] < config["few_shot_per_class"]:
            few_shot_idx.append(i)
            class_counts[labels[i]] += 1

    if config.get("balance_method") == "oversample":
        # only oversampling cats as it performs worse
        few_shot_idx = oversample_indices(
            few_shot_idx,
            labels,
            target_label=0,
            multiplier=config.get("oversample_multiplier", 3),
        )

    # Create validation set from remaining training data (after first split)
    remaining_train = [i for i in train_idx if i not in few_shot_idx]
    if len(remaining_train) > 0:
        val_idx, _ = train_test_split(
            remaining_train,
            test_size=0.5,
            stratify=[labels[i] for i in remaining_train],
            random_state=42,
        )
    else:
        val_idx = []

    # Plot distribution before creating datasets
    split_indices = {"train": few_shot_idx, "val": val_idx, "test": test_idx}
    plot_dataset_distribution(split_indices, labels)

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Create data loaders with balance method info
    loaders = {}
    for split, indices in [
        ("train", few_shot_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        dataset = ContrastiveDataset(
            [wavs[i] for i in indices],
            processor,
            is_train=(split == "train"),
            # balance_method=balance_method,
            # apply_augment=(split == 'train')  # Only augment during training
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            collate_fn=contrastive_collate,
        )

    return loaders, processor


def build_classification_loader(config):
    """Build data loaders using ClassificationDataset instead of ContrastiveDataset"""
    # Gather all audio files
    wavs = sorted(Path(config["data_dir"]).rglob("*.wav"))
    labels = [0 if p.parent.name.startswith("cat") else 1 for p in wavs]

    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(wavs))), test_size=0.3, stratify=labels, random_state=42
    )

    # Few-shot selection from training set
    few_shot_idx = []
    class_counts = {0: 0, 1: 0}
    for i in train_idx:
        if class_counts[labels[i]] < config["few_shot_per_class"]:
            few_shot_idx.append(i)
            class_counts[labels[i]] += 1

    if config.get("balance_method") == "oversample":
        # only oversampling cats as it performs worse
        few_shot_idx = oversample_indices(
            few_shot_idx,
            labels,
            target_label=0,
            multiplier=config.get("oversample_multiplier", 3),
        )

    # Create validation set from remaining training data
    remaining_train = [i for i in train_idx if i not in few_shot_idx]
    val_idx, _ = train_test_split(
        remaining_train,
        test_size=0.5,
        stratify=[labels[i] for i in remaining_train],
        random_state=42,
    )

    # Plot distribution before creating datasets
    split_indices = {"train": few_shot_idx, "val": val_idx, "test": test_idx}
    plot_dataset_distribution(split_indices, labels)

    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Create data loaders using ClassificationDataset
    loaders = {}
    for split, indices in [
        ("train", few_shot_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        dataset = ClassificationDataset(
            [wavs[i] for i in indices], processor, is_train=(split == "train")
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            collate_fn=classification_collate,
            num_workers=4,
            persistent_workers=True,
        )

    return loaders, processor
