import torch

import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from pathlib import Path

from src.losses import contrastive_loss, supervised_contrastive_loss

from src.plots import plot_tsne_embeddings
from src.dataloader import build_contrastive_loader, build_classification_loader
from src.models import ContrastiveModel

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def train_epoch(model, loader, optimizer, device):
    """Train one epoch using triplet-based contrastive loss"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # Move to device
        def to_device(x):
            return {k: v.to(device) for k, v in x.items()}

        anchor_inputs = to_device(batch["anchor"])
        positive_inputs = to_device(batch["positive"])
        negative_inputs = to_device(batch["negative"])

        # Encode all triplets
        anchor_emb = model.encode_audio(anchor_inputs)
        positive_emb = model.encode_audio(positive_inputs)
        negative_emb = model.encode_audio(negative_inputs)

        # Compute loss
        loss, pos_sim, neg_sim = contrastive_loss(
            anchor_emb, positive_emb, negative_emb
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        batch_size = anchor_emb.size(0)
        total_loss += loss.item() * batch_size
        correct += (pos_sim > neg_sim).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_epoch_classification(model, loader, optimizer, device, temperature=0.07):
    """Train one epoch using supervised contrastive loss on classification dataset"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Move to device
        labels = batch["labels"].to(device)
        audio_inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_features", "attention_mask"}
        }

        # Debug info for first few batches
        if batch_idx < 3:
            print(f"Batch {batch_idx}: size={len(labels)}, labels={labels.tolist()}")

        # Skip batches that are too small or have only one class
        unique_labels = torch.unique(labels)
        if len(labels) < 2 or len(unique_labels) < 2:
            print(
                f"Skipping batch {batch_idx}: size={len(labels)}, unique_labels={len(unique_labels)}"
            )
            continue

        try:
            # Get normalized embeddings
            embeddings = model.encode_audio(
                audio_inputs
            )  # Already normalized in encode_audio

            # Double-check normalization
            norms = torch.norm(embeddings, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                print(f"embeddings not normalized. Norms: {norms}")
                embeddings = F.normalize(embeddings, dim=1)

            # Compute supervised contrastive loss
            loss = supervised_contrastive_loss(embeddings, labels, temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.projection_head.parameters(), max_norm=1.0
            )

            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            valid_batches += 1

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

        num_batches += 1

    if valid_batches == 0:
        print("No valid batches processed!")
        return 0.0

    avg_loss = total_loss / valid_batches
    print(f"Processed {valid_batches}/{num_batches} batches successfully")
    return avg_loss


def train_epoch_classification(model, loader, optimizer, device, temperature=0.07):
    """Train one epoch using supervised contrastive loss on classification dataset"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Move to device
        labels = batch["labels"].to(device)
        audio_inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_features", "attention_mask"}
        }

        # Debug info for first few batches
        if batch_idx < 3:
            print(f"Batch {batch_idx}: size={len(labels)}, labels={labels.tolist()}")

        # Skip batches that are too small or have only one class
        unique_labels = torch.unique(labels)
        if len(labels) < 2 or len(unique_labels) < 2:
            print(
                f"Skipping batch {batch_idx}: size={len(labels)}, unique_labels={len(unique_labels)}"
            )
            continue

        try:
            # Get normalized embeddings
            embeddings = model.encode_audio(
                audio_inputs
            )  # Already normalized in encode_audio

            # Double-check normalization
            norms = torch.norm(embeddings, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                print(f"embeddings not normalized. Norms: {norms}")
                embeddings = F.normalize(embeddings, dim=1)

            # Compute supervised contrastive loss
            loss = supervised_contrastive_loss(embeddings, labels, temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.projection_head.parameters(), max_norm=1.0
            )

            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            valid_batches += 1

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

        num_batches += 1

    if valid_batches == 0:
        print("No valid batches processed!")
        return 0.0

    avg_loss = total_loss / valid_batches
    print(f"Processed {valid_batches}/{num_batches} batches successfully")
    return avg_loss


def compute_class_centroids(model, train_loader, device):
    """Compute class centroids from training data"""
    model.eval()

    class_embeddings = {0: [], 1: []}

    with torch.no_grad():
        for batch in train_loader:
            # For classification loader, we have labels directly
            if "labels" in batch:
                labels = batch["labels"]
                audio_inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in {"input_features", "attention_mask"}
                }

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
            # if no samples, initialize a zero vector
            centroids[class_id] = torch.zeros(
                model.projection_head.out_features, device=device
            )

    return centroids


def evaluate_with_centroids(model, eval_loader, train_loader, device):
    """Evaluate using centroid based classification"""
    model.eval()

    # Compute class centroids from training data
    centroids = compute_class_centroids(model, train_loader, device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        # Check if eval_loader is classification or contrastive type
        for batch in eval_loader:
            if "labels" in batch:
                # Classification loader - we have labels and can batch process
                labels = batch["labels"].to(device)
                audio_inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in {"input_features", "attention_mask"}
                }

                # Get embeddings for the batch
                embeddings = model.encode_audio(audio_inputs)

                # Compute similarities to centroids for each sample
                for i, embedding in enumerate(embeddings):
                    similarities = {}
                    for class_id, centroid in centroids.items():
                        similarities[class_id] = F.cosine_similarity(
                            embedding.unsqueeze(0), centroid.unsqueeze(0)
                        ).item()

                    # Predict class with highest similarity
                    predicted_class = max(
                        similarities.keys(), key=lambda k: similarities[k]
                    )

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
                            embedding.unsqueeze(0), centroid.unsqueeze(0)
                        ).item()

                    predicted_class = max(
                        similarities.keys(), key=lambda k: similarities[k]
                    )

                    all_predictions.append(predicted_class)
                    all_labels.append(true_label)
                break  # Only process once for the whole dataset

    # Compute accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(
        all_labels
    )

    return accuracy, all_predictions, all_labels


def compute_train_embeddings(model, train_loader, device):

    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in train_loader:
            if "labels" in batch:
                labels = batch["labels"]  # shape (B,)
                # pick only the audio inputs
                audio_inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in {"input_features", "attention_mask"}
                }
                embeddings = model.encode_audio(audio_inputs)
                # normalize
                embeddings = F.normalize(embeddings, dim=1)

                all_embs.append(embeddings.cpu())
                all_labels.append(labels.cpu())

            else:
                train_dataset = train_loader.dataset
                for i in range(len(train_dataset.paths)):
                    inputs, label = train_dataset._load_audio(i)
                    # inputs is a dict of Tensors; label is an int
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                    emb = model.encode_audio(inputs).squeeze(0)  # shape (D,)
                    emb = F.normalize(emb, dim=0)  # unit-norm
                    all_embs.append(emb.cpu().unsqueeze(0))  # shape (1, D)
                    all_labels.append(torch.tensor([label]))  # shape (1,)

                break

    # concatenate everything into one big tensor / list
    train_embeddings = torch.cat(all_embs, dim=0).to(device)  # (N_train, D)
    train_labels = torch.cat(all_labels, dim=0).tolist()  # [int, int, ..., int]
    return train_embeddings, train_labels


def evaluate_with_knn(model, eval_loader, train_loader, device, k=5):

    model.eval()

    train_embeddings, train_labels = compute_train_embeddings(
        model, train_loader, device
    )

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in eval_loader:
            if "labels" in batch:
                labels = batch["labels"].to(device)  # (B,)
                audio_inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in {"input_features", "attention_mask"}
                }
                emb_batch = model.encode_audio(audio_inputs)  # (B, D)
                emb_batch = F.normalize(emb_batch, dim=1)  # (B, D)

                # cosine similariy
                sims = emb_batch @ train_embeddings.t()

                # for each sample in the batch, pick top k indices
                topk_inds = sims.topk(k=k, dim=1).indices  # shape (B, k)
                for i in range(emb_batch.size(0)):
                    neighbor_inds = (
                        topk_inds[i].cpu().tolist()
                    )  # k indices into train set
                    neighbor_labels = [train_labels[idx] for idx in neighbor_inds]
                    # majority vote
                    pred = max(set(neighbor_labels), key=neighbor_labels.count)
                    all_preds.append(pred)
                    all_trues.append(labels[i].item())

            else:
                eval_dataset = eval_loader.dataset
                for i in range(len(eval_dataset.paths)):
                    inputs, true_label = eval_dataset._load_audio(i)
                    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                    emb = model.encode_audio(inputs).squeeze(0)  # (D,)
                    emb = F.normalize(emb, dim=0).unsqueeze(0)  # (1, D)

                    sims = emb @ train_embeddings.t()  # (1, N_train)
                    topk_inds = sims.topk(k=k, dim=1).indices.squeeze(0).cpu().tolist()
                    neighbor_labels = [train_labels[idx] for idx in topk_inds]
                    pred = max(set(neighbor_labels), key=neighbor_labels.count)

                    all_preds.append(pred)
                    all_trues.append(true_label)

                break

    correct = sum(p == t for p, t in zip(all_preds, all_trues))
    accuracy = correct / len(all_trues)
    return accuracy, all_preds, all_trues


def train_contrastive_model(
    config,
    eval_methods=("centroid", "knn"),
    knn_k=4,
    checkpoint_path="best_contrastive_model.pth",
):
    """
    Train model using triplet-based contrastive loss
    track best‐val CMs for centroid & KNN, and plot them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build contrastive data loaders
    loaders, processor = build_contrastive_loader(config)
    dataset_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Dataset sizes: {dataset_sizes}")

    model = ContrastiveModel(device)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=config["lr"])

    best_val_acc = 0.0
    best_epoch = 0

    # To store the best‐validation confusion matrices
    best_val_cm = {}

    for epoch in range(1, config["epochs"] + 1):
        # Train with triplet loss
        train_loss, train_triplet_acc = train_epoch(
            model, loaders["train"], optimizer, device
        )

        # Validation: compute both centroid & KNN if requested
        val_acc_centroid = val_acc_knn = None
        preds_centroid = labels_centroid = None
        preds_knn = labels_knn = None

        if "centroid" in eval_methods:
            val_acc_centroid, preds_centroid, labels_centroid = evaluate_with_centroids(
                model,
                eval_loader=loaders["val"],
                train_loader=loaders["train"],
                device=device,
            )
        if "knn" in eval_methods:
            val_acc_knn, preds_knn, labels_knn = evaluate_with_knn(
                model,
                eval_loader=loaders["val"],
                train_loader=loaders["train"],
                device=device,
                k=knn_k,
            )

        # which validation accuracy to use for "best"
        if "knn" in eval_methods and val_acc_knn is not None:
            current_val_acc = val_acc_knn
        else:
            current_val_acc = val_acc_centroid

        # save best checkpoint and record confusion matrices
        if current_val_acc is not None and current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

            if "centroid" in eval_methods and preds_centroid is not None:
                cm_c = confusion_matrix(labels_centroid, preds_centroid, labels=[0, 1])
                best_val_cm["centroid"] = cm_c
            if "knn" in eval_methods and preds_knn is not None:
                cm_k = confusion_matrix(labels_knn, preds_knn, labels=[0, 1])
                best_val_cm["knn"] = cm_k

        # Print training & validation stats
        metrics_parts = []
        if val_acc_centroid is not None:
            metrics_parts.append(f"Val Centroid Acc={val_acc_centroid:.3f}")
        if val_acc_knn is not None:
            metrics_parts.append(f"Val KNN Acc={val_acc_knn:.3f}")
        metrics_str = " | ".join(metrics_parts)

        print(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"Train Loss={train_loss:.3f}, Triplet Acc={train_triplet_acc:.3f} | {metrics_str}"
        )

        # Early stopping
        if epoch - best_epoch > 5 and epoch > 10:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load the best checkpoint before final evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(
        f"\nLoaded model from best epoch {best_epoch} with val_acc={best_val_acc:.3f}"
    )

    # best‐validation confusion matrices side by side
    methods_to_plot = [m for m in eval_methods if m in best_val_cm]
    n_methods = len(methods_to_plot)

    if n_methods > 0:
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, methods_to_plot):
            cm = best_val_cm[method]
            title = f"Best‐Val {method.capitalize()} CM"
            if method == ("knn" if "knn" in methods_to_plot else "centroid"):
                title += f" (Acc={best_val_acc:.3f})"
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(title)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(["Cat", "Dog"])
            ax.set_yticklabels(["Cat", "Dog"])
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        plt.tight_layout()
        plt.show()
    else:
        print("No best val confusion matrices to plot.")

    # evaluation on test set, with confusion matrices
    cms_test = {}
    test_results = {}

    if "centroid" in eval_methods:
        test_acc_c, test_preds_c, test_labels_c = evaluate_with_centroids(
            model,
            eval_loader=loaders["test"],
            train_loader=loaders["train"],
            device=device,
        )
        cm_c_test = confusion_matrix(test_labels_c, test_preds_c, labels=[0, 1])
        cms_test["centroid"] = cm_c_test
        test_results["centroid"] = test_acc_c

    if "knn" in eval_methods:
        test_acc_k, test_preds_k, test_labels_k = evaluate_with_knn(
            model,
            eval_loader=loaders["test"],
            train_loader=loaders["train"],
            device=device,
            k=knn_k,
        )
        cm_k_test = confusion_matrix(test_labels_k, test_preds_k, labels=[0, 1])
        cms_test["knn"] = cm_k_test
        test_results["knn"] = test_acc_k

    print("\nFinal Results (Triplet Contrastive):")
    for method, acc in test_results.items():
        print(f"Test {method.capitalize()} Acc: {acc:.3f}")

    # Plot test confusion matrices side by side
    methods_test = list(cms_test.keys())
    n_test = len(methods_test)
    if n_test:
        fig, axes = plt.subplots(1, n_test, figsize=(5 * n_test, 4))
        if n_test == 1:
            axes = [axes]
        for ax, method in zip(axes, methods_test):
            cm = cms_test[method]
            acc = test_results[method]
            ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"Test {method.capitalize()} CM (Acc={acc:.3f})")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels([0, 1])
            ax.set_yticklabels([0, 1])
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        plt.tight_layout()
        plt.show()

    return model, {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_val_confusion_matrices": best_val_cm,
        "test_results": test_results,
        "test_confusion_matrices": cms_test,
    }


def train_supervised_contrastive_model(
    config, eval_methods=("centroid", "knn"), knn_k=4, checkpoint_path="best_model.pth"
):
    """Train model with SupCon loss, track best‐val CMs for centroid & KNN, and plot them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build loaders
    loaders, processor = build_classification_loader(config)
    dataset_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Dataset sizes: {dataset_sizes}")

    model = ContrastiveModel(device)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=config["lr"])

    best_val_acc = 0.0
    best_epoch = 0

    # Placeholders for best‐val confusion matrices & labels/preds
    best_val_cm = {}
    best_val_details = {}

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch_classification(
            model, loaders["train"], optimizer, device
        )
        plot_tsne_embeddings(model, loaders["train"], device, epoch)

        # compute both accuracy & preds/labels
        val_acc_centroid = val_acc_knn = None
        preds_centroid = labels_centroid = None
        preds_knn = labels_knn = None

        if "centroid" in eval_methods:
            val_acc_centroid, preds_centroid, labels_centroid = evaluate_with_centroids(
                model,
                eval_loader=loaders["val"],
                train_loader=loaders["train"],
                device=device,
            )
        if "knn" in eval_methods:
            val_acc_knn, preds_knn, labels_knn = evaluate_with_knn(
                model,
                eval_loader=loaders["val"],
                train_loader=loaders["train"],
                device=device,
                k=knn_k,
            )

        # Determine current validation metric (prefer KNN if both)
        if "knn" in eval_methods and val_acc_knn is not None:
            current_val_acc = val_acc_knn
        else:
            current_val_acc = val_acc_centroid

        # If new best, save checkpoint & validation CM
        if current_val_acc is not None and current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

            # Save confusion matrices for the best validation epoch
            if "centroid" in eval_methods and preds_centroid is not None:
                cm_c = confusion_matrix(labels_centroid, preds_centroid, labels=[0, 1])
                best_val_cm["centroid"] = cm_c
            if "knn" in eval_methods and preds_knn is not None:
                cm_k = confusion_matrix(labels_knn, preds_knn, labels=[0, 1])
                best_val_cm["knn"] = cm_k

        # Print summary for this epoch
        metrics_parts = []
        if val_acc_centroid is not None:
            metrics_parts.append(f"Val Centroid Acc={val_acc_centroid:.3f}")
        if val_acc_knn is not None:
            metrics_parts.append(f"Val KNN Acc={val_acc_knn:.3f}")
        metrics_str = " | ".join(metrics_parts)

        print(
            f"Epoch {epoch:2d}/{config['epochs']} | "
            f"Train SupCon Loss={train_loss:.3f} | {metrics_str}"
        )

        # Early stopping
        if epoch - best_epoch > 5 and epoch > 10:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(
        f"\nLoaded model from best epoch {best_epoch} with val_acc={best_val_acc:.3f}"
    )

    # bestval confusion matrices side by side
    methods_to_plot = [m for m in eval_methods if m in best_val_cm]
    n_methods = len(methods_to_plot)

    if n_methods > 0:
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, methods_to_plot):
            cm = best_val_cm[method]
            acc_label = (
                best_val_acc
                if method == ("knn" if "knn" in methods_to_plot else "centroid")
                else None
            )
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            title = f"Best Val {method.capitalize()} CM"
            if acc_label is not None:
                title += f" (Acc={acc_label:.3f})"
            ax.set_title(title)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels([0, 1])
            ax.set_yticklabels([0, 1])
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        plt.tight_layout()
        plt.show()
    else:
        print(
            "No validation confusion matrices to plot (ensure eval_methods includes 'centroid' or 'knn')."
        )

    # evaluation on test set
    cms_test = {}
    test_results = {}

    if "centroid" in eval_methods:
        test_acc_c, test_preds_c, test_labels_c = evaluate_with_centroids(
            model,
            eval_loader=loaders["test"],
            train_loader=loaders["train"],
            device=device,
        )
        cm_c_test = confusion_matrix(test_labels_c, test_preds_c, labels=[0, 1])
        cms_test["centroid"] = cm_c_test
        test_results["centroid"] = test_acc_c

    if "knn" in eval_methods:
        test_acc_k, test_preds_k, test_labels_k = evaluate_with_knn(
            model,
            eval_loader=loaders["test"],
            train_loader=loaders["train"],
            device=device,
            k=knn_k,
        )
        cm_k_test = confusion_matrix(test_labels_k, test_preds_k, labels=[0, 1])
        cms_test["knn"] = cm_k_test
        test_results["knn"] = test_acc_k

    print("Final Results (Supervised Contrastive):")
    for method, acc in test_results.items():
        print(f"Test {method.capitalize()} Acc: {acc:.3f}")

    # Plot test confusion matrices side by side
    methods_test = list(cms_test.keys())
    n_test = len(methods_test)
    if n_test:
        fig, axes = plt.subplots(1, n_test, figsize=(5 * n_test, 4))
        if n_test == 1:
            axes = [axes]
        for ax, method in zip(axes, methods_test):
            cm = cms_test[method]
            acc = test_results[method]
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"Test {method.capitalize()} CM (Acc={acc:.3f})")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels([0, 1])
            ax.set_yticklabels([0, 1])
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        plt.tight_layout()
        plt.show()

    return model, {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_val_confusion_matrices": best_val_cm,
        "test_results": test_results,
        "test_confusion_matrices": cms_test,
    }
