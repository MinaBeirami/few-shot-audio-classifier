import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import ClapProcessor
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.losses import contrastive_loss, supervised_contrastive_loss

from src.plots import plot_tsne_embeddings, plot_confusion_matrix
from src.dataloader import build_contrastive_loader, build_classification_loader
from src.models import ContrastiveModel

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
        
        anchor_inputs = to_device(batch['anchor'])
        positive_inputs = to_device(batch['positive'])
        negative_inputs = to_device(batch['negative'])
        
        # Encode all triplets
        anchor_emb = model.encode_audio(anchor_inputs)
        positive_emb = model.encode_audio(positive_inputs)
        negative_emb = model.encode_audio(negative_inputs)
        
        # Compute loss
        loss, pos_sim, neg_sim = contrastive_loss(anchor_emb, positive_emb, negative_emb)
        
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

def train_contrastive_model(config):
    """Train model using triplet-based contrastive loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build contrastive data loaders
    loaders, processor = build_contrastive_loader(config)
    dataset_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Dataset sizes: {dataset_sizes}")
    
    # Initialize model and optimizer
    model = ContrastiveModel(device)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=config["lr"])
    
    print(f"Training triplet contrastive model for {config['epochs']} epochs...")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, config["epochs"] + 1):
        # Train using triplet loss
        train_loss, train_triplet_acc = train_epoch(model, loaders['train'], optimizer, device)

        # Validate using centroid-based classification
        val_acc, _, _ = evaluate_with_centroids(model, loaders['val'], loaders['train'], device)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        print(f"Epoch {epoch:2d}/{config['epochs']} | "
              f"Train: Loss={train_loss:.3f}, Triplet Acc={train_triplet_acc:.3f} | "
              f"Val Centroid Acc={val_acc:.3f}")
        
        # Early stopping
        if epoch - best_epoch > 5 and epoch > 10:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break
    
    # Final evaluation on test set
    test_acc, test_preds, test_labels = evaluate_with_centroids(
        model, loaders['test'], loaders['train'], device
    )
    
    print(f"\nFinal Results (Triplet Contrastive):")
    print(f"Best Val Acc: {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"Test Centroid Acc: {test_acc:.3f}")
    
    # Print confusion matrix info
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print(f"Test Confusion Matrix:")
    print(f"  Cat predicted as: Cat={cm[0,0]}, Dog={cm[0,1]}")
    print(f"  Dog predicted as: Cat={cm[1,0]}, Dog={cm[1,1]}")
    plot_confusion_matrix(test_labels, test_preds, test_acc, title='Contrastive Model')
    return model, {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_confusion_matrix': cm
    }



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
    
    plot_confusion_matrix(test_labels, test_preds, test_acc, title='Supervised Contrastive Model')

    return model, {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_confusion_matrix': cm
    }
