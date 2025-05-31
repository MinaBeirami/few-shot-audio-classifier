import numpy as np



def balance_dataset_explicit(wavs, labels, balance_method='oversample', max_multiplier=3):
    """
    balance dataset by oversampling or undersampling
    """
    
    # Count samples per class
    cat_indices = [i for i, label in enumerate(labels) if label == 0]
    dog_indices = [i for i, label in enumerate(labels) if label == 1]
    
    print(f"Original distribution: {len(cat_indices)} cats, {len(dog_indices)} dogs")
    
    if balance_method == 'oversample':
        # Oversample minority class (dogs)
        if len(dog_indices) < len(cat_indices):
            target_dog_count = min(len(cat_indices), len(dog_indices) * max_multiplier)
            
            # Create additional dog samples by replicating existing ones
            additional_dogs_needed = target_dog_count - len(dog_indices)
            
            balanced_wavs = wavs.copy()
            balanced_labels = labels.copy()
            
            # Add replicated dog samples
            for _ in range(additional_dogs_needed):
                random_dog_idx = np.random.choice(dog_indices)
                balanced_wavs.append(wavs[random_dog_idx])
                balanced_labels.append(1)
            
            print(f"Added {additional_dogs_needed} replicated dog samples")
            print(f"New distribution: {sum(1 for l in balanced_labels if l == 0)} cats, {sum(1 for l in balanced_labels if l == 1)} dogs")
            
            return balanced_wavs, balanced_labels
        
    elif balance_method == 'undersample':
        # Undersample majority class (cats)
        min_samples = min(len(cat_indices), len(dog_indices))
        
        # Randomly select samples to keep
        selected_cat_indices = resample(cat_indices, n_samples=min_samples, random_state=42)
        selected_indices = selected_cat_indices + dog_indices
        
        balanced_wavs = [wavs[i] for i in selected_indices]
        balanced_labels = [labels[i] for i in selected_indices]
        
        print(f"Undersampled to {min_samples} cats, {len(dog_indices)} dogs")
        
        return balanced_wavs, balanced_labels
    
    # Return original if no balancing
    return wavs, labels