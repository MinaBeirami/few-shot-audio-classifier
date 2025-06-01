import torch
import torch.nn.functional as F
from torch import nn
from transformers import ClapModel


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

# def contrastive_loss(anchor, positive, negative, margin=0.5):
#     """Compute margin ranking loss for contrastive learning"""
#     # Compute cosine similarities
#     pos_sim = F.cosine_similarity(anchor, positive)
#     neg_sim = F.cosine_similarity(anchor, negative)
    
#     # Margin ranking loss: encourage pos_sim > neg_sim + margin
#     targets = torch.ones_like(pos_sim)
#     loss = F.margin_ranking_loss(pos_sim, neg_sim, targets, margin=margin)
    
#     return loss, pos_sim, neg_sim