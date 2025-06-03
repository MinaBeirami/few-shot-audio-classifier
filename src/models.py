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
        # self.projection_head = nn.Linear(proj_dim, 128, bias=False)
        self.projection_head = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

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
