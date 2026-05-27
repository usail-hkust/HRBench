"""
MLP Classifier model for adaptive think/nothink switching.
Architecture: prompt_embedding(4096) -> MLP(128) + scalar_features(4) -> classifier -> P(think)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Encode prompt embedding + scalar features into a combined representation."""

    def __init__(self, embedding_dim: int = 4096, prompt_hidden_dim: int = 128):
        super().__init__()
        self.prompt_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, prompt_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, scalar_features: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalar_features: (batch, 4) — [mean_entropy, max_entropy, max_eos_prob, mean_eos_prob]
            prompt_embedding: (batch, embedding_dim) — prompt hidden state
        Returns:
            combined: (batch, prompt_hidden_dim + 4)
        """
        prompt_feat = self.prompt_mlp(prompt_embedding)
        return torch.cat([scalar_features, prompt_feat], dim=-1)


class MLPClassifier(nn.Module):
    """
    Binary classifier: should we switch from nothink to think mode?

    Input: scalar features (4-dim) + prompt embedding (4096-dim)
    Output: P(switch to think) in [0, 1]
    """

    def __init__(
        self,
        embedding_dim: int = 4096,
        prompt_hidden_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_dim, prompt_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(prompt_hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, scalar_features: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        combined = self.feature_encoder(scalar_features, prompt_embedding)
        return self.classifier(combined)
