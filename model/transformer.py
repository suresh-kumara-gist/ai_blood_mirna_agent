import torch
import torch.nn as nn


class TabularMiRNATransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model)
        self.feature_embedding = nn.Embedding(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        # x: [batch, features]
        b, f = x.shape
        feat_ids = torch.arange(f, device=x.device).unsqueeze(0).expand(b, f)
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embedding(feat_ids)
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)
