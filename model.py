"""
Multimodal transformer for grasping force prediction.

This module implements the core model requested by the user:
  * A 2D image encoder that converts RGB frames into a compact sequence of tokens.
  * A 1D tactile encoder that ingests temporal sensor traces shaped (500, 6).
  * A transformer encoder that fuses both modalities through self-attention.
  * A small regression head that maps the fused representation to a single force value.

The implementation intentionally avoids external dependencies beyond PyTorch so
it can be trained end-to-end or fine-tuned together with downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MultimodalTransformerConfig:
    """
    Configuration container for ``MultimodalForceTransformer``.

    Attributes:
        image_in_channels: Number of input channels for the image stream (default: RGB → 3).
        image_token_grid: Spatial resolution (height == width) after the image encoder.
            The number of image tokens equals ``image_token_grid ** 2``.
        tactile_channels: Number of sensor channels in the tactile stream.
        tactile_tokens: Number of tokens that represent the temporal tactile signal.
        d_model: Transformer embedding dimension. Both encoders project into this space.
        nhead: Number of attention heads in the transformer encoder.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Hidden size of the transformer's feed-forward sublayer.
        dropout: Dropout probability applied at several stages for regularisation.
    """

    image_in_channels: int = 3
    image_token_grid: int = 14
    tactile_channels: int = 6
    tactile_tokens: int = 20
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    def num_image_tokens(self) -> int:
        return self.image_token_grid * self.image_token_grid


class ImageEncoder2D(nn.Module):
    """
    Lightweight convolutional encoder that maps images to a fixed grid of tokens.

    The encoder uses a small convolutional stem with progressive downsampling followed
    by an adaptive pooling stage. This ensures a consistent number of output tokens
    regardless of the input image size, provided the height and width are at least 8×
    ``image_token_grid`` (e.g. 112px when the grid is 14).
    """

    def __init__(self, in_channels: int, embed_dim: int, grid_size: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim // 2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Tensor with shape ``(batch, num_tokens, embed_dim)`` where
            ``num_tokens == grid_size ** 2``.
        """
        features = self.stem(images)
        pooled = self.pool(features)
        embedded = self.proj(pooled)  # (B, embed_dim, grid, grid)
        tokens = embedded.flatten(2).transpose(1, 2)  # (B, num_tokens, embed_dim)

        tokens = self.dropout(tokens)
        tokens = self.norm(tokens)
        return tokens


class TactileEncoder1D(nn.Module):
    """
    Temporal encoder for tactile sequences shaped ``(batch, 500, 6)``.

    The encoder applies a stack of 1D convolutions to summarise local temporal
    structure before projecting towards the transformer embedding dimension.
    """

    def __init__(self, in_channels: int, embed_dim: int, num_tokens: int, dropout: float = 0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(num_tokens),
        )

        self.proj = nn.Linear(256, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile: Tensor shaped ``(batch, seq_len, channels)`` or ``(batch, channels, seq_len)``.
                The canonical representation expected by downstream code is ``(batch, 500, 6)``.

        Returns:
            Tensor of tokens shaped ``(batch, num_tokens, embed_dim)``.
        """
        if tactile.dim() != 3:
            raise ValueError(f"Expected tactile input with 3 dimensions, got shape {tuple(tactile.shape)}")

        if tactile.size(1) == 6:
            tactile = tactile.transpose(1, 2)  # Convert (B, channels, seq) → (B, seq, channels)

        if tactile.size(2) != 6:
            raise ValueError(f"Expected tactile channel dimension to be 6, got {tactile.size(2)}")

        tactile = tactile.transpose(1, 2)  # (B, channels, seq_len) for Conv1d
        features = self.network(tactile)  # (B, 256, num_tokens)
        tokens = features.transpose(1, 2)  # (B, num_tokens, 256)

        tokens = self.proj(tokens)
        tokens = self.dropout(tokens)
        tokens = self.norm(tokens)
        return tokens


class MultimodalForceTransformer(nn.Module):
    """
    Full multimodal model that fuses image and tactile encoders with a transformer.

    Typical usage:
        >>> config = MultimodalTransformerConfig()
        >>> model = MultimodalForceTransformer(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> tactile = torch.randn(4, 500, 6)
        >>> force = model(images, tactile)
        >>> force.shape
        torch.Size([4, 1])
    """

    def __init__(self, config: Optional[MultimodalTransformerConfig] = None):
        super().__init__()
        self.config = config or MultimodalTransformerConfig()

        self.image_encoder = ImageEncoder2D(
            in_channels=self.config.image_in_channels,
            embed_dim=self.config.d_model,
            grid_size=self.config.image_token_grid,
            dropout=self.config.dropout,
        )
        self.tactile_encoder = TactileEncoder1D(
            in_channels=self.config.tactile_channels,
            embed_dim=self.config.d_model,
            num_tokens=self.config.tactile_tokens,
            dropout=self.config.dropout,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.image_positional = nn.Parameter(
            torch.randn(1, self.config.num_image_tokens(), self.config.d_model) * 0.02
        )
        self.tactile_positional = nn.Parameter(
            torch.randn(1, self.config.tactile_tokens, self.config.d_model) * 0.02
        )
        self.modal_dropout = nn.Dropout(self.config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.transformer_norm = nn.LayerNorm(self.config.d_model)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(
        self, images: torch.Tensor, tactile: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates token sequences for both modalities, augmented with positional encodings.

        Returns:
            cls_tokens: Expanded CLS token, shape ``(batch, 1, d_model)``.
            image_tokens: Image tokens with positional encoding applied.
            tactile_tokens: Tactile tokens with positional encoding applied.
        """
        image_tokens = self.image_encoder(images) + self.image_positional
        tactile_tokens = self.tactile_encoder(tactile) + self.tactile_positional

        batch_size = images.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        return cls_tokens, image_tokens, tactile_tokens

    def forward(
        self, images: torch.Tensor, tactile: torch.Tensor, return_tokens: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Batch of RGB images ``(batch, 3, H, W)``.
            tactile: Batch of tactile sequences ``(batch, 500, 6)``.
            return_tokens: When ``True``, the method also returns the fused token sequence.

        Returns:
            force: Regression output ``(batch, 1)``.
            tokens (optional): Token sequence after the transformer ``(batch, total_tokens, d_model)``.
        """
        cls_tokens, image_tokens, tactile_tokens = self.forward_features(images, tactile)
        tokens = torch.cat([cls_tokens, image_tokens, tactile_tokens], dim=1)
        tokens = self.modal_dropout(tokens)

        fused = self.transformer(tokens)
        fused = self.transformer_norm(fused)

        cls_output = fused[:, 0, :]
        force = self.regression_head(cls_output)

        if return_tokens:
            return force, fused
        return force


__all__ = [
    "MultimodalTransformerConfig",
    "ImageEncoder2D",
    "TactileEncoder1D",
    "MultimodalForceTransformer",
]
