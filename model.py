"""
Multimodal transformer for predicting gripper motion deltas.

This module implements the core model requested by the user:
  * A 2D image encoder (DINOv3 or CLIP) that converts RGB frames into a compact token stream.
  * A Perceiver-style resampler that compresses ViT patch tokens into 10 latent tokens.
  * A 1D tactile encoder that ingests temporal sensor traces shaped (≈50, 6) and emits 3 tokens.
  * A transformer encoder that fuses both modalities through self-attention.
  * A regression head that maps the fused representation to the delta gripper position target.

The implementation intentionally avoids external dependencies beyond PyTorch (plus ``transformers``
for the vision backbones) so it can be trained end-to-end together with downstream code.


  | Component               | Line | Shape                  | Description           |
  |-------------------------|------|------------------------|-----------------------|
  | PerceiverResampler      | 617  | (B,196,256)→(B,10,256) | Compress 196→10       |
  | DINOv3 register tokens  | 615  | (B, 4, 256)            | keep same             |
  | Image tokens concat     | 619  | (B, 14, 256)           | 4 register+ 10 patch  |
  | Fusion CLS token        | 631  | (B, 1, 256)            | learnable CLS         |
  | Tactile tokens          | 628  | (B, 3, 256)            | Conv1D 3 token        |
  | Final transformer input | 649  | (B, 18, 256)           | 1+14+3=18 tokens      |

  Final transformer input using dinov3:
  [CLS] [Reg1, Reg2, Reg3, Reg4] [P1'...P10'] [Tac1, Tac2, Tac3]

  Final transformer input using clip:
  [CLS] [P1'...P10'] [Tac1, Tac2, Tac3]

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


@dataclass
class MultimodalTransformerConfig:
    """
    Configuration container for ``MultimodalForceTransformer``.

    Attributes:
        image_in_channels: Number of input channels for the image stream (default: RGB → 3).
        image_token_grid: Spatial resolution (height == width) after the image encoder when using
            the lightweight CNN. The number of image tokens equals ``image_token_grid ** 2``.
        image_encoder_type: Which vision backbone to instantiate. ``"dino_v3"`` uses a pretrained
            ViT while ``"conv"`` selects the legacy CNN stem.
        dinov3_model_name: Hugging Face checkpoint identifier for the pretrained DINOv3 backbone.
        dinov3_freeze_backbone: When ``True`` the ViT weights stay frozen during training.
        dinov3_drop_cls_token: Drop the CLS token before projecting into the fusion space.
        dinov3_normalize_inputs: Apply ImageNet mean/std normalisation before DINO forwarding.

        tactile_channels: Number of sensor channels in the tactile stream.
        tactile_tokens: Number of tokens that represent the temporal tactile signal.

        perceiver_num_latents: Number of learnable latents used by the perceiver resampler.
        perceiver_num_layers: Depth of the perceiver resampler cross-attention stack.
        perceiver_num_heads: Attention heads inside the perceiver resampler.
        use_perceiver_resampler: Enable the perceiver resampler for patch tokens.
        clip_model_name: Hugging Face checkpoint identifier when using the CLIP encoder.
        clip_freeze_backbone: When ``True`` the CLIP vision encoder stays frozen.
        d_model: Transformer embedding dimension. Both encoders project into this space.
        nhead: Number of attention heads in the transformer encoder.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Hidden size of the transformer's feed-forward sublayer.
        dropout: Dropout probability applied at several stages for regularisation.
    """

    image_in_channels: int = 3
    image_token_grid: int = 14
    image_encoder_type: str = "dino_v3"                                  # Options: "dino_v3", "clip"
    dinov3_model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m" # smallest model, not perform the best accordingly to their github
    dinov3_freeze_backbone: bool = True                                 # freeze DINOv3 weights during training, no learn until in the latent token 
    dinov3_drop_cls_token: bool = True                                  # drop CLS token before fusion
    dinov3_normalize_inputs: bool = True

    tactile_channels: int = 6
    tactile_tokens: int = 3

    perceiver_num_latents: int = 10                                     # resample tokens for image from 196 to 10 
    perceiver_num_layers: int = 2
    perceiver_num_heads: int = 8
    use_perceiver_resampler: bool = True

    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_freeze_backbone: bool = True
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    def num_image_tokens(self) -> int:
        return self.image_token_grid * self.image_token_grid


# CNN version
# class ImageEncoder2D(nn.Module):
#     """
#     Lightweight convolutional encoder that maps images to a fixed grid of tokens.
#
#     The encoder uses a small convolutional stem with progressive downsampling followed
#     by an adaptive pooling stage. This ensures a consistent number of output tokens
#     regardless of the input image size, provided the height and width are at least 8×
#     ``image_token_grid`` (e.g. 112px when the grid is 14).
#     """
#
#     def __init__(self, in_channels: int, embed_dim: int, grid_size: int, dropout: float = 0.1):
#         super().__init__()
#         hidden_dim = embed_dim // 2
#         self.num_tokens = grid_size * grid_size
#
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
#         self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             images: Float tensor with shape ``(batch, channels, height, width)``.
#
#         Returns:
#             Tensor with shape ``(batch, num_tokens, embed_dim)`` where
#             ``num_tokens == grid_size ** 2``.
#         """
#         features = self.stem(images)
#         pooled = self.pool(features)
#         embedded = self.proj(pooled)  # (B, embed_dim, grid, grid)
#         tokens = embedded.flatten(2).transpose(1, 2)  # (B, num_tokens, embed_dim)
#
#         tokens = self.dropout(tokens)
#         tokens = self.norm(tokens)
#         return tokens


class DinoV3ImageEncoder(nn.Module):
    """
    Wrapper around a pretrained DINOv3 vision transformer loaded via Hugging Face ``transformers``.

    The module keeps the backbone in eval mode by default and optionally freezes its weights. Inputs
    are resized to the backbone's expected image size (usually 224) and normalised with ImageNet
    statistics before being forwarded through the transformer. Patch tokens (CLS removed by default)
    are then projected into ``embed_dim`` so they can be consumed by the downstream multimodal model.
    """

    def __init__(
        self,
        embed_dim: int,
        model_name: str,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        drop_cls_token: bool = True,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise ImportError(
                "The 'transformers' package is required for the DINOv3 image encoder. "
                "Install it via `pip install transformers safetensors`."
            ) from exc

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # hidden_size = # token of features in each token vector inside the transformer
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"DINO backbone '{model_name}' does not expose 'hidden_size'.")

        # Use dinov3 as a frozen encoder, no train it 
        self.backbone.eval()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.drop_cls_token = drop_cls_token
        self.normalize_inputs = normalize_inputs
        self.expected_image_size = getattr(self.backbone.config, "image_size", None)
        patch_size = getattr(self.backbone.config, "patch_size", None)
        num_patches = getattr(self.backbone.config, "num_patches", None)
        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))

        if num_patches is None and self.expected_image_size and patch_size:
            if isinstance(patch_size, (tuple, list)):
                patch_h, patch_w = patch_size[0], patch_size[-1]
            else:
                patch_h = patch_w = patch_size
            grid_h = self.expected_image_size // patch_h
            grid_w = self.expected_image_size // patch_w
            num_patches = grid_h * grid_w

        if num_patches is None:
            raise ValueError(
                f"Unable to infer number of image tokens from DINO config: {self.backbone.config}"
            )

        # Total tokens = patches + register tokens + CLS (always produced, optionally dropped).
        total_tokens = int(num_patches) + num_register_tokens + 1
        self.num_patch_tokens = int(num_patches)
        self.num_register_tokens = int(num_register_tokens)
        self.num_tokens = total_tokens - 1 if drop_cls_token else total_tokens

        self.project = (
            nn.Linear(hidden_size, embed_dim, bias=False)
            if hidden_size != embed_dim
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        """
        fixed constatn from ImagNet
        param used in dinov3 github: 
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        CROP_DEFAULT_SIZE = 224
        view(batch size, 3 RGB, H, W)
        """
        # calibration 
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4: #(B, C, H, W)
            raise ValueError(f"Expected 4D image tensor, got shape {tuple(images.shape)}")

        if self.normalize_inputs:
            pixel_mean = self.pixel_mean.to(dtype=images.dtype, device=images.device)
            pixel_std = self.pixel_std.to(dtype=images.dtype, device=images.device)
            images = (images - pixel_mean) / pixel_std

        # if img size is not 224x224, bilinear resize it 
        if self.expected_image_size is not None and (
            images.shape[-2] != self.expected_image_size or images.shape[-1] != self.expected_image_size
        ):
            images = F.interpolate(
                images,
                size=(self.expected_image_size, self.expected_image_size),
                mode="bilinear",
                align_corners=False,
            )

        # feed to Dinov3 transformer 
        outputs = self.backbone(pixel_values=images)
        hidden = outputs.last_hidden_state

        # split output tokens (order: CLS → Register → Patch)
        # source: https://huggingface.co/docs/transformers/main/en/model_doc/dinov3?utm_source=chatgpt.com
        cls_token = hidden[:, :1, :]
        register_tokens = hidden[:, 1 : 1 + self.num_register_tokens, :]
        patch_tokens = hidden[:, 1 + self.num_register_tokens :, :]
        if not self.drop_cls_token:
            register_tokens = torch.cat([cls_token, register_tokens], dim=1)

        patch_tokens = self.project(patch_tokens)
        patch_tokens = self.dropout(patch_tokens)
        patch_tokens = self.norm(patch_tokens)

        if register_tokens.numel() > 0:
            register_tokens = self.project(register_tokens)
            register_tokens = self.dropout(register_tokens)
            register_tokens = self.norm(register_tokens)
        else:
            register_tokens = patch_tokens.new_zeros(
                patch_tokens.size(0), 0, patch_tokens.size(-1), requires_grad=False
            )

        return patch_tokens, register_tokens


class ClipImageEncoder(nn.Module):
    """
    Image encoder that wraps a pretrained CLIP vision transformer.

    The implementation mirrors :class:`DinoV3ImageEncoder` but leverages CLIP checkpoints
    from Hugging Face. Only the vision tower is used; projection heads are discarded.
    """

    def __init__(
        self,
        embed_dim: int,
        model_name: str,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        drop_cls_token: bool = True,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'transformers' package is required for the CLIP image encoder. "
                "Install it via `pip install transformers safetensors`."
            ) from exc

        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        vision_config = self.backbone.config
        hidden_size = getattr(vision_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"CLIP backbone '{model_name}' does not expose 'hidden_size'.")

        self.backbone.eval()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.drop_cls_token = drop_cls_token
        self.normalize_inputs = normalize_inputs
        self.expected_image_size = int(getattr(vision_config, "image_size", 224))
        patch_size = getattr(vision_config, "patch_size", 16)
        grid = self.expected_image_size // patch_size
        self.num_patch_tokens = grid * grid
        # clip does not use register tokens
        self.num_register_tokens = 0

        self.project = (
            nn.Linear(hidden_size, embed_dim, bias=False)
            if hidden_size != embed_dim
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # mean and std source
        # https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L85

        pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        pixel_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, got shape {tuple(images.shape)}")

        if self.normalize_inputs:
            pixel_mean = self.pixel_mean.to(dtype=images.dtype, device=images.device)
            pixel_std = self.pixel_std.to(dtype=images.dtype, device=images.device)
            images = (images - pixel_mean) / pixel_std

        if images.shape[-2] != self.expected_image_size or images.shape[-1] != self.expected_image_size:
            images = F.interpolate(
                images,
                size=(self.expected_image_size, self.expected_image_size),
                mode="bilinear",
                align_corners=False,
            )

        outputs = self.backbone(pixel_values=images)
        hidden = outputs.last_hidden_state
        cls_token = hidden[:, :1, :]
        patch_tokens = hidden[:, 1 : 1 + self.num_patch_tokens, :]

        if not self.drop_cls_token:
            register_tokens = cls_token
        else:
            register_tokens = torch.zeros(
                (images.size(0), 0, patch_tokens.size(-1)),
                dtype=patch_tokens.dtype,
                device=patch_tokens.device,
            )

        patch_tokens = self.project(patch_tokens)
        patch_tokens = self.dropout(patch_tokens)
        patch_tokens = self.norm(patch_tokens)

        if register_tokens.numel() > 0:
            register_tokens = self.project(register_tokens)
            register_tokens = self.dropout(register_tokens)
            register_tokens = self.norm(register_tokens)

        return patch_tokens, register_tokens
        # Clip does not have register_tokens


class PerceiverResamplerLayer(nn.Module):
    """Single cross-attention + MLP block used inside the perceiver resampler."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.latent_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # stabilize latent-toekn distribution before attention, signal calibration 
        query = self.latent_norm(latents)
        # latent queries and input values on the same scale 
        key_value = self.input_norm(inputs)
        # smart fusion, cross-attention
        # no care weights, only care fusion result 
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        latents = latents + self.attn_dropout(attn_out)
        # MLP refinement
        latents = latents + self.mlp_dropout(self.mlp(latents))
        return latents


class PerceiverResampler(nn.Module):
    """
    Learnable resampler inspired by Perceiver/Flamingo that compresses long token sequences.
    Learnable latent tokens performing corss-attention on input tokens to produce a fixed-size output.
    """

    def __init__(
        self,
        dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # lernable query vectors will extract info from input tokens via cross-attention
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.layers = nn.ModuleList(
            PerceiverResamplerLayer(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        latents = self.latents.expand(tokens.size(0), -1, -1)
        for layer in self.layers:
            latents = layer(latents, tokens)
        return latents


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
        >>> tactile = torch.randn(4, 50, 6)
        >>> delta_gripper = model(images, tactile)
        >>> delta_gripper.shape
        torch.Size([4, 1])
    """

    def __init__(self, config: Optional[MultimodalTransformerConfig] = None):
        super().__init__()
        self.config = config or MultimodalTransformerConfig()

        encoder_type = self.config.image_encoder_type.lower()
        if encoder_type == "dino_v3":
            self.image_encoder = DinoV3ImageEncoder(
                embed_dim=self.config.d_model,
                model_name=self.config.dinov3_model_name,
                dropout=self.config.dropout,
                freeze_backbone=self.config.dinov3_freeze_backbone,
                drop_cls_token=self.config.dinov3_drop_cls_token,
                normalize_inputs=self.config.dinov3_normalize_inputs,
            )
        elif encoder_type == "clip":
            self.image_encoder = ClipImageEncoder(
                embed_dim=self.config.d_model,
                model_name=self.config.clip_model_name,
                dropout=self.config.dropout,
                freeze_backbone=self.config.clip_freeze_backbone,
                drop_cls_token=self.config.dinov3_drop_cls_token,
                normalize_inputs=self.config.dinov3_normalize_inputs,
            )
        else:
            raise ValueError(f"Unsupported image_encoder_type '{self.config.image_encoder_type}'.")

        logging.getLogger(__name__).info(
            "Instantiated image encoder %s (type=%s, source=%s)",
            self.image_encoder.__class__.__name__,
            self.config.image_encoder_type,
            getattr(
                self.config,
                "dinov3_model_name" if encoder_type == "dino_v3" else "clip_model_name",
                "n/a",
            ),
        )

        patch_tokens_available = getattr(self.image_encoder, "num_patch_tokens", self.config.num_image_tokens())
        self.num_register_tokens = getattr(self.image_encoder, "num_register_tokens", 0)
        if self.config.use_perceiver_resampler:
            self.image_resampler = PerceiverResampler(
                dim=self.config.d_model,
                num_latents=self.config.perceiver_num_latents,
                num_layers=self.config.perceiver_num_layers,
                num_heads=self.config.perceiver_num_heads,
                dropout=self.config.dropout,
            )
            self.image_patch_token_count = self.config.perceiver_num_latents
        else:
            self.image_resampler = None
            self.image_patch_token_count = patch_tokens_available

        total_image_tokens = self.image_patch_token_count + self.num_register_tokens
        self.tactile_encoder = TactileEncoder1D(
            in_channels=self.config.tactile_channels,
            embed_dim=self.config.d_model,
            num_tokens=self.config.tactile_tokens,
            dropout=self.config.dropout,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.image_positional = nn.Parameter(torch.randn(1, total_image_tokens, self.config.d_model) * 0.02)
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
        patch_tokens, register_tokens = self.image_encoder(images)
        if self.image_resampler is not None:
            patch_tokens = self.image_resampler(patch_tokens)
        if register_tokens is not None and register_tokens.numel() > 0:
            image_tokens = torch.cat([patch_tokens, register_tokens], dim=1)
        else:
            image_tokens = patch_tokens

        if image_tokens.size(1) != self.image_positional.size(1):
            raise ValueError(
                f"Image token count mismatch (expected {self.image_positional.size(1)}, got {image_tokens.size(1)})."
            )
        image_tokens = image_tokens + self.image_positional
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
            delta_gripper: Regression output ``(batch, 1)`` representing the gripper delta.
            tokens (optional): Token sequence after the transformer ``(batch, total_tokens, d_model)``.
        """
        cls_tokens, image_tokens, tactile_tokens = self.forward_features(images, tactile)
        tokens = torch.cat([cls_tokens, image_tokens, tactile_tokens], dim=1)
        tokens = self.modal_dropout(tokens)

        fused = self.transformer(tokens)
        fused = self.transformer_norm(fused)

        cls_output = fused[:, 0, :]
        delta_gripper = self.regression_head(cls_output)

        if return_tokens:
            return delta_gripper, fused
        return delta_gripper


__all__ = [
    "MultimodalTransformerConfig",
    "DinoV3ImageEncoder",
    "ClipImageEncoder",
    "PerceiverResampler",
    "TactileEncoder1D",
    "MultimodalForceTransformer",
]
