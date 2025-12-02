# Multimodal Force Transformer Architecture

## Mermaid Diagram

```mermaid
graph TB
    subgraph Inputs["Inputs"]
        IL["images_left<br/>(B, 3, 224, 224)"]
        IR["images_right<br/>(B, 3, 224, 224)"]
        TAC["tactile<br/>(B, 500, 6)"]
    end

    subgraph ImageEncoder["Image Encoder (DINOv3/CLIP)"]
        IE_L["DINOv3 Encoder<br/>Left Camera"]
        IE_R["DINOv3 Encoder<br/>Right Camera"]
        IE_L --> PT_L["patch_tokens_left<br/>(B, 196, 256)"]
        IE_L --> RT_L["register_tokens_left<br/>(B, 4, 256)"]
        IE_R --> PT_R["patch_tokens_right<br/>(B, 196, 256)"]
        IE_R --> RT_R["register_tokens_right<br/>(B, 4, 256)"]
    end

    subgraph PerceiverResampler["Perceiver Resampler"]
        PR_L["Resampler Left<br/>(2 layers, 8 heads)"]
        PR_R["Resampler Right<br/>(2 layers, 8 heads)"]
        PT_L --> PR_L
        PT_R --> PR_R
        PR_L --> RPT_L["resampled_patch_left<br/>(B, 10, 256)"]
        PR_R --> RPT_R["resampled_patch_right<br/>(B, 10, 256)"]
    end

    subgraph Masking["Masking (Training Only)"]
        MASK_L["Image Mask Left<br/>ratio=0.5"]
        MASK_R["Image Mask Right<br/>ratio=0.5"]
        MASK_T["Tactile Mask<br/>ratio=0.3"]
        RPT_L --> MASK_L
        RPT_R --> MASK_R
    end

    subgraph ImageTokens["Image Token Assembly"]
        CAT_L["Concat<br/>[patch, register]"]
        CAT_R["Concat<br/>[patch, register]"]
        CAT_ALL["Concat<br/>[left, right]"]
        MASK_L --> CAT_L
        RT_L --> CAT_L
        MASK_R --> CAT_R
        RT_R --> CAT_R
        CAT_L --> CAT_ALL
        CAT_R --> CAT_ALL
        CAT_ALL --> IT["image_tokens<br/>(B, 28, 256)<br/>14 per camera × 2"]
        IT --> POS_IMG["+ Positional Encoding<br/>(B, 28, 256)"]
    end

    subgraph TactileEncoder["Tactile Encoder"]
        TE["Conv1D Stack<br/>(64→128→256→256)"]
        TAC --> TE
        TE --> TT["tactile_tokens<br/>(B, 3, 256)"]
        TT --> MASK_T
        MASK_T --> POS_TAC["+ Positional Encoding<br/>(B, 3, 256)"]
    end

    subgraph TransformerInput["Transformer Input Assembly"]
        CLS["CLS Token<br/>(B, 1, 256)"]
        POS_IMG --> CONCAT["Concat<br/>[CLS, image, tactile]"]
        POS_TAC --> CONCAT
        CLS --> CONCAT
        CONCAT --> TOKENS["All Tokens<br/>(B, 32, 256)<br/>1+28+3=32"]
        TOKENS --> DROP["Dropout<br/>(0.1)"]
    end

    subgraph TransformerEncoder["Transformer Encoder"]
        TE1["Layer 1<br/>(8 heads, FFN=512)"]
        TE2["Layer 2"]
        TE3["Layer 3"]
        TE4["Layer 4"]
        TE5["Layer 5"]
        TE6["Layer 6"]
        TE7["Layer 7"]
        TE8["Layer 8"]
        DROP --> TE1
        TE1 --> TE2
        TE2 --> TE3
        TE3 --> TE4
        TE4 --> TE5
        TE5 --> TE6
        TE6 --> TE7
        TE7 --> TE8
        TE8 --> NORM["LayerNorm"]
    end

    subgraph RegressionHead["Regression Head"]
        CLS_OUT["CLS Output<br/>(B, 256)"]
        NORM --> CLS_OUT
        CLS_OUT --> FC1["Linear(256→128)<br/>+ GELU + Dropout"]
        FC1 --> FC2["Linear(128→10)"]
        FC2 --> OUTPUT["action_chunk<br/>(B, 10)"]
    end

    IL --> IE_L
    IR --> IE_R

    style Inputs fill:#e1f5ff
    style ImageEncoder fill:#fff4e1
    style PerceiverResampler fill:#f0e1ff
    style Masking fill:#ffe1e1
    style ImageTokens fill:#e1ffe1
    style TactileEncoder fill:#ffe1f5
    style TransformerInput fill:#e1e1ff
    style TransformerEncoder fill:#ffffe1
    style RegressionHead fill:#e1ffff
```

## Architecture Details

### Token Composition (DINOv3)
- **CLS Token**: 1 token (learnable)
- **Left Camera**: 10 patch tokens + 4 register tokens = 14 tokens
- **Right Camera**: 10 patch tokens + 4 register tokens = 14 tokens
- **Tactile Tokens**: 3 tokens
- **Total**: 1 + 14 + 14 + 3 = **32 tokens**

### Transformer Configuration
- **Layers**: 8 (updated from 4)
- **d_model**: 256
- **nhead**: 8
- **dim_feedforward**: 512
- **dropout**: 0.1
- **Activation**: GELU
- **Norm**: LayerNorm (pre-norm architecture)

### Masking Strategy
- **Image Mask Ratio**: 0.5 (50% of patch tokens masked)
- **Tactile Mask Ratio**: 0.3 (30% of tactile tokens masked)
- **Masking**: Only during training (`self.training == True`)
- **Mask Tokens**: Learned parameters (separate for image and tactile)

### Action Chunking
- **Output Size**: 10 future steps
- **Shape**: `(batch, 10)`
- **Meaning**: Predicts gripper position deltas for next 10 timesteps

### Key Components

1. **DINOv3 Image Encoder**
   - Frozen backbone (`freeze_backbone=True`)
   - Outputs: 196 patch tokens + 4 register tokens
   - Hidden size: 256 (projected from DINOv3's hidden size)

2. **Perceiver Resampler**
   - Compresses 196 patch tokens → 10 latent tokens
   - 2 cross-attention layers
   - 8 attention heads
   - Learnable latent queries

3. **Tactile Encoder**
   - Conv1D stack: 6 → 64 → 128 → 256 → 256
   - Adaptive pooling to 3 tokens
   - Processes temporal sequences (500 timesteps → 3 tokens)

4. **Transformer Encoder**
   - 8 layers of self-attention
   - Processes all tokens together (CLS + image + tactile)
   - Pre-norm architecture (LayerNorm before attention/FFN)

5. **Regression Head**
   - Takes CLS token output
   - 2-layer MLP: 256 → 128 → 10
   - Outputs action chunk (10 future deltas)
