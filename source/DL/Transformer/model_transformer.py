import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection 
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len] or [batch_size, 1, seq_len]
            
        Returns:
            [batch_size, seq_len, embed_dim]
        """
        B, N, C = x.shape
        

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

# Transformer Layer

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
      
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            # Pre-normalization (more stable training)
            x = x + self.dropout1(self.attn(self.norm1(x), mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            # Post-normalization (original transformer)
            x = self.norm1(x + self.dropout1(self.attn(x, mask)))
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        
        return x

class MeanPooling(nn.Module):
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: Optional [batch_size, seq_len] mask
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x = x * mask
            return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return x.mean(dim=1)


class MaxPooling(nn.Module):
    """Max pooling over sequence dimension."""
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, float('-inf'))
        return x.max(dim=1)[0]


class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable query."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        nn.init.trunc_normal_(self.query, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)



# Model

class ParticleCloudTransformer(nn.Module):
   
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pooling: str = "mean",
        pre_norm: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling_type = pooling.lower()
       
        self.input_embed = nn.Linear(num_features, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                pre_norm=pre_norm,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm)
        self.final_norm = nn.LayerNorm(embed_dim) if pre_norm else nn.Identity()
        
        # Pooling layer
        if self.pooling_type == "mean":
            self.pool = MeanPooling()
        elif self.pooling_type == "max":
            self.pool = MaxPooling()
        elif self.pooling_type == "attention":
            self.pool = AttentionPooling(embed_dim)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}. Choose from: mean, max, attention")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_particles, num_features]
            mask: Optional [batch_size, num_particles] padding mask
            
        Returns:
            logits: [batch_size, num_classes]
        """

        x = self.input_embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        
        x = self.pool(x, mask)
        logits = self.classifier(x)
        
        return logits



def build_transformer_model(
    num_features: int,
    embed_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 4,
    ffn_dim: int = 256,
    num_classes: int = 2,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    pooling: str = "mean",
    pre_norm: bool = True,
    **kwargs
) -> ParticleCloudTransformer:
    """
    Build a Particle Cloud Transformer model.
    
    Args:
        num_features: Number of input features per particle
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ffn_dim: Feed-forward network dimension
        num_classes: Number of output classes
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        pooling: Pooling type (mean, max, attention)
        pre_norm: Use pre-normalization (recommended)
    """
    model = ParticleCloudTransformer(
        num_features=num_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        num_classes=num_classes,
        dropout=dropout,
        attention_dropout=attention_dropout,
        pooling=pooling,
        pre_norm=pre_norm,
    )
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Particle Cloud Transformer")
    print(f"  Input features: {num_features}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")
    print(f"  FFN dim: {ffn_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Pooling: {pooling}")
    print(f"  Pre-norm: {pre_norm}")
    print(f"  Output classes: {num_classes}")
    print(f"  Total parameters: {n_params:,}")
    
    return model
