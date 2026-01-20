"""
Semantic learning modules for VITS with Whisper teacher and student semantics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SemPred(nn.Module):
    """Semantic Predictor - predicts student semantics from mel-aligned text features."""
    def __init__(self, hidden_channels, whisper_dim, kernel_size=5, p_dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.whisper_dim = whisper_dim
        
        # Project z_p_base to intermediate
        self.proj_in = nn.Linear(hidden_channels, hidden_channels)
        
        # Convolutional layers
        padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        
        # Output projection to Whisper dimension
        self.proj_out = nn.Linear(hidden_channels, whisper_dim)
    
    def forward(self, z_p_base):
        """
        Args:
            z_p_base: [B, T, hidden_channels] - text-aligned features from flow
        Returns:
            S_hat: [B, T, whisper_dim] - predicted semantics
        """
        x = self.proj_in(z_p_base)  # [B, T, hidden_channels]
        x = x.transpose(1, 2)  # [B, hidden_channels, T]
        x = self.convs(x)  # [B, hidden_channels, T]
        x = x.transpose(1, 2)  # [B, T, hidden_channels]
        S_hat = self.proj_out(x)  # [B, T, whisper_dim]
        return S_hat


class SemanticCrossAttnAdapter(nn.Module):
    """
    Cross-attention adapter that fuses teacher/student semantics before prior stats.
    
    Computes attention between z_p_base (query) and S_used (key/value),
    produces residual contribution.
    """
    def __init__(self, hidden_channels, whisper_dim, num_heads=4, attn_drop=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.whisper_dim = whisper_dim
        self.num_heads = num_heads
        
        # Project semantics to hidden_channels for attention
        self.proj_sem = nn.Linear(whisper_dim, hidden_channels)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_channels, 
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        # Small residual gate (starts small, can grow during training)
        self.gate_init = 0.1
        self.gate = nn.Parameter(torch.tensor(self.gate_init))
        
        # Output projection
        self.proj_out = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, z_p_base, S_used, length_mask=None):
        """
        Args:
            z_p_base: [B, T, hidden_channels] - query (text features)
            S_used: [B, T, whisper_dim] - key/value (semantics)
            length_mask: [B, T] - 1 for valid, 0 for padding (optional)
        Returns:
            attn_out: [B, T, hidden_channels] - attention output
            attn_weights: [B, num_heads, T, T] - attention weights
        """
        # Project semantics
        K_V = self.proj_sem(S_used)  # [B, T, hidden_channels]
        
        # Cross-attention: Q=z_p_base, K=V=K_V
        attn_out, attn_weights = self.attention(
            z_p_base, K_V, K_V,
            key_padding_mask=None,  # Could use length_mask if needed
            need_weights=True
        )  # attn_out: [B, T, hidden_channels], attn_weights: [B, num_heads, T, T]
        
        # Project output
        attn_out = self.proj_out(attn_out)  # [B, T, hidden_channels]
        
        # Residual with small gate (allows training to modulate contribution)
        attn_out = attn_out * self.gate
        
        return attn_out, attn_weights


class SemanticLoss(nn.Module):
    """L1 loss between predicted and teacher semantics."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, S_hat, S_mix, mask=None):
        """
        Args:
            S_hat: [B, T, whisper_dim] - student predictions
            S_mix: [B, T, whisper_dim] - teacher (Whisper) features
            mask: [B, T] - 1 for valid, 0 for padding (optional)
        Returns:
            loss: scalar
        """
        loss = F.l1_loss(S_hat, S_mix, reduction='none')  # [B, T, whisper_dim]
        loss = loss.mean(dim=-1)  # [B, T]
        
        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / (mask.float().sum() + 1e-8)
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss
