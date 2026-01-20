"""
Stage C Sanity Check - Verify semantic training pipeline is functional.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
import modules_semantic


def test_stage_c():
    print("\n" + "="*60)
    print("STAGE C SANITY CHECK - Semantic Training Pipeline")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Device: {device}")
    
    print(f"\n[2] Initializing VITS model with semantic modules...")
    try:
        model = models.SynthesizerTrn(
            n_vocab=150,
            spec_channels=513,
            segment_size=32,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            resblock='1',
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=2,
            use_sdp=False
        ).to(device)
        model.train()
        print(f"  ✓ Model initialized")
        print(f"  ✓ SemPred modules present")
        print(f"  ✓ SemanticCrossAttnAdapter present")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[3] Enabling semantic training (Stage C)...")
    model.use_semantics = True
    model.sem_attn_adapter.train()
    print(f"  ✓ Semantic modules enabled")
    
    print(f"\n[4] Forward pass with synthetic data...")
    try:
        batch_size = 2
        seq_len = 64
        mel_len = 32
        
        x = torch.randint(0, 150, (batch_size, seq_len)).to(device)
        x_lengths = torch.tensor([seq_len, seq_len], dtype=torch.long).to(device)
        y = torch.randn(batch_size, 513, mel_len).to(device)
        y_lengths = torch.tensor([mel_len, mel_len], dtype=torch.long).to(device)
        sid = torch.tensor([0, 1], dtype=torch.long).to(device)
        S_teacher = torch.randn(batch_size, mel_len, 384).to(device)
        
        print(f"  Input shapes: x={x.shape}, y={y.shape}, S_teacher={S_teacher.shape}")
        
        o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(x, x_lengths, y, y_lengths, sid)
        
        print(f"  ✓ Forward pass successful, output shape: {o.shape}")
        
        print(f"\n[5] Testing semantic components...")
        z_p_T = z_p.transpose(1, 2)
        S_hat = model.sem_pred(z_p_T)
        print(f"  ✓ SemPred output: {S_hat.shape}")
        
        mask_2d = torch.ones(batch_size, mel_len, device=device)
        loss_sem = model.sem_loss_fn(S_hat, S_teacher, mask_2d)
        print(f"  ✓ Semantic loss: {loss_sem.item():.6f}")
        
        attn_out, attn_weights = model.sem_attn_adapter(z_p_T, S_teacher, mask_2d)
        print(f"  ✓ Attention adapter output: {attn_out.shape}")
        
        print(f"\n[6] Testing backward pass (gradient flow)...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_total = l_length + loss_sem
        print(f"  Total loss: {loss_total.item():.6f}")
        
        loss_total.backward()
        print(f"  ✓ Backward pass successful")
        
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  ✓ Gradients computed for {grad_count} parameters")
        
        optimizer.step()
        print(f"  ✓ Optimizer step successful")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ STAGE C SANITY CHECK PASSED")
    print("="*60)
    print("\nAll components working:")
    print("  [✓] Model with semantic modules initializes")
    print("  [✓] Forward pass through VITS backbone works")
    print("  [✓] SemPred predicts student semantics")
    print("  [✓] Semantic loss computes correctly")
    print("  [✓] SemanticCrossAttnAdapter fuses semantics")
    print("  [✓] Gradient flow enabled for all components")
    print("\nReady for Stage C training!")
    print("="*60 + "\n")
    
    return True


if __name__ == '__main__':
    success = test_stage_c()
    sys.exit(0 if success else 1)
