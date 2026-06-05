"""
GaussianImageDistribution - Quick Test

This script performs a quick sanity check of the distribution alignment model.
Tests basic functionality without full training pipeline.

Usage:
    python examples/quick_test.py
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.dist_align_model import DistributionAlignmentModel
from losses.dist_align_losses import DistributionAlignmentLoss, CombinedDistributionLoss
import config
from utils.seed import set_seed


def quick_test():
    """Run quick test of model and loss functions."""
    print("=" * 60)
    print("Quick Test - Distribution Alignment Model")
    print("=" * 60)

    # Set seed
    set_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Test 1: Model creation
    print("\n" + "-" * 60)
    print("Test 1: Model Creation")
    print("-" * 60)

    try:
        model = DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging="moment_matching",
            dropout_rate=0.1
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = model.num_trainable_parameters()

        print(f"[OK] Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

    # Test 2: Forward pass
    print("\n" + "-" * 60)
    print("Test 2: Forward Pass")
    print("-" * 60)

    try:
        batch_size = 2
        num_captions = 5
        max_seq_len = 77

        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_captions = torch.randint(0, 49408, (batch_size, num_captions, max_seq_len))
        dummy_attention_mask = torch.ones(batch_size, num_captions, max_seq_len)

        print(f"Input pixel_values shape: {dummy_images.shape}")
        print(f"Input input_ids shape: {dummy_captions.shape}")
        print(f"Input attention_mask shape: {dummy_attention_mask.shape}")

        with torch.no_grad():
            outputs = model(dummy_images, dummy_captions, dummy_attention_mask)

        print(f"[OK] Forward pass successful")
        print(f"Output shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Loss computation
    print("\n" + "-" * 60)
    print("Test 3: Loss Computation")
    print("-" * 60)

    try:
        criterion = DistributionAlignmentLoss(
            lambda_contrastive=1.0,
            lambda_kl=0.5,
            temperature=0.07,
            kl_type="symmetric"
        )

        loss, loss_dict = criterion(
            outputs['img_features'],
            outputs['text_features'],
            outputs['img_mu'],
            outputs['img_logvar'],
            outputs['text_mu'],
            outputs['text_logvar']
        )

        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss_dict['total']:.4f}")
        print(f"  Contrastive loss: {loss_dict['contrastive']:.4f}")
        print(f"  KL loss: {loss_dict['kl']:.4f}")

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Combined loss
    print("\n" + "-" * 60)
    print("Test 4: Combined Loss (with variance regularization)")
    print("-" * 60)

    try:
        combined_criterion = CombinedDistributionLoss(
            lambda_contrastive=1.0,
            lambda_kl=0.5,
            lambda_var=0.1,
            temperature=0.07,
            kl_type="symmetric"
        )

        combined_loss, combined_dict = combined_criterion(
            outputs['img_features'],
            outputs['text_features'],
            outputs['img_mu'],
            outputs['img_logvar'],
            outputs['text_mu'],
            outputs['text_logvar']
        )

        print(f"✓ Combined loss computation successful")
        print(f"  Total loss: {combined_dict['total']:.4f}")
        print(f"  Contrastive loss: {combined_dict['contrastive']:.4f}")
        print(f"  KL loss: {combined_dict['kl']:.4f}")
        print(f"  Variance loss: {combined_dict.get('variance', 0):.4f}")

    except Exception as e:
        print(f"✗ Combined loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Backward pass
    print("\n" + "-" * 60)
    print("Test 5: Backward Pass")
    print("-" * 60)

    try:
        import torch.optim as optim

        # Recompute forward pass to get gradients
        batch_size = 2
        num_captions = 5
        max_seq_len = 77

        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_captions = torch.randint(0, 49408, (batch_size, num_captions, max_seq_len))
        dummy_attention_mask = torch.ones(batch_size, num_captions, max_seq_len)

        # Forward pass (without no_grad)
        outputs = model(dummy_images, dummy_captions, dummy_attention_mask)

        # Compute loss
        loss, loss_dict = criterion(
            outputs['img_features'],
            outputs['text_features'],
            outputs['img_mu'],
            outputs['img_logvar'],
            outputs['text_mu'],
            outputs['text_logvar']
        )

        # Backward pass
        optimizer = optim.Adam(model.trainable_parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✓ Backward pass successful")
        print(f"  Loss value: {loss_dict['total']:.4f}")

        # Check gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        if has_gradients:
            print(f"  Gradients computed correctly")
        else:
            print(f"  Warning: No gradients found")

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 60)
    print("✓ All quick tests passed!")
    print("=" * 60)
    print("\nThe distribution alignment model is working correctly.")
    print("You can proceed with full training using:")
    print("  python main.py --task train_dist_align")

    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Quick test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
