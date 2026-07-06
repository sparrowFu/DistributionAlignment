"""
GaussianImageDistribution - Quick Test (MSDA)

A quick sanity check of the MSDA distribution alignment model: model creation,
forward pass, MSDA loss computation, and a backward step. For the full pipeline
test see examples/test_dist_align.py.

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
from losses.dist_align_losses import MSDALoss
import config
from utils.seed import set_seed


def _dummy_outputs(model, device):
    """Run a forward pass on dummy data and return the output dict."""
    B, K, L = 2, 5, 77
    images = torch.randn(B, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 49408, (B, K, L), device=device)
    attention_mask = torch.ones(B, K, L, dtype=torch.long, device=device)
    return model(images, input_ids, attention_mask)


def quick_test():
    """Run quick test of the MSDA model and loss."""
    print("=" * 60)
    print("Quick Test - MSDA Distribution Alignment Model")
    print("=" * 60)

    set_seed(42)
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
            dropout_rate=0.1,
            cov_rank=config.MSDA_COV_RANK,
        ).to(device)
        print(f"[OK] Model created. Trainable params: {model.num_trainable_parameters():,}")
        print(f"  cov_rank r = {model.cov_rank}")
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

    # Test 2: Forward pass
    print("\n" + "-" * 60)
    print("Test 2: Forward Pass")
    print("-" * 60)

    try:
        with torch.no_grad():
            outputs = _dummy_outputs(model, device)
        print(f"[OK] Forward pass successful. Output shapes:")
        for key, value in outputs.items():
            shape = value.shape if value is not None else None
            print(f"  {key}: {shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: MSDA loss computation
    print("\n" + "-" * 60)
    print("Test 3: MSDA Loss Computation")
    print("-" * 60)

    try:
        criterion = MSDALoss(
            tau=config.MSDA_TAU, m_pos=config.MSDA_M_POS, target_var=config.MSDA_TARGET_VAR
        )
        loss, d = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs['text_Us'],
        )
        print(f"✓ MSDA loss computed: total={d['total']:.4f}")
        print(f"  set-NCE={d['set_nce']:.4f} var={d['var']:.4f} "
              f"cover={d['cover']:.4f} cov={d['cov']:.4f} reg={d['reg']:.4f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Backward pass
    print("\n" + "-" * 60)
    print("Test 4: Backward Pass")
    print("-" * 60)

    try:
        import torch.optim as optim
        outputs = _dummy_outputs(model, device)
        loss, d = criterion(
            outputs['img_mu'], outputs['img_logvar'], outputs['img_U'],
            outputs['text_mu'], outputs['text_logvar'],
            outputs['text_mus'], outputs['text_logvars'], outputs['text_Us'],
        )
        optimizer = optim.Adam(model.trainable_parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"✓ Backward pass successful. Loss={d['total']:.4f}")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All quick tests passed!")
    print("=" * 60)
    print("\nThe MSDA distribution alignment model is working correctly.")
    print("Full training:  python main.py --task train_dist_align")
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
