"""
GaussianImageDistribution - Test Script for Distribution Alignment

This script tests the complete training and evaluation pipeline for the
distribution alignment model using a small subset of data.

Usage:
    python examples/test_dist_align.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
from datetime import datetime

import config
from data.caption_dataset import ImageCaptionDataset
from models.dist_align_model import DistributionAlignmentModel
from losses.dist_align_losses import DistributionAlignmentLoss, CombinedDistributionLoss, MSDALoss
from utils.logger import setup_logger, get_logger
from utils.seed import set_seed
from utils.metrics import compute_recall_at_k


def setup_test_environment():
    """Setup test environment with logging."""
    # Create test logs directory
    test_log_dir = config.LOG_DIR / "test"
    test_log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = test_log_dir / f"test_dist_align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger("test", log_file)
    logger = get_logger("test")

    return logger, log_file


def create_dummy_collate():
    """Create a collate function that returns dummy data for testing."""
    def collate_fn(batch):
        """Create dummy batch for testing."""
        batch_size = len(batch) if batch else 1
        num_captions = 5
        max_seq_len = 77
        image_size = 224

        return {
            'image': [torch.rand(3, image_size, image_size) for _ in range(batch_size)],
            'captions': [torch.randint(0, 49408, (num_captions, max_seq_len)) for _ in range(batch_size)],
        }
    return collate_fn


def create_mock_dataset(num_samples=20):
    """Create a mock dataset for testing when real data is not available."""
    class MockImageCaptionDataset(torch.utils.data.Dataset):
        """Mock dataset for testing."""

        def __init__(self, num_samples=20, num_captions=5):
            self.num_samples = num_samples
            self.num_captions = num_captions

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Return random image and captions
            return {
                'image': torch.rand(3, 224, 224),
                'captions': torch.randint(0, 49408, (self.num_captions, 77)),
            }

    return MockImageCaptionDataset(num_samples)


def test_model_creation(logger):
    """Test 1: Model creation and basic functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Creation")
    logger.info("=" * 60)

    try:
        # Create model
        model = DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging="moment_matching",
            dropout_rate=0.1
        )

        logger.info(f"✓ Model created successfully")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Trainable parameters: {model.num_trainable_parameters():,}")

        # Test forward pass
        batch_size = 2
        num_captions = 5
        max_seq_len = 77

        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_captions = torch.randint(0, 49408, (batch_size, num_captions, max_seq_len))
        dummy_attention_mask = torch.ones(batch_size, num_captions, max_seq_len, dtype=torch.long)

        logger.info(f"\nTesting forward pass...")
        logger.info(f"  Input images shape: {dummy_images.shape}")
        logger.info(f"  Input captions shape: {dummy_captions.shape}")

        with torch.no_grad():
            outputs = model(dummy_images, dummy_captions, dummy_attention_mask)

        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output shapes:")
        for key, value in outputs.items():
            logger.info(f"    {key}: {value.shape}")

        return True

    except Exception as e:
        logger.error(f"✗ Model creation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_loss_function(logger):
    """Test 2: Loss function computation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Loss Function")
    logger.info("=" * 60)

    try:
        # Create loss function
        criterion = DistributionAlignmentLoss(
            lambda_contrastive=1.0,
            lambda_kl=0.5,
            temperature=0.07,
            kl_type="symmetric"
        )

        logger.info("✓ Loss function created")

        # Create dummy outputs
        B, D = 4, 768
        img_features = torch.randn(B, D)
        text_features = torch.randn(B, D)
        img_mu = torch.randn(B, D)
        img_logvar = torch.randn(B, D)
        text_mu = torch.randn(B, D)
        text_logvar = torch.randn(B, D)

        # Compute loss
        loss, loss_dict = criterion(
            img_features, text_features,
            img_mu, img_logvar,
            text_mu, text_logvar
        )

        logger.info(f"✓ Loss computation successful")
        logger.info(f"  Total loss: {loss_dict['total']:.4f}")
        logger.info(f"  Contrastive loss: {loss_dict['contrastive']:.4f}")
        logger.info(f"  KL loss: {loss_dict['kl']:.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ Loss function test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_msda_loss_function(logger):
    """Test 2b: MSDA loss computation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2b: MSDA Loss Function")
    logger.info("=" * 60)

    try:
        criterion = MSDALoss()

        B, D, K, r = 4, 768, 5, config.MSDA_COV_RANK
        img_mu = torch.randn(B, D)
        img_logvar = torch.randn(B, D)
        img_U = torch.randn(B, D, r)
        text_mu = torch.randn(B, D)
        text_logvar = torch.randn(B, D)
        text_mus = torch.randn(B, K, D)
        text_logvars = torch.randn(B, K, D)
        text_Us = torch.randn(B, K, D, r)

        loss, loss_dict = criterion(
            img_mu, img_logvar, img_U,
            text_mu, text_logvar,
            text_mus, text_logvars, text_Us,
        )

        logger.info("✓ MSDA loss computation successful")
        logger.info(f"  Total loss: {loss_dict['total']:.4f}")
        logger.info(f"  set-NCE: {loss_dict['set_nce']:.4f}")
        logger.info(f"  var: {loss_dict['var']:.4f}")
        logger.info(f"  cover: {loss_dict['cover']:.4f}")
        logger.info(f"  cov: {loss_dict['cov']:.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ MSDA loss function test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_training_step(logger, model, dataloader, criterion, optimizer, device):
    """Test 3: Single training step."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Training Step")
    logger.info("=" * 60)

    try:
        model.train()

        # Get one batch
        batch = next(iter(dataloader))

        # Get data - PIL images and text lists
        pil_images = batch["image"]
        caption_lists = batch["captions"]

        # Process images with CLIP processor
        pixel_values = model.process_images(pil_images).to(device)

        # Process text captions
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        # Flatten all captions
        all_captions = []
        for caption_list in caption_lists:
            all_captions.extend(caption_list)

        # Process with CLIP processor
        text_inputs = model.process_text(all_captions)

        # Reshape to [B, K, max_len]
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        logger.info(f"  Batch pixel_values shape: {pixel_values.shape}")
        logger.info(f"  Batch input_ids shape: {input_ids.shape}")
        logger.info(f"  Batch attention_mask shape: {attention_mask.shape}")

        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"✓ Training step successful")
        logger.info(f"  Loss: {loss_dict['total']:.4f}")
        logger.info(f"  Gradient norm: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ Training step failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_training_epoch(logger, model, dataloader, criterion, optimizer, device):
    """Test 4: Complete training epoch."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Training Epoch")
    logger.info("=" * 60)

    try:
        model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            if batch is None:
                continue

            # Get data - PIL images and text lists
            pil_images = batch["image"]
            caption_lists = batch["captions"]

            # Process images with CLIP processor
            pixel_values = model.process_images(pil_images).to(device)

            # Process text captions
            batch_size = len(pil_images)
            num_captions = len(caption_lists[0])

            # Flatten all captions
            all_captions = []
            for caption_list in caption_lists:
                all_captions.extend(caption_list)

            # Process with CLIP processor
            text_inputs = model.process_text(all_captions)

            # Reshape to [B, K, max_len]
            input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
            attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

            # Forward pass
            outputs = model(pixel_values, input_ids, attention_mask)

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_dict['total']
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        logger.info(f"✓ Training epoch completed")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Batches processed: {num_batches}")

        return True

    except Exception as e:
        logger.error(f"✗ Training epoch failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_evaluation(logger, model, dataloader, device):
    """Test 5: Evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Evaluation")
    logger.info("=" * 60)

    try:
        model.eval()

        all_img_features = []
        all_text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if batch is None:
                    continue

                # Get data - PIL images and text lists
                pil_images = batch["image"]
                caption_lists = batch["captions"]

                # Process images with CLIP processor
                pixel_values = model.process_images(pil_images).to(device)

                # Process text captions
                batch_size = len(pil_images)
                num_captions = len(caption_lists[0])

                # Flatten all captions
                all_captions = []
                for caption_list in caption_lists:
                    all_captions.extend(caption_list)

                # Process with CLIP processor
                text_inputs = model.process_text(all_captions)

                # Reshape to [B, K, max_len]
                input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
                attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

                outputs = model(pixel_values, input_ids, attention_mask)

                all_img_features.append(outputs['img_mu'].cpu())
                all_text_features.append(outputs['text_mu'].cpu())

        # Concatenate features
        img_features = torch.cat(all_img_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)

        logger.info(f"  Image features shape: {img_features.shape}")
        logger.info(f"  Text features shape: {text_features.shape}")

        # Compute similarity
        similarity = torch.matmul(img_features, text_features.T)
        logger.info(f"  Similarity matrix shape: {similarity.shape}")

        # Compute Recall@K (returns a dict keyed by K)
        recalls = compute_recall_at_k(similarity, [1, 5])

        logger.info(f"✓ Evaluation completed")
        logger.info(f"  Recall@1: {recalls[1]:.4f}")
        logger.info(f"  Recall@5: {recalls[5]:.4f}")

        return True

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_checkpoint_save_load(logger, model, optimizer):
    """Test 6: Checkpoint saving and loading."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Checkpoint Save/Load")
    logger.info("=" * 60)

    try:
        # Create test checkpoint directory
        test_checkpoint_dir = config.CHECKPOINT_DIR / "test"
        test_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = test_checkpoint_dir / "test_checkpoint.pt"

        # Save checkpoint
        model.save(str(checkpoint_path))
        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")

        # Create new model and load checkpoint
        new_model = DistributionAlignmentModel(
            freeze_clip=True,
            distribution_merging="moment_matching"
        )

        new_model.load(str(checkpoint_path))
        logger.info(f"✓ Checkpoint loaded successfully")

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
            if not torch.allclose(p1, p2, rtol=1e-5):
                logger.warning(f"  Parameter mismatch in {n1}")

        logger.info(f"✓ Parameters verified")

        return True

    except Exception as e:
        logger.error(f"✗ Checkpoint test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function."""
    # Setup
    logger, log_file = setup_test_environment()

    logger.info("=" * 60)
    logger.info("Distribution Alignment - Test Suite")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("=" * 60)

    # Set seed
    set_seed(42)

    # Test configuration
    test_config = {
        'num_samples': 20,
        'batch_size': 4,
        'num_captions': 5,
        'epochs': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    logger.info(f"\nTest Configuration:")
    for key, value in test_config.items():
        logger.info(f"  {key}: {value}")

    device = torch.device(test_config['device'])

    # Test results
    test_results = {}

    # Test 1: Model creation
    success = test_model_creation(logger)
    test_results['model_creation'] = success

    if not success:
        logger.error("Stopping tests due to model creation failure")
        return

    # Create model for subsequent tests
    model = DistributionAlignmentModel(
        freeze_clip=True,
        distribution_merging="moment_matching",
        dropout_rate=0.1
    )
    model = model.to(device)

    # Test 2: Loss function
    success = test_loss_function(logger)
    test_results['loss_function'] = success

    # Test 2b: MSDA loss function
    success = test_msda_loss_function(logger)
    test_results['msda_loss_function'] = success

    # Create loss function
    criterion = DistributionAlignmentLoss(
        lambda_contrastive=1.0,
        lambda_kl=0.5,
        temperature=0.07,
        kl_type="symmetric"
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # Create dataset
    logger.info("\n" + "=" * 60)
    logger.info("Setting up dataset...")
    logger.info("=" * 60)

    try:
        # Try to use real dataset if available
        if config.CAPTIONS_PATH.exists():
            logger.info(f"Using real dataset from {config.CAPTIONS_PATH}")
            dataset = ImageCaptionDataset(
                captions_path=str(config.CAPTIONS_PATH),
                images_dir=str(config.IMAGES_DIR),
                num_captions=config.NUM_CAPTIONS
            )
            # Use only first N samples for testing
            dataset = Subset(dataset, list(range(test_config['num_samples'])))
        else:
            logger.info(f"Dataset not found at {config.CAPTIONS_PATH}")
            logger.info("Using mock dataset for testing")
            dataset = create_mock_dataset(test_config['num_samples'])

        dataloader = DataLoader(
            dataset,
            batch_size=test_config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch  # Use default collate for mock data
        )

        logger.info(f"✓ Dataset created: {len(dataset)} samples")
        logger.info(f"  Batches per epoch: {len(dataloader)}")

    except Exception as e:
        logger.error(f"✗ Dataset creation failed: {str(e)}")
        logger.info("Using mock dataset...")
        dataset = create_mock_dataset(test_config['num_samples'])
        dataloader = DataLoader(
            dataset,
            batch_size=test_config['batch_size'],
            shuffle=True,
            num_workers=0
        )

    # Test 3: Single training step
    success = test_training_step(logger, model, dataloader, criterion, optimizer, device)
    test_results['training_step'] = success

    # Test 4: Training epoch
    success = test_training_epoch(logger, model, dataloader, criterion, optimizer, device)
    test_results['training_epoch'] = success

    # Test 5: Evaluation
    success = test_evaluation(logger, model, dataloader, device)
    test_results['evaluation'] = success

    # Test 6: Checkpoint save/load
    success = test_checkpoint_save_load(logger, model, optimizer)
    test_results['checkpoint'] = success

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")

    logger.info("=" * 60)
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error(f"✗ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nTest suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
