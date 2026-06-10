"""
GaussianImageDistribution - ProLIP (B3) Evaluation Script

Evaluates ProLIP using image-text retrieval R@K.

Usage:
    python scripts/evaluate_prolip.py
    python main.py --task eval_prolip
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.caption_dataset import ImageCaptionDataset, filter_none_collate
from models.prolip_model import ProLIPModel
from utils.logger import get_logger, log_exception
from utils.seed import set_seed


logger = get_logger("eval_prolip", config.EVAL_PROLIP_LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ProLIP (B3)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--captions-path", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, dataloader, device, num_samples=None):
    model.eval()
    all_img, all_text = [], []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]
        batch_size = len(pil_images)
        num_captions = len(caption_lists[0])

        pixel_values = model.process_images(pil_images).to(device)
        all_captions = []
        for cl in caption_lists:
            all_captions.extend(cl)
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(batch_size, num_captions, -1).to(device)
        attention_mask = text_inputs["attention_mask"].view(batch_size, num_captions, -1).to(device)

        outputs = model(pixel_values, input_ids, attention_mask)
        all_img.append(outputs['img_mu'].cpu())
        all_text.append(outputs['text_mu'].cpu())

        sample_count += batch_size
        if num_samples and sample_count >= num_samples:
            break

    img = torch.cat(all_img, dim=0)[:num_samples] if num_samples else torch.cat(all_img, dim=0)
    text = torch.cat(all_text, dim=0)[:num_samples] if num_samples else torch.cat(all_text, dim=0)
    return img, text


def compute_recall_chunked(img_features, text_features, k_values, chunk_size=1000):
    n = img_features.shape[0]
    img_features = F.normalize(img_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    hits = {k: 0 for k in k_values}

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks"):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(img_features[start:end], text_features.T)
        ranked_indices = torch.argsort(sim_chunk, dim=1, descending=True)
        for k in k_values:
            top_k = ranked_indices[:, :k]
            gt = torch.arange(start, end).unsqueeze(1)
            hits[k] += (top_k == gt).any(dim=1).sum().item()

    return {f'recall@{k}': hits[k] / n for k in k_values}


def main():
    args = parse_args()
    set_seed(config.SEED)

    checkpoint_path = args.checkpoint or str(config.PROLIP_BEST_CKPT)
    logger.info(f"Loading ProLIP from {checkpoint_path}")

    model = ProLIPModel(freeze_clip=True, dropout_rate=config.DIST_ALIGN_DROPOUT_RATE)
    model.load(checkpoint_path)
    model = model.to(args.device)

    captions_path = args.captions_path or config.CAPTIONS_PATH
    images_dir = args.images_dir or config.IMAGES_DIR
    dataset = ImageCaptionDataset(
        captions_path=captions_path, images_dir=images_dir,
        num_captions=config.NUM_CAPTIONS,
    )

    num_samples = args.num_samples
    if num_samples and num_samples < len(dataset):
        generator = torch.Generator().manual_seed(config.SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, collate_fn=filter_none_collate)

    img_features, text_features = extract_features(model, dataloader, args.device, args.num_samples)
    metrics = compute_recall_chunked(img_features, text_features, args.recall_at_k)

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    output_path = args.output_path or str(config.PROLIP_EVAL_RESULTS_PATH)
    with open(output_path, 'w') as f:
        json.dump({'checkpoint': str(checkpoint_path), 'num_samples': num_samples, 'metrics': metrics}, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "ProLIP evaluation failed")
        raise
