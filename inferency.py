# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pillow",
#     "safetensors",
#     "torch",
#     "torch",
#     "Torchvision",
#     "transformers>=5.9.0",
# ]
# ///

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Choose which fine-tuned model to use, m1 or m2
MODEL = "m1"

INPUT_ROOT = f"./india_tiles/14"
OUTPUT_ROOT = f"./india_tiles_inference_{MODEL}/14"

MODEL_DIRS = {
    "m1": "./results/segformer-m1-results-2026-may-30/final_best_model",
    "m2": "./results/segformer-m2-results-2026-may-29/final_best_model",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SegFormer inference recursively on india_tiles/14."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help=f"Folder to scan recursively for images (default: {INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=(
            "Folder where predictions are written, preserving subfolder structure "
            f"(default: {OUTPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the positive class when saving the binary mask.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device override, e.g. "cpu", "cuda", or "mps".',
    )
    parser.add_argument(
        "--save-probability",
        action="store_true",
        help="Also save an 8-bit probability map for the positive class.",
    )
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def predict_mask(
    image_path: Path,
    processor: SegformerImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = (
            torch.softmax(upsampled_logits, dim=1)[0, 1].detach().cpu().numpy()
        )

    return np.array(image), probabilities


def save_outputs(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    rgb_image: np.ndarray,
    probability_map: np.ndarray,
    threshold: float,
    save_probability: bool,
) -> None:
    relative_path = image_path.relative_to(input_root)
    destination_dir = output_root / relative_path.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    binary_mask = (probability_map >= threshold).astype(np.uint8) * 255

    mask_path = destination_dir / f"{stem}_mask.png"
    Image.fromarray(binary_mask, mode="L").save(mask_path)

    overlay = rgb_image.copy()
    positive = probability_map >= threshold
    overlay[positive] = (
        0.35 * overlay[positive] + 0.65 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)
    overlay_path = destination_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    if save_probability:
        probability_u8 = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
        probability_path = destination_dir / f"{stem}_prob.png"
        Image.fromarray(probability_u8, mode="L").save(probability_path)


args = parse_args()

input_root = args.input_root.resolve()
output_root = args.output_root.resolve()

device = choose_device(args.device)

model_dir = MODEL_DIRS[MODEL]

processor = SegformerImageProcessor.from_pretrained(model_dir)

model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.to(device)
model.eval()

image_paths = list(iter_image_files(input_root))
if not image_paths:
    raise FileNotFoundError(f"No images found under: {input_root}")

print(f"Using model choice: {MODEL}")
print(f"Model directory: {model_dir}")
print(f"Device: {device}")
print(f"Found {len(image_paths)} image(s) under {input_root}")
print(f"Writing outputs to {output_root}")

inference_start = time.perf_counter()

for index, image_path in enumerate(image_paths, start=1):
    print(f"[{index}/{len(image_paths)}] {image_path}")
    rgb_image, probability_map = predict_mask(image_path, processor, model, device)
    save_outputs(
        image_path=image_path,
        input_root=input_root,
        output_root=output_root,
        rgb_image=rgb_image,
        probability_map=probability_map,
        threshold=args.threshold,
        save_probability=args.save_probability,
    )

inference_elapsed = time.perf_counter() - inference_start
print(f"Inference loop completed in {inference_elapsed:.2f} seconds.")
print("Done.")
