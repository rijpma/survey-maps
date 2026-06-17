#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.12.0",
#   "datasets>=4.4.2",
#   "evaluate>=0.4.6",
#   "numpy>=2.0.0",
#   "pillow>=12.0.0",
#   "scikit-learn>=1.7.2",
#   "torch>=2.9.1",
#   "torchvision>=0.24.1",
#   "transformers>=4.57.3",
# ]
# ///

import argparse
import json
import random
import string
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from sklearn.model_selection import train_test_split
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--train-on-cpu", action="store_true")  # = default false
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument(
    "--skip-augmentations", dest="skip_augmentations", action="store_true"
)
parser.add_argument("--model", type=str, default="nvidia/mit-b1")
parser.add_argument("--output-dir", type=str, default="segformer-dry-run")
parser.add_argument("--num-epochs", type=int, default=20)
parser.add_argument("--bw-probability", type=float, default=0.08)
parser.add_argument("--color-probability", type=float, default=0.15)
parser.add_argument("--text-probability", type=float, default=0.08)
parser.add_argument("--grid-line-probability", type=float, default=0.08)
parser.add_argument("--blur-probability", type=float, default=0.06)
parser.add_argument("--noise-probability", type=float, default=0.06)
parser.add_argument("--resume-from-checkpoint", type=str)
parser.add_argument(
    "--loss",
    type=str,
    default="ce",
    choices=("ce", "weighted-ce", "dice", "ce-dice", "weighted-ce-dice"),
)
parser.add_argument("--class-weight-object", type=float, default=3.0)
parser.add_argument("--dice-smooth", type=float, default=1.0)
parser.add_argument("--dice-weight", type=float, default=1.0)
args = parser.parse_args()

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("MPS detected, but training keps on CPU to avoid MPS SegFormer issues.")


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = args.model

HARD_DIR = REPO_ROOT / "img" / "hard"
DATA_ROOT = REPO_ROOT / "labelled" / "batch1"
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "masks"

OUTPUT_DIR = REPO_ROOT / "results" / args.output_dir
GENERATED_MASKS_DIR = OUTPUT_DIR / "generated_masks"
PREVIEW_DIR = OUTPUT_DIR / "previews"
PREDICTION_DIR = OUTPUT_DIR / "prediction_samples"
LOG_PATH = OUTPUT_DIR / "training.log"
EVAL_LOG_PATH = OUTPUT_DIR / "eval_results.jsonl"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_best_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_IMAGE_NAMES = [
    "7401_lat_17.11979_lon_76.02539.png",
    "7028_lat_24.76678_lon_95.29541.png",
]

POSITIVE_IMAGE_NAMES = [
    "6908_lat_27.13737_lon_80.41992.png",
    "6968_lat_25.95804_lon_79.40918.png",
    "6815_lat_28.94086_lon_78.99170.png",
    "6962_lat_26.07652_lon_81.18896.png",
    "6962_lat_26.07652_lon_81.18896.png",
]


IMAGE_SIZE = 512  #  switch to 384 for b2?
TEST_SIZE = 0.2
SEED = 42

NUM_EPOCHS = args.num_epochs
TRAIN_BATCH_SIZE = 8  # switch to 4 for b2?
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

SKIP_AUGMENTATIONS = args.skip_augmentations
SKIP_ALL_AUGMENTATIONS = False  # redundant but keep for compatibility
BW_PROBABILITY = args.bw_probability
COLOR_PROBABILITY = args.color_probability
TEXT_PROBABILITY = args.text_probability
GRID_LINE_PROBABILITY = args.grid_line_probability
BLUR_PROBABILITY = args.blur_probability
NOISE_PROBABILITY = args.noise_probability
LOSS_NAME = args.loss
CLASS_WEIGHT_OBJECT = args.class_weight_object
DICE_SMOOTH = args.dice_smooth
DICE_WEIGHT = args.dice_weight
TEXT_MIN_FONT_SIZE = 24
TEXT_MAX_FONT_SIZE = 36
TEXT_MAX_TEXTS = 1
TEXT_MIN_LENGTH = 3
TEXT_MAX_LENGTH = 12


id2label = {0: "background", 1: "object"}
label2id = {v: k for k, v in id2label.items()}

TRAIN_ON_CPU = args.train_on_cpu
SMOKE_TEST_ONLY = args.smoke_test
RESUME_FROM_CHECKPOINT = args.resume_from_checkpoint

print("Checkpoint:", CHECKPOINT)
print("Output dir:", str(OUTPUT_DIR))
print("Data root:", str(DATA_ROOT))
print("Image size:", IMAGE_SIZE)
print("Skip augmentations:", SKIP_AUGMENTATIONS)
print("Skip all augmentations:", SKIP_ALL_AUGMENTATIONS)
print("BW probability:", BW_PROBABILITY)
print("Color probability:", COLOR_PROBABILITY)
print("Text probability:", TEXT_PROBABILITY)
print("Grid-line probability:", GRID_LINE_PROBABILITY)
print("Blur probability:", BLUR_PROBABILITY)
print("Noise probability:", NOISE_PROBABILITY)
print("Loss:", LOSS_NAME)
print("Class weight object:", CLASS_WEIGHT_OBJECT)
print("Dice smooth:", DICE_SMOOTH)
print("Dice weight:", DICE_WEIGHT)
print("Train on CPU:", TRAIN_ON_CPU)
print("Smoke test only:", SMOKE_TEST_ONLY)
print("Resume from checkpoint:", RESUME_FROM_CHECKPOINT)


if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
    raise FileNotFoundError(f"Expected local dataset under {DATA_ROOT}")

print("Using local data in:", DATA_ROOT)
print("Available images:", len(list(IMAGES_DIR.glob("*.png"))))
print("Available masks:", len(list(MASKS_DIR.glob("*.png"))))


def normalize_mask_array(mask):
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    return (mask_np > 0).astype(np.uint8)


def normalize_mask(mask_img):
    return Image.fromarray(normalize_mask_array(mask_img), mode="L")


def make_empty_mask_like(image_path):
    img = Image.open(image_path)
    width, height = img.size
    return Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")


records = []
positive_count = 0
negative_count = 0

# make empty mask when no mask file exist
for image_path in sorted(IMAGES_DIR.glob("*.png")):
    mask_path = MASKS_DIR / image_path.name

    if mask_path.exists():
        normalized_mask = normalize_mask(Image.open(mask_path))
        generated_mask_path = GENERATED_MASKS_DIR / image_path.name
        normalized_mask.save(generated_mask_path)
        final_mask_path = generated_mask_path
        positive_count += 1
    else:
        empty_mask = make_empty_mask_like(image_path)
        generated_mask_path = GENERATED_MASKS_DIR / image_path.name
        empty_mask.save(generated_mask_path)
        final_mask_path = generated_mask_path
        negative_count += 1

    records.append(
        {
            "image_path": str(image_path),
            "mask_path": str(final_mask_path),
            "has_object_mask": 1 if mask_path.exists() else 0,
            "file_name": image_path.name,
        }
    )

print("Total records:", len(records))
print("Positive images with masks:", positive_count)
print("Negative images without masks:", negative_count)
assert len(records) > 0


indices = list(range(len(records)))
stratify_labels = [r["has_object_mask"] for r in records]

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=stratify_labels,
)

train_records = [records[i] for i in train_idx]
test_records = [records[i] for i in test_idx]

print("Train size:", len(train_records))
print("Test size:", len(test_records))
print("Train positives:", sum(r["has_object_mask"] for r in train_records))
print("Test positives:", sum(r["has_object_mask"] for r in test_records))


train_ds = Dataset.from_list(train_records)
test_ds = Dataset.from_list(test_records)
raw_datasets = DatasetDict({"train": train_ds, "test": test_ds})


def make_overlay(image, mask, *, alpha=1.0):
    image_np = np.array(image).astype(np.float32)
    overlay = image_np.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    mask_bool = normalize_mask_array(mask) > 0
    overlay[mask_bool] = (1 - alpha) * image_np[mask_bool] + alpha * red
    return Image.fromarray(overlay.astype(np.uint8))


def write_image_and_masks(destination_dir, image, masks_by_suffix, output_file_name):
    destination_dir.mkdir(parents=True, exist_ok=True)
    image.save(destination_dir / f"{output_file_name}_image.png")
    for suffix, mask in masks_by_suffix.items():
        mask_np = normalize_mask_array(mask)
        Image.fromarray(mask_np * 255).save(
            destination_dir / f"{output_file_name}_{suffix}.png"
        )


def write_triplet(destination_dir, image, mask, output_file_name):
    write_image_and_masks(
        destination_dir,
        image,
        {"mask": mask},
        output_file_name,
    )
    make_overlay(image, mask, alpha=1.0).save(
        destination_dir / f"{output_file_name}_overlay.png"
    )


def get_examples_by_name(records, file_names):
    records_by_name = {record["file_name"]: record for record in records}
    selected = []
    for file_name in file_names:
        example = records_by_name.get(file_name)
        if example is None:
            raise FileNotFoundError(f"Image {file_name} not found in dataset records")
        selected.append(example)
    return selected


def write_transform_previews(records):
    selected = get_examples_by_name(records, PREVIEW_IMAGE_NAMES)

    augmentation_showcases = [
        (
            "original",
            lambda image, mask: image,
        ),
        (
            "bw",
            lambda image, mask: convert_to_bw_rgb(image, probability=1.0),
        ),
        (
            "color-balance",
            lambda image, mask: adjust_color_balance(image, probability=1.0),
        ),
        (
            "text-overlay",
            lambda image, mask: overlay_short_text(image, probability=1.0, mask=mask),
        ),
        (
            "grid-line",
            lambda image, mask: add_single_grid_line(image, probability=1.0),
        ),
        (
            "gaussian-blur",
            lambda image, mask: apply_gaussian_blur(image, probability=1.0),
        ),
        (
            "speckle-noise",
            lambda image, mask: add_speckle_noise(image, probability=1.0),
        ),
    ]

    for example in selected:
        image, mask_arr = load_image_and_mask(example)
        label = "positive" if example["has_object_mask"] else "negative"
        stem = Path(example["file_name"]).stem

        for augmentation_name, augmentation_fn in augmentation_showcases:
            augmented_image = augmentation_fn(image.copy(), mask_arr)
            write_triplet(
                PREVIEW_DIR,
                augmented_image,
                mask_arr,
                f"{label}_{stem}_{augmentation_name}",
            )


processor = SegformerImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=True,
    size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    do_reduce_labels=False,
)


def load_image_and_mask(example):
    image = Image.open(example["image_path"]).convert("RGB")
    mask_arr = normalize_mask_array(Image.open(example["mask_path"]))
    return image, mask_arr


def convert_to_bw_rgb(image, probability=0.2):
    if random.random() >= probability:
        return image
    return image.convert("L").convert("RGB")


def adjust_color_balance(
    image,
    probability=0.4,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.2),
    color_range=(0.75, 1.25),
    sharpness_range=(0.9, 1.1),
):
    if random.random() >= probability:
        return image

    out = image
    out = ImageEnhance.Brightness(out).enhance(random.uniform(*brightness_range))
    out = ImageEnhance.Contrast(out).enhance(random.uniform(*contrast_range))
    out = ImageEnhance.Color(out).enhance(random.uniform(*color_range))
    out = ImageEnhance.Sharpness(out).enhance(random.uniform(*sharpness_range))
    return out


def _random_place_name_string(min_length=3, max_length=12):
    if min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length")

    lengths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    weights = [18, 18, 16, 14, 10, 8, 6, 4, 3, 3]
    valid_pairs = [
        (length, weight)
        for length, weight in zip(lengths, weights)
        if min_length <= length <= max_length
    ]
    chosen_length = random.choices(
        [length for length, _ in valid_pairs],
        weights=[weight for _, weight in valid_pairs],
        k=1,
    )[0]

    all_digits = random.random() < 0.2
    if all_digits:
        alphabet = string.digits
    else:
        alphabet = string.ascii_uppercase

    return "".join(random.choice(alphabet) for _ in range(chosen_length)).title()


def overlay_short_text(
    image,
    probability=0.15,
    mask=None,
    min_font_size=24,
    max_font_size=36,
    max_texts=1,
    alpha_range=(110, 190),
    text_color_choices=((0, 0, 0), (35, 35, 35), (70, 70, 70)),
    min_length=3,
    max_length=12,
):
    if random.random() >= probability:
        return image

    base = image.convert("RGBA")
    width, height = base.size
    n_texts = random.randint(1, max_texts)

    mask_array = None
    if mask is not None:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_array = (mask_array > 0).astype(np.uint8)

    for _ in range(n_texts):
        text = _random_place_name_string(min_length=min_length, max_length=max_length)
        font_size = random.randint(min_font_size, max_font_size)
        alpha = random.randint(*alpha_range)
        rgb = random.choice(text_color_choices)
        ink = (*rgb, alpha)

        font = ImageFont.load_default()

        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_w = int(max(1, bbox[2] - bbox[0]))
        text_h = int(max(1, bbox[3] - bbox[1]))

        pad_x = 8
        pad_y = 6
        text_layer = Image.new(
            "RGBA",
            (text_w + 2 * pad_x, text_h + 2 * pad_y),
            (0, 0, 0, 0),
        )
        text_draw = ImageDraw.Draw(text_layer)
        text_draw.text((pad_x, pad_y), text, fill=ink, font=font)

        max_x = max(0, width - text_layer.size[0])
        max_y = max(0, height - text_layer.size[1])

        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0

        if mask_array is not None and mask_array.any():
            ys, xs = np.nonzero(mask_array)
            anchor_index = random.randrange(len(xs))
            anchor_x = int(xs[anchor_index])
            anchor_y = int(ys[anchor_index])

            target_x = anchor_x - text_layer.size[0] // 2
            target_y = anchor_y - text_layer.size[1] // 2
            x = min(max(target_x, 0), max_x)
            y = min(max(target_y, 0), max_y)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay.alpha_composite(text_layer, dest=(x, y))
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")


def add_single_grid_line(
    image,
    probability=0.2,
    thickness_range=(2, 4),
    alpha_range=(120, 220),
    blur_radius_range=(0.0, 0.6),
):
    if random.random() >= probability:
        return image

    base = image.convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    thickness = random.randint(*thickness_range)
    alpha = random.randint(*alpha_range)
    tone = random.choice([0, 20, 40, 60])
    color = (tone, tone, tone, alpha)

    if random.random() < 0.5:
        x = random.randint(0, max(0, width - 1))
        draw.line((x, 0, x, height), fill=color, width=thickness)
    else:
        y = random.randint(0, max(0, height - 1))
        draw.line((0, y, width, y), fill=color, width=thickness)

    blur_radius = random.uniform(*blur_radius_range)
    if blur_radius > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_gaussian_blur(image, probability=0.15, radius_range=(0.3, 1.2)):
    if random.random() >= probability:
        return image

    radius = random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_speckle_noise(image, probability=0.15, std_range=(4.0, 14.0)):
    if random.random() >= probability:
        return image

    arr = np.asarray(image).astype(np.float32)
    std = random.uniform(*std_range)
    noise = np.random.normal(loc=0.0, scale=std, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_map_augmentations(
    image,
    *,
    mask=None,
    skip_all=False,
    bw_probability=0.2,
    color_probability=0.4,
    text_probability=0.15,
    grid_line_probability=0.2,
    blur_probability=0.15,
    noise_probability=0.15,
    text_min_font_size=24,
    text_max_font_size=36,
    text_max_texts=1,
    text_min_length=3,
    text_max_length=12,
):
    out = image.convert("RGB")
    if skip_all:
        return out
    out = convert_to_bw_rgb(out, probability=bw_probability)
    out = adjust_color_balance(out, probability=color_probability)
    out = overlay_short_text(
        out,
        probability=text_probability,
        mask=mask,
        min_font_size=text_min_font_size,
        max_font_size=text_max_font_size,
        max_texts=text_max_texts,
        min_length=text_min_length,
        max_length=text_max_length,
    )
    out = add_single_grid_line(out, probability=grid_line_probability)
    out = apply_gaussian_blur(out, probability=blur_probability)
    out = add_speckle_noise(out, probability=noise_probability)
    return out


write_transform_previews(records)
print("Saved transform previews to:", PREVIEW_DIR)


def train_transforms(example_batch):
    images = []
    labels = []
    for image_path, mask_path in zip(
        example_batch["image_path"], example_batch["mask_path"]
    ):
        image, mask_arr = load_image_and_mask(
            {"image_path": image_path, "mask_path": mask_path}
        )

        if not SKIP_AUGMENTATIONS:
            image = apply_map_augmentations(
                image,
                mask=mask_arr,
                skip_all=SKIP_ALL_AUGMENTATIONS,
                bw_probability=BW_PROBABILITY,
                color_probability=COLOR_PROBABILITY,
                text_probability=TEXT_PROBABILITY,
                grid_line_probability=GRID_LINE_PROBABILITY,
                blur_probability=BLUR_PROBABILITY,
                noise_probability=NOISE_PROBABILITY,
                text_min_font_size=TEXT_MIN_FONT_SIZE,
                text_max_font_size=TEXT_MAX_FONT_SIZE,
                text_max_texts=TEXT_MAX_TEXTS,
                text_min_length=TEXT_MIN_LENGTH,
                text_max_length=TEXT_MAX_LENGTH,
            )

        images.append(image)
        labels.append(mask_arr)

    inputs = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return inputs


transformed_datasets = raw_datasets.with_transform(train_transforms)


model = SegformerForSemanticSegmentation.from_pretrained(
    CHECKPOINT,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,}")
print(f"Total params: {total_params:,}")


metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    metrics = (
        metric.compute(
            predictions=F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            .argmax(dim=1)
            .cpu()
            .numpy(),
            references=labels,
            num_labels=2,
            ignore_index=255,
            reduce_labels=False,
        )
        or {}
    )

    per_category_iou = metrics.get("per_category_iou")
    if per_category_iou is None:
        per_category_iou = [0.0, 0.0]

    return {
        "mean_iou": metrics.get("mean_iou", 0.0),
        "mean_accuracy": metrics.get("mean_accuracy", 0.0),
        "iou_background": float(per_category_iou[0]),
        "iou_object": float(per_category_iou[1]),
    }


class ValidationLogger(TrainerCallback):
    def __init__(self, log_path=EVAL_LOG_PATH, training_log_path=LOG_PATH):
        self.log_path = Path(log_path)
        self.training_log_path = Path(training_log_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = {"epoch": state.epoch, "step": state.global_step, **logs}
            line = json.dumps(payload)
            print(line)
            with open(self.training_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if "eval_loss" in logs:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")


class WeightedLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_name="ce",
        class_weight_object=3.0,
        dice_smooth=1.0,
        dice_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.class_weight_object = class_weight_object
        self.dice_smooth = dice_smooth
        self.dice_weight = dice_weight

    def _dice_loss(self, logits, labels):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        labels = labels.float()
        intersection = (probs * labels).sum(dim=(1, 2))
        denominator = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
        dice_score = (2 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        return 1.0 - dice_score.mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        upsampled_logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        labels = labels.long()
        loss = None

        if self.loss_name in {"ce", "weighted-ce", "ce-dice", "weighted-ce-dice"}:
            ce_weight = None
            if self.loss_name in {"weighted-ce", "weighted-ce-dice"}:
                ce_weight = torch.tensor(
                    [1.0, self.class_weight_object],
                    device=upsampled_logits.device,
                    dtype=upsampled_logits.dtype,
                )
            ce_loss = F.cross_entropy(upsampled_logits, labels, weight=ce_weight)
            loss = ce_loss

        if self.loss_name in {"dice", "ce-dice", "weighted-ce-dice"}:
            dice_loss = self._dice_loss(upsampled_logits, labels)
            loss = (
                self.dice_weight * dice_loss
                if loss is None
                else loss + self.dice_weight * dice_loss
            )

        if loss is None:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_mean_accuracy",
    greater_is_better=True,
    dataloader_num_workers=0,
    report_to="none",
    fp16=(torch.cuda.is_available() and not TRAIN_ON_CPU),
    bf16=False,
    use_cpu=TRAIN_ON_CPU,
    dataloader_pin_memory=False,
    weight_decay=WEIGHT_DECAY,
    seed=SEED,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=transformed_datasets["train"],
    eval_dataset=transformed_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[ValidationLogger()],
    loss_name=LOSS_NAME,
    class_weight_object=CLASS_WEIGHT_OBJECT,
    dice_smooth=DICE_SMOOTH,
    dice_weight=DICE_WEIGHT,
)

print(training_args)


print("Starting training on device:", trainer.args.device)
if SMOKE_TEST_ONLY:
    train_result = None
    train_result_payload = {"skipped_training": True}
    print(json.dumps({"train_result": train_result_payload}, sort_keys=True))
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"train_result": train_result_payload}, sort_keys=True) + "\n"
        )
else:
    train_result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    train_result_payload = train_result.metrics
    print(json.dumps({"train_result": train_result_payload}, sort_keys=True))
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"train_result": train_result_payload}, sort_keys=True) + "\n"
        )


if SMOKE_TEST_ONLY:
    eval_metrics = {"skipped_eval": True}
else:
    eval_metrics = trainer.evaluate()
print(json.dumps({"final_eval": eval_metrics}, sort_keys=True))
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(json.dumps({"final_eval": eval_metrics}, sort_keys=True) + "\n")


final_model_path = FINAL_MODEL_PATH
trainer.save_model(str(final_model_path))
processor.save_pretrained(str(final_model_path))
print("Best model saved to:", final_model_path)
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(
        json.dumps({"best_model_path": str(final_model_path)}, sort_keys=True) + "\n"
    )


PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(inference_device)
model.eval()


def predict_mask(image):
    inputs = processor(images=image, return_tensors="pt").to(inference_device)
    with torch.no_grad():
        outputs = model(**inputs)
    upsampled_logits = F.interpolate(
        outputs.logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return prediction


def save_prediction_triplet(example, destination_dir, prefix):
    image = Image.open(example["image_path"]).convert("RGB")
    gt_mask = normalize_mask_array(Image.open(example["mask_path"]))
    pred_mask = predict_mask(image)

    write_image_and_masks(
        destination_dir,
        image,
        {"gt": gt_mask, "pred": pred_mask},
        prefix,
    )
    make_overlay(image, pred_mask, alpha=0.7).save(
        destination_dir / f"{prefix}_overlay.png"
    )


positive_examples = get_examples_by_name(records, POSITIVE_IMAGE_NAMES)
for i, example in enumerate(positive_examples):
    prefix = f"test_positive_{i:02d}_{Path(example['file_name']).stem}"
    save_prediction_triplet(example, PREDICTION_DIR, prefix)

# negative examples as well? Usually this is fine.

for hard_image_path in sorted(HARD_DIR.rglob("*.png")):
    implicit_label = hard_image_path.parent.name
    prefix = f"hard_{implicit_label}_{hard_image_path.stem}"
    hard_example = {
        "image_path": str(hard_image_path),
        "mask_path": str(
            GENERATED_MASKS_DIR / f"{hard_image_path.stem}_hard_empty_mask.png"
        ),
        "has_object_mask": 0,
        "file_name": hard_image_path.name,
    }
    make_empty_mask_like(hard_image_path).save(hard_example["mask_path"])
    save_prediction_triplet(hard_example, PREDICTION_DIR, prefix)

print("Saved prediction samples to:", PREDICTION_DIR)
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(
        json.dumps(
            {
                "preview_dir": str(PREDICTION_DIR),
                "prediction_dir": str(PREDICTION_DIR),
                "hard_dir": str(HARD_DIR),
            },
            sort_keys=True,
        )
        + "\n"
    )


print("Prediction files written to:", PREDICTION_DIR)
print("Training run complete. Results are available under:", OUTPUT_DIR)
print("Training log:", LOG_PATH)
print("Eval log:", EVAL_LOG_PATH)
print("Best model:", FINAL_MODEL_PATH)
