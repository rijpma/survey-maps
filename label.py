# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "napari[all]",    # Includes the GUI backend (PyQt5)
#     "scikit-image",   # For loading image collections
#     "pillow",            # For reading/writing PNGs
#     "numpy",
# ]
# ///

import argparse
import os
import random
import shutil
from pathlib import Path

import napari
import numpy as np
from PIL import Image


def setup_labeling_session(src_dir, dest_root, sample_size=20):
    # 1. Setup Folders
    img_dest = Path(dest_root) / "images"
    mask_dest = Path(dest_root) / "masks"
    img_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)

    # 2. Sample random tiles recursively
    src_path = Path(src_dir)
    all_tiles = list(src_path.rglob("*.png"))
    sampled_tiles = random.sample(all_tiles, min(sample_size, len(all_tiles)))

    print(f"Sampled {len(sampled_tiles)} tiles into {img_dest}")

    # Copy files and load into memory for Napari
    image_list = []
    grayscale_list = []
    for tile_path in sampled_tiles:
        tile_name = tile_path.name
        shutil.copy(tile_path, img_dest / tile_name)
        # Load color image (force RGB)
        color_image = np.array(Image.open(img_dest / tile_name).convert("RGB"))
        image_list.append(color_image)
        # Convert to grayscale for labels (one layer, no color blabla)
        grayscale_image = np.array(Image.open(img_dest / tile_name).convert("L"))
        grayscale_list.append(grayscale_image)

    # Stack images into 3D arrays
    image_stack = np.stack(image_list, axis=0)  # Shape: (N, H, W, 3)
    grayscale_stack = np.stack(grayscale_list, axis=0)  # Shape: (N, H, W)
    print(f"Image stack shape (color): {image_stack.shape}")
    print(f"Grayscale stack shape: {grayscale_stack.shape}")

    # 3. Launch Napari
    viewer = napari.Viewer()
    viewer.add_image(image_stack, name="Tiles")

    # Add empty labels layer (0 = background, 1 = road)
    label_layer = viewer.add_labels(
        np.zeros(grayscale_stack.shape, dtype=np.uint8), name="Road_Labels"
    )

    # Pre-select label 1 and the paintbrush tool
    label_layer.selected_label = 1
    label_layer.mode = "paint"
    label_layer.brush_size = 5

    print("\n--- INSTRUCTIONS ---")
    print("1. Use the 'Paintbrush' tool to draw over roads.")
    print("2. Scroll through the 20 images using the slider at the bottom.")
    print("3. Use 'Eraser' to fix mistakes.")
    print(
        "4. IMPORTANT: Close the Napari window when you are finished to save your work."
    )

    napari.run()

    # 4. Save individual masks after Napari closes
    print("Saving masks...")
    final_masks = label_layer.data  # This is the (N, H, W) array you painted

    for i, tile_path in enumerate(sampled_tiles):
        tile_name = tile_path.name
        mask_path = mask_dest / tile_name
        # Save as 0 and 1 (standard for Transformers/PyTorch)
        Image.fromarray(final_masks[i].astype(np.uint8)).save(mask_path)

    print(f"Successfully saved {len(sampled_tiles)} masks to {mask_dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a Napari labeling session.")
    parser.add_argument("--src_dir", type=str, default="./img/india_tiles/14/", help="Directory containing source tiles.")
    parser.add_argument("--dest_root", type=str, default="./labelled/roads14", help="Root directory for saving images and masks.")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of random tiles to sample.")

    args = parser.parse_args()

    setup_labeling_session(
        src_dir=args.src_dir,
        dest_root=args.dest_root,
        sample_size=args.sample_size
    )
