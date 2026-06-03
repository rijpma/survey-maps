# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "napari[all]",    # Includes the GUI backend (PyQt5)
#     "scikit-image",   # For loading image collections
#     "pillow",         # For reading/writing PNGs
#     "numpy",
# ]
# ///

import argparse
import shutil
from pathlib import Path

import napari
import numpy as np
from PIL import Image


def setup_labeling_session(src_dir, dest_root, limit=None):
    src_dir = Path(src_dir)
    dest_root = Path(dest_root)

    # 1. Setup folders
    img_dest = dest_root / "images"
    mask_dest = dest_root / "masks"
    complete_dest = dest_root / "complete"
    img_dest.mkdir(parents=True, exist_ok=True)
    mask_dest.mkdir(parents=True, exist_ok=True)
    complete_dest.mkdir(parents=True, exist_ok=True)

    # 2. Find all tiles recursively and skip ones already labeled
    src_path = Path(src_dir)
    all_tiles = sorted(src_path.rglob("*.png"))
    pending_tiles = []

    for tile_path in all_tiles:
        tile_name = tile_path.name
        mask_path = mask_dest / tile_name
        complete_path = complete_dest / f"{tile_name}.done"
        if mask_path.exists() or complete_path.exists():
            continue
        pending_tiles.append(tile_path)

    if limit is not None:
        pending_tiles = pending_tiles[:limit]

    if not pending_tiles:
        print("No unlabeled tiles remaining.")
        return

    print(f"Found {len(all_tiles)} total tiles")
    print(f"Labeling {len(pending_tiles)} unlabeled tiles")

    # Copy pending files and load into memory for Napari
    image_list = []
    grayscale_list = []
    for tile_path in pending_tiles:
        tile_name = tile_path.name
        copied_tile_path = img_dest / tile_name
        if not copied_tile_path.exists():
            shutil.copy(tile_path, copied_tile_path)

        # Load color image (force RGB)
        color_image = np.array(Image.open(copied_tile_path).convert("RGB"))
        image_list.append(color_image)
        # Convert to grayscale for labels (one layer, no color blabla)
        grayscale_image = np.array(Image.open(copied_tile_path).convert("L"))
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
    print("2. Scroll through the images using the slider at the bottom.")
    print("3. Use 'Eraser' to fix mistakes.")
    print(
        "4. IMPORTANT: Close the Napari window when you are finished to save your work."
    )

    napari.run()

    # 4. Save results after Napari closes
    print("Saving results...")
    final_masks = label_layer.data  # This is the (N, H, W) array you painted

    saved_count = 0
    completed_empty_count = 0
    for i, tile_path in enumerate(pending_tiles):
        tile_name = tile_path.name
        mask_array = final_masks[i].astype(np.uint8)
        complete_path = complete_dest / f"{tile_name}.done"

        if np.any(mask_array):
            mask_path = mask_dest / tile_name
            # Save as 0 and 1 (standard for Transformers/PyTorch)
            Image.fromarray(mask_array).save(mask_path)
            saved_count += 1
        else:
            complete_path.touch()
            completed_empty_count += 1

    print(f"Saved {saved_count} masks to {mask_dest}")
    print(f"Marked {completed_empty_count} empty tiles as complete in {complete_dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a Napari labeling session.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--src_dir",
        type=Path,
        default=base_dir / "india_tiles" / "14",
        help="Directory containing source tiles.",
    )
    parser.add_argument(
        "--dest_root",
        type=Path,
        default=base_dir / "napari_out" / "roads14",
        help="Root directory for saving images and masks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only open the first N unlabeled tiles.",
    )

    args = parser.parse_args()

    setup_labeling_session(
        src_dir=args.src_dir,
        dest_root=args.dest_root,
        limit=args.limit,
    )
