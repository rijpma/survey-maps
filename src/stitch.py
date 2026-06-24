import os
import re

from PIL import Image

# Increase the max image pixels to allow for very large stitched maps
Image.MAX_IMAGE_PIXELS = None

IN_DIR = "india_tiles-vm"
OUT_NAME = "vm"
TILE_SIZE = 256


def stitch_tiles(zoom):
    zoom_dir = os.path.join(IN_DIR, str(zoom))
    if not os.path.exists(zoom_dir):
        print(f"Zoom directory not found: {zoom_dir}")
        return

    # Find all tiles and determine coordinate boundaries
    tiles = []
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for x_str in os.listdir(zoom_dir):
        if not x_str.isdigit():
            continue
        x = int(x_str)
        x_path = os.path.join(zoom_dir, x_str)

        if not os.path.isdir(x_path):
            continue

        for filename in os.listdir(x_path):
            if not filename.endswith(".png"):
                continue

            # Extract y from filename (format: {y}_lat_..._lon_....png)
            match = re.match(r"^(\d+)_", filename)
            if match:
                y = int(match.group(1))
            else:
                # Fallback just in case
                y_str = filename.split(".")[0]
                if not y_str.isdigit():
                    continue
                y = int(y_str)

            tiles.append((x, y, os.path.join(x_path, filename)))

            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

    if not tiles:
        print(f"No tiles found for zoom level {zoom}.")
        return

    print(f"--- Zoom Level {zoom} ---")
    print(f"Found {len(tiles)} tiles.")
    print(f"X range: {x_min} to {x_max}")
    print(f"Y range: {y_min} to {y_max}")

    # Calculate final image dimensions
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE
    print(f"Output image size: {width}x{height} pixels")

    # Create a blank transparent image canvas
    stitched = Image.new("RGBA", (width, height))

    # Paste each tile into its correct position
    for x, y, path in tiles:
        try:
            with Image.open(path) as tile_img:
                # Calculate pixel offsets
                x_offset = (x - x_min) * TILE_SIZE
                y_offset = (y - y_min) * TILE_SIZE
                stitched.paste(tile_img, (x_offset, y_offset))
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save output
    out_filename = f"stitched_india_z{zoom}_{OUT_NAME}.png"
    print(f"Saving to {out_filename}...")
    stitched.save(out_filename, "PNG")
    print(f"Successfully created {out_filename}\n")


if __name__ == "__main__":
    # The download script fetched only zooms 14
    stitch_tiles(14)
