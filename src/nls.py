import math
import os
import random
import time

import requests

# XYZ tile server
URL_TEMPLATE = "https://geo.nls.uk/mapdata2/india-half/{z}/{x}/{y}.png"

# India bounding box (sample)
WEST, SOUTH, EAST, NORTH = 77.32122, 8.103021, 77.563616, 8.211740

# India bounding box (full)
WEST, SOUTH, EAST, NORTH = 65.557617, 6.699017, 101.341483, 36.135965

# Sample or everything?
SAMPLE = 0.01  # 0.01 = 1%; >1 is everything

# zoom levels to download
# ZOOMS = range(14, 16)
# 14 is the max detail mentioned in source
# range ... 16, goes up to 15

OUT_DIR = "india_tiles"
os.makedirs(OUT_DIR, exist_ok=True)


def latlon_to_tile(lat, lon, z):
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_to_latlon(x, y, z):
    """
    Returns the top-left (North-West) latitude and longitude of the tile.
    """
    n = 2.0**z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


# don't loop anymore, z = 14 is the optimum
z = 14
x_min, y_max = latlon_to_tile(SOUTH, WEST, z)
x_max, y_min = latlon_to_tile(NORTH, EAST, z)

print(f"Zoom {z}: x {x_min}-{x_max}, y {y_min}-{y_max}")

for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        if random.random() > SAMPLE:
            print(f"Skipped {y}, {x}")
            continue
        url = URL_TEMPLATE.format(z=z, x=x, y=y)

        out_path = os.path.join(OUT_DIR, str(z), str(x))
        os.makedirs(out_path, exist_ok=True)

        # Get the top-left coordinates of the tile to include in the filename
        lat, lon = tile_to_latlon(x, y, z)
        filename = os.path.join(out_path, f"{y}_lat_{lat:.5f}_lon_{lon:.5f}.png")

        if os.path.exists(filename):
            continue

        try:
            r = requests.get(url, timeout=30)
            print(url)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
            else:
                print("missing", z, x, y)
            time.sleep(0.2)  # sleep, but only if sampling this particular coord set

        except Exception as e:
            print("error", z, x, y, e)
