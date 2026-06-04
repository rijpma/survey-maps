# mmmmaps: Historical Map Processing Pipeline

This repository contains a set of scripts for downloading, reconstructing, and creating machine learning datasets from historical map tiles (specifically from the National Library of Scotland - NLS). 

The pipeline consists of four main scripts that handle data acquisition, visual reconstruction, manual segmentation (labeling), and finetuning of a Segformer model.

## Pipeline Overview and Data Flow

The following scripts are in the `src` folders:

1. `nls.py`: Pulls individual 256x256 tiles from the web into a structured grid format.
2. `stitch.py`: Combines those individual tiles into a single, cohesive, high-resolution map for viewing.
3. `label.py`: Samples a subset of the downloaded tiles to create pairs of images and binary masks for training computer vision models. Launches Napari.
4. `segformer_colab_t4.ipynb`: Notebook to finetune a segformer model. [Launch it here on Google Colab](https://githubtocolab.com/rijpma/survey-maps/blob/main/src//segformer_colab_t4.ipynb). Make sure to put the runtime to T4.
5. `inference.py` run inference using the finetuned models.

### Step 1: Downloading Tiles (`nls.py`)
The pipeline starts with `nls.py`. Standard XYZ tile servers divide the world map into a grid. This script takes a bounding box (latitude and longitude) and converts it into the standard `x`, `y`, and `z` (zoom) coordinates required by the server.

- **Process**: It loops over the coordinate ranges for zoom level 14, requesting the 256x256 pixel tiles.
- **Output**: The tiles are saved in an organized folder structure: `india_tiles/{z}/{x}/`. 
- **Naming**: To retain geographical information without needing complex metadata files, the script calculates the North-West latitude and longitude of each tile and appends it to the filename (e.g., `7816_lat_8.18826_lon_77.41516.png`).

### Step 2: Stitching the Map (`stitch.py`)
Because XYZ tiles are mathematical subdivisions of the Web Mercator projection, they touch perfectly edge-to-edge without overlapping. 

- **Process**: `stitch.py` reads the downloaded directory structure. Since the `x` and `y` coordinates represent precise column and row indices, the script calculates exact pixel offsets `((x - x_min) * 256)`.
- **Output**: It generates a large, transparent RGBA canvas, pastes every tile into its exact mathematical position, and saves a single high-resolution image called `stitched_india_z14.png` in the project root.

### Step 3: Labeling a Dataset (`label.py`)
To train a computer vision model (like a semantic segmentation model to detect roads), you need a dataset of images and matching ground-truth masks.

- **Process**: `label.py` recursively searches the source directory for tiles, randomly samples a specified number of them (default: 100), and copies them to a new dataset directory.
- **Napari GUI**: It stacks the selected RGB images into a 3D array and opens them in Napari. A blank label layer is superimposed on top, allowing you to manually paint over features (like roads).
- **Output**: When you close Napari, the script saves your painted layers as binary masks (0 for background, 1 for road) directly corresponding to the sampled images.
