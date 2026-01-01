# K3NK Image Grab
Standalone ComfyUI node to grab frames from a directory as a single batch, with optional frame stride and ordering controls.
This node is designed to feed video continuation workflows, improving temporal consistency when working with interpolated frames. It is ideal for use with **WanVideo ClipVision Encode** or other video processing nodes in ComfyUI.
**Credit:** This node is a modified version of the original BoyoNodes by DragonDiffusion (https://github.com/DragonDiffusionbyBoyo/Boyonodes). Thanks to the author for the original implementation.
---
## Features
- Grab the last N frames from a folder (or first N frames with `reverse_logic`)
- Optional frame stride: skip frames between selected frames (up to 1000)
- Auto-fill: if stride is too large, automatically grabs the last available frame
- Reverse order: invert the final selection order
- Reverse logic: start from oldest frames instead of newest
- Returns a single batch tensor for ComfyUI nodes
- VAE encoding support for proper latent generation
- Standalone: no dependencies on other custom nodes
---
## Inputs
| Name | Type | Description |
|------|------|-------------|
| `directory_path` | STRING | Path to the folder containing frames |
| `num_images` | INT | Number of frames to grab |
| `frame_stride` | INT | Number of frames to skip between selected frames (0-1000) |
| `reverse_order` | BOOLEAN | Reverse the order of selected images (default: False) |
| `reverse_logic` | BOOLEAN | Start from oldest frame instead of newest (default: False) |
| `file_extensions` | STRING | Comma-separated list of file extensions (default: jpg,jpeg,png,bmp,tiff,webp) |
| `vae` | VAE (optional) | VAE for proper latent encoding |
---
## Outputs
| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch tensor containing selected frames |
| `extra_latents` | LATENT | Encoded latents (requires VAE input) |
| `filenames` | STRING | Names of the selected files |
| `full_paths` | STRING | Full paths of the selected files |
| `timestamp` | FLOAT | Latest modification timestamp among selected files |
---
## Example Usage
**Grab 4 frames with a stride of 5:** `num_images = 4`, `frame_stride = 5`. This will select the last frame, skip 5 frames backward, grab another, and repeat until 4 frames are collected.

**Grab first and last frame:** `num_images = 2`, `frame_stride = 1000`. This will grab the newest frame, then skip forward and automatically grab the oldest frame available.

**Start from oldest frames:** Enable `reverse_logic = True` to begin selection from the oldest frame in the directory instead of the newest.

**Reverse final order:** Enable `reverse_order = True` to invert the order of the selected frames after selection.
---
## Installation
### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI  
2. Go to **Manager → Install Custom Node**  
3. Search for `"K3NK Image Grab"`  
4. Click **Install** and restart ComfyUI
### Method 2: Git Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/K3NK3/ComfyUI-K3NKImageGrab
```
Restart ComfyUI afterward.
