# K3NK Image Grab

Standalone ComfyUI node to grab the last N frames from a directory as a single batch, with optional frame stride.

This node is designed to feed video continuation workflows, improving temporal consistency when working with interpolated frames. It is ideal for use with **WanVideo ClipVision Encode** or other video processing nodes in ComfyUI.

**Credit:** This node is a modified version of the original BoyoNodes by DragonDiffusion (https://github.com/DragonDiffusionbyBoyo/Boyonodes). Thanks to the author for the original implementation.

---

## Features

- Grab the last N frames from a folder
- Optional frame stride: skip frames between selected frames
- Returns a single batch tensor for ComfyUI nodes
- Latent output for direct integration with diffusion pipelines
- Automatic size matching: latent dimensions match input image sizes
- Standalone: no dependencies on other custom nodes

---

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `directory_path` | STRING | Path to the folder containing frames |
| `num_images` | INT | Number of frames to grab |
| `frame_stride` | INT | Number of frames to skip between selected frames |
| `file_extensions` | STRING | Comma-separated list of file extensions (default: jpg,jpeg,png,bmp,tiff,webp) |

---

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Batch tensor containing selected frames |
| `latent` | LATENT | Empty latents matching image dimensions for diffusion pipelines |
| `filenames` | STRING | Names of the selected files |
| `full_paths` | STRING | Full paths of the selected files |
| `timestamp` | FLOAT | Latest modification timestamp among selected files |

---

## Example Usage

Grab 4 frames with a stride of 5 frames between them: `num_images = 4`, `frame_stride = 5`. This will select the last frame, skip 5 frames backward, grab another, and repeat until 4 frames are collected.

The latent output automatically matches the dimensions of your input images (divided by 8 for VAE compatibility), making it ready for direct use in diffusion workflows.

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
