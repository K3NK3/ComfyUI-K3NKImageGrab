# ComfyUI-K3NKImageGrab

A custom ComfyUI node that loads the latest frames from a directory and outputs them as a single image batch.  
Useful for video continuation, frame analysis, and workflows requiring temporal consistency.  
Supports frame stride selection, file extension filtering, and provides metadata such as file names and timestamps.

---

## Features

- Loads the most recent N image files from a directory.
- Supports frame stride (e.g., load every X frames).
- Supports filtering by file extension.
- Outputs:
  - Image batch tensor
  - File names
  - Full file paths
  - Timestamp of the latest modified frame
- Designed for use in video and temporal workflows (e.g., WanVideo ClipVision Encode).
- Standalone implementation with no external node dependencies.

---

## Installation

### Using ComfyUI Manager
1. Open ComfyUI.
2. Go to **Manager → Install from URL**.
3. Paste the repository URL:

```
https://github.com/K3NK3/ComfyUI-K3NKImageGrab
```

4. Install and restart ComfyUI.

### Manual Installation
Clone or download this repository into the ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/ComfyUI-K3NKImageGrab
```

Restart ComfyUI afterward.

---

## Node Inputs

| Input | Type | Description |
|-------|--------|-------------|
| `directory_path` | string | Path to the directory containing the image frames. |
| `num_images` | int | Number of recent frames to load. |
| `frame_stride` | int | Load every X frames (stride). |
| `file_extensions` | string | Comma-separated list of extensions (`png,jpg,jpeg`). |

---

## Node Outputs

| Output | Type | Description |
|--------|--------|-------------|
| `IMAGE` | tensor | Batch of loaded frames. |
| `filenames` | list | File names of the selected frames. |
| `full_paths` | list | Absolute paths of each frame. |
| `latest_timestamp` | float | Modification timestamp of the newest frame. |

---

## Usage Notes

- Frames are sorted by modification time, newest first.
- Stride is applied after sorting.
- If fewer frames exist than requested, the node returns as many as available.
- Works well in video workflows and any temporal conditioning setup.

---

## Example Workflow

1. Generate or capture frames into a folder.
2. Use **K3NK Image Grab** to load the latest batch with a specific stride.
3. Feed the batch into your temporal encoder or video model.
4. Generate continuation or condition the next frames.

---

## Requirements

Dependencies are listed in `requirements.txt`.  
ComfyUI Manager installs them automatically when possible.

---

## License

MIT License.

---

## Credits

This node is adapted and "standaloned" from the original BoyoNodes version:  
https://github.com/dragon9918/BoyoNodes
