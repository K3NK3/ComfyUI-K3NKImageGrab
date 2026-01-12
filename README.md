# K3NK Image Grab
Advanced ComfyUI node for loading and processing images/latent files with full WanVideo SVI compatibility.
This node is specifically designed for video continuation workflows with **WanVideoWrapper SVI nodes**, allowing seamless loading of saved `.latent` files and proper formatting for temporal consistency.

**Credit:** This node is a modified version of the original BoyoNodes by DragonDiffusion (https://github.com/DragonDiffusionbyBoyo/Boyonodes). Thanks to the author for the original implementation.

## Features
- **Multi-format support**: Loads `.latent` files (safetensors, pickle, ComfyUI SaveLatent format), plus standard images
- **WanVideo 5D format**: Outputs tensors in `[1, 16, T, H, W]` format compatible with WanVideo SVI nodes
- **Dual output system**: Separate `anchor_frame` and `latent_batch` outputs for SVI workflows
- **Flexible file selection**: Grab N frames with configurable stride and ordering
- **Independent anchor control**: Anchor frame selection independent of batch processing
- **Frame limiting**: Select specific frame ranges from loaded batches
- **Sequential numbering**: Detects file order by numeric sequence in filenames

## Inputs
| Name | Type | Description |
|------|------|-------------|
| `directory_path` | STRING | Path to folder containing frames/latent files |
| `num_images` | INT | Number of files to process (1-100) |
| `frame_stride` | INT | Number of files to skip between selections (0-1000) |
| `reverse_order` | BOOLEAN | Reverse the order of selected files in output |
| `reverse_logic` | BOOLEAN | Start selection from oldest (lowest number) instead of newest |
| `max_batch_frames` | INT | Maximum frames in latent_batch (0=all frames) |
| `batch_start_frame` | INT | Starting frame index for latent_batch output |
| `anchor_from_start` | BOOLEAN | True: first frame of oldest file, False: last frame of newest file |
| `file_extensions` | STRING | File extensions to load (default: jpg,jpeg,png,bmp,tiff,webp,latent) |
| `vae` | VAE (optional) | VAE for encoding images to latents |

## Outputs
| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image batch for preview (black placeholders for .latent files) |
| `latent_batch` | LATENT | All frames in WanVideo format `[1, 16, T, H, W]` for `prev_samples` |
| `anchor_frame` | LATENT | Single frame in WanVideo format `[1, 16, 1, H, W]` for `anchor_frame` |
| `filenames` | STRING | Names of processed files |
| `full_paths` | STRING | Full paths of processed files |
| `timestamp` | FLOAT | Latest modification timestamp |

## Key Workflows

### Video Continuation with SVI
Saved .latent files → K3NK Image Grab → WanVideo SVI nodes → Continue generation
- Load previously saved `.latent` files from SaveLatent node
- Automatically converts to WanVideo 5D format
- Provides proper `anchor_frame` and `prev_samples` for SVI

### Anchor Frame Strategies
- `anchor_from_start=True`: First frame of oldest (lowest number) file
- `anchor_from_start=False`: Last frame of newest (highest number) file
- Independent of `reverse_logic` and `reverse_order` settings

### Frame Selection Examples
| Goal | Settings |
|------|----------|
| Last 4 frames | `max_batch_frames=4`, `batch_start_frame=0` |
| Frames 10-15 | `batch_start_frame=10`, `max_batch_frames=5` |
| All frames | `max_batch_frames=0` (default) |
| Skip every other file | `frame_stride=1` |
| Start from oldest | `reverse_logic=True` |

### File Naming Convention
The node detects numeric sequences in filenames:
- `0203_00138_.latent` → number `138`
- `frame_050.png` → number `50`
- `output_100.jpg` → number `100`
Oldest = lowest number, Newest = highest number

## Example Workflows

### 1. Continue from Last Frame
Settings:
- anchor_from_start: False
- reverse_logic: False (start from newest)
- num_images: 2 (load 2 most recent files)
- max_batch_frames: 4 (last 4 frames total)
Result:
- anchor_frame: Last frame of newest .latent file
- latent_batch: Last 4 frames from 2 newest files

### 2. Start New Sequence from Beginning
Settings:
- anchor_from_start: True
- reverse_logic: True (start from oldest)
- num_images: 1 (just the first file)
- max_batch_frames: 0 (all frames)
Result:
- anchor_frame: First frame of oldest .latent file
- latent_batch: All frames from first file

### 3. Middle Section of Sequence
Settings:
- anchor_from_start: False
- reverse_logic: False
- num_images: 3
- batch_start_frame: 10
- max_batch_frames: 15
Result:
- anchor_frame: Last frame of newest file
- latent_batch: Frames 10-24 from 3 newest files

## Technical Details

### Latent File Support
- **Safetensors format**: Looks for `latent_tensor` or `samples` keys
- **ComfyUI SaveLatent**: Supports JSON format with metadata
- **PyTorch pickle**: Direct tensor or dictionary loading
- **WanVideo format**: Auto-converts to `[1, 16, T, H, W]`

### Output Format
- `latent_batch`: `[1, 16, total_frames, height//8, width//8]`
- `anchor_frame`: `[1, 16, 1, height//8, width//8]`
- Compatible with WanVideoWrapper SVI `anchor_frame` and `prev_samples` inputs

### Processing Order
1. Anchor frame: Selected independently based on `anchor_from_start`
2. File selection: Based on `num_images`, `frame_stride`, `reverse_logic`
3. Frame concatenation: Temporal frames concatenated with newest files first
4. Frame selection: Applied via `batch_start_frame` and `max_batch_frames`

## Installation
### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI
2. Go to Manager → Install Custom Node
3. Search for "K3NK Image Grab"
4. Click Install and restart ComfyUI

### Method 2: Manual Installation
cd ComfyUI/custom_nodes
git clone https://github.com/K3NK3/ComfyUI-K3NKImageGrab
Restart ComfyUI after installation.

### Method 3: Direct Download
1. Download `K3NKImageGrab.py` from GitHub
2. Place in `ComfyUI/custom_nodes/ComfyUI-K3NKImageGrab/`
3. Restart ComfyUI

## Tips & Best Practices

### For SVI Workflows
- Use `.latent` files saved from WanVideoEncode for best compatibility
- Set `anchor_from_start=False` to continue from where you left off
- Use `max_batch_frames=4-8` for optimal SVI performance
- Ensure consistent dimensions across all loaded files

### File Management
- Use sequential numbering in filenames for proper ordering
- Keep all files in the same directory
- Use consistent file extensions
- The node will skip files it cannot load and continue with others

### Performance
- Loading `.latent` files is faster than encoding images
- Use `frame_stride` to skip processing unnecessary files
- `max_batch_frames` limits memory usage in downstream nodes

## Troubleshooting

### "not enough values to unpack" error
- Ensure you're using the latest version
- Check that `.latent` files are in correct WanVideo format
- Verify tensor shapes in console output

### Anchor frame not as expected
- Check `anchor_from_start` setting
- Verify file numbering sequence
- Check console for which file is being used

### No frames loaded
- Verify `directory_path` is correct
- Check `file_extensions` includes your file type
- Ensure files have numeric sequences in names

### VAE encoding issues
- Connect a VAE node for image encoding
- For `.latent` files, VAE is not required
- Check image dimensions are compatible with VAE

## Version History
- v2.0: WanVideo 5D format support, SVI compatibility, anchor/batch separation
- v1.5: `.latent` file support, sequential numbering detection
- v1.0: Initial release with basic image loading

**Note**: This node is optimized for WanVideoWrapper SVI workflows. For standard image workflows, consider using the original BoyoNodes implementation.
