# K3NK Image Grab
Advanced ComfyUI node for loading and processing images/latent files with full WanVideoWrapper SVI compatibility.
This node is specifically designed for video continuation workflows with **WanVideoWrapper SVI nodes**, allowing seamless loading of saved `.latent` files and proper formatting for temporal consistency.

**Credit:** This node is a modified version of the original BoyoNodes by DragonDiffusion (https://github.com/DragonDiffusionbyBoyo/Boyonodes). Thanks to the author for the original implementation.

## Features
- **Multi-format support**: Loads `.latent` files (safetensors, pickle, ComfyUI SaveLatent format), plus standard images
- **WanVideoWrapper 5D format**: Outputs tensors in `[1, 16, T, H, W]` format compatible with WanVideoWrapper SVI nodes
- **Dual output system**: Separate `anchor_frame` and `latent_batch` outputs for SVI workflows
- **Flexible file selection**: Grab N frames with configurable stride and ordering
- **Independent anchor control**: Anchor frame selection independent of batch processing
- **Frame group selection**: Select specific frame groups (4-frame blocks) from loaded batches
- **Sequential numbering**: Detects file order by numeric sequence in filenames
- **Improved anchor selection**: New `anchor_frame_index` parameter for precise frame choice within anchor file
- **Enhanced frame trimming**: Added `batch_end_frame` parameter to remove frame groups from the end of the batch
- **Latent stride control**: `latent_frame_stride` toggles whether `frame_stride` applies to frames inside `.latent` files

## Important: WanVideoWrapper Frame Groups
WanVideoWrapper processes frames in groups of 4 (frame groups). This affects how parameters work:
- `batch_start_frame`: Starting **frame group** index (not individual frame)
- `batch_end_frame`: **Frame groups** to remove from END (not individual frames)
- Each frame group contains 4 frames
- Example: `batch_start_frame=1` skips the first 4 frames
- Example: `batch_end_frame=2` removes the last 8 frames (2 groups × 4 frames)

## Inputs
| Name | Type | Description |
|------|------|-------------|
| `directory_path` | STRING | Path to folder containing frames/latent files |
| `num_images` | INT | Number of files to process (1-10000) |
| `frame_stride` | INT | Number of files to skip between selections (0-10000). Applies to images AND latent frames if `latent_frame_stride=True` |
| `reverse_order` | BOOLEAN | Reverse the order of selected files in output |
| `reverse_logic` | BOOLEAN | Start selection from oldest (lowest number) instead of newest |
| `max_batch_frames` | INT | Maximum frames in latent_batch (0=all frames, has priority over `batch_end_frame`) |
| `batch_start_frame` | INT | Starting **frame group** index for latent_batch output (0=first group, 1=second group, etc.) |
| `batch_end_frame` | INT | **Frame groups** to remove from END of batch (0=keep all, 1=remove last group, 2=remove last 2 groups, etc.) |
| `anchor_from_start` | BOOLEAN | True: first frame of lowest-number file, False: last frame of highest-number file |
| `anchor_frame_index` | INT | Frame index to use as anchor (0=first/last depending on `anchor_from_start`, 1=second/second-to-last, etc.) |
| `latent_frame_stride` | BOOLEAN | Apply `frame_stride` to latent files (skip frames inside `.latent` files) |
| `file_extensions` | STRING (optional) | File extensions to load (default: jpg,jpeg,png,bmp,tiff,webp,latent) |
| `vae` | VAE (optional) | VAE for encoding images to latents |

## Outputs
| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Image batch for preview (black placeholders for .latent files) |
| `latent_batch` | LATENT | All frames in WanVideoWrapper format `[1, 16, T, H, W]` for `prev_samples` |
| `anchor_frame` | LATENT | Single frame in WanVideoWrapper format `[1, 16, 1, H, W]` for `anchor_frame` |
| `filenames` | STRING | Names of processed files |
| `full_paths` | STRING | Full paths of processed files |
| `timestamp` | FLOAT | Latest modification timestamp |

## Key Workflows

### Video Continuation with SVI
Saved .latent files → K3NK Image Grab → WanVideoWrapper SVI nodes → Continue generation
- Load previously saved `.latent` files from SaveLatent node
- Automatically converts to WanVideoWrapper 5D format
- Provides proper `anchor_frame` and `prev_samples` for SVI

### Anchor Frame Strategies
- `anchor_from_start=True`: Select from lowest-number .latent file
  - `anchor_frame_index=0`: First frame of file
  - `anchor_frame_index=1`: Second frame of file
  - `anchor_frame_index=N`: (N+1)th frame of file
- `anchor_from_start=False`: Select from highest-number .latent file
  - `anchor_frame_index=0`: Last frame of file
  - `anchor_frame_index=1`: Second-to-last frame of file
  - `anchor_frame_index=N`: (N+1)th frame from the end

### Frame Group Selection Examples (WanVideoWrapper 4-frame groups)
| Goal | Settings | Explanation |
|------|----------|-------------|
| Last 4 frames (1 group) | `max_batch_frames=4`, `batch_start_frame=0` | Takes last 4 frames as one group |
| Groups 2-4 (12 frames) | `batch_start_frame=1`, `max_batch_frames=12` | Skips first group (4 frames), takes next 3 groups |
| All except last group | `batch_start_frame=0`, `batch_end_frame=1` | Removes last 4 frames |
| Groups 3-5 only | `batch_start_frame=2`, `batch_end_frame=2`, `max_batch_frames=12` | Takes groups 3,4,5 (12 frames) |
| Skip every other file | `frame_stride=1` | Normal file skipping |
| Start from oldest | `reverse_logic=True` | Selection logic reversed |

### File Naming Convention
The node detects numeric sequences in filenames:
- `0203_00138_.latent` → number `138`
- `frame_050.png` → number `50`
- `output_100.jpg` → number `100`
Oldest = lowest number, Newest = highest number

## Example Workflows

### 1. Continue from Specific Frame in Latest File
Settings:
- anchor_from_start: False
- anchor_frame_index: 2 (third-to-last frame)
- reverse_logic: False (start from newest)
- num_images: 2
- max_batch_frames: 4 (one WanVideoWrapper group)
Result:
- anchor_frame: Third-to-last frame of newest .latent file
- latent_batch: Last 4 frames from 2 newest files (1 frame group)

### 2. Start from Middle Groups
Settings:
- anchor_from_start: True
- anchor_frame_index: 10 (11th frame)
- reverse_logic: True (start from oldest)
- num_images: 1
- batch_start_frame: 2 (skip first 8 frames)
- batch_end_frame: 1 (remove last 4 frames)
Result:
- anchor_frame: 11th frame of oldest .latent file
- latent_batch: Groups 3 onward, excluding last group

### 3. Trimmed Middle Section with Group Removal
Settings:
- anchor_from_start: False
- anchor_frame_index: 0 (last frame)
- reverse_logic: False
- num_images: 3
- batch_start_frame: 2 (skip first 8 frames)
- batch_end_frame: 2 (remove last 8 frames)
- max_batch_frames: 16 (4 groups maximum)
Result:
- anchor_frame: Last frame of newest file
- latent_batch: Groups 3 onward from 3 newest files, removing last 2 groups, up to max 4 groups

### 4. Stride Inside Latent Files
Settings:
- frame_stride: 1
- latent_frame_stride: True
- num_images: 2
Result:
- Skips every other frame BOTH between files AND inside .latent files
- Useful for downsampling temporal resolution

## Technical Details

### Latent File Support
- **Safetensors format**: Looks for `latent_tensor` or `samples` keys
- **ComfyUI SaveLatent**: Supports JSON format with metadata
- **PyTorch pickle**: Direct tensor or dictionary loading
- **WanVideoWrapper format**: Auto-converts to `[1, 16, T, H, W]`

### Output Format
- `latent_batch`: `[1, 16, total_frames, height//8, width//8]`
- `anchor_frame`: `[1, 16, 1, height//8, width//8]`
- Compatible with WanVideoWrapper SVI `anchor_frame` and `prev_samples` inputs

### Processing Order
1. Anchor frame: Selected independently based on `anchor_from_start` and `anchor_frame_index`
2. File selection: Based on `num_images`, `frame_stride`, `reverse_logic`
3. Frame concatenation: Temporal frames concatenated with newest files first
4. Frame group selection: Applied via `batch_start_frame`, `max_batch_frames`, and `batch_end_frame`

### Priority Rules
1. `max_batch_frames > 0`: Limits total frames (has highest priority)
2. `batch_start_frame`: Sets starting frame group (multiplied by 4 for frames)
3. `batch_end_frame`: Removes frame groups from end (multiplied by 4 for frames, applied after `max_batch_frames` if both set)

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
- Use `.latent` files saved from WanVideoWrapper Encode for best compatibility
- Set `anchor_from_start=False` and `anchor_frame_index=0` to continue from where you left off
- Use `max_batch_frames=4-8` for optimal SVI performance (1-2 frame groups)
- Ensure consistent dimensions across all loaded files

### Anchor Frame Selection
- `anchor_frame_index` allows precise control over which frame to use as anchor
- When continuing generation, use the last frame of the previous sequence
- When starting a new shot, choose a frame from the middle for better temporal coherence

### Frame Group Management
- WanVideoWrapper processes frames in groups of 4
- `batch_start_frame=1` skips the first 4 frames (entire first group)
- `batch_end_frame=2` removes the last 8 frames (2 groups)
- Always work with multiples of 4 for clean SVI integration

### File Management
- Use sequential numbering in filenames for proper ordering
- Keep all files in the same directory
- Use consistent file extensions
- The node will skip files it cannot load and continue with others

### Performance
- Loading `.latent` files is faster than encoding images
- Use `frame_stride` to skip processing unnecessary files
- `max_batch_frames` limits memory usage in downstream nodes
- `batch_end_frame` helps remove unwanted frame groups without re-saving

## Troubleshooting

### "not enough values to unpack" error
- Ensure you're using the latest version
- Check that `.latent` files are in correct WanVideoWrapper format
- Verify tensor shapes in console output

### Anchor frame not as expected
- Check `anchor_from_start` and `anchor_frame_index` settings
- Verify file numbering sequence
- Check console for which file and frame index is being used

### No frames loaded
- Verify `directory_path` is correct
- Check `file_extensions` includes your file type
- Ensure files have numeric sequences in names

### Frame selection issues
- Remember WanVideoWrapper uses 4-frame groups
- `batch_start_frame` and `batch_end_frame` work with groups, not individual frames
- Set `max_batch_frames=0` to disable frame limit
- Example: To skip first 8 frames, use `batch_start_frame=2`

### VAE encoding issues
- Connect a VAE node for image encoding
- For `.latent` files, VAE is not required
- Check image dimensions are compatible with VAE

## Version History
- v2.2: Updated for WanVideoWrapper frame groups (4-frame blocks). Clarified `batch_start_frame` and `batch_end_frame` as frame group indices.
- v2.1: Added `anchor_frame_index`, `batch_end_frame`, `latent_frame_stride` parameters
- v2.0: WanVideoWrapper 5D format support, SVI compatibility, anchor/batch separation
- v1.5: `.latent` file support, sequential numbering detection
- v1.0: Initial release with basic image loading

**Note**: This node is optimized for WanVideoWrapper SVI workflows. For standard image workflows, consider using the original BoyoNodes implementation.
