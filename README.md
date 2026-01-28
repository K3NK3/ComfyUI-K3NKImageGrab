# K3NK Image Loader with Blending
Advanced ComfyUI node for loading image sequences with intelligent frame blending for seamless video transitions.

This node is specifically designed for **multi-sequence video workflows**, automatically blending overlapping frames between sequences to create smooth transitions without visible cuts.

## Features
- **Intelligent sequence detection**: Automatically calculates complete sequences and remaining frames
- **Adaptive frame blending**: Seamlessly blends overlapping frames between sequences using linear interpolation
- **Incomplete sequence handling**: Properly processes remaining frames even when they don't form a complete sequence
- **Flexible file patterns**: Support for multiple image formats via glob patterns
- **Sequential numbering**: Detects and sorts files by numeric sequence in filenames
- **Frame stride compatibility**: Works seamlessly with interpolated sequences (2x, 4x, etc.)
- **Debug output**: Detailed console logging for monitoring blend operations

## Key Concepts

### Sequence-Based Processing
The node divides your image directory into sequences of N frames (e.g., 81 frames per sequence). When you have multiple sequences, it automatically blends the overlapping frames at sequence boundaries to create smooth transitions.

### Frame Blending Mechanism
- **Overlap frames**: The last N frames of one sequence blend with the first N frames of the next
- **Linear interpolation**: Uses progressive alpha blending (e.g., 20%, 40%, 60%, 80% across 4 frames)
- **Frame replacement**: Blended frames replace the original overlapping frames (no duplication)
- **Incomplete sequences**: Automatically handles leftover frames with adaptive blending

### Output Frame Count
**Important**: The output will have FEWER frames than input due to blending:
- Input: 162 frames (2 sequences of 81)
- Overlap: 4 frames
- Output: 162 - 4 = **158 frames** (blending replaces, doesn't add)
- Formula: `total_frames - (overlap_frames × number_of_transitions)`

## Inputs
| Name | Type | Description |
|------|------|-------------|
| `directory_path` | STRING | Path to folder containing image sequence |
| `sequence_frames` | INT | Number of frames per sequence (1-10000, default: 81) |
| `overlap_frames` | INT | Number of frames to blend at sequence boundaries (1-20, default: 4) |
| `file_pattern` | STRING | Glob pattern for image files (default: *.png) |

## Outputs
| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Blended image tensor in format `[total_frames, H, W, 3]` for ComfyUI processing |

## Usage Examples

### Example 1: Standard Two-Sequence Workflow
**Scenario**: 162 frames total (2 sequences of 81 frames each)

Settings:
```
sequence_frames: 81
overlap_frames: 4
```

Processing:
- Sequence 1: Frames 0-80 (added complete)
- Sequence 2: Frames 81-161
  - Blend: Last 4 frames of Seq1 (76-79) with first 4 of Seq2 (81-84)
  - Add: Remaining frames 85-161
- **Output**: 158 frames total (162 - 4 blended frames)

### Example 2: Incomplete Final Sequence
**Scenario**: 161 frames total (81 + 80 frames)

Settings:
```
sequence_frames: 81
overlap_frames: 4
```

Processing:
- Sequence 1: Frames 0-80 (complete)
- Remaining: 80 frames (not a complete sequence)
  - Adaptive blend: Uses min(4, 80) = 4 frames for blending
  - Blend: Frames 76-79 with frames 81-84
  - Add: Remaining frames 85-160
- **Output**: 157 frames total

### Example 3: Three Sequences with Interpolation
**Scenario**: 486 frames (3 sequences × 162 interpolated frames)

Settings:
```
sequence_frames: 162  # 81 × 2 (interpolated)
overlap_frames: 8     # 4 × 2 (scaled for interpolation)
```

Processing:
- Creates 2 blend transitions (between 3 sequences)
- Total blended frames: 8 × 2 = 16 frames
- **Output**: 486 - 16 = **470 frames**

### Example 4: Single Sequence (No Blending)
**Scenario**: 81 frames total

Settings:
```
sequence_frames: 81
overlap_frames: 4
```

Processing:
- Only one complete sequence detected
- No blending needed
- **Output**: 81 frames (unchanged)

## Interpolation Guidelines

When working with interpolated sequences, scale both parameters proportionally:

| Original | 2× Interpolation | 4× Interpolation |
|----------|------------------|------------------|
| 81 frames, 4 blend | 162 frames, 8 blend | 324 frames, 16 blend |
| 121 frames, 6 blend | 242 frames, 12 blend | 484 frames, 24 blend |

**Formula**: `interpolated_value = original_value × interpolation_factor`

## File Naming Convention

The node extracts numeric sequences from filenames for proper ordering:
- `frame_0001.png` → number `1`
- `output_00138.png` → number `138`
- `render_050_final.png` → number `50`

Files are sorted by the **last number found** in the filename.

## Supported File Formats

Default pattern `*.png` includes PNG files. Customize with:
- `*.jpg` - JPEG files only
- `*.{png,jpg}` - Multiple formats (requires bash-style brace expansion support)
- `frame_*.png` - Specific prefix
- `*_final.png` - Specific suffix

The node automatically converts all images to RGB and normalizes to float32 [0-1] range.

## Technical Details

### Blending Algorithm
```python
blended_frame = frame1 × (1.0 - alpha) + frame2 × alpha
```
Where `alpha` progressively increases across overlap frames:
- Frame 1: alpha = 1/5 = 0.20 (20% new, 80% old)
- Frame 2: alpha = 2/5 = 0.40 (40% new, 60% old)
- Frame 3: alpha = 3/5 = 0.60 (60% new, 40% old)
- Frame 4: alpha = 4/5 = 0.80 (80% new, 20% old)

### Processing Order
1. **File discovery**: Glob pattern matching in directory
2. **Sequential sorting**: Extract and sort by numeric values
3. **Image loading**: Load all images, convert to RGB tensors
4. **Sequence calculation**: Determine complete sequences and remainders
5. **First sequence**: Add all frames without modification
6. **Subsequent sequences**: Blend overlap, add remaining frames
7. **Incomplete sequences**: Adaptive blending with available frames
8. **Output stacking**: Concatenate all frames into single tensor

### Memory Considerations
- All images are loaded into memory simultaneously
- Output tensor size: `[total_frames, height, width, 3]` in float32
- Example: 500 frames at 1280×768 = ~4.7GB RAM
- Consider batch processing for very long sequences

## Console Output Example
```
📁 Found 162 images
✅ Loaded 162 images
🎯 81 frames per sequence, 4 overlap frames

🔧 Processing sequences...
  Complete sequences: 2
  Remaining frames: 0

  Added first sequence: frames 0-80

  Sequence 2:
    Start index: 81
    Blending frames 77-80 with 81-84
    Added frames 85-161

📊 Final: 158 frames (original: 162)

🔍 Checking for actual blending...
   4/10 first frames were modified
```

## Best Practices

### Sequence Planning
- Plan your renders in multiples of `sequence_frames` when possible
- Account for lost frames in blending: `output = input - (overlap × transitions)`
- For 3 sequences: expect to lose `overlap_frames × 2` frames total

### Overlap Selection
- **Small overlaps (2-4 frames)**: Faster transitions, more abrupt
- **Medium overlaps (4-8 frames)**: Balanced, recommended for most cases
- **Large overlaps (8-16 frames)**: Very smooth, slower transitions
- Scale overlap proportionally with interpolation factor

### Quality Tips
- Use consistent lighting across sequences for better blending
- Avoid drastic scene changes at sequence boundaries
- Higher overlap values create smoother transitions but reduce output length
- Test with a small sequence first to verify settings

### Workflow Integration
- Use with VHS Video Combine for final video output
- Pair with frame interpolation nodes for smooth slow-motion
- Compatible with any ComfyUI node accepting IMAGE input
- Works seamlessly with AnimateDiff, Wan Video, and other video models

## Troubleshooting

### No blending visible in output
- Check console for "X/10 first frames were modified"
- Verify you have multiple sequences (more than `sequence_frames` total)
- Ensure `overlap_frames > 0`
- Check that images actually differ at sequence boundaries

### Unexpected output frame count
- Remember: blending REPLACES frames, doesn't add them
- Formula: `total_input - (overlap_frames × number_of_transitions)`
- Example: 243 frames (3 sequences) with 4 overlap = 243 - 8 = 235 output

### "No files found" error
- Verify `directory_path` is correct and absolute
- Check `file_pattern` matches your files
- Ensure files have proper extensions
- Verify file permissions

### Images in wrong order
- Check filename numbering is sequential
- The node uses the LAST number found in each filename
- Rename files if necessary: `frame_001.png`, `frame_002.png`, etc.

### Out of memory
- Reduce total frame count
- Process in smaller batches
- Lower image resolution before processing
- Use a machine with more RAM

## Performance Notes

- **Loading speed**: ~0.1-0.5s per image depending on resolution and disk speed
- **Blending speed**: Nearly instant (GPU tensor operations)
- **Bottleneck**: Usually file I/O, not computation
- **Optimization**: Use SSD storage for faster loading

## Version History
- **v1.2**: Improved incomplete sequence handling with adaptive blending
- **v1.1**: Fixed edge cases with remaining frames, enhanced debug output
- **v1.0**: Initial release with multi-sequence blending support

## Credits

Built for ComfyUI video workflows. Designed to work seamlessly with:
- WanVideoWrapper
- AnimateDiff
- VHS Video Combine
- Frame interpolation nodes

## License

MIT License - Free to use and modify

---

**Note**: This node is optimized for multi-sequence video generation workflows where smooth transitions between generated segments are critical. For single-sequence workflows or when transitions aren't needed, consider using standard image loader nodes.
