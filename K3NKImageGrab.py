import os
import glob
from PIL import Image, ImageOps
import torch
import numpy as np
import time
import sys

class K3NKImageGrab:
    """
    Standalone ComfyUI node.
    Grabs the last N frames from a directory as a single batch,
    with optional frame stride between selected frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "Directory path"}),
                "num_images": ("INT", {"default": 2, "min": 1, "max": 100}),
                "frame_stride": ("INT", {"default": 5, "min": 0, "max": 100,
                                         "tooltip": "Number of frames to skip between selected frames"})
            },
            "optional": {
                "file_extensions": ("STRING", {"default": "jpg,jpeg,png,bmp,tiff,webp"})
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "latent", "filenames", "full_paths", "timestamp")
    FUNCTION = "grab_latest_images"
    CATEGORY = "K3NK/loaders"

    @classmethod
    def IS_CHANGED(cls, directory_path, num_images=2, frame_stride=5, file_extensions="jpg,jpeg,png,bmp,tiff,webp"):
        if not os.path.exists(directory_path):
            return time.time()
        return time.time()

    def grab_latest_images(self, directory_path, num_images=2, frame_stride=5, file_extensions="jpg,jpeg,png,bmp,tiff,webp"):
        extensions = [e.strip().lower() for e in file_extensions.split(",")]
        all_files = []
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))

        if len(all_files) == 0:
            raise ValueError(f"No frames found in '{directory_path}'")

        # Sort files by modification time descending
        all_files = list({f: os.path.getmtime(f) for f in all_files}.items())
        all_files.sort(key=lambda x: x[1], reverse=True)

        # Safe selection with stride
        selected_files = []
        index = 0
        while index < len(all_files) and len(selected_files) < num_images:
            selected_files.append(all_files[index][0])
            index += frame_stride + 1

        tensors = []
        filenames, full_paths, timestamps = [], [], []
        image_sizes = []

        for f in selected_files:
            img = Image.open(f)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None,])
            filenames.append(os.path.basename(f))
            full_paths.append(f)
            timestamps.append(os.path.getmtime(f))
            image_sizes.append(img.size)

        batch = torch.cat(tensors, dim=0)
        
        # Create latent output matching image dimensions
        batch_size = batch.shape[0]
        if image_sizes:
            width, height = image_sizes[0]
            latent_width = width // 8
            latent_height = height // 8
        else:
            latent_width = 64
            latent_height = 64
            
        latent = torch.zeros([batch_size, 4, latent_height, latent_width])
        latent_output = {"samples": latent}
        
        return batch, latent_output, "\n".join(filenames), "\n".join(full_paths), float(max(timestamps))

NODE_CLASS_MAPPINGS = {
    "K3NKImageGrab": K3NKImageGrab
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3NKImageGrab": "K3NK Image Grab"
}

print("### K3NK Image Grab: Loaded ###")