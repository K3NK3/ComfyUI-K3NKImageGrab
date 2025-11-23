import os
import glob
from PIL import Image, ImageOps
import torch
import numpy as np
import time
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class K3NKImageGrab:
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
                "file_extensions": ("STRING", {"default": "jpg,jpeg,png,bmp,tiff,webp"}),
                "vae": ("VAE",)  # Add VAE input for proper latent encoding
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "extra_latents", "filenames", "full_paths", "timestamp")
    FUNCTION = "grab_latest_images"
    CATEGORY = "K3NK/loaders"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def grab_latest_images(self, directory_path, num_images=2, frame_stride=5, file_extensions="jpg,jpeg,png,bmp,tiff,webp", vae=None):
        extensions = [e.strip().lower() for e in file_extensions.split(",")]
        all_files = []
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))

        if len(all_files) == 0:
            raise ValueError(f"No frames found in '{directory_path}'")

        all_files = list({f: os.path.getmtime(f) for f in all_files}.items())
        all_files.sort(key=lambda x: x[1], reverse=True)

        selected_files = []
        index = 0
        while index < len(all_files) and len(selected_files) < num_images:
            selected_files.append(all_files[index][0])
            index += frame_stride + 1

        tensors = []
        filenames, full_paths, timestamps = [], [], []

        for f in selected_files:
            img = Image.open(f)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None,]
            tensors.append(tensor)
            filenames.append(os.path.basename(f))
            full_paths.append(f)
            timestamps.append(os.path.getmtime(f))

        batch = torch.cat(tensors, dim=0)
        
        # Create proper latents for WanVideo
        if vae is not None:
            # Use VAE to encode images to proper latents
            latent = vae.encode(batch)
        else:
            # Fallback: create empty latents with correct dimensions
            batch_size = batch.shape[0]
            height = batch.shape[1] // 8
            width = batch.shape[2] // 8
            latent = torch.zeros([batch_size, 4, height, width])
            latent = {"samples": latent}
        
        return (batch, latent, "\n".join(filenames), "\n".join(full_paths), float(max(timestamps)))

NODE_CLASS_MAPPINGS = {"K3NKImageGrab": K3NKImageGrab}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKImageGrab": "K3NK Image Grab"}
print("✅ K3NK Image Grab: Loaded")
