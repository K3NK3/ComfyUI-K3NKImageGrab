import os
import glob
import json
import base64
import struct
import io
import re
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
                "num_images": ("INT", {"default": 2, "min": 1, "max": 10000}),
                "frame_stride": ("INT", {"default": 5, "min": 0, "max": 10000,
                                         "tooltip": "Number of frames to skip between selected frames (applies to images AND latent frames)"}),
                "reverse_order": ("BOOLEAN", {"default": False, 
                                              "tooltip": "Reverse the order of selected images in output"}),
                "reverse_logic": ("BOOLEAN", {"default": False, 
                                              "tooltip": "Reverse selection logic: start from oldest instead of newest"}),
                "max_batch_frames": ("INT", {"default": 0, "min": 0, "max": 1000,
                                            "tooltip": "Max frames to output in latent_batch (0 = all frames)"}),
                "batch_start_frame": ("INT", {"default": 0, "min": 0, "max": 1000,
                                             "tooltip": "Starting frame index from END (0=keep all frames, 1=remove last frame, 2=remove last 2 frames, etc.)"}),
                "anchor_from_start": ("BOOLEAN", {"default": False,
                                                 "tooltip": "True: first frame of lowest-number file\nFalse: last frame of highest-number file"}),
                "latent_frame_stride": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Apply frame_stride to latent files (skip frames inside .latent files)"})
            },
            "optional": {
                "file_extensions": ("STRING", {"default": "jpg,jpeg,png,bmp,tiff,webp,latent"}),
                "vae": ("VAE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "latent_batch", "anchor_frame", "filenames", "full_paths", "timestamp")
    FUNCTION = "grab_latest_images"
    CATEGORY = "K3NK/loaders"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()
    
    def extract_number_from_filename(self, filename):
        """Extract numeric sequence from filename"""
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    
    def load_latent_file(self, filepath):
        """Load a latent file - handles multiple formats"""
        try:
            print(f"⏳ Loading latent file: {os.path.basename(filepath)}")
            
            try:
                import safetensors.torch
                data = safetensors.torch.load_file(filepath)
                print(f"Loaded as safetensors, keys: {list(data.keys())}")
                
                if 'latent_tensor' in data:
                    tensor = data['latent_tensor']
                    print(f"Found 'latent_tensor' with shape: {tensor.shape}")
                    return tensor
                
                if 'samples' in data:
                    tensor = data['samples']
                    print(f"Found 'samples' with shape: {tensor.shape}")
                    return tensor
                
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"Using tensor '{key}' with shape: {value.shape}")
                        return value
                        
            except Exception as e:
                print(f"Safetensors load failed: {e}")
            
            try:
                data = torch.load(filepath, map_location='cpu', weights_only=False)
                print(f"Loaded as torch/pickle, type: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"Dict keys: {list(data.keys())}")
                    for key in ['latent', 'samples', 'latent_tensor']:
                        if key in data and isinstance(data[key], torch.Tensor):
                            tensor = data[key]
                            print(f"Found '{key}' with shape: {tensor.shape}")
                            return tensor
                    
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            print(f"Using tensor '{key}' with shape: {value.shape}")
                            return value
                
                elif isinstance(data, torch.Tensor):
                    print(f"Direct tensor with shape: {data.shape}")
                    return data
                    
            except Exception as e:
                print(f"Binary load failed: {e}")
            
            print(f"⚠️ Could not load latent, using fallback")
            return torch.zeros([1, 16, 1, 64, 112])
            
        except Exception as e:
            print(f"❌ Critical error: {e}")
            return torch.zeros([1, 16, 1, 64, 112])
    
    def prepare_for_wanvideo(self, latent_tensor, num_frames=1):
        """Prepare tensor for WanVideo - ensure 5D [B, C, T, H, W]"""
        print(f"Input tensor shape: {latent_tensor.shape}")
        
        if len(latent_tensor.shape) == 5:
            if latent_tensor.shape[1] == 16:
                print(f"Already in WanVideo 5D format: {latent_tensor.shape}")
                return latent_tensor
            elif latent_tensor.shape[0] == 16:
                latent_tensor = latent_tensor.unsqueeze(0)
                print(f"Added batch dimension: {latent_tensor.shape}")
                return latent_tensor
        
        elif len(latent_tensor.shape) == 4:
            if latent_tensor.shape[1] == 16:
                latent_tensor = latent_tensor.unsqueeze(2)
                print(f"Added temporal dimension: {latent_tensor.shape}")
                return latent_tensor
            elif latent_tensor.shape[0] == 16:
                if latent_tensor.shape[1] == num_frames:
                    latent_tensor = latent_tensor.unsqueeze(0)
                    print(f"Added batch dimension: {latent_tensor.shape}")
                    return latent_tensor
                else:
                    latent_tensor = latent_tensor.unsqueeze(0).unsqueeze(2)
                    print(f"Added batch and temporal dimensions: {latent_tensor.shape}")
                    return latent_tensor
        
        elif len(latent_tensor.shape) == 3:
            latent_tensor = latent_tensor.unsqueeze(0).unsqueeze(2)
            print(f"Added batch and temporal dimensions: {latent_tensor.shape}")
            return latent_tensor
        
        print(f"⚠️ Could not determine format, using default 5D")
        return torch.zeros([1, 16, num_frames, 64, 112])
    
    def grab_latest_images(self, directory_path, num_images=2, frame_stride=5, 
                          reverse_order=False, reverse_logic=False, 
                          max_batch_frames=0, batch_start_frame=0,
                          anchor_from_start=False, latent_frame_stride=True,
                          file_extensions="jpg,jpeg,png,bmp,tiff,webp,latent", 
                          vae=None):
        extensions = [e.strip().lower() for e in file_extensions.split(",")]
        all_files = []
        
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in '{directory_path}'")
        
        print(f"📁 Found {len(all_files)} files in directory")
        
        # Extraer información de TODOS los archivos
        all_files_info = []
        for f in all_files:
            filename = os.path.basename(f)
            number = self.extract_number_from_filename(filename)
            all_files_info.append((f, filename, number, os.path.getmtime(f)))
        
        # Ordenar por número para identificar lowest/highest
        all_files_info.sort(key=lambda x: x[2])  # Sort by number ascending
        
        lowest_file = all_files_info[0] if all_files_info else None
        highest_file = all_files_info[-1] if all_files_info else None
        
        print(f"📊 File numbers: {lowest_file[2] if lowest_file else 'N/A'} to {highest_file[2] if highest_file else 'N/A'}")
        
        # ===== ANCHOR FRAME =====
        print(f"\n🎯 ANCHOR FRAME:")
        
        if anchor_from_start and lowest_file:
            # Primer frame del archivo más viejo (número más bajo)
            anchor_file = lowest_file
            print(f"  anchor_from_start=True → First frame of LOWEST number: {anchor_file[1]}")
            
            if anchor_file[0].lower().endswith('.latent'):
                anchor_latent = self.load_latent_file(anchor_file[0])
                anchor_tensor = self.prepare_for_wanvideo(anchor_latent)
                anchor_frame_output = {"samples": anchor_tensor[:, :, 0:1, :, :]}
            else:
                anchor_frame_output = {"samples": torch.zeros([1, 16, 1, 64, 112])}
            
        elif not anchor_from_start and highest_file:
            # Último frame del archivo más nuevo (número más alto)
            anchor_file = highest_file
            print(f"  anchor_from_start=False → Last frame of HIGHEST number: {anchor_file[1]}")
            
            if anchor_file[0].lower().endswith('.latent'):
                anchor_latent = self.load_latent_file(anchor_file[0])
                anchor_tensor = self.prepare_for_wanvideo(anchor_latent)
                last_frame_idx = anchor_tensor.shape[2] - 1
                anchor_frame_output = {"samples": anchor_tensor[:, :, last_frame_idx:last_frame_idx+1, :, :]}
            else:
                anchor_frame_output = {"samples": torch.zeros([1, 16, 1, 64, 112])}
        
        else:
            anchor_frame_output = {"samples": torch.zeros([1, 16, 1, 64, 112])}
        
        print(f"  Anchor shape: {anchor_frame_output['samples'].shape}")
        
        # ===== LATENT BATCH (selección normal) =====
        print(f"\n📦 LATENT BATCH (normal selection):")
        print(f"  frame_stride: {frame_stride} (applies to {'images AND latents' if latent_frame_stride else 'images only'})")
        
        # Ordenar para selección basado en reverse_logic
        if reverse_logic:
            files_for_selection = sorted(all_files_info, key=lambda x: x[2])  # Números ascendentes
            print(f"  reverse_logic=True → Select from OLDEST files first")
        else:
            files_for_selection = sorted(all_files_info, key=lambda x: x[2], reverse=True)  # Números descendentes
            print(f"  reverse_logic=False → Select from NEWEST files first")
        
        # Seleccionar archivos
        selected_files_info = []
        index = 0
        
        while index < len(files_for_selection) and len(selected_files_info) < num_images:
            selected_files_info.append(files_for_selection[index])
            index += frame_stride + 1
        
        print(f"  Selected {len(selected_files_info)} files with stride {frame_stride}")
        for i, (_, name, num, _) in enumerate(selected_files_info):
            print(f"    {i}: {name} (number: {num})")
        
        if reverse_order:
            selected_files_info.reverse()
            print(f"  reverse_order=True → Reversed selection order")
        
        # Procesar archivos para latent_batch
        wanvideo_tensors = []
        image_tensors = []
        filenames, full_paths, timestamps = [], [], []
        
        for f, filename, file_number, file_time in selected_files_info:
            file_ext = os.path.splitext(f)[1].lower()
            print(f"\n  Processing for batch: {filename} (number: {file_number})")
            
            if file_ext == ".latent":
                latent_tensor = self.load_latent_file(f)
                wanvideo_tensor = self.prepare_for_wanvideo(latent_tensor)
                
                # Aplicar frame_stride a los frames internos del .latent
                if latent_frame_stride and frame_stride > 0 and wanvideo_tensor.shape[2] > 1:
                    total_frames = wanvideo_tensor.shape[2]
                    selected_frames_indices = list(range(0, total_frames, frame_stride + 1))
                    
                    if len(selected_frames_indices) > 0:
                        print(f"    Applying frame_stride={frame_stride} inside .latent file")
                        print(f"    Original frames: {total_frames}, Selected frames: {len(selected_frames_indices)}")
                        
                        # Crear nuevo tensor con solo los frames seleccionados
                        selected_frames = []
                        for idx in selected_frames_indices:
                            selected_frames.append(wanvideo_tensor[:, :, idx:idx+1, :, :])
                        
                        if selected_frames:
                            wanvideo_tensor = torch.cat(selected_frames, dim=2)
                            print(f"    New tensor shape: {wanvideo_tensor.shape}")
                
                wanvideo_tensors.append(wanvideo_tensor)
                
                # Placeholder para imagen
                if len(wanvideo_tensor.shape) == 5:
                    _, _, _, h_latent, w_latent = wanvideo_tensor.shape
                else:
                    h_latent, w_latent = 64, 112
                
                height = h_latent * 8
                width = w_latent * 8
                placeholder = torch.zeros((1, height, width, 3), dtype=torch.float32)
                image_tensors.append(placeholder)
            
            else:
                try:
                    img = Image.open(f)
                    img = ImageOps.exif_transpose(img)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    arr = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr)[None,]
                    image_tensors.append(tensor)
                    
                    height = tensor.shape[1] // 8
                    width = tensor.shape[2] // 8
                    wanvideo_tensor = torch.zeros([1, 16, 1, height, width])
                    wanvideo_tensors.append(wanvideo_tensor)
                    
                except Exception as img_error:
                    print(f"  Error loading image: {img_error}")
                    placeholder = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                    image_tensors.append(placeholder)
                    wanvideo_tensors.append(torch.zeros([1, 16, 1, 64, 112]))
            
            filenames.append(filename)
            full_paths.append(f)
            timestamps.append(file_time)
        
        if not wanvideo_tensors:
            raise ValueError("No valid frames loaded for batch")
        
        # Asegurar batch size = 1
        for i, tensor in enumerate(wanvideo_tensors):
            if tensor.shape[0] != 1:
                wanvideo_tensors[i] = tensor[0:1]
        
        # ===== CONCATENACIÓN CORREGIDA =====
        print(f"\n🔧 CONCATENATING FRAMES (FIXED for newest-first):")
        
        # Invertir el orden de los tensores si reverse_logic=False (selección desde los más nuevos)
        if not reverse_logic and not reverse_order:
            print(f"  reverse_logic=False → Reversing tensor order for concatenation")
            wanvideo_tensors.reverse()
        
        temporal_slices = []
        total_frames_list = []
        
        for i, tensor in enumerate(wanvideo_tensors):
            t_frames = tensor.shape[2]
            total_frames_list.append(t_frames)
            print(f"  Tensor {i}: {t_frames} frames")
            
            for t in range(t_frames):
                temporal_slice = tensor[:, :, t:t+1, :, :]
                temporal_slices.append(temporal_slice)
        
        # Concatenar todos los frames
        if temporal_slices:
            wanvideo_batch = torch.cat(temporal_slices, dim=2)
        else:
            wanvideo_batch = torch.zeros([1, 16, 1, 64, 112])
        
        total_frames = wanvideo_batch.shape[2]
        print(f"  Total frames concatenated: {total_frames}")
        print(f"  Batch shape before selection: {wanvideo_batch.shape}")
        
        # ===== APLICAR SELECCIÓN DE FRAMES (INVERTIDA - desde el FINAL) =====
        print(f"\n✂️ APPLYING FRAME SELECTION (INVERTED - from END):")
        print(f"  batch_start_frame: {batch_start_frame} (frames to remove from END)")
        print(f"  max_batch_frames: {max_batch_frames} (0 = all frames)")
        
        # Calcular el frame de inicio desde el FINAL
        if batch_start_frame > 0:
            # batch_start_frame indica cuántos frames eliminar desde el final
            frames_to_remove = min(batch_start_frame, total_frames)
            end_frame = total_frames - frames_to_remove
            start_frame = 0  # Siempre empezamos desde el principio
            
            print(f"  Removing {frames_to_remove} frames from the END")
            print(f"  Keeping frames 0 to {end_frame} (total {end_frame} frames)")
        else:
            # Si es 0, mantener todos los frames
            start_frame = 0
            end_frame = total_frames
            print(f"  batch_start_frame=0 → Keeping ALL frames")
        
        # Ahora aplicar max_batch_frames (si está configurado)
        if max_batch_frames > 0:
            # Ajustar para NO exceder max_batch_frames
            available_frames = end_frame - start_frame
            if available_frames > max_batch_frames:
                # Para mantener el comportamiento de tomar los últimos N frames
                start_frame = max(0, end_frame - max_batch_frames)
                print(f"  max_batch_frames={max_batch_frames} → Adjusting to frames {start_frame}:{end_frame}")
        
        # Finalmente, calcular cuántos frames tomar
        frames_to_take = end_frame - start_frame
        
        print(f"  Final selection: frames {start_frame}:{end_frame} ({frames_to_take} frames)")
        
        if frames_to_take > 0:
            selected_batch = wanvideo_batch[:, :, start_frame:end_frame, :, :]
        else:
            # Si no quedan frames, tomar al menos 1 frame
            if total_frames > 0:
                selected_batch = wanvideo_batch[:, :, 0:1, :, :]
                print(f"  Warning: No frames left after removal, keeping first frame")
            else:
                selected_batch = wanvideo_batch[:, :, -1:, :, :]
        
        print(f"  Selected batch shape: {selected_batch.shape}")
        
        # Crear output
        latent_batch_output = {"samples": selected_batch}
        
        # Concatenar imágenes para preview
        if image_tensors:
            image_batch = torch.cat(image_tensors, dim=0)
        else:
            image_batch = torch.zeros([len(wanvideo_tensors), 512, 512, 3])
        
        print(f"\n✅ FINAL:")
        print(f"  anchor_frame: {anchor_frame_output['samples'].shape}")
        print(f"  latent_batch: {latent_batch_output['samples'].shape} ({selected_batch.shape[2]} frames)")
        print(f"  image: {image_batch.shape}")
        print(f"  Applied frame_stride to latents: {latent_frame_stride}")
        
        return (image_batch, latent_batch_output, anchor_frame_output, 
                "\n".join(filenames), "\n".join(full_paths), float(max(timestamps) if timestamps else time.time()))

NODE_CLASS_MAPPINGS = {"K3NKImageGrab": K3NKImageGrab}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKImageGrab": "K3NK Image Grab"}
print("✅ K3NK Image Grab (Inverted batch_start_frame): Loaded")
print("   - latent_batch ahora empieza desde los frames del .latent más nuevo")
print("   - Concatenación corregida para poner frames nuevos al FINAL")
print("   - batch_start_frame ahora funciona desde el FINAL:")
print("     0 = mantener todos los frames")
print("     1 = eliminar el último frame")
print("     2 = eliminar los últimos 2 frames")
print("     ... etc.")
print("   - frame_stride ahora aplica también a frames dentro de archivos .latent")
