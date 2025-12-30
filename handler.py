import runpod
import os
import torch
import base64
import safetensors.torch
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from diffusers.utils import load_image

# --- 1. Startup: Official 4-Bit loading logic ---
device = torch.device("cuda")
cache_dir = "/runpod-volume"
uri_base = "Qwen/Qwen-Image-Edit-2509"
uri_lora = "huawei-bayerlab/windowseat-reflection-removal-v1-0"

if not os.path.exists(cache_dir):
    raise RuntimeError(f"❌ Network volume not found at {cache_dir}!")

print(f"🚀 Loading Network Components (4-bit mode)...")

def fetch_state_dict(pretrained_model_name_or_path_or_dict, weight_name, subfolder=None):
    file_path = hf_hub_download(pretrained_model_name_or_path_or_dict, weight_name, subfolder=subfolder, cache_dir=cache_dir)
    return safetensors.torch.load_file(file_path)

# 1a. Load VAE
vae = AutoencoderKLQwenImage.from_pretrained(uri_base, subfolder="vae", torch_dtype=torch.bfloat16, device_map=device, cache_dir=cache_dir)
vae.to(device, dtype=torch.bfloat16)

# 1b. Load Transformer in 4-bit (NF4)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(uri_base, subfolder="transformer", torch_dtype=torch.bfloat16, quantization_config=nf4_config, device_map=device, cache_dir=cache_dir)

# 1c. Load LoRA
lora_config = LoraConfig.from_pretrained(uri_lora, subfolder="transformer_lora", cache_dir=cache_dir)
transformer.add_adapter(lora_config)
state_dict = fetch_state_dict(uri_lora, "pytorch_lora_weights.safetensors", subfolder="transformer_lora")
transformer.load_state_dict(state_dict, strict=False)

# 1d. Load Text Embeddings
embeds_dict = fetch_state_dict(uri_lora, "state_dict.safetensors", subfolder="text_embeddings")
for k in embeds_dict:
    embeds_dict[k] = embeds_dict[k].to(device=device, dtype=torch.bfloat16)

print("✅ Worker Loaded with High-Res Tiling Support")

# --- 2. Inference & Flow Matching ---

def encode_img(image_tensor):
    out = vae.encode(image_tensor.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=out.device, dtype=out.dtype)).view(1, vae.config.z_dim, 1, 1, 1)
    return (out - latents_mean) * latents_std_inv

def decode_latents(latents):
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    return out.sample[:, :, 0]

def flow_step(model_input):
    B = model_input.shape[0]
    C, H, W = model_input.shape[1], model_input.shape[3], model_input.shape[4]
    
    prompt_embeds = embeds_dict["prompt_embeds"].expand(B, -1, -1)
    prompt_mask = (embeds_dict["prompt_mask"] > 0).expand(B, -1)

    packed_input = QwenImageEditPipeline._pack_latents(model_input[:, :, 0], B, C, H, W).to(torch.bfloat16)
    timestep = torch.full((B,), 499.0 / 1000.0, device=device, dtype=torch.bfloat16)
    img_shapes = [[(1, H // 2, W // 2)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # Check for attention_kwargs (from official windowseat.py)
    if getattr(transformer, "attention_kwargs", None) is None:
        attention_kwargs = {}
    else:
        attention_kwargs = transformer.attention_kwargs

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    # Dynamic VAE scale factor (from official windowseat.py)
    temperal_downsample = vae.config.get("temperal_downsample", None)
    if temperal_downsample is not None:
        vae_scale_factor = 2 ** len(temperal_downsample)
    else:
        vae_scale_factor = 8
    
    model_pred = QwenImageEditPipeline._unpack_latents(model_pred, height=H * vae_scale_factor, width=W * vae_scale_factor, vae_scale_factor=vae_scale_factor)
    return model_input.to(vae.dtype) - model_pred.to(vae.dtype)

# --- 3. Tiling & Stitching Engine ---

def get_tile_starts(size, tile_size, overlap):
    if size <= tile_size: return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size: starts.append(size - tile_size)
    return starts

def accumulate_tile(tile_out, x, y, tile_size, acc, wsum):
    """Helper to validate, resize, and accumulate a single tile with triangular weighting."""
    h, w = tile_out.shape[-2:]
    if (h != tile_size) or (w != tile_size):
        tile_out = TF.resize(tile_out, [tile_size, tile_size], antialias=True)
        h, w = tile_size, tile_size
    
    # Triangular window for smooth blending
    wx = 1 - (2 * torch.arange(w, device=device).float() / max(w - 1, 1) - 1).abs()
    wy = 1 - (2 * torch.arange(h, device=device).float() / max(h - 1, 1) - 1).abs()
    tile_mask = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
    
    acc[:, y:y+h, x:x+w] += tile_out * tile_mask
    wsum[y:y+h, x:x+w] += tile_mask

@torch.no_grad()  # From official windowseat.py - prevents gradient accumulation
def process_high_res(pil_img, max_res=1280): # Default to 1280 if not specified
    W_orig, H_orig = pil_img.size
    
    # --- Resolution Cap Logic ---
    if max(W_orig, H_orig) > max_res:
        scale = max_res / max(W_orig, H_orig)
        new_W, new_H = int(W_orig * scale), int(H_orig * scale)
        pil_img = pil_img.resize((new_W, new_H), Image.LANCZOS)
        print(f"📏 Downscaled to {max_res}px limit: {W_orig}x{H_orig} → {new_W}x{new_H}")
    else:
        print(f"🖼️ Processing at original resolution: {W_orig}x{H_orig} (Limit: {max_res}px)")
    # --------------------------

    W_working, H_working = pil_img.size
    tile_size = 768
    overlap = 128
    
    # Ensure image is large enough for at least one tile
    if W_working < tile_size or H_working < tile_size:
        scale = max(tile_size/W_working, tile_size/H_working)
        pil_img = pil_img.resize((int(W_working*scale)+1, int(H_working*scale)+1), Image.LANCZOS)
    
    W, H = pil_img.size
    img_tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device=device, dtype=torch.bfloat16) * 2.0 - 1.0 # [1, 3, H, W]
    
    xs = get_tile_starts(W, tile_size, overlap)
    ys = get_tile_starts(H, tile_size, overlap)
    
    acc = torch.zeros((3, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((H, W), device=device, dtype=torch.float32)
    
    total_tiles = len(xs) * len(ys)
    print(f"🧩 Processing {total_tiles} tiles...")

    # Dynamic batch size based on available VRAM (conservative for 4-bit)
    batch_size = 4 if total_tiles > 10 else 2
    
    batch_tiles = []
    batch_coords = []
    tiles_processed = 0
    
    for y0 in ys:
        for x0 in xs:
            tile_tensor = img_tensor[:, :, y0:y0+tile_size, x0:x0+tile_size]
            batch_tiles.append(tile_tensor)
            batch_coords.append((x0, y0))
            
            # Process in batches
            if len(batch_tiles) >= batch_size:
                try:
                    inp = torch.cat(batch_tiles, dim=0)
                    latents = encode_img(inp)
                    refined = flow_step(latents)
                    out = (decode_latents(refined) + 1.0) / 2.0
                        
                    for i in range(len(batch_tiles)):
                        x, y = batch_coords[i]
                        accumulate_tile(out[i], x, y, tile_size, acc, wsum)
                    
                    tiles_processed += len(batch_tiles)
                    if tiles_processed % 10 == 0:
                        print(f"   Processed {tiles_processed}/{total_tiles} tiles")
                except Exception as e:
                    print(f"⚠️ Batch processing error: {e}, falling back to single-tile mode")
                    # Fallback: process tiles individually
                    for tile, (x, y) in zip(batch_tiles, batch_coords):
                        latents = encode_img(tile)
                        refined = flow_step(latents)
                        out = (decode_latents(refined) + 1.0) / 2.0
                        accumulate_tile(out[0], x, y, tile_size, acc, wsum)
                        tiles_processed += 1
                
                batch_tiles = []
                batch_coords = []

    # Final batch
    if batch_tiles:
        try:
            inp = torch.cat(batch_tiles, dim=0)
            latents = encode_img(inp)
            refined = flow_step(latents)
            out = (decode_latents(refined) + 1.0) / 2.0
            for i in range(len(batch_tiles)):
                x, y = batch_coords[i]
                accumulate_tile(out[i], x, y, tile_size, acc, wsum)
        except Exception as e:
            print(f"⚠️ Final batch error: {e}")
            for tile, (x, y) in zip(batch_tiles, batch_coords):
                latents = encode_img(tile)
                refined = flow_step(latents)
                out = (decode_latents(refined) + 1.0) / 2.0
                accumulate_tile(out[0], x, y, tile_size, acc, wsum)

    print(f"✅ All {total_tiles} tiles processed, stitching final image...")
    stitched = (acc / wsum.clamp_min(1e-6)).clamp(0, 1)
    final_pil = TF.to_pil_image(stitched.cpu())
    
    # Return at processed resolution (NOT upscaled back to original if we capped it)
    return final_pil

# --- 4. Main Handler ---

def handler(event):
    try:
        data = event["input"]
        image_url = data.get("image_url")
        if not image_url: return {"error": "Missing 'image_url'"}

        # Determine Max Resolution based on Plan
        plan_title = data.get("title", "free") # Default to free
        
        if plan_title == "pro_plus":
            max_res = 4096 # 4K
        elif plan_title == "pro":
            max_res = 2560 # 2K
        else: 
            # "free" or unknown
            max_res = 1280 # HD
            
        print(f"📝 Plan: {plan_title} | Max Res: {max_res}px")

        input_image = load_image(image_url).convert("RGB")
        output_image = process_high_res(input_image, max_res=max_res)

        # Log VRAM usage stats
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            vram_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            vram_peak = torch.cuda.max_memory_allocated() / 1024**3   # GB
            print(f"📊 VRAM Stats: Current={vram_allocated:.2f}GB | Peak={vram_peak:.2f}GB | Reserved={vram_reserved:.2f}GB")
            torch.cuda.reset_peak_memory_stats()  # Reset for next request

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        return {"output_image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8")}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
