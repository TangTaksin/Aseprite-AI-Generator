#!/usr/bin/env python3
"""
เซิร์ฟเวอร์ Local AI Generator สำหรับ Aseprite - Optimized for RTX 5070 Ti
การสร้าง Pixel Art ระดับมืออาชีพโดยใช้ Stable Diffusion พร้อมรองรับ LoRA และการลบพื้นหลังด้วย BiRefNet
เวอร์ชัน 2.4.1 - Pylance Fixed + Fast Boot + Fast Model Switch + Blackwell Architecture
"""

# pyright: reportMissingImports=false
# pyright: reportPrivateImportUsage=false
from diffusers.utils import logging as diffusers_logging
import logging
import os
import sys
import json
import platform
import base64
import io
import warnings
import glob
import gc
import time
import random
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union

from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from diffusers import (  # type: ignore
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from transformers import AutoModelForImageSegmentation  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image, ImageEnhance, ImageFilter  # type: ignore
import numpy as np  # type: ignore

# ปิดการแจ้งเตือนคำเตือนต่างๆ เพื่อให้ Console สะอาด
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
warnings.filterwarnings("ignore", message=".*CLIPTextModelWithProjection.*")

# ปิดคำเตือนของ Flask development server
diffusers_logging.set_verbosity_error()
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)


class PixelArtSDServer:
    """เซิร์ฟเวอร์สำหรับสร้าง Pixel Art ด้วย Stable Diffusion"""
    
    def __init__(self, default_model: Optional[str] = None):
        self.pipeline: Optional[Any] = None
        self.segmentation_model: Optional[Any] = None
        self.segmentation_processor: Optional[Any] = None
        self.model_loaded: bool = False
        self.current_model: Optional[str] = None
        self.current_lora: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache: Dict[str, Any] = {}
        self.offline_mode: bool = False
        self.default_model: Optional[str] = default_model

        is_windows = platform.system() == "Windows"
        self.use_compile: bool = False if is_windows else True

        # 🎯 RTX 5070 Ti Optimized Settings
        self.optimized_settings: Dict[str, Any] = {
            "use_bf16": torch.cuda.is_bf16_supported(),
            "use_compile": self.use_compile,
            "enable_xformers": not is_windows,
            "enable_attention_slicing": True,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",
        }

        # ตั้งค่าพื้นฐานสำหรับการสร้างภาพ Pixel Art
        self.default_settings: Dict[str, Any] = {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels",
        }

        print(f"🚀 Local AI Generator Server v2.4.1 [Pylance Fixed + Fast Switch Mode]")
        print(f"📱 อุปกรณ์ที่ใช้: {self.device}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"💾 VRAM: {props.total_memory / 1024**3:.1f}GB")
            print(f"⚡ Compute Capability: {props.major}.{props.minor}")
            print(f"🔷 BF16 Support: {torch.cuda.is_bf16_supported()}")
            self._apply_global_cuda_optimizations()

    # ─────────────────────────────────────────────────────────────────────────
    # CUDA / Hardware Optimizations
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_global_cuda_optimizations(self) -> None:
        """Apply global CUDA optimizations for RTX 5070 Ti"""
        if self.device != "cuda":
            return

        print("\n🔧 Applying RTX 5070 Ti Global Optimizations...")

        if self.optimized_settings["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
            print("   ✅ cuDNN Benchmark: Enabled")

        precision = self.optimized_settings["float32_matmul_precision"]
        try:
            torch.set_float32_matmul_precision(precision)
            print(f"   ✅ TF32 Precision: {precision}")
        except Exception:
            print(f"   ⚠️ Could not set matmul precision")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   ✅ TF32 Acceleration: Enabled")
        print("   ✅ Global CUDA optimizations applied!\n")

    def _optimize_pipeline_for_blackwell(self, pipeline: Any, model_type: str = "sdxl") -> Any:
        if self.device != "cuda":
            return pipeline

        print(f"🎯 Optimizing {model_type.upper()} pipeline for RTX 5070 Ti...")

        dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
        pipeline = pipeline.to(self.device, dtype=dtype)
        print(f"   ✅ Precision: {dtype}")
        print("   ✅ Memory Layout: contiguous (default)")

        try:
            if hasattr(pipeline, "disable_attention_slicing"):
                pipeline.disable_attention_slicing()
            print("   ✅ Attention: Native SDPA (Blackwell optimized)")
        except Exception as e:
            print(f"   ⚠️ Attention setup note: {e}")

        try:
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
            print("   ✅ VAE Slicing & Tiling: Enabled (Faster Decode)")
        except Exception:
            pass

        if self.optimized_settings["use_compile"]:
            print("   🚀 Compiling U-Net with torch.compile (First run will take 1-3 minutes!)...")
            pipeline.unet = torch.compile(
                pipeline.unet, mode="reduce-overhead", fullgraph=True, dynamic=True
            )
            print("   ✅ torch.compile: Enabled")

        if model_type == "sdxl" and hasattr(pipeline, "scheduler"):
            try:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
                print("   ✅ Scheduler: DPM++ 2M Karras")
            except Exception:
                pass

        print("   ✅ Pipeline optimization complete!\n")
        return pipeline

    # ─────────────────────────────────────────────────────────────────────────
    # ✅ Fast Model Switch - Offload to CPU Cache
    # ─────────────────────────────────────────────────────────────────────────

    def _offload_current_pipeline(self) -> None:
        """ย้ายโมเดลปัจจุบันออกจาก GPU ไปเก็บไว้ใน CPU Cache"""
        if self.pipeline is None:
            return

        print(f"   📤 Offloading '{self.current_model}' to CPU cache...")
        try:
            if self.current_lora and hasattr(self.pipeline, "unload_lora_weights"):
                print(f"   🧹 Unloading LoRA before offload: {self.current_lora}")
                self.pipeline.unload_lora_weights()
                self.current_lora = None

            self.pipeline.to("cpu")
            if self.current_model:
                self.model_cache[self.current_model] = self.pipeline

        except Exception as e:
            print(f"   ⚠️ Offload warning (will clear instead): {e}")
            if self.current_model:
                self.model_cache.pop(self.current_model, None)

        finally:
            self.pipeline = None
            self.model_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   🧹 VRAM after offload → Used: {vram_used:.2f}GB | Reserved: {vram_reserved:.2f}GB")

    # ─────────────────────────────────────────────────────────────────────────
    # Segmentation (BiRefNet)
    # ─────────────────────────────────────────────────────────────────────────

    def load_segmentation_model(self) -> bool:
        """โหลดโมเดล BiRefNet (Lazy Load)"""
        if self.segmentation_model and self.segmentation_processor:
            return True

        print("📦 Loading BiRefNet for background removal...")
        try:
            model_name = "zhengpeng7/BiRefNet"
            self.segmentation_processor = transforms.Compose(
                [
                    transforms.Resize(
                        (352, 352), interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode,
                dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            if self.segmentation_model is None:
                raise RuntimeError("Failed to initialize segmentation_model")

            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()

            if self.device == "cuda":
                try:
                    self.segmentation_model.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            if self.optimized_settings["use_compile"]:
                print("   🚀 Compiling BiRefNet...")
                self.segmentation_model = torch.compile(
                    self.segmentation_model, mode="reduce-overhead"
                )

            print("✅ BiRefNet loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error loading BiRefNet: {e}")
            return False

    def remove_background(self, pil_image: Image.Image) -> Image.Image:
        """ใช้ BiRefNet ลบพื้นหลัง"""
        if not self.load_segmentation_model():
            raise Exception("ไม่สามารถโหลดโมเดลลบพื้นหลังได้")

        print("🎭 Removing background with BiRefNet...")
        try:
            with torch.inference_mode():
                rgb_image = pil_image.convert("RGB")
                
                if self.segmentation_processor is None:
                    raise RuntimeError("segmentation_processor not initialized")
                    
                input_tensor = (
                    self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)
                )

                if self.segmentation_model is None:
                    raise RuntimeError("segmentation_model not initialized")
                    
                model_dtype = next(self.segmentation_model.parameters()).dtype
                input_tensor = input_tensor.to(dtype=model_dtype)

                outputs = self.segmentation_model(input_tensor)
                logits = outputs[0]

                mask = F.interpolate(
                    logits,
                    size=pil_image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.sigmoid(mask).squeeze()
                binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)

            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)

            print("✅ Background removal complete")
            return rgba_image

        except Exception as e:
            print(f"❌ Error during background removal: {e}")
            return pil_image.convert("RGBA")

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def image_to_base64(self, image: Image.Image) -> str:
        """แปลงภาพ PIL เป็น base64 encoded string"""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return base64.b64encode(image.tobytes()).decode()

    # ─────────────────────────────────────────────────────────────────────────
    # ✅ load_model — Fast Switch via CPU Cache
    # ─────────────────────────────────────────────────────────────────────────

    def load_model(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0") -> bool:
        """โหลดโมเดล AI พร้อม Fast Switch"""
        try:
            local_only = self.offline_mode

            # Path 1: โมเดลนี้อยู่บน GPU อยู่แล้ว
            if (
                self.model_loaded
                and self.current_model == model_name
                and self.pipeline is not None
            ):
                print(f"⚡ '{model_name}' already on GPU. Skipping load.")
                return True

            # Offload โมเดลปัจจุบันออก GPU ก่อนเสมอ
            if self.pipeline is not None:
                print(f"🔄 Model switch: '{self.current_model}' → '{model_name}'")
                self._offload_current_pipeline()

            # Path 2: มีใน CPU cache → restore ไป GPU
            if model_name in self.model_cache:
                print(f"⚡ Restoring '{model_name}' from CPU cache → GPU...")
                self.pipeline = self.model_cache[model_name]

                dtype = (
                    torch.bfloat16
                    if self.optimized_settings["use_bf16"]
                    else torch.float16
                )
                
                if self.pipeline is not None:
                    self.pipeline.to(self.device, dtype=dtype)

                self.current_model = model_name
                self.model_loaded = True
                print(f"✅ Model restored to GPU from cache!")
                return True

            # Path 3: Load จาก disk / HuggingFace
            print(f"📥 Loading model from disk: {model_name}")

            is_sdxl = "xl" in model_name.lower()
            precision = (
                torch.bfloat16
                if (self.device == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            print(f"   ✅ Precision: {precision}")

            local_model_path = os.path.join("models", model_name)
            is_local_file = os.path.isfile(local_model_path)

            if is_local_file:
                print(f"📂 Found local model: {local_model_path}")
                if is_sdxl:
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=precision,
                        local_files_only=local_only,
                    )
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        local_model_path,
                        vae=vae,
                        torch_dtype=precision,
                        use_safetensors=True,
                        config="stabilityai/stable-diffusion-xl-base-1.0",
                    )
                else:
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        local_model_path,
                        torch_dtype=precision,
                        load_safety_checker=False,
                    )
            else:
                if is_sdxl:
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=precision,
                        local_files_only=local_only,
                    )
                    self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_name,
                        vae=vae,
                        torch_dtype=precision,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )
                else:
                    self.pipeline = StableDiffusionPipeline.from_pretrained(
                        model_name,
                        torch_dtype=precision,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )

            model_type = "sdxl" if is_sdxl else "sd15"
            self.pipeline = self._optimize_pipeline_for_blackwell(
                self.pipeline, model_type
            )

            # Warmup
            if self.device == "cuda" and self.pipeline is not None:
                print("🔥 Running warmup inference...")
                try:
                    with torch.inference_mode():
                        _ = self.pipeline(
                            prompt="pixel",
                            width=512 if not is_sdxl else 1024,
                            height=512 if not is_sdxl else 1024,
                            num_inference_steps=1,
                            guidance_scale=1.0,
                        )
                    print("✅ Warmup complete - Ready for fast inference!")
                except Exception as e:
                    print(f"⚠️ Warmup skipped: {e}")

            self.model_cache[model_name] = self.pipeline
            self.current_model = model_name
            self.model_loaded = True

            print(f"✅ Model loaded and optimized: {model_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # ✅ generate_image — LoRA Smart Load
    # ─────────────────────────────────────────────────────────────────────────

    def generate_image(
        self, 
        prompt: str, 
        lora_model: Optional[str] = None, 
        lora_strength: float = 1.0, 
        **kwargs: Any
    ) -> Tuple[Image.Image, int]:
        """สร้างภาพด้วยการควบคุมสไตล์ผ่าน LoRA"""
        if not self.model_loaded or self.pipeline is None:
            raise Exception("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")

        lora_active = lora_model and lora_model.lower() not in ["none", ""]

        try:
            print(f"🎨 Generating: '{prompt[:50]}...'")
            pipeline_kwargs: Dict[str, Any] = {}

            # ── LoRA Smart Load ───────────────────────────────────────────────
            if lora_active and lora_model:
                if self.current_lora != lora_model:
                    if self.current_lora and hasattr(self.pipeline, "unload_lora_weights"):
                        print(f"   🔄 Unloading previous LoRA: {self.current_lora}")
                        self.pipeline.unload_lora_weights()

                    print(f"🎭 Loading LoRA: {lora_model} (strength: {lora_strength})")
                    
                    full_lora_path: str = lora_model
                    if not os.path.exists(full_lora_path):
                        full_lora_path = os.path.join("loras", lora_model)

                    if os.path.exists(full_lora_path):
                        lora_dir, weight_name = os.path.split(full_lora_path)
                        self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
                    else:
                        self.pipeline.load_lora_weights(lora_model)  # type: ignore

                    self.current_lora = lora_model
                else:
                    print(f"⚡ Reusing cached LoRA: {lora_model} (strength: {lora_strength})")

                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}

            elif self.current_lora:
                print(f"   🧹 Unloading LoRA (not needed): {self.current_lora}")
                if hasattr(self.pipeline, "unload_lora_weights"):
                    self.pipeline.unload_lora_weights()
                self.current_lora = None

            # ── Generation Parameters ─────────────────────────────────────────
            gen_params = self.default_settings.copy()
            gen_params.update(kwargs)

            if "width" not in kwargs or "height" not in kwargs:
                if self.current_model and "xl" in self.current_model.lower():
                    gen_params.setdefault("width", 1024)
                    gen_params.setdefault("height", 1024)
                else:
                    gen_params.setdefault("width", 512)
                    gen_params.setdefault("height", 512)

            if "pixel art" not in prompt.lower():
                prompt += gen_params["pixel_art_prompt_suffix"]

            seed = gen_params.get("seed", -1)
            generator = torch.Generator(device=self.device)

            if seed is not None and int(seed) != -1:
                generator.manual_seed(int(seed))
                print(f"🎲 Using seed: {seed}")
            else:
                random_seed = random.randint(0, 2**32 - 1)
                generator.manual_seed(random_seed)
                print(f"🎲 Random seed: {random_seed}")
                seed = random_seed

            pipeline_kwargs.update({
                "prompt": prompt,
                "negative_prompt": gen_params["negative_prompt"],
                "width": gen_params["width"],
                "height": gen_params["height"],
                "num_inference_steps": int(gen_params["num_inference_steps"]),
                "guidance_scale": float(gen_params["guidance_scale"]),
                "generator": generator,
            })

            print(f"⚙️ Settings: {gen_params['width']}x{gen_params['height']}, {gen_params['num_inference_steps']} steps")

            with torch.inference_mode():
                result = self.pipeline(**pipeline_kwargs)  # type: ignore

            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()  # type: ignore

        except Exception as e:
            self.current_lora = None
            raise e

    # ─────────────────────────────────────────────────────────────────────────
    # Pixel Art Post-Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_for_pixel_art(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (64, 64),
        colors: int = 16,
        use_dithering: bool = False,
        alpha_threshold: int = 128,
        enhance_contrast: float = 1.0,
        sharpen_amount: float = 2.0,
    ) -> Image.Image:
        """ประมวลผลภาพให้เป็น Pixel Art ที่คมและสะอาดตา"""

        print(f"🖼️ กำลังแปลงเป็น Pixel Art: ขนาด {target_size}, จำนวนสี {colors}")

        if colors < 2:
            raise ValueError("❌ จำนวนสีต้องมากกว่าหรือเท่ากับ 2")

        alpha: Optional[Image.Image] = None
        if image.mode in ("RGBA", "LA", "PA", "P"):
            try:
                temp_image = image.convert("RGBA")
                extracted_alpha = temp_image.getchannel("A")
                
                # --- FIX: Use a lookup table (LUT) instead of a lambda to resolve Pylance typing ---
                # A 256-element list covers all 8-bit pixel values, and runs much faster natively in Pillow
                lut = [255 if i >= alpha_threshold else 0 for i in range(256)]
                alpha = extracted_alpha.point(lut)
                
                image = temp_image.convert("RGB")
            except Exception as e:
                print(f"⚠️ Warning processing alpha: {e}")
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")

        if enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhance_contrast)

        if sharpen_amount > 0:
            image = image.filter(
                ImageFilter.UnsharpMask(
                    radius=1, percent=int(100 * sharpen_amount), threshold=2
                )
            )

        image = image.resize(target_size, Image.NEAREST)  # type: ignore

        if colors > 0:
            dither_mode = Image.FLOYDSTEINBERG if use_dithering else Image.NONE  # type: ignore
            image = image.quantize(
                colors=colors, method=Image.MEDIANCUT, dither=dither_mode  # type: ignore
            ).convert("RGB")

        if alpha is not None:
            alpha = alpha.resize(target_size, Image.NEAREST)  # type: ignore
            image = image.convert("RGBA")
            image.putalpha(alpha)

        print("✅ ประมวลผล Pixel Art สำเร็จ (คมชัด + สีสันสดใส)")
        return image


# ─────────────────────────────────────────────────────────────────────────────
# Server Instance
# ─────────────────────────────────────────────────────────────────────────────

sd_server = PixelArtSDServer()


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Any, int]:
    """Endpoint หลักสำหรับรับคำสั่งสร้างภาพ"""
    try:
        data = request.get_json()
        prompt = data.get("prompt") if data else None

        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400

        print(f"\n🎯 New generation request: {prompt[:30]}...")

        requested_model = (
            data.get("model_name") if data else None
        ) or sd_server.default_model or "stabilityai/stable-diffusion-xl-base-1.0"

        if not sd_server.load_model(requested_model):
            return (
                jsonify({"success": False, "error": f"Failed to load model: {requested_model}"}),
                500,
            )

        defaults = sd_server.default_settings
        kwargs = {
            "lora_model": data.get("lora_model") if data else None,
            "lora_strength": data.get("lora_strength", 1.0) if data else 1.0,
            "num_inference_steps": data.get("steps", defaults.get("num_inference_steps")) if data else defaults.get("num_inference_steps"),
            "guidance_scale": data.get("guidance_scale", defaults.get("guidance_scale")) if data else defaults.get("guidance_scale"),
            "seed": data.get("seed", -1) if data else -1,
            "negative_prompt": data.get("negative_prompt", defaults.get("negative_prompt")) if data else defaults.get("negative_prompt"),
            "width": data.get("width", 1024) if data else 1024,
            "height": data.get("height", 1024) if data else 1024,
        }

        start_time = time.time()
        image, used_seed = sd_server.generate_image(prompt=prompt, **kwargs)

        if data and data.get("remove_background", False):
            image = sd_server.remove_background(image)

        pixel_width = int(data.get("pixel_width", 64)) if data else 64
        pixel_height = int(data.get("pixel_height", 64)) if data else 64
        colors = int(data.get("colors", 16)) if data else 16

        pixel_image = sd_server.process_for_pixel_art(
            image, target_size=(pixel_width, pixel_height), colors=colors
        )

        img_base64 = sd_server.image_to_base64(pixel_image)

        generation_time = time.time() - start_time
        print(f"⏱️ Total time: {generation_time:.2f}s")

        return jsonify({
            "success": True,
            "image": {
                "base64": img_base64,
                "width": pixel_width,
                "height": pixel_height,
                "mode": "rgba",
            },
            "seed": used_seed,
            "prompt": prompt,
            "generation_time": generation_time,
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check() -> Any:
    """ตรวจสอบสถานะของเซิร์ฟเวอร์"""
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return jsonify({
        "status": "healthy",
        "model_loaded": sd_server.model_loaded,
        "current_model": sd_server.current_model,
        "current_lora": sd_server.current_lora,
        "device": sd_server.device,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "cached_models": list(sd_server.model_cache.keys()),
        "version": "2.4.1-rtx5070ti-pylance-fixed",
    })


@app.route("/load_model", methods=["POST"])
def load_model_route() -> Tuple[Any, int]:
    """โหลดโมเดล AI ที่ระบุ"""
    try:
        data = request.get_json()
        model_name = data.get("model_name") if data else None

        if not model_name:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ model_name"}), 400

        print(f"📦 Loading model via API: {model_name}")

        if sd_server.load_model(model_name):
            return jsonify({
                "success": True,
                "model": model_name,
                "device": sd_server.device,
                "cached_models": list(sd_server.model_cache.keys()),
            }), 200
        else:
            return (
                jsonify({"success": False, "error": f"ไม่สามารถโหลด {model_name} ได้"}),
                500,
            )

    except Exception as e:
        print(f"❌ Model load error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models() -> Any:
    """แสดงรายการโมเดลที่มีให้เลือก"""
    models: List[str] = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]

    model_directory = "models"
    os.makedirs(model_directory, exist_ok=True)

    if os.path.exists(model_directory):
        for filename in os.listdir(model_directory):
            if filename.endswith(".safetensors") or filename.endswith(".ckpt"):
                if filename not in models:
                    models.append(filename)

    return jsonify({"models": models})


@app.route("/loras", methods=["GET"])
def list_loras() -> Any:
    """แสดงรายการ LoRA"""
    lora_models: List[str] = ["None", "nerijs/pixel-art-xl", "ntc-ai/SDXL-LoRA-slider.pixel-art"]

    lora_directory = "loras"
    os.makedirs(lora_directory, exist_ok=True)

    if os.path.exists(lora_directory):
        for filename in os.listdir(lora_directory):
            if filename.endswith(".safetensors"):
                if filename not in lora_models:
                    lora_models.append(filename)

    return jsonify({"loras": lora_models})


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(default_model_to_load: Optional[str] = None, offline: bool = False) -> None:
    """ฟังก์ชันหลักสำหรับเริ่มการทำงานของเซิร์ฟเวอร์"""
    print("\n" + "=" * 60)
    print("🎮 LOCAL AI GENERATOR SERVER v2.4.1 [PYLANCE FIXED + FAST SWITCH]")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)

    sd_server.offline_mode = offline
    sd_server.default_model = default_model_to_load

    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {sd_server.device}")
    print(f"   • Offline Mode: {offline}")
    print(f"   • Default Model: {default_model_to_load} (Lazy Load)")

    if torch.cuda.is_available():
        print(f"   • BF16 Enabled: {torch.cuda.is_bf16_supported()}")
        print(f"   • Optimizations: TF32, DPM++ Scheduler, CPU Cache Switch")

    print("\n✅ Server ready! Connect at http://127.0.0.1:5000")
    print("🎨 Aseprite plugin can now connect and generate!")
    print("⚡ Model will load automatically on first request.")
    print("🔄 Model switching uses CPU cache for fast reload.")
    print("\n" + "=" * 60 + "\n")

    try:
        cli = sys.modules.get("flask.cli")
        if cli and hasattr(cli, "show_server_banner"):
            setattr(cli, "show_server_banner", lambda *x: None)

        app.run(
            host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main(default_model_to_load="stabilityai/stable-diffusion-xl-base-1.0")