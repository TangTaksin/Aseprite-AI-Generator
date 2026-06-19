import os
import gc
import platform
import random
import torch
from typing import Optional, Dict, Any, Tuple
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)

class ModelManager:
    """จัดการเกี่ยวกับการโหลดโมเดล Stable Diffusion, LoRA และ VRAM Optimization"""
    
    def __init__(self, default_model: Optional[str] = None):
        self.pipeline: Optional[Any] = None
        self.model_loaded: bool = False
        self.current_model: Optional[str] = None
        self.current_lora: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cache: Dict[str, Any] = {}
        self.offline_mode: bool = False
        self.default_model: Optional[str] = default_model
        self._max_cache_size: int = 2

        is_windows = platform.system() == "Windows"
        self.use_compile: bool = False if is_windows else True

        # 🎯 Settings สำหรับ RTX 5070 Ti
        self.optimized_settings: Dict[str, Any] = {
            "use_bf16": torch.cuda.is_bf16_supported(),
            "use_compile": self.use_compile,
            "enable_xformers": not is_windows,
            "enable_attention_slicing": True,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",
        }

        self.default_settings: Dict[str, Any] = {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality, worst quality, lowres, jpeg artifacts, watermark, signature, username, out of focus, hazy, painting, oil painting, sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed",
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels",
        }

        if torch.cuda.is_available():
            self._apply_global_cuda_optimizations()

    def _apply_global_cuda_optimizations(self) -> None:
        """ตั้งค่าเพิ่มประสิทธิภาพ CUDA ทั่วไป"""
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
        print("   ✅ TF32 Acceleration: Enabled\n")

    def _optimize_pipeline_for_blackwell(self, pipeline: Any, model_type: str = "sdxl") -> Any:
        """รีดประสิทธิภาพ Pipeline สำหรับ GPU สถาปัตยกรรม Blackwell/RTX 50 Series"""
        if self.device != "cuda":
            return pipeline

        print(f"🎯 Optimizing {model_type.upper()} pipeline for RTX 5070 Ti...")
        dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
        pipeline = pipeline.to(self.device, dtype=dtype)
        print(f"   ✅ Precision: {dtype}")

        try:
            if hasattr(pipeline, "disable_attention_slicing"):
                pipeline.disable_attention_slicing()
            print("   ✅ Attention: Native SDPA (Blackwell optimized)")
        except Exception as e:
            print(f"   ⚠️ Attention setup note: {e}")

        try:
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
            print("   ✅ VAE Slicing & Tiling: Enabled")
        except Exception:
            pass

        if self.optimized_settings["use_compile"]:
            print("   🚀 Compiling U-Net with torch.compile...")
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

    def _offload_current_pipeline(self) -> None:
        """ย้ายโมเดลปัจจุบันออกจาก GPU เพื่อลดการใช้ VRAM"""
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
                self._add_to_cache(self.current_model, self.pipeline)

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
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   🧹 VRAM after offload → Used: {vram_used:.2f}GB")

    def _add_to_cache(self, model_name: str, pipeline: Any) -> None:
        """Add pipeline to CPU cache with LRU eviction (max 2 models)"""
        if model_name in self.model_cache:
            del self.model_cache[model_name]
        while len(self.model_cache) >= self._max_cache_size:
            evict_key = next(iter(self.model_cache))
            print(f"   🗑️ Evicting '{evict_key}' from CPU cache (limit: {self._max_cache_size})")
            del self.model_cache[evict_key]
            gc.collect()
        self.model_cache[model_name] = pipeline

    def load_model(self, model_name: str) -> bool:
        """โหลดโมเดลหลักพร้อมแคชโมเดลบน CPU RAM สำหรับการสลับอย่างรวดเร็ว"""
        try:
            local_only = self.offline_mode

            if self.model_loaded and self.current_model == model_name and self.pipeline is not None:
                print(f"⚡ '{model_name}' already on GPU. Skipping load.")
                return True

            if self.pipeline is not None:
                print(f"🔄 Model switch: '{self.current_model}' → '{model_name}'")
                self._offload_current_pipeline()

            if model_name in self.model_cache:
                print(f"⚡ Restoring '{model_name}' from CPU cache → GPU...")
                self.pipeline = self.model_cache.pop(model_name)
                dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
                if self.pipeline is not None:
                    self.pipeline.to(self.device, dtype=dtype)
                self.current_model = model_name
                self.model_loaded = True
                print("✅ Model restored to GPU from cache!")
                return True

            print(f"📥 Loading model: {model_name}")
            is_sdxl = "xl" in model_name.lower()
            precision = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

            local_model_path = os.path.join("models", model_name)
            is_local_file = os.path.isfile(local_model_path)

            if is_local_file:
                print(f"📂 Found local model: {local_model_path}")
                if is_sdxl:
                    try:
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
                    except Exception as vae_err:
                        print(f"⚠️ Failed to load external VAE, falling back to embedded VAE: {vae_err}")
                        self.pipeline = StableDiffusionXLPipeline.from_single_file(
                            local_model_path,
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
                    try:
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
                    except Exception as vae_err:
                        print(f"⚠️ Failed to load external VAE, loading without it: {vae_err}")
                        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                            model_name,
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
            self.pipeline = self._optimize_pipeline_for_blackwell(self.pipeline, model_type)

            # Warmup
            if self.device == "cuda" and self.pipeline is not None:
                print("🔥 Running warmup inference...")
                try:
                    with torch.inference_mode():
                        _ = self.pipeline(
                            prompt="pixel",
                            width=512,
                            height=512,
                            num_inference_steps=1,
                            guidance_scale=1.0,
                        )
                    print("✅ Warmup complete!")
                except Exception as e:
                    print(f"⚠️ Warmup skipped: {e}")

            self._add_to_cache(model_name, self.pipeline)
            self.current_model = model_name
            self.model_loaded = True
            return True

        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {str(e)}")
            return False

    def generate_image(
        self, 
        prompt: str, 
        lora_model: Optional[str] = None, 
        lora_strength: float = 1.0, 
        **kwargs: Any
    ) -> Tuple[Any, int]:
        """สร้างรูปภาพด้วย Stable Diffusion พร้อมประยุกต์ใช้ LoRA"""
        if not self.model_loaded or self.pipeline is None:
            raise Exception("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")

        lora_active = lora_model and lora_model.lower() not in ["none", ""]
        try:
            print(f"🎨 Generating: '{prompt[:50]}...'")
            pipeline_kwargs: Dict[str, Any] = {}

            # LoRA Smart Load
            if lora_active and lora_model:
                if self.current_lora != lora_model:
                    if self.current_lora and hasattr(self.pipeline, "unload_lora_weights"):
                        print(f"   🔄 Unloading previous LoRA: {self.current_lora}")
                        self.pipeline.unload_lora_weights()

                    print(f"🎭 Loading LoRA: {lora_model} (strength: {lora_strength})")
                    full_lora_path = lora_model
                    if not os.path.exists(full_lora_path):
                        full_lora_path = os.path.join("loras", lora_model)

                    try:
                        if os.path.exists(full_lora_path):
                            lora_dir, weight_name = os.path.split(full_lora_path)
                            self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
                        else:
                            self.pipeline.load_lora_weights(lora_model)
                        self.current_lora = lora_model
                    except Exception as lora_err:
                        print(f"❌ LoRA Loading Failed (VRAM cleanup initiated): {lora_err}")
                        if hasattr(self.pipeline, "unload_lora_weights"):
                            try:
                                self.pipeline.unload_lora_weights()
                            except Exception:
                                pass
                        self.current_lora = None
                        
                        # Generate a friendly error message for size mismatch (SD 1.5 vs SDXL mismatch)
                        err_msg = str(lora_err)
                        if "size mismatch" in err_msg.lower():
                            friendly_err = "Model and LoRA incompatible! Please match versions."
                        else:
                            friendly_err = f"Failed to load LoRA weights: {err_msg}"
                        raise Exception(friendly_err)
                else:
                    print(f"⚡ Reusing cached LoRA: {lora_model}")

                pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
            elif self.current_lora:
                print(f"   🧹 Unloading LoRA (not needed): {self.current_lora}")
                if hasattr(self.pipeline, "unload_lora_weights"):
                    self.pipeline.unload_lora_weights()
                self.current_lora = None

            # Setup parameters
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

            with torch.inference_mode():
                result = self.pipeline(**pipeline_kwargs)

            print("✅ Image generation complete")
            return result.images[0], generator.initial_seed()

        except Exception as e:
            self.current_lora = None
            raise e
