import gc
import logging
import os
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

# ปิดคำเตือน PyTorch Dynamo / Triton (cosmetic, ไม่กระทบการทำงาน)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", message=".*Dynamo detected a call to a functools.lru_cache.*")
warnings.filterwarnings("ignore", message=".*Cannot find a working triton installation.*")

log = logging.getLogger(__name__)


class ModelManager:
    """จัดการการโหลดโมเดล Stable Diffusion, LoRA และ VRAM Optimization"""

    # ─── Constants ────────────────────────────────────────────────────────────
    _MAX_CACHE_SIZE: int = 2

    def __init__(self, default_model: Optional[str] = None) -> None:
        self.pipeline: Optional[Any] = None
        self.model_loaded: bool = False
        self.current_model: Optional[str] = None
        self.current_lora: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # LRU cache: เก็บ pipeline บน CPU RAM สูงสุด 2 โมเดล
        self.model_cache: Dict[str, Any] = {}
        self.offline_mode: bool = False
        self.default_model: Optional[str] = default_model

        # ⚙️ Settings สำหรับ RTX 40/50 Series (Blackwell / Ada Lovelace)
        self.optimized_settings: Dict[str, Any] = {
            "use_bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "cudnn_benchmark": True,
            "float32_matmul_precision": "high",
        }

        self.default_settings: Dict[str, Any] = {
            "num_inference_steps": 16,
            "guidance_scale": 7.5,
            "negative_prompt": (
                "blurry, smooth, antialiased, realistic, photographic, 3d render, "
                "low quality, worst quality, lowres, jpeg artifacts, watermark, "
                "signature, username, out of focus, hazy, painting, oil painting, "
                "sketch, drawing, smooth shading, gradients, noise, extra fingers, deformed"
            ),
            "pixel_art_prompt_suffix": ", pixel art, 8bit style, game sprite, masterpiece, sharp pixels",
        }

        if torch.cuda.is_available():
            self._apply_global_cuda_optimizations()

    # ─── CUDA / Global Optimizations ─────────────────────────────────────────

    def _apply_global_cuda_optimizations(self) -> None:
        """ตั้งค่าเพิ่มประสิทธิภาพ CUDA ทั่วไป"""
        if self.device != "cuda":
            return

        log.info("Applying RTX Global Optimizations...")

        if self.optimized_settings["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
            log.info("  [OK] cuDNN Benchmark: Enabled")

        precision = self.optimized_settings["float32_matmul_precision"]
        try:
            torch.set_float32_matmul_precision(precision)
            log.info("  [OK] TF32 Precision: %s", precision)
        except Exception:
            log.warning("  [WARNING] Could not set matmul precision")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("  [OK] TF32 Acceleration: Enabled")

    def _optimize_pipeline(self, pipeline: Any, model_type: str = "sdxl",
                           model_name: Optional[str] = None) -> Any:
        """รีดประสิทธิภาพ Pipeline สำหรับ GPU RTX 40/50 Series"""
        if self.device != "cuda":
            return pipeline

        log.info("Optimizing %s pipeline...", model_type.upper())

        dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
        pipeline = pipeline.to(self.device, dtype=dtype)
        log.info("  [OK] Precision: %s", dtype)

        # ปิด attention slicing (ใช้ full attention แทน — เร็วกว่าถ้ามี VRAM พอ)
        try:
            if hasattr(pipeline, "disable_attention_slicing"):
                pipeline.disable_attention_slicing()
        except Exception as e:
            log.warning("  Attention slicing disable failed: %s", e)

        # ใช้ Native SDPA (PyTorch built-in) เป็นค่าเริ่มต้น
        log.info("  [OK] Attention: Native SDPA (PyTorch built-in)")

        # VAE slicing ช่วยลด VRAM, VAE tiling เปิดเฉพาะภาพใหญ่ (เพิ่ม overhead สำหรับภาพปกติ)
        try:
            pipeline.enable_vae_slicing()
            log.info("  [OK] VAE Slicing: Enabled")
        except Exception as e:
            log.warning("  VAE slicing failed: %s", e)

        # Scheduler
        if model_type == "sdxl" and hasattr(pipeline, "scheduler"):
            self._apply_scheduler(pipeline, model_name)

        log.info("  [OK] Pipeline optimization complete!")
        return pipeline

    def _apply_scheduler(self, pipeline: Any, model_name: Optional[str]) -> None:
        """ตั้งค่า Scheduler ให้เหมาะสมกับโมเดล"""
        is_lcm = model_name and "lcm" in model_name.lower()
        try:
            if is_lcm:
                pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
                log.info("  [OK] Scheduler: LCM")
            else:
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    use_karras_sigmas=True,
                )
                log.info("  [OK] Scheduler: DPM++ 2M Karras")
        except Exception as e:
            log.warning("  [WARNING] Scheduler setup failed: %s", e)

    # ─── CPU Cache Helpers ────────────────────────────────────────────────────

    def _add_to_cache(self, model_name: str, pipeline: Any) -> None:
        """เพิ่ม pipeline (ที่ย้ายมา CPU แล้ว) เข้า LRU cache (สูงสุด 2 โมเดล)"""
        # ถ้ามีอยู่แล้วให้ลบก่อน (เพื่อ refresh ลำดับ LRU)
        self.model_cache.pop(model_name, None)

        # Evict โมเดลเก่าสุดถ้าเต็ม
        while len(self.model_cache) >= self._MAX_CACHE_SIZE:
            evict_key = next(iter(self.model_cache))
            log.info("  Evicting '%s' from CPU cache (limit: %d)", evict_key, self._MAX_CACHE_SIZE)
            del self.model_cache[evict_key]
            gc.collect()

        self.model_cache[model_name] = pipeline

    # ─── Pipeline Offload / Cleanup ───────────────────────────────────────────

    def _offload_current_pipeline(self) -> None:
        """ย้าย pipeline ปัจจุบันออกจาก GPU แล้วเก็บใน CPU cache"""
        if self.pipeline is None:
            return

        log.info("  Offloading '%s' to CPU cache...", self.current_model)
        try:
            self._remove_lora(silent=False)
            self.pipeline.to("cpu")
            if self.current_model:
                self._add_to_cache(self.current_model, self.pipeline)
        except Exception as e:
            log.warning("  Offload warning (clearing instead): %s", e)
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
                log.info("  VRAM after offload: %.2f GB used", vram_used)

    # ─── LoRA Helpers ─────────────────────────────────────────────────────────

    def _remove_lora(self, silent: bool = True) -> None:
        """ถอด LoRA weights ออกจาก pipeline ปัจจุบัน"""
        if not self.current_lora:
            return
        if self.pipeline is None:
            self.current_lora = None
            return
        if not hasattr(self.pipeline, "unload_lora_weights"):
            self.current_lora = None
            return

        if not silent:
            log.info("  Unloading LoRA: %s", self.current_lora)
        try:
            self.pipeline.unload_lora_weights()
        except Exception as e:
            log.warning("  LoRA unload warning: %s", e)
        finally:
            self.current_lora = None

    def _apply_lora(self, lora_model: str, lora_strength: float) -> None:
        """โหลดและ apply LoRA เข้า pipeline ปัจจุบัน

        Raises:
            Exception: ถ้าโหลด LoRA ไม่สำเร็จ (พร้อม friendly error message)
        """
        # ถอด LoRA เก่าออกก่อน (ถ้ามี)
        if self.current_lora and self.current_lora != lora_model:
            self._remove_lora(silent=False)

        # หา path ของ LoRA
        lora_path = lora_model
        if not os.path.exists(lora_path):
            lora_path = os.path.join("loras", lora_model)

        log.info("  Loading LoRA: %s (strength: %.2f)", lora_model, lora_strength)
        try:
            if os.path.exists(lora_path):
                lora_dir, weight_name = os.path.split(lora_path)
                self.pipeline.load_lora_weights(lora_dir, weight_name=weight_name)
            else:
                # fallback: โหลดจาก HuggingFace Hub (ถ้าไม่ได้ offline mode)
                self.pipeline.load_lora_weights(lora_model)
            self.current_lora = lora_model
            log.info("  [OK] LoRA loaded: %s", lora_model)
        except Exception as e:
            # cleanup ถ้า partial load
            try:
                if hasattr(self.pipeline, "unload_lora_weights"):
                    self.pipeline.unload_lora_weights()
            except Exception:
                pass
            self.current_lora = None

            err_msg = str(e)
            if "size mismatch" in err_msg.lower():
                raise Exception("Model และ LoRA ไม่รองรับกัน! กรุณาเลือก version ให้ตรงกัน (SD 1.5 / SDXL)") from e
            raise Exception(f"โหลด LoRA ไม่สำเร็จ: {err_msg}") from e

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _load_pipeline_from_disk(self, model_name: str, is_sdxl: bool,
                                  precision: torch.dtype, local_only: bool) -> Any:
        """โหลด pipeline จาก local file หรือ HuggingFace Hub

        Returns:
            pipeline ที่โหลดสำเร็จ (ยังไม่ได้ move ไป GPU)

        Raises:
            Exception: ถ้าโหลดไม่สำเร็จ
        """
        local_model_path = os.path.join("models", model_name)
        is_local_file = os.path.isfile(local_model_path)
        log.info("  Source: %s", local_model_path if is_local_file else f"Hub ({model_name})")

        if is_sdxl:
            vae = self._load_sdxl_vae(precision, local_only)
            kwargs: Dict[str, Any] = {"torch_dtype": precision, "use_safetensors": True}
            if vae is not None:
                kwargs["vae"] = vae
            if is_local_file:
                kwargs["config"] = "stabilityai/stable-diffusion-xl-base-1.0"
                return StableDiffusionXLPipeline.from_single_file(local_model_path, **kwargs)
            kwargs["local_files_only"] = local_only
            return StableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)

        if is_local_file:
            return StableDiffusionPipeline.from_single_file(
                local_model_path, torch_dtype=precision, load_safety_checker=False)
        return StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=precision, use_safetensors=True, local_files_only=local_only)

    def _load_sdxl_vae(self, precision: torch.dtype,
                        local_only: bool) -> Optional[Any]:
        """โหลด VAE สำหรับ SDXL (fp16-fix) — คืน None ถ้าโหลดไม่ได้"""
        try:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=precision,
                local_files_only=local_only,
            )
            log.info("  [OK] Loaded SDXL VAE (fp16-fix)")
            return vae
        except Exception as e:
            log.warning("  [WARNING] Could not load external VAE, using embedded: %s", e)
            return None

    def _warmup_pipeline(self) -> None:
        """Run warmup inference to pre-load CUDA kernels"""
        if self.device != "cuda" or self.pipeline is None:
            return
        log.info("Running warmup inference...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.inference_mode():
                    _ = self.pipeline(
                        prompt="pixel", width=512, height=512,
                        num_inference_steps=1, guidance_scale=1.0,
                    )
            log.info("Warmup complete")
        except Exception as e:
            log.warning("Warmup skipped: %s", e)

    def load_model(self, model_name: str) -> bool:
        """โหลดโมเดลหลักพร้อม CPU LRU cache สำหรับการสลับโมเดลอย่างรวดเร็ว

        Returns:
            True ถ้าโหลดสำเร็จ, False ถ้าล้มเหลว
        """
        try:
            # 1. โมเดลนี้อยู่บน GPU แล้ว — ไม่ต้องทำอะไร
            if self.model_loaded and self.current_model == model_name and self.pipeline is not None:
                log.info("'%s' already on GPU. Skipping.", model_name)
                return True

            # 2. มีโมเดลอื่นอยู่ — offload ก่อน
            if self.pipeline is not None:
                log.info("Model switch: '%s' → '%s'", self.current_model, model_name)
                self._offload_current_pipeline()

            # 3. ดึงจาก CPU cache (ถ้ามี)
            if model_name in self.model_cache:
                log.info("Restoring '%s' from CPU cache → GPU...", model_name)
                self.pipeline = self.model_cache.pop(model_name)
                dtype = torch.bfloat16 if self.optimized_settings["use_bf16"] else torch.float16
                self.pipeline.to(self.device, dtype=dtype)
                self.current_model = model_name
                self.model_loaded = True
                log.info("[OK] Model restored to GPU from cache!")
                return True

            # 4. โหลดจาก disk / Hub
            log.info("Loading model: %s", model_name)
            is_sdxl = "xl" in model_name.lower()
            local_only = self.offline_mode
            precision = (
                torch.bfloat16
                if (self.device == "cuda" and torch.cuda.is_bf16_supported())
                else torch.float16
            )

            self.pipeline = self._load_pipeline_from_disk(model_name, is_sdxl, precision, local_only)

            # 5. Optimize + warmup
            model_type = "sdxl" if is_sdxl else "sd15"
            self.pipeline = self._optimize_pipeline(self.pipeline, model_type, model_name)
            self._warmup_pipeline()

            # 6. อัปเดต state (cache เฉพาะตอน offload ครั้งถัดไป ไม่ใช่ตอนนี้)
            self.current_model = model_name
            self.model_loaded = True
            log.info("[OK] Model '%s' ready on GPU!", model_name)
            return True

        except Exception as e:
            log.error("Failed to load model '%s': %s", model_name, e, exc_info=True)
            self.pipeline = None
            self.model_loaded = False
            self.current_model = None
            return False

    # ─── Image Generation ─────────────────────────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        lora_model: Optional[str] = None,
        lora_strength: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[Any, int]:
        """สร้างรูปภาพด้วย Stable Diffusion พร้อมประยุกต์ใช้ LoRA

        Returns:
            Tuple (PIL.Image, seed ที่ใช้)

        Raises:
            Exception: ถ้าโมเดลยังไม่ถูกโหลด หรือเกิดข้อผิดพลาดระหว่าง generate
        """
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("ยังไม่ได้โหลดโมเดลหลัก กรุณาโหลดโมเดลก่อนสร้างภาพ")

        lora_active = bool(lora_model and lora_model.strip().lower() not in ("none", ""))

        log.info("Generating: '%s...'", prompt[:50])
        pipeline_kwargs: Dict[str, Any] = {}

        # ─── LoRA Management ───────────────────────────────────────────────
        if lora_active and lora_model:
            if self.current_lora != lora_model:
                # โหลด LoRA ใหม่ (จะถอดตัวเก่าออกโดยอัตโนมัติใน _apply_lora)
                self._apply_lora(lora_model, lora_strength)
            else:
                log.info("  Reusing loaded LoRA: %s", lora_model)
            pipeline_kwargs["cross_attention_kwargs"] = {"scale": float(lora_strength)}
        elif self.current_lora:
            # ไม่ต้องการ LoRA แล้ว — ถอดออก
            self._remove_lora(silent=False)

        # ─── Build Generation Parameters ──────────────────────────────────
        gen_params = self.default_settings.copy()
        gen_params.update(kwargs)

        # ขนาด default ตามประเภทโมเดล
        is_xl = self.current_model and "xl" in self.current_model.lower()
        default_size = 1024 if is_xl else 512
        gen_params.setdefault("width", default_size)
        gen_params.setdefault("height", default_size)

        # เติม pixel art suffix ถ้ายังไม่มี
        if "pixel art" not in prompt.lower():
            prompt += gen_params["pixel_art_prompt_suffix"]

        # ─── Seed ─────────────────────────────────────────────────────────
        raw_seed = gen_params.get("seed", -1)
        generator = torch.Generator(device=self.device)
        try:
            seed_int = int(raw_seed) if raw_seed is not None else -1
        except (ValueError, TypeError):
            seed_int = -1

        if seed_int != -1:
            generator.manual_seed(seed_int)
            log.info("  Seed: %d (fixed)", seed_int)
        else:
            random_seed = random.randint(0, 2**32 - 1)
            generator.manual_seed(random_seed)
            log.info("  Seed: %d (random)", random_seed)

        # เปิด VAE tiling เฉพาะภาพใหญ่ (>1024px) เพื่อป้องกัน OOM โดยไม่เพิ่ม overhead ภาพเล็ก
        gen_width = int(gen_params["width"])
        gen_height = int(gen_params["height"])
        use_tiling = gen_width > 1024 or gen_height > 1024
        try:
            if use_tiling:
                self.pipeline.enable_vae_tiling()
            else:
                self.pipeline.disable_vae_tiling()
        except Exception:
            pass

        pipeline_kwargs.update({
            "prompt": prompt,
            "negative_prompt": gen_params["negative_prompt"],
            "width": gen_width,
            "height": gen_height,
            "num_inference_steps": int(gen_params["num_inference_steps"]),
            "guidance_scale": float(gen_params["guidance_scale"]),
            "generator": generator,
        })

        # ─── Inference ────────────────────────────────────────────────────
        with torch.inference_mode():
            result = self.pipeline(**pipeline_kwargs)

        log.info("[OK] Image generation complete")
        return result.images[0], generator.initial_seed()
