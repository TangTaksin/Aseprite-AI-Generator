import logging
import os
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

import torch
from diffusers.utils import logging as diffusers_logging
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.image_processing import ImageProcessor
from src.models_manager import ModelManager, is_sdxl_model, is_sdxl_lora
from src.version import __version__

# ─── Logging Setup ────────────────────────────────────────────────────────────
import warnings
# ซ่อน UserWarning เกี่ยวกับ token length ของ transformers เพื่อไม่ให้แสดงคำเตือนที่น่าสับสนใน log
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# ปิด log ที่ไม่จำเป็น
diffusers_logging.set_verbosity_error()
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

server_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

# สร้าง global instances
model_manager = ModelManager()
image_processor = ImageProcessor(device=model_manager.device)

# ─── Helper Functions ─────────────────────────────────────────────────────────

def _check_model_lora_compatibility(model_name: str, lora_name: str) -> Optional[str]:
    """ตรวจสอบความเข้ากันได้ของ Model และ LoRA (SD 1.5 / SDXL)"""
    is_model_sdxl = is_sdxl_model(model_name)
    is_lora_sdxl = is_sdxl_lora(lora_name)
    if is_model_sdxl != is_lora_sdxl:
        return "Model and LoRA incompatible! Please match versions (SD 1.5 / SDXL)."
    return None


def _get_request_params(data: dict, defaults: dict) -> dict:
    """ดึงค่าพารามิเตอร์จาก request data โดยใช้ค่าเริ่มต้นจาก defaults"""
    return {
        "lora_model": data.get("lora_model"),
        "lora_strength": data.get("lora_strength", 1.0),
        "num_inference_steps": data.get("steps", defaults.get("num_inference_steps")),
        "guidance_scale": data.get("guidance_scale", defaults.get("guidance_scale")),
        "seed": data.get("seed", -1),
        "negative_prompt": data.get("negative_prompt", defaults.get("negative_prompt")),
        "width": data.get("width", 1024),
        "height": data.get("height", 1024),
    }


def _list_files_in_dir(directory: str, extensions: tuple) -> List[str]:
    """แสดงรายการไฟล์ในโฟลเดอร์ที่มีนามสกุลตรงตามที่กำหนด"""
    try:
        os.makedirs(directory, exist_ok=True)
        return [f for f in os.listdir(directory) if f.endswith(extensions)]
    except Exception as e:
        log.error("Error listing files in directory %s: %s", directory, e)
        return []


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Any, int]:
    server_lock.acquire()
    try:
        data = request.get_json()
        prompt = data.get("prompt") if data else None

        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400

        log.info("New generation request: %s...", prompt[:30])

        requested_model = (
            data.get("model_name")
        ) or model_manager.default_model or "stabilityai/stable-diffusion-xl-base-1.0"

        lora_model = data.get("lora_model")

        # ตรวจสอบความเข้ากันได้ของ Model และ LoRA (โมเดลประเภท SD 1.5 และ SDXL ไม่สามารถใช้ร่วมกันได้)
        if lora_model and lora_model.strip().lower() not in ["none", ""]:
            compat_error = _check_model_lora_compatibility(requested_model, lora_model)
            if compat_error:
                return jsonify({"success": False, "error": compat_error}), 400

        # โหลดโมเดลหลัก
        if not model_manager.load_model(requested_model):
            return (
                jsonify({"success": False, "error": f"Failed to load model: {requested_model}"}),
                500,
            )

        defaults = model_manager.default_settings
        kwargs = _get_request_params(data, defaults)

        start_time = time.time()
        image, used_seed = model_manager.generate_image(prompt=prompt, **kwargs)

        # ลบพื้นหลัง
        if data.get("remove_background", False):
            bg_threshold = float(data.get("remove_background_threshold", 0.5))
            image = image_processor.remove_background(image, threshold=bg_threshold)
            # Segmentation model remains resident in memory for performance
            # To manually offload, use the /offload_segmentation endpoint or call offload_segmentation_model() directly

        pixel_width = int(data.get("pixel_width", 64))
        pixel_height = int(data.get("pixel_height", 64))
        colors = int(data.get("colors", 16))
        pixel_snapping = bool(data.get("pixel_snapping", False))
        pixel_size_override = float(data.get("pixel_size", 0.0))

        # ทำพิกเซลอาร์ต
        pixel_image = image_processor.process_for_pixel_art(
            image,
            target_size=(pixel_width, pixel_height),
            colors=colors,
            pixel_snapping=pixel_snapping,
            pixel_size_override=pixel_size_override
        )

        img_base64 = image_processor.image_to_base64(pixel_image)
        generation_time = time.time() - start_time
        log.info("Total time: %.2fs", generation_time)

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
        log.error("Generation error: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        server_lock.release()


@app.route("/health", methods=["GET"])
def health_check() -> Any:
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "current_model": model_manager.current_model,
        "default_model": model_manager.default_model,
        "current_lora": model_manager.current_lora,
        "device": model_manager.device,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "cached_models": list(model_manager.model_cache.keys()),
        "version": __version__,
    })


@app.route("/load_model", methods=["POST"])
def load_model_route() -> Tuple[Any, int]:
    server_lock.acquire()
    try:
        data = request.get_json()
        model_name = data.get("model_name") if data else None

        if not model_name:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ model_name"}), 400

        log.info("Loading model via API: %s", model_name)

        if model_manager.load_model(model_name):
            return jsonify({
                "success": True,
                "model": model_name,
                "device": model_manager.device,
                "cached_models": list(model_manager.model_cache.keys()),
            }), 200
        else:
            return jsonify({"success": False, "error": f"ไม่สามารถโหลด {model_name} ได้"}), 500

    except Exception as e:
        log.error("Model load error: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        server_lock.release()


@app.route("/models", methods=["GET"])
def list_models() -> Any:
    models: List[str] = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
    ]

    for filename in _list_files_in_dir("models", (".safetensors", ".ckpt")):
        if filename not in models:
            models.append(filename)

    return jsonify({"models": models})


@app.route("/loras", methods=["GET"])
def list_loras() -> Any:
    lora_models: List[str] = ["None", "nerijs/pixel-art-xl", "ntc-ai/SDXL-LoRA-slider.pixel-art"]

    for filename in _list_files_in_dir("loras", (".safetensors",)):
        if filename not in lora_models:
            lora_models.append(filename)

    return jsonify({"loras": lora_models})


@app.route("/offload_segmentation", methods=["POST"])
def offload_segmentation_route() -> Tuple[Any, int]:
    """Manually offload the segmentation model to free VRAM"""
    server_lock.acquire()
    try:
        image_processor.offload_segmentation_model()
        return jsonify({
            "success": True,
            "message": "Segmentation model offloaded to CPU"
        }), 200
    except Exception as e:
        log.error("Error offloading segmentation model: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        server_lock.release()


def main(default_model_to_load: Optional[str] = None, offline: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"LOCAL AI GENERATOR SERVER v{__version__}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)

    model_manager.offline_mode = offline
    model_manager.default_model = default_model_to_load
    image_processor.offline_mode = offline

    log.info("Server Configuration:")
    log.info("  Host        : 127.0.0.1")
    log.info("  Port        : 5000")
    log.info("  Device      : %s", model_manager.device)
    log.info("  Offline Mode: %s", offline)
    log.info("  Default Model: %s (Lazy Load)", default_model_to_load)
    log.info("Server ready at http://127.0.0.1:5000")
    log.info("Model will load automatically on first request.")
    print("=" * 60 + "\n")

    try:
        cli = sys.modules.get("flask.cli")
        if cli and hasattr(cli, "show_server_banner"):
            setattr(cli, "show_server_banner", lambda *x: None)

        app.run(
            host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False
        )
    except KeyboardInterrupt:
        log.info("Shutting down server...")
    except Exception as e:
        log.error("Server error: %s", e, exc_info=True)
