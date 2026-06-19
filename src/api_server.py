import os
import sys
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional, Tuple, Any, List
from diffusers.utils import logging as diffusers_logging
import logging

from src.models_manager import ModelManager
from src.image_processing import ImageProcessor

# ปิดคำเตือนที่ไม่จำเป็น
diffusers_logging.set_verbosity_error()
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# สร้าง global instances
model_manager = ModelManager()
image_processor = ImageProcessor(device=model_manager.device)

@app.route("/generate", methods=["POST"])
def generate() -> Tuple[Any, int]:
    try:
        data = request.get_json()
        prompt = data.get("prompt") if data else None

        if not prompt:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ Prompt"}), 400

        print(f"\n🎯 New generation request: {prompt[:30]}...")

        requested_model = (
            data.get("model_name") if data else None
        ) or model_manager.default_model or "stabilityai/stable-diffusion-xl-base-1.0"

        # โหลดโมเดลหลัก
        if not model_manager.load_model(requested_model):
            return (
                jsonify({"success": False, "error": f"Failed to load model: {requested_model}"}),
                500,
            )

        defaults = model_manager.default_settings
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

        # เผื่อมีการอัปเดต offline mode ล่าสุด
        image_processor.offline_mode = model_manager.offline_mode

        start_time = time.time()
        image, used_seed = model_manager.generate_image(prompt=prompt, **kwargs)

        # ลบพื้นหลัง
        if data and data.get("remove_background", False):
            image = image_processor.remove_background(image)

        pixel_width = int(data.get("pixel_width", 64)) if data else 64
        pixel_height = int(data.get("pixel_height", 64)) if data else 64
        colors = int(data.get("colors", 16)) if data else 16

        # ทำพิกเซลอาร์ต
        pixel_image = image_processor.process_for_pixel_art(
            image, target_size=(pixel_width, pixel_height), colors=colors
        )

        img_base64 = image_processor.image_to_base64(pixel_image)
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
    vram_used = 0.0
    vram_total = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "current_model": model_manager.current_model,
        "current_lora": model_manager.current_lora,
        "device": model_manager.device,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "cached_models": list(model_manager.model_cache.keys()),
        "version": "1.0.4",
    })


@app.route("/load_model", methods=["POST"])
def load_model_route() -> Tuple[Any, int]:
    try:
        data = request.get_json()
        model_name = data.get("model_name") if data else None

        if not model_name:
            return jsonify({"success": False, "error": "ไม่ได้ระบุ model_name"}), 400

        print(f"📦 Loading model via API: {model_name}")

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
        print(f"❌ Model load error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models() -> Any:
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
    lora_models: List[str] = ["None", "nerijs/pixel-art-xl", "ntc-ai/SDXL-LoRA-slider.pixel-art"]
    lora_directory = "loras"
    os.makedirs(lora_directory, exist_ok=True)

    if os.path.exists(lora_directory):
        for filename in os.listdir(lora_directory):
            if filename.endswith(".safetensors"):
                if filename not in lora_models:
                    lora_models.append(filename)

    return jsonify({"loras": lora_models})


def main(default_model_to_load: Optional[str] = None, offline: bool = False) -> None:
    print("\n" + "=" * 60)
    print("🎮 LOCAL AI GENERATOR SERVER v1.0.4")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("loras", exist_ok=True)

    model_manager.offline_mode = offline
    model_manager.default_model = default_model_to_load
    image_processor.offline_mode = offline

    print("\n🌐 Server Configuration:")
    print(f"   • Host: 127.0.0.1")
    print(f"   • Port: 5000")
    print(f"   • Device: {model_manager.device}")
    print(f"   • Offline Mode: {offline}")
    print(f"   • Default Model: {default_model_to_load} (Lazy Load)")

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
