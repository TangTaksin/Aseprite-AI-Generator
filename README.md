# Aseprite AI Generator (AAG)

**Aseprite AI Generator (AAG)** is a bridge that connects local Generative AI (Stable Diffusion) directly with the Aseprite pixel art editor. It converts your text prompts into ready-to-use game sprites within seconds, running 100% locally on your machine's GPU.

---

## Key Highlights

* **Local & Private:** All processing is done locally. No images are sent to the cloud, keeping your project assets private and secure.
* **Professional Pixel Quantization:** Uses high-quality Color Quantization (Median Cut) and Nearest Neighbor scaling to turn AI images into clean, sharp pixel art with customizable palette color counts.
* **AI Background Removal:** Powered by **BiRefNet**, a high-end background segmentation model that automatically isolates sprites and outputs transparency-ready PNGs.
* **LoRA Support:** Integrates LoRA models to direct artistic styles (e.g., 8-bit, 16-bit, Isometric, or specific artist styles) on top of your base models.
* **Optimized for SDXL & SD 1.5:** Supports high-quality Stable Diffusion XL models as well as lightweight Stable Diffusion 1.5 models for fast iteration.
* **Next-Gen GPU Acceleration:** Optimized for NVIDIA RTX GPUs (including Blackwell/RTX 50 Series architecture) with custom configurations for bfloat16 precision, cuDNN benchmarks, and VRAM memory offloading.

---

## Generation Examples

Here are some examples of what you can generate using AAG, along with the prompts and settings used.

| Output Image | Prompt & Settings |
| :---: | :--- |
| <img src="sample/example_1.png" width="200" alt="Chibi Boa Profile"> | **Prompt:** `score_9, score_8_up, score_7_up, score_6_up, white background, 1girl, solo, portrait, looking at viewer, BREAK, Boa Hancock, large breasts, cleavage, off-shoulder dress, collarbone, long black hair, blue eyes, smug smile, gold earrings, one piece style, <lora:shirosu0011:1>, shirosu00, chibi` <br> **Negative Prompt:** `score_6, score_5, score_4, score_3, score_2, score_1, realistic, 3d, photorealistic, blurry, lowres, bad anatomy, bad hands, extra fingers, extra arms, extra legs, malformed limbs, deformed face, mutated hands, text, watermark, signature, duplicate, cropped, worst quality` <br> **Model:** [ponyDiffusionV6XL_v6StartWithThisOne.safetensors](https://civitai.com/models/257749/pony-diffusion-v6-xl) <br> **LoRA:** [Pony_shirosu0011.safetensors](https://civitai.com/models/493151/pixel-art-shirosu-artist-style-pony) (Strength: `1.0`) <br> **Settings:** Size: `64x64` \| Steps: `30` \| CFG: `7` \| Colors: `64` |

---

## Repository Structure

The project features a clean, modular design separating the backend server from the Aseprite frontend extension:

```
├── Aseprite-AI-Generator (Python Backend Server)
│   ├── sd_server.py           # Main entry point (proxies requests to src/api_server.py)
│   ├── startup_script.py      # Interactive dependency installer and setup script
│   ├── Start Server.bat       # Double-click script to run the local API server
│   ├── requirements.txt       # List of Python dependencies
│   ├── src/                   # Core modular backend code
│   │   ├── api_server.py      # Flask REST API endpoints and router logic
│   │   ├── models_manager.py  # Model loading, CPU memory caching, and GPU memory offload
│   │   └── image_processing.py # Color quantization, filters, and BiRefNet segmentation
│   ├── models/                # Local Stable Diffusion checkpoint directory (.safetensors / .ckpt)
│   ├── loras/                 # LoRA weights folder for custom pixel art styles
│   └── cache/                 # Hugging Face downloaded model cache
│
└── Aseprite Extension (Aseprite Extension files)
    ├── package.json           # Aseprite extension configuration manifest
    ├── main.lua               # Menu wrapper that registers the generator command
    ├── local-ui-main.lua      # Main dialog GUI layout and canvas rendering script
    └── libs/                  # Helper Lua scripts
        ├── http-client.lua    # API requester for communication with python server
        ├── json.lua           # Lua table JSON encoder/decoder
        └── base64.lua         # Image string decoder
```

---

## Prerequisites

Before setting up the project, please ensure your system meets the following specifications:
* **OS:** Windows 10 or 11
* **Python:** 3.10.x - 3.11.x (Recommended)
* **GPU:** NVIDIA GPU with 8GB+ VRAM (e.g., RTX 30/40/50 Series with CUDA support)
* **Storage:** 10GB+ of free space
* **Aseprite:** v1.2.10 or newer

---

## Getting Started

### 1. Initialize Python Backend
1. Clone this repository to your local drive.
2. Double-click `Start Server.bat` inside the folder. The script will:
   * Create a Python virtual environment (`venv/`) if it does not exist.
   * Auto-install `PyTorch` (CUDA-supported), `diffusers`, `transformers`, `Flask`, and all image-processing dependencies.
   * Ask you to select your default model (SDXL or SD 1.5) and prompt you to run in Online Mode on the first startup to download the models.
3. Once downloaded and loaded, you will see a text confirmation: **Server ready!** Keep this window open during your Aseprite sessions.

### 2. Install Aseprite Extension

1. Simply double-click the **`PixelAI.aseprite-extension`** file located in the root of this repository.
2. Aseprite will launch and prompt you to install the extension automatically. Click **Install**.
3. Restart Aseprite to complete the setup.

Once installed, you can launch the plugin via **File > Local AI Generator**. Simply enter your prompt and click **Generate** to create pixel art directly on your canvas!

---

## Adding Custom Models & LoRAs

You can easily expand AAG by adding custom models (Checkpoints) and style adapters (LoRAs) downloaded from platforms like Hugging Face or [Civitai](https://civitai.com/).

### 1. Adding Custom Checkpoints
Place your Stable Diffusion base models (e.g., SD 1.5, SDXL, or Pony/Illustrious-based checkpoints) inside the `models/` directory:
```text
models/
```
* **Naming Guide:** If your model is based on **SDXL** (or Pony/Illustrious), ensure the filename contains **`xl`** (case-insensitive) in it (e.g., `hyphoria_xl.safetensors`). This allows the server to automatically detect and load the correct pipeline architecture.
* **Startup Selection:** After placing files in the `models/` folder, run `startup_script.py` (or double-click `Start Server.bat`). The script will automatically detect your local models and display them as options in the CLI menu.

### 2. Adding Custom LoRAs
Place your style LoRA adapters inside the `loras/` directory:
```text
loras/
```
* **Aseprite Integration:** Once files are placed, restart the Python server. The backend API will automatically scan this folder and make your custom LoRAs available in the dropdown selection inside the Aseprite Extension interface.

---

## Advanced Settings Guide

* **Steps:** Controls the number of noise reduction iterations. Standard recommended range: `25 - 35`.
* **Guidance / CFG Scale:** Determines how strictly the model adheres to your text prompt. Standard value: `5.0 - 8.0` (7.0 default).
* **Seed:** Image genetic code. Set to `-1` for random generations. Use specific integer seeds to lock in characters/poses while editing prompt details.
* **Negative Prompt (for SDXL / Pony XL):**
  > `score_4, score_5, score_6, lowres, (bad anatomy), blurry, 3d, photographic, realistic, gradient`

---

## Credits & Acknowledgments

This project is an enhanced and modularized development of the original works by:
* **Original Creator:** [Red335](https://red335.itch.io/pixelai-local-ai-directly-in-aseprite) (Creator of **PixelAI**).

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

Copyright (c) 2026 **TangTaksin**