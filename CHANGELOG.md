# Changelog — Aseprite AI Generator (AAG)

## Python Server

### [1.0.4] — 2026-06-19

#### Added
- Added model and LoRA compatibility validation to prevent mixing SD 1.5 and SDXL/Pony assets
- Added a clean, single-line error response message for incompatible assets

#### Changed
- Refactored entire codebase into modular structure (`api_server.py`, `models_manager.py`, `image_processing.py`)
- Moved image generation and model management logic out of `sd_server.py` into `src/`
- Updated type annotations to pass Pylance strict mode

#### Fixed
- Resolved Pylance type-checking errors in `models_manager.py` and `image_processing.py`
- Fixed import path in `sd_server.py` to support running from any working directory

#### Optimized
- Tuned settings specifically for RTX 5070 Ti (Blackwell architecture)
- Auto-enable TF32 Acceleration and cuDNN Benchmark on startup
- Enable bfloat16 precision when GPU supports it
- Added VAE Slicing & Tiling to reduce VRAM usage
- Added Generation Examples gallery in `README.md` featuring a Chibi Boa Profile setup (200px preview)
- Formatted configuration files (`ai_profiles.json`) with pretty-print spacing for better readability

---

### [1.3.0]

#### Added
- Support loading `.safetensors` checkpoints directly from `models/` folder (local checkpoint)
- Support for Pony Diffusion XL on SDXL pipeline
- CPU model cache system for fast model switching without full reload (`model_cache`)
- LoRA smart load/unload with per-session cache

#### Changed
- Changed default scheduler to DPM++ 2M Karras (SDE) for SDXL
- Changed default `num_inference_steps` to 20

---

### [1.2.0]

#### Added
- `startup_script.py`: Interactive setup wizard for first-time configuration
- Model selection via interactive menu on startup
- Automatic Python version and GPU spec verification
- Online / Offline mode support

#### Changed
- Updated PyTorch installation to use CUDA 13.0 index
- Pinned `diffusers==0.38.0`, `transformers==4.57.6`, `accelerate==1.13.0` for stability

---

### [1.1.0]

#### Added
- BiRefNet support for automatic background removal (Lazy Load)
- Pixel Art Color Quantization using Median Cut algorithm
- Nearest Neighbor downscale to preserve sharp pixel edges
- LoRA support on SDXL pipeline

#### Changed
- Added `remove_background`, `pixel_width`, `pixel_height`, `colors` options to `/generate` endpoint

---

### [1.0.0]

#### Added
- Flask REST API server (`sd_server.py`)
- Endpoints: `POST /generate`, `GET /health`, `POST /load_model`, `GET /models`, `GET /loras`
- Support for SDXL and SD 1.5 on a unified pipeline
- CORS support for requests from Aseprite Extension (localhost)

---

*See also [README.md](./README.md)*
*Aseprite Extension changelog → [extension-aseprite-generator/CHANGELOG.md](C:\Users\taksi\AppData\Roaming\Aseprite\extensions\extension-aseprite-generator\CHANGELOG.md)*
