# Changelog — Aseprite AI Generator (AAG)

## Python Server

### [1.0.8] — 2026-07-12

#### Added
- Added automatic architecture detection for model checkpoints and LoRAs via embedded `.safetensors` metadata (`modelspec.architecture` and `ss_base_model_version`).
- Added file size threshold (> 5.0 GB) detection to automatically identify local SDXL checkpoints.
- Added support for new short-form and custom model/LoRA keywords (`hyphoria`, `noob`, `walnut`, `plantmilk`, `ilu`) in filename classification.
- Added a second generation example (`example_2.png`) to `README.md` featuring the `hyphoria_v002_2` model with the `[Ilu] Pixel_Resin_x16_contrast_ep20` LoRA using 32 colors constraint.
- Added a "Local Model & LoRA Compatibility Guide" section to `README.md` separating Illustrious XL and Pony Diffusion XL assets to prevent mis-matching.

#### Fixed
- Fixed checkpoint loading failures for models like `hyphoria_v002_2.safetensors` and `plantMilkModelSuite_walnut.safetensors` which previously failed to load because the system classified them as SD 1.5 due to missing "xl" keywords.
- Fixed asset compatibility validation bypass/errors for Illustrious LoRAs using bracketed short tags like `[Ilu]` which caused false mismatch warnings when paired with SDXL checkpoints.

### [1.0.7] — 2026-06-25

#### Fixed
- Fixed background removal memory residency to immediately offload BiRefNet from VRAM (`image_processor.offload_segmentation_model()`) after execution to comply with memory safety rules.

### [1.0.6] — 2026-06-25

#### Added
- Integrated Sprite Fusion Pixel Snapper grid-snapping algorithm into [src/image_processing.py](file:///D:/Aseprite-AI-Generator/src/image_processing.py) for cleaning up sub-pixel rendering (mixels) in AI-generated pixel art
- Added support for `pixel_snapping` and `pixel_size` parameters in the `/generate` endpoint in [src/api_server.py](file:///D:/Aseprite-AI-Generator/src/api_server.py)
- Added default values for `pixel_snapping` and `pixel_size_override` in the Aseprite Extension configuration ([libs/settings-store.lua](file:///C:/Users/taksi/AppData/Roaming/Aseprite/extensions/extension-aseprite-generator/libs/settings-store.lua))
- Added custom controls (Enable Pixel Snapping checkbox and Pixel Size Override numeric input) to the Aseprite Extension Advanced Settings dialog ([local-ui-main.lua](file:///C:/Users/taksi/AppData/Roaming/Aseprite/extensions/extension-aseprite-generator/local-ui-main.lua))

#### Changed
- Optimized background removal preprocessor resolution to `1024x1024` for BiRefNet to achieve high-fidelity edges and cleaner background isolation
- Updated background removal logic to use the highest resolution prediction layer (`outputs[-1]`) from the BiRefNet output stages
- Integrated `remove_background_threshold` into the `/generate` endpoint to allow custom transparency thresholds

#### Removed
- Removed the Floyd-Steinberg dithering system entirely from image processing and API endpoints

#### Optimized
- Optimized RGB pixel resampling via packed uint32 1D unique value extraction, achieving a **6.13x speedup** (from 314ms to 51ms)
- Optimized Alpha channel resampling using `np.bincount` majority voting, yielding a **1.46x speedup**

### [1.0.5] — 2026-06-22

#### Added
- Added LoRA caching system to reduce disk I/O when switching LoRAs
- Added manual endpoint `/offload_segmentation` to control VRAM usage for background removal model
- Added safety check for StopIteration when retrieving model parameters
- Added infinite prompt token length support (> 77 tokens) using a custom token chunking and embedding concatenation system for both SD 1.5 and SDXL
- Added token threshold check (under 77 tokens) to fallback to raw string prompts, ensuring 100% exact output match with previous versions

#### Changed
- Modified background removal model to load once and remain resident in GPU memory (instead of loading/unloading per request)
- Removed redundant offline mode assignment per request
- Avoided unnecessary image conversions by checking image mode before conversion
- Replaced xformers with PyTorch native SDPA for stable performance on newer GPUs
- Translated all Thai comments, docstrings, error messages, and logs in image_processing.py to English for code standardization
- Updated VAE slicing and tiling API calls to use the newer direct VAE module methods (.vae.enable_slicing, .vae.enable_tiling, .vae.disable_tiling) to suppress deprecation FutureWarnings

#### Fixed
- Replaced bare `except: pass` with specific exception logging in image_processing.py
- Changed generic Exception to RuntimeError for background loading failure
- Refined exception handling in model optimization to avoid swallowing critical errors
- Improved error messages for attention setup, VAE slicing/tiling, and scheduler setup failures
- Suppressed noisy Hugging Face transformers token length UserWarnings in api_server.py console

#### Removed
- Removed xformers support in favor of PyTorch native SDPA for stability and identical performance
- Removed unused `scipy` and `torchaudio` dependencies from requirements.txt and startup_script.py to speed up installation and save disk space
- Removed empty test logs and `.qodo` editor cache directories

#### Optimized
- Reduced latency for consecutive background removal requests by keeping model resident
- Reduced LoRA switch latency via caching mechanism

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