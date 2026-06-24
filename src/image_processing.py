import math
import base64
import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

log = logging.getLogger(__name__)


# ─── Sprite Fusion Pixel Snapper Helper Functions ─────────────────────────────────

def _compute_profiles(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = img.shape
    if w < 3 or h < 3:
        raise ValueError("Image too small for snapping (minimum 3x3)")

    gray = (
        0.299 * img[:, :, 0].astype(np.float64)
        + 0.587 * img[:, :, 1].astype(np.float64)
        + 0.114 * img[:, :, 2].astype(np.float64)
    )

    col_proj = np.zeros(w, dtype=np.float64)
    row_proj = np.zeros(h, dtype=np.float64)

    col_proj[1:-1] = np.sum(np.abs(gray[:, 2:] - gray[:, :-2]), axis=0)
    row_proj[1:-1] = np.sum(np.abs(gray[2:, :] - gray[:-2, :]), axis=1)

    return col_proj, row_proj


def _estimate_step_size(profile: np.ndarray, threshold_multiplier: float = 0.2, distance_filter: int = 4) -> Optional[float]:
    if len(profile) == 0:
        return None
    max_val = float(np.max(profile))
    if max_val == 0.0:
        return None
    threshold = max_val * threshold_multiplier

    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if (
            profile[i] > threshold
            and profile[i] > profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)
    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > (distance_filter - 1):
            clean_peaks.append(p)
    if len(clean_peaks) < 2:
        return None

    diffs = np.diff(clean_peaks)
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def _resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    pixel_size_override: float = 0.0,
    max_step_ratio: float = 1.8,
    fallback_target_segments: int = 64
) -> Tuple[float, float]:
    if pixel_size_override != 0.0:
        return pixel_size_override, pixel_size_override

    if step_x_opt is not None and step_y_opt is not None:
        sx, sy = step_x_opt, step_y_opt
        ratio = sx / sy if sx > sy else sy / sx
        if ratio > max_step_ratio:
            smaller = min(sx, sy)
            return smaller, smaller
        avg = (sx + sy) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback = (min(width, height) / float(fallback_target_segments)) if fallback_target_segments else 1.0
    return max(fallback, 1.0), max(fallback, 1.0)


def _walk(
    profile: np.ndarray, step_size: float, limit: int,
    search_window_ratio: float = 0.35, min_search_window: float = 2.0, strength_threshold: float = 0.5
) -> List[int]:
    if len(profile) == 0:
        raise ValueError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * search_window_ratio, min_search_window)
    mean_val = float(np.mean(profile))

    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1))
        end_search = min(int(target + search_window), limit)

        if end_search <= start_search:
            current_pos = target
            continue

        segment = np.asarray(profile[start_search:end_search])
        max_rel = int(np.argmax(segment))
        max_val = float(segment[max_rel]) if segment.size else -1.0
        max_idx = start_search + max_rel

        if max_val > mean_val * strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target
    return cuts


def _sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    if limit == 0:
        return [0]
    
    clamped_cuts = {max(0, min(v, limit)) for v in cuts}
    clamped_cuts.add(0)
    clamped_cuts.add(limit)
    return sorted(clamped_cuts)


def _snap_uniform_cuts(
    profile: np.ndarray,
    limit: int,
    target_step: float,
    min_required: int,
    search_window_ratio: float = 0.35,
    min_search_window: float = 2.0,
    strength_threshold: float = 0.5
) -> List[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    desired_cells = int(round(limit / target_step)) if target_step > 0 and math.isfinite(target_step) else 0
    desired_cells = max(desired_cells, min_required - 1, 1)
    desired_cells = min(desired_cells, limit)

    cell_width = limit / float(desired_cells)
    search_window = max(cell_width * search_window_ratio, min_search_window)
    mean_val = float(np.mean(profile)) if len(profile) else 0.0

    cuts: List[int] = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = int(math.floor(target - search_window))
        start = max(start, prev + 1, 0)
        end = int(math.ceil(target + search_window))
        end = min(end, limit - 1)
        if end < start:
            start = prev + 1
            end = start

        segment = np.asarray(profile[start:min(end + 1, len(profile))])
        if segment.size:
            best_rel = int(np.argmax(segment))
            best_val = float(segment[best_rel])
            best_idx = start + best_rel
        else:
            best_val = -1.0
            best_idx = start

        strength_threshold_val = mean_val * strength_threshold
        if best_val < strength_threshold_val:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return _sanitize_cuts(cuts, limit)


def _stabilize_cuts(
    profile: np.ndarray,
    cuts: List[int],
    limit: int,
    sibling_cuts: Sequence[int],
    sibling_limit: int,
    min_cuts_per_axis: int = 4,
    max_step_ratio: float = 1.8,
    fallback_target_segments: int = 64,
    search_window_ratio: float = 0.35,
    min_search_window: float = 2.0,
    strength_threshold: float = 0.5
) -> List[int]:
    if limit == 0:
        return [0]

    cuts = _sanitize_cuts(cuts, limit)
    min_required = max(min_cuts_per_axis, 2)
    min_required = min(min_required, limit + 1)

    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = sibling_limit > 0 and sibling_cells >= (min_required - 1) and sibling_cells > 0
    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / float(axis_cells)
        sibling_step = sibling_limit / float(sibling_cells)
        step_ratio = axis_step / sibling_step
        steps_skewed = (
            step_ratio > max_step_ratio
            or step_ratio < 1.0 / max_step_ratio
        )

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / float(sibling_cells)
    elif fallback_target_segments > 1:
        target_step = limit / float(fallback_target_segments)
    elif axis_cells > 0:
        target_step = limit / float(axis_cells)
    else:
        target_step = float(limit)

    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return _snap_uniform_cuts(
        profile, limit, target_step, min_required,
        search_window_ratio=search_window_ratio,
        min_search_window=min_search_window,
        strength_threshold=strength_threshold
    )


def _stabilize_both_axes(
    profile_x: np.ndarray,
    profile_y: np.ndarray,
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    min_cuts_per_axis: int = 4,
    max_step_ratio: float = 1.8,
    fallback_target_segments: int = 64,
    search_window_ratio: float = 0.35,
    min_search_window: float = 2.0,
    strength_threshold: float = 0.5
) -> Tuple[List[int], List[int]]:
    col_cuts_pass1 = _stabilize_cuts(
        profile_x, list(raw_col_cuts), width, raw_row_cuts, height,
        min_cuts_per_axis=min_cuts_per_axis, max_step_ratio=max_step_ratio,
        fallback_target_segments=fallback_target_segments,
        search_window_ratio=search_window_ratio, min_search_window=min_search_window,
        strength_threshold=strength_threshold
    )
    row_cuts_pass1 = _stabilize_cuts(
        profile_y, list(raw_row_cuts), height, raw_col_cuts, width,
        min_cuts_per_axis=min_cuts_per_axis, max_step_ratio=max_step_ratio,
        fallback_target_segments=fallback_target_segments,
        search_window_ratio=search_window_ratio, min_search_window=min_search_window,
        strength_threshold=strength_threshold
    )

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / float(col_cells)
    row_step = height / float(row_cells)
    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            final_cols = _snap_uniform_cuts(
                profile_x, width, target_step, min_cuts_per_axis,
                search_window_ratio=search_window_ratio,
                min_search_window=min_search_window,
                strength_threshold=strength_threshold
            )
        else:
            final_cols = col_cuts_pass1

        if row_step > target_step * 1.2:
            final_rows = _snap_uniform_cuts(
                profile_y, height, target_step, min_cuts_per_axis,
                search_window_ratio=search_window_ratio,
                min_search_window=min_search_window,
                strength_threshold=strength_threshold
            )
        else:
            final_rows = row_cuts_pass1
        return final_cols, final_rows

    return col_cuts_pass1, row_cuts_pass1


def _resample(img: np.ndarray, cols: Sequence[int], rows: Sequence[int]) -> np.ndarray:
    if len(cols) < 2 or len(rows) < 2:
        raise ValueError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue
            cell = img[ys:ye, xs:xe]
            if cell.size == 0:
                continue
            
            # Pack RGB pixels into uint32 for fast 1D unique calculation (5x+ speedup)
            pixels = cell.reshape(-1, 3)
            packed = (
                (pixels[:, 0].astype(np.uint32) << 16) |
                (pixels[:, 1].astype(np.uint32) << 8) |
                pixels[:, 2].astype(np.uint32)
            )
            values, counts = np.unique(packed, return_counts=True)
            best_val = values[np.argmax(counts)]
            
            # Unpack RGB values back to output channels
            final_img[y_i, x_i, 0] = (best_val >> 16) & 0xFF
            final_img[y_i, x_i, 1] = (best_val >> 8) & 0xFF
            final_img[y_i, x_i, 2] = best_val & 0xFF

    return final_img


def _resample_alpha(alpha_np: np.ndarray, cols: Sequence[int], rows: Sequence[int]) -> np.ndarray:
    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_alpha = np.zeros((out_h, out_w), dtype=np.uint8)

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue
            cell = alpha_np[ys:ye, xs:xe]
            if cell.size == 0:
                continue
            
            # Optimize alpha channel majority vote using np.bincount (extremely fast)
            counts = np.bincount(cell.flat)
            final_alpha[y_i, x_i] = np.argmax(counts)

    return final_alpha


class ImageProcessor:
    """Processes images: background removal (BiRefNet) and pixel art color quantization."""
    
    def __init__(self, device: str = "cuda", offline_mode: bool = False):
        self.device = device
        self.offline_mode = offline_mode
        self.segmentation_model: Optional[Any] = None
        self.segmentation_processor: Optional[Any] = None

    def load_segmentation_model(self) -> bool:
        """Loads the BiRefNet model for background removal (Load Once, Keep Resident)."""
        if self.segmentation_model and self.segmentation_processor:
            # Model already loaded, ensure it's on the correct device
            model_device = next(self.segmentation_model.parameters()).device
            if model_device != torch.device(self.device):
                self.segmentation_model.to(self.device)
            return True

        log.info("Loading BiRefNet for background removal (will keep resident in memory)...")
        try:
            model_name = "zhengpeng7/BiRefNet"
            self.segmentation_processor = transforms.Compose(
                [
                    transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            if self.segmentation_model is None:
                raise RuntimeError("Failed to initialize segmentation_model")

            self.segmentation_model.to(self.device)
            self.segmentation_model.eval()

            if self.device == "cuda" and self.segmentation_model is not None:
                try:
                    self.segmentation_model.to(memory_format=torch.channels_last)
                except Exception as e:
                    log.warning("Could not set memory format to channels_last: %s", e)

            log.info("BiRefNet loaded successfully and will remain resident in memory")
            return True

        except Exception as e:
            log.error("Error loading BiRefNet: %s", e)
            return False

    def offload_segmentation_model(self) -> None:
        """Move BiRefNet to CPU to free VRAM after use"""
        if self.segmentation_model is not None:
            log.info("Offloading BiRefNet to CPU...")
            self.segmentation_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("BiRefNet offloaded to CPU")

    def remove_background(self, pil_image: Image.Image, threshold: float = 0.5) -> Image.Image:
        """Removes the background of the image using BiRefNet."""
        if not self.load_segmentation_model():
            raise RuntimeError("Failed to load background removal model")

        log.info("Removing background with BiRefNet...")
        try:
            with torch.inference_mode():
                rgb_image = pil_image.convert("RGB")
                
                if self.segmentation_processor is None:
                    raise RuntimeError("segmentation_processor not initialized")
                    
                input_tensor = self.segmentation_processor(rgb_image).unsqueeze(0).to(self.device)
                
                if self.segmentation_model is None:
                    raise RuntimeError("segmentation_model not initialized")

                try:
                    model_dtype = next(self.segmentation_model.parameters()).dtype
                except StopIteration:
                    raise RuntimeError("Segmentation model has no parameters")
                input_tensor = input_tensor.to(dtype=model_dtype)

                outputs = self.segmentation_model(input_tensor)
                logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs

                mask = F.interpolate(
                    logits,
                    size=pil_image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.sigmoid(mask).squeeze(0).squeeze(0)
                binary_mask = (mask > threshold).cpu().numpy().astype(np.uint8)

            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            rgba_image = pil_image.convert("RGBA")
            rgba_image.putalpha(mask_image)

            log.info("Background removal complete")
            return rgba_image

        except Exception as e:
            log.error("Error during background removal: %s", e)
            return pil_image.convert("RGBA")

    def process_for_pixel_art(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (64, 64),
        colors: int = 16,
        alpha_threshold: int = 128,
        enhance_contrast: float = 1.0,
        sharpen_amount: float = 2.0,
        pixel_snapping: bool = False,
        pixel_size_override: float = 0.0,
    ) -> Image.Image:
        """Processes the image to convert it into pixel art with sharp edges and a fixed color palette."""
        log.info("Converting to Pixel Art: size %s, colors %d, snapping %s", target_size, colors, pixel_snapping)

        if colors < 2:
            raise ValueError("Number of colors must be greater than or equal to 2")

        alpha: Optional[Image.Image] = None
        if image.mode in ("RGBA", "LA", "PA", "P"):
            try:
                temp_image = image.convert("RGBA")
                extracted_alpha = temp_image.getchannel("A")

                # Use a Lookup Table (LUT) to threshold the alpha channel
                lut = [255 if i >= alpha_threshold else 0 for i in range(256)]
                alpha = extracted_alpha.point(lut)

                image = temp_image.convert("RGB")
            except Exception as e:
                log.warning("Warning processing alpha: %s", e)
                if image.mode != "RGB":
                    image = image.convert("RGB")
        else:
            if image.mode != "RGB":
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

        if pixel_snapping:
            log.info("Running Pixel Snapping workflow...")
            try:
                # 1. Quantize the high-res image first to get clean edges for grid detection
                quantized_image = image.quantize(
                    colors=colors, method=Image.MEDIANCUT, dither=Image.NONE
                ).convert("RGB")
                
                # 2. Run snapper to get cuts
                img_np = np.array(quantized_image)
                h, w, _ = img_np.shape
                
                # Calculate automatic pixel size if override is not provided
                px_override = pixel_size_override
                if px_override == 0.0:
                    px_w = w / target_size[0]
                    px_h = h / target_size[1]
                    px_override = (px_w + px_h) / 2.0
                    log.info("Auto-calculated pixel size override: %s", px_override)
                
                profile_x, profile_y = _compute_profiles(img_np)
                step_x_opt = _estimate_step_size(profile_x)
                step_y_opt = _estimate_step_size(profile_y)
                step_x, step_y = _resolve_step_sizes(
                    step_x_opt, step_y_opt, w, h,
                    pixel_size_override=px_override,
                    fallback_target_segments=target_size[0]
                )
                
                raw_col_cuts = _walk(profile_x, step_x, w)
                raw_row_cuts = _walk(profile_y, step_y, h)
                
                col_cuts, row_cuts = _stabilize_both_axes(
                    profile_x, profile_y, raw_col_cuts, raw_row_cuts, w, h,
                    fallback_target_segments=target_size[0]
                )
                
                # 3. Resample RGB and Alpha
                snapped_rgb_np = _resample(img_np, col_cuts, row_cuts)
                image = Image.fromarray(snapped_rgb_np)
                
                if alpha is not None:
                    alpha_np = np.array(alpha)
                    snapped_alpha_np = _resample_alpha(alpha_np, col_cuts, row_cuts)
                    alpha = Image.fromarray(snapped_alpha_np)
                    
            except Exception as e:
                log.error("Error during Pixel Snapping: %s. Falling back to default resizing.", e, exc_info=True)
                # Fallback to standard resize if snapper fails
                image = image.resize(target_size, Image.NEAREST)
                if colors > 0:
                    image = image.quantize(
                        colors=colors, method=Image.MEDIANCUT, dither=Image.NONE
                    ).convert("RGB")
                if alpha is not None:
                    alpha = alpha.resize(target_size, Image.NEAREST)
        else:
            # Standard workflow
            image = image.resize(target_size, Image.NEAREST)
            if colors > 0:
                image = image.quantize(
                    colors=colors, method=Image.MEDIANCUT, dither=Image.NONE
                ).convert("RGB")
            if alpha is not None:
                alpha = alpha.resize(target_size, Image.NEAREST)

        # Force output to exactly target_size to ensure Aseprite canvas alignment
        if image.size != target_size:
            image = image.resize(target_size, Image.NEAREST)
        if alpha is not None and alpha.size != target_size:
            alpha = alpha.resize(target_size, Image.NEAREST)

        if alpha is not None:
            image = image.convert("RGBA")
            image.putalpha(alpha)

        log.info("Pixel art processing successful")
        return image

    def image_to_base64(self, image: Image.Image) -> str:
        """Converts a PIL Image into a base64 encoded string for API transmission."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return base64.b64encode(image.tobytes()).decode()
