import base64
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

log = logging.getLogger(__name__)


class ImageProcessor:
    """Processes images: background removal (BiRefNet) and pixel art color quantization."""
    
    def __init__(self, device: str = "cuda", offline_mode: bool = False):
        self.device = device
        self.offline_mode = offline_mode
        self.segmentation_model: Optional = None
        self.segmentation_processor: Optional = None

    def load_segmentation_model(self) -> bool:
        """Loads the BiRefNet model for background removal (Load Once, Keep Resident)."""
        if self.segmentation_model and self.segmentation_processor:
            # Model already loaded, ensure it's on the correct device
            if self.segmentation_model.device != torch.device(self.device):
                self.segmentation_model.to(self.device)
            return True

        log.info("Loading BiRefNet for background removal (will keep resident in memory)...")
        try:
            model_name = "zhengpeng7/BiRefNet"
            self.segmentation_processor = transforms.Compose(
                [
                    transforms.Resize((352, 352), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.offline_mode,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

    def remove_background(self, pil_image: Image.Image) -> Image.Image:
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
        use_dithering: bool = False,
        alpha_threshold: int = 128,
        enhance_contrast: float = 1.0,
        sharpen_amount: float = 2.0,
    ) -> Image.Image:
        """Processes the image to convert it into pixel art with sharp edges and a fixed color palette."""
        log.info("Converting to Pixel Art: size %s, colors %d", target_size, colors)

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

        image = image.resize(target_size, Image.NEAREST)

        if colors > 0:
            dither_mode = Image.FLOYDSTEINBERG if use_dithering else Image.NONE
            image = image.quantize(
                colors=colors, method=Image.MEDIANCUT, dither=dither_mode
            ).convert("RGB")

        if alpha is not None:
            alpha = alpha.resize(target_size, Image.NEAREST)
            image = image.convert("RGBA")
            image.putalpha(alpha)

        log.info("Pixel art processing successful")
        return image

    def image_to_base64(self, image: Image.Image) -> str:
        """Converts a PIL Image into a base64 encoded string for API transmission."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return base64.b64encode(image.tobytes()).decode()
