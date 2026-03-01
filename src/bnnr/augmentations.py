"""Core augmentation primitives, registry, and built-in augmentation operators."""

from __future__ import annotations

import abc
import math
import random
import warnings
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
import torch
from torch import Tensor

AugT = TypeVar("AugT", bound="BaseAugmentation")


class BaseAugmentation(abc.ABC):
    name: str = "base"
    device_compatible: bool = False

    def __init__(
        self,
        probability: float = 1.0,
        random_state: int | None = None,
        intensity: float = 1.0,
        name_override: str | None = None,
    ) -> None:
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be in [0, 1]")
        if not (0.0 <= intensity <= 2.0):
            raise ValueError("intensity must be in [0, 2]")
        self.probability = probability
        self.intensity = intensity
        self.random_state = random_state
        self._rnd = random.Random(random_state)
        if name_override is not None:
            self.name = name_override

    def validate_input(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim != 3:
            raise ValueError("Expected image shape (H, W, C)")
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.shape[2] != 3:
            raise ValueError("Expected 3 channels")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    @abc.abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_batch(self, images: np.ndarray) -> np.ndarray:
        out = images.copy()
        input_channels = images.shape[-1]
        for idx in range(images.shape[0]):
            if self._rnd.random() <= self.probability:
                aug_image = self.apply(images[idx])
                if aug_image.ndim == 2:
                    aug_image = aug_image[..., None]
                # Keep channel contract of the batch (e.g. MNIST is HxWx1).
                if input_channels == 1 and aug_image.shape[-1] == 3:
                    gray = cv2.cvtColor(aug_image, cv2.COLOR_RGB2GRAY)
                    aug_image = gray[..., None]
                elif input_channels == 3 and aug_image.shape[-1] == 1:
                    aug_image = np.repeat(aug_image, 3, axis=2)
                # Blend with original based on intensity (1.0 = full effect).
                if self.intensity < 1.0:
                    aug_image = cv2.addWeighted(
                        images[idx], 1.0 - self.intensity,
                        aug_image, self.intensity, 0,
                    )
                # OpenCV may drop a singleton channel (HWC1 -> HW). Restore shape.
                if aug_image.ndim == 2:
                    aug_image = aug_image[..., None]
                if input_channels == 1 and aug_image.shape[-1] == 3:
                    gray = cv2.cvtColor(aug_image, cv2.COLOR_RGB2GRAY)
                    aug_image = gray[..., None]
                elif input_channels == 3 and aug_image.shape[-1] == 1:
                    aug_image = np.repeat(aug_image, 3, axis=2)
                out[idx] = aug_image
        return out

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        raise NotImplementedError("Tensor-native augmentation is not implemented")

    def apply_tensor(self, images: Tensor) -> Tensor:
        if self.device_compatible:
            return self.apply_tensor_native(images)

        # Default fallback path for augmentations that do not implement GPU-native variant.
        np_images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        if np_images.max() <= 1.0:
            np_images = (np_images * 255.0).astype(np.uint8)
        else:
            np_images = np_images.astype(np.uint8)
        aug = self.apply_batch(np_images)
        tensor = torch.as_tensor(aug, device=images.device, dtype=images.dtype).permute(0, 3, 1, 2)
        if images.max() <= 1.0:
            tensor = tensor / 255.0
        return tensor

    def __repr__(self) -> str:
        parts = f"name={self.name}, probability={self.probability}"
        if self.intensity != 1.0:
            parts += f", intensity={self.intensity}"
        return f"{self.__class__.__name__}({parts})"

    def __str__(self) -> str:
        return self.name


class AugmentationRegistry:
    _registry: dict[str, type[BaseAugmentation]] = {}
    _cpu_warning_emitted: bool = False

    @classmethod
    def register(cls, name: str) -> Callable[[type[AugT]], type[AugT]]:
        def decorator(aug_cls: type[AugT]) -> type[AugT]:
            cls._registry[name] = aug_cls
            aug_cls.name = name
            return aug_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseAugmentation]:
        if name not in cls._registry:
            raise KeyError(f"Augmentation '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseAugmentation:
        aug = cls.get(name)(**kwargs)
        if not aug.device_compatible and not cls._cpu_warning_emitted:
            warnings.warn(
                "Built-in augmentations are CPU-bound (NumPy/OpenCV path). "
                f"First requested augmentation: '{name}'. "
                "For high-throughput training consider a tensor-native augmentation.",
                RuntimeWarning,
                stacklevel=2,
            )
            cls._cpu_warning_emitted = True
        return aug

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry


def _line_partitions(height: int, width: int, num_lines: int, rnd: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    region_id = np.zeros((height, width), dtype=np.int32)
    for bit in range(num_lines):
        angle = rnd.uniform(0, math.pi)
        a = math.cos(angle)
        b = math.sin(angle)
        cx = width * 0.5 + rnd.uniform(-0.2 * width, 0.2 * width)
        cy = height * 0.5 + rnd.uniform(-0.2 * height, 0.2 * height)
        c = -(a * cx + b * cy)
        signed = a * xx + b * yy + c
        region_id |= ((signed >= 0).astype(np.int32) << bit)
    return region_id


def _np_rng(rnd: random.Random) -> np.random.Generator:
    return np.random.default_rng(rnd.randrange(0, 2**32 - 1))  # type: ignore[return-value]


@AugmentationRegistry.register("augmentation_1")
@AugmentationRegistry.register("church_noise")
class ChurchNoise(BaseAugmentation):
    device_compatible: bool = True

    def __init__(self, num_lines: int = 3, noise_strength_range: tuple[float, float] = (5.0, 14.0), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_lines = num_lines
        self.noise_strength_range = noise_strength_range

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """GPU-native noise augmentation on BCHW float32 tensors in [0,1]."""
        if self._rnd.random() > self.probability:
            return images
        b, c, h, w = images.shape
        std = self._rnd.uniform(*self.noise_strength_range) / 255.0  # scale to [0,1]
        noise = torch.randn(b, 1, h, w, device=images.device, dtype=images.dtype) * std
        result = (images + noise).clamp(0.0, 1.0)
        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        rnd = self._rnd
        regions = _line_partitions(h, w, max(1, int(self.num_lines)), rnd)
        out: np.ndarray = image.astype(np.float32).copy()

        for region in np.unique(regions):
            mask = regions == region
            std = rnd.uniform(*self.noise_strength_range)
            noise_kind = rnd.choice(["white", "gaussian", "pink"])
            np_rng = _np_rng(rnd)
            if noise_kind == "white":
                noise = np_rng.uniform(-std * math.sqrt(3), std * math.sqrt(3), size=(h, w)).astype(np.float32)
            elif noise_kind == "gaussian":
                noise = np_rng.normal(0.0, std, size=(h, w)).astype(np.float32)
            else:
                spectrum = np_rng.normal(size=(h, w)) + 1j * np_rng.normal(size=(h, w))
                fy = np.fft.fftfreq(h).reshape(-1, 1)
                fx = np.fft.fftfreq(w).reshape(1, -1)
                radius = np.sqrt(fx * fx + fy * fy)
                radius[0, 0] = 1.0
                pink = np.fft.ifft2(spectrum / radius).real
                pink = (pink - pink.mean()) / (pink.std() + 1e-8)
                noise = (pink * std).astype(np.float32)

            noise3 = np.repeat(noise[:, :, None], 3, axis=2)
            out[mask] = np.clip(out[mask] + noise3[mask], 0, 255)
        return out.astype(np.uint8)


@AugmentationRegistry.register("augmentation_3")
@AugmentationRegistry.register("basic_augmentation")
class BasicAugmentation(BaseAugmentation):
    def __init__(self, num_lines: int = 1, global_blur_sigma: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_lines = num_lines
        self.global_blur_sigma = global_blur_sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        regions = _line_partitions(h, w, max(1, self.num_lines), self._rnd)
        out = image.copy()

        for region in np.unique(regions):
            mask = regions == region
            if self._rnd.random() < 0.5:
                dx = self._rnd.uniform(-2.2, 2.2)
                dy = self._rnd.uniform(-2.2, 2.2)
                transform_matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
                r = cv2.warpAffine(out[:, :, 0], transform_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
                b = cv2.warpAffine(out[:, :, 2], -transform_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
                region_aug = np.stack([r, out[:, :, 1], b], axis=2)
            else:
                imgf = out.astype(np.float32) / 255.0
                gamma = self._rnd.uniform(0.85, 1.15)
                imgf = np.clip(imgf**gamma, 0, 1)
                hsv = cv2.cvtColor((imgf * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._rnd.uniform(0.85, 1.15), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self._rnd.uniform(0.85, 1.15), 0, 255)
                region_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            out[mask] = region_aug[mask]

        if self.global_blur_sigma > 0:
            k = max(3, int(2 * round(3 * self.global_blur_sigma) + 1))
            out = cv2.GaussianBlur(out, (k, k), sigmaX=self.global_blur_sigma, sigmaY=self.global_blur_sigma)
        return out


@AugmentationRegistry.register("augmentation_5")
@AugmentationRegistry.register("dif_presets")
class DifPresets(BaseAugmentation):
    device_compatible: bool = True

    def __init__(self, num_circles_range: tuple[int, int] = (3, 6), radius_range: tuple[int, int] = (15, 60), feather: int = 35, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_circles_range = num_circles_range
        self.radius_range = radius_range
        self.feather = feather

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """GPU-native DifPresets: color temperature shifts on BCHW float32 tensors."""
        if self._rnd.random() > self.probability:
            return images
        b, c, h, w = images.shape
        kind = self._rnd.choice(["warm", "cold", "vivid", "fade"])
        if kind == "warm":
            shifts = torch.tensor(
                [self._rnd.uniform(15, 40) / 255.0, self._rnd.uniform(5, 25) / 255.0, self._rnd.uniform(-15, 5) / 255.0],
                device=images.device, dtype=images.dtype,
            )
        elif kind == "cold":
            shifts = torch.tensor(
                [self._rnd.uniform(-5, 10) / 255.0, self._rnd.uniform(-15, 5) / 255.0, self._rnd.uniform(20, 45) / 255.0],
                device=images.device, dtype=images.dtype,
            )
        elif kind == "vivid":
            # Increase contrast/saturation via scaling around mean
            scale = self._rnd.uniform(1.1, 1.4)
            mean = images.mean(dim=(2, 3), keepdim=True)
            result = mean + (images - mean) * scale
            result = result.clamp(0.0, 1.0)
            if self.intensity < 1.0:
                result = images * (1.0 - self.intensity) + result * self.intensity
            return result
        else:  # fade
            scale = self._rnd.uniform(0.5, 0.8)
            mean = images.mean(dim=(2, 3), keepdim=True)
            result = mean + (images - mean) * scale
            result = result.clamp(0.0, 1.0)
            if self.intensity < 1.0:
                result = images * (1.0 - self.intensity) + result * self.intensity
            return result

        if c >= 3:
            result = images.clone()
            result[:, 0:3] = result[:, 0:3] + shifts.view(1, 3, 1, 1)
            result = result.clamp(0.0, 1.0)
        else:
            result = (images + shifts[0]).clamp(0.0, 1.0)
        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    def _apply_augmentation(self, img: np.ndarray, kind: str) -> np.ndarray:
        if kind == "warm":
            shifts = (self._rnd.randint(15, 40), self._rnd.randint(5, 25), self._rnd.randint(-15, 5))
        elif kind == "cold":
            shifts = (self._rnd.randint(-5, 10), self._rnd.randint(-15, 5), self._rnd.randint(20, 45))
        else:
            shifts = (0, 0, 0)
        if kind in {"warm", "cold"}:
            b, g, r = cv2.split(img.astype(np.int16))
            out = cv2.merge([
                np.clip(b + shifts[2], 0, 255),
                np.clip(g + shifts[1], 0, 255),
                np.clip(r + shifts[0], 0, 255),
            ]).astype(np.uint8)
            return out
        if kind == "vivid" or kind == "fade":
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            sat = self._rnd.uniform(1.2, 1.7) if kind == "vivid" else self._rnd.uniform(0.4, 0.8)
            val = self._rnd.uniform(1.1, 1.5) if kind == "vivid" else self._rnd.uniform(0.7, 1.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if kind == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(img, -1, kernel)
        if kind == "blur":
            return cv2.GaussianBlur(img, (self._rnd.choice([5, 7, 9, 11]),) * 2, 0)
        return img

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        final: np.ndarray = image.copy().astype(np.float32)
        kinds = ["warm", "cold", "sharpen", "blur", "vivid", "fade"]
        n = self._rnd.randint(*self.num_circles_range)
        for _ in range(n):
            radius = min(self._rnd.randint(*self.radius_range), min(h, w) // 2)
            cx = self._rnd.randint(radius, max(radius + 1, w - radius))
            cy = self._rnd.randint(radius, max(radius + 1, h - radius))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (self.feather * 2 + 1, self.feather * 2 + 1), 0).astype(np.float32) / 255.0
            aug: np.ndarray = self._apply_augmentation(image, self._rnd.choice(kinds)).astype(np.float32)
            final = aug * mask[..., None] + final * (1.0 - mask[..., None])
        return np.clip(final, 0, 255).astype(np.uint8)


@AugmentationRegistry.register("augmentation_6")
@AugmentationRegistry.register("drust")
class Drust(BaseAugmentation):
    def __init__(self, layers: int = 2, base_particles: int = 500, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.layers = layers
        self.base_particles = base_particles

    def _layer(self, h: int, w: int, num_particles: int, intensity_range: tuple[int, int], max_blur: int) -> np.ndarray:
        overlay = np.zeros((h, w), dtype=np.float32)
        for _ in range(num_particles):
            x = self._rnd.randint(0, w - 1)
            y = self._rnd.randint(0, h - 1)
            overlay[y, x] += self._rnd.uniform(*intensity_range)
        k = self._rnd.choice([3, 5, max_blur])
        return cv2.GaussianBlur(overlay, (k, k), sigmaX=1)

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        dust = np.zeros((h, w), dtype=np.float32)
        for _ in range(self.layers):
            particles = int(self.base_particles * self._rnd.uniform(0.8, 1.2))
            layer = self._layer(h, w, particles, (40, 160), self._rnd.choice([3, 5, 7]))
            dust += layer
        dust = np.clip(dust, 0, 255).astype(np.uint8)
        dust3 = cv2.merge([dust] * 3)  # type: ignore[list-item]
        out: np.ndarray = cv2.addWeighted(image, 1.0, dust3, 0.5, 0)
        noise = _np_rng(self._rnd).normal(0, 3, out.shape).astype(np.float32)
        return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@AugmentationRegistry.register("augmentation_7")
@AugmentationRegistry.register("luxfer_glass")
class LuxferGlass(BaseAugmentation):
    def __init__(self, grid_range: tuple[int, int] = (100, 200), glass_thickness: tuple[float, float] = (0.03, 0.08), wave_strength: tuple[float, float] = (0.3, 0.8), blur_kernel: tuple[int, int] = (1, 1), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grid_range = grid_range
        self.glass_thickness = glass_thickness
        self.wave_strength = wave_strength
        self.blur_kernel = blur_kernel

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w = image.shape[:2]
        out = image.copy()
        grid = self._rnd.randint(min(self.grid_range[0], min(h, w)), min(self.grid_range[1], min(h, w)))
        thickness = self._rnd.uniform(*self.glass_thickness)
        wave = self._rnd.uniform(*self.wave_strength)
        blur_k = self._rnd.randint(*self.blur_kernel)
        if blur_k % 2 == 0:
            blur_k += 1
        img_blur = cv2.GaussianBlur(image, (blur_k, blur_k), 0) if blur_k > 0 else image

        for y0 in range(0, h, grid):
            for x0 in range(0, w, grid):
                y1, x1 = min(y0 + grid, h), min(x0 + grid, w)
                bh, bw = y1 - y0, x1 - x0
                if bh < 2 or bw < 2:
                    continue
                block = img_blur[y0:y1, x0:x1]
                map_x, map_y = np.meshgrid(np.arange(bw), np.arange(bh))
                map_x = map_x.astype(np.float32)
                map_y = map_y.astype(np.float32)
                norm_x = (map_x / bw - 0.5) * 2
                norm_y = (map_y / bh - 0.5) * 2
                radius = np.sqrt(norm_x**2 + norm_y**2)
                distortion = 1 + (radius**3) * thickness
                wave_x = np.sin(norm_y * np.pi * 4 + self._rnd.uniform(0, 2 * np.pi)) * wave
                wave_y = np.cos(norm_x * np.pi * 4 + self._rnd.uniform(0, 2 * np.pi)) * wave
                cx, cy = bw / 2.0, bh / 2.0
                map_x_new = cx + (map_x - cx) * distortion + wave_x
                map_y_new = cy + (map_y - cy) * distortion + wave_y
                out[y0:y1, x0:x1] = cv2.remap(block, map_x_new, map_y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        return out


@AugmentationRegistry.register("augmentation_8")
@AugmentationRegistry.register("procam")
class ProCAM(BaseAugmentation):
    device_compatible: bool = True

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """GPU-native ProCAM: white balance + gamma on BCHW float32 tensors in [0,1]."""
        if self._rnd.random() > self.probability:
            return images
        b, c, h, w = images.shape
        # Random per-channel white balance shift
        shifts = torch.tensor(
            [self._rnd.uniform(-5, 5) / 255.0 for _ in range(min(c, 3))],
            device=images.device, dtype=images.dtype,
        )
        # Pad if fewer than c channels
        if c > len(shifts):
            shifts = torch.cat([shifts, torch.zeros(c - len(shifts), device=images.device)])
        result = images + shifts.view(1, c, 1, 1)
        # Random gamma correction
        gamma = self._rnd.uniform(0.9, 1.1)
        result = result.clamp(1e-8, 1.0).pow(1.0 / gamma)
        result = result.clamp(0.0, 1.0)
        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    def _adjust_wb(self, img: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
        b, g, r = cv2.split(img.astype(np.int16))
        return cv2.merge([
            np.clip(b + shift[2], 0, 255),
            np.clip(g + shift[1], 0, 255),
            np.clip(r + shift[0], 0, 255),
        ]).astype(np.uint8)

    def _gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = self.validate_input(image)
        profile = self._rnd.choice(["cheap", "smartphone", "pro", "webcam", "darkroom"])
        if profile == "cheap":
            img = self._adjust_wb(img, (self._rnd.randint(-5, 3), self._rnd.randint(-3, 3), self._rnd.randint(-3, 5)))
            img = np.clip((img - img.mean()) * self._rnd.uniform(0.85, 1.0) + img.mean(), 0, 255).astype(np.uint8)
        elif profile == "smartphone":
            img = self._adjust_wb(img, (self._rnd.randint(-2, 5), self._rnd.randint(-2, 5), self._rnd.randint(-2, 5)))
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._rnd.uniform(1.05, 1.15), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        elif profile == "pro":
            img = self._adjust_wb(img, (self._rnd.randint(-2, 2), self._rnd.randint(-2, 2), self._rnd.randint(-2, 2)))
            img = self._gamma(img, self._rnd.uniform(0.95, 1.05))
        elif profile == "webcam":
            img = self._adjust_wb(img, (self._rnd.randint(-5, 3), self._rnd.randint(0, 5), self._rnd.randint(-3, 3)))
        else:
            img = self._adjust_wb(img, (self._rnd.randint(0, 5), self._rnd.randint(-2, 2), self._rnd.randint(0, 5)))
            img = self._gamma(img, self._rnd.uniform(1.0, 1.15))
        return img


@AugmentationRegistry.register("augmentation_9")
@AugmentationRegistry.register("smugs")
class Smugs(BaseAugmentation):
    def __init__(self, num_streaks: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_streaks = num_streaks

    def _mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        x = self._rnd.randint(0, w - 1)
        y = self._rnd.randint(0, h - 1)
        pts = [(x, y)]
        for _ in range(self._rnd.randint(10, 22)):
            x = int(np.clip(x + self._rnd.randint(-w // 10, w // 10), 0, w - 1))
            y = int(np.clip(y + self._rnd.randint(-h // 10, h // 10), 0, h - 1))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(mask, pts[i], pts[i + 1], 255, self._rnd.randint(15, 40))
        return cv2.GaussianBlur(mask, (51, 51), sigmaX=15).astype(np.float32) / 255.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        out: np.ndarray = image.astype(np.float32).copy()
        for _ in range(self.num_streaks):
            mask = self._mask(h, w)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._rnd.uniform(0.5, 2.0), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self._rnd.uniform(0.75, 2.0), 0, 255)
            aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            alpha = mask * self._rnd.uniform(0.15, 0.45)
            out = out * (1 - alpha[..., None]) + aug * alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


@AugmentationRegistry.register("augmentation_10")
@AugmentationRegistry.register("tea_stains")
class TeaStains(BaseAugmentation):
    def _palette(self, image: np.ndarray, n_colors: int = 5, patch_size: int = 20) -> list[tuple[float, float, float]]:
        h, w = image.shape[:2]
        colors: list[tuple[float, float, float]] = []
        p_h = min(h, patch_size)
        p_w = min(w, patch_size)
        for _ in range(n_colors):
            y = self._rnd.randint(0, h - p_h)
            x = self._rnd.randint(0, w - p_w)
            roi = image[y : y + p_h, x : x + p_w]
            colors.append(cv2.mean(roi)[:3])
        return colors

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w = image.shape[:2]
        out: np.ndarray = image.astype(np.float32).copy()
        current_scale = self._rnd.randint(15, 35)
        current_intensity = self._rnd.uniform(0.7, 0.95)
        shape_threshold = self._rnd.randint(160, 185)
        texture_threshold = self._rnd.randint(80, 120)

        small_h, small_w = max(1, h // current_scale), max(1, w // current_scale)
        noise_low = _np_rng(self._rnd).integers(0, 255, size=(small_h, small_w), dtype=np.uint8)
        noise_low = cv2.resize(noise_low, (w, h), interpolation=cv2.INTER_CUBIC)
        _, mask_shape = cv2.threshold(noise_low, shape_threshold, 255, cv2.THRESH_BINARY)

        noise_high = _np_rng(self._rnd).integers(0, 255, size=(h, w), dtype=np.uint8)
        noise_high = cv2.GaussianBlur(noise_high, (3, 3), 0)
        _, mask_texture = cv2.threshold(noise_high, texture_threshold, 255, cv2.THRESH_BINARY)

        final_mask = cv2.bitwise_and(mask_shape, mask_texture)
        final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
        palette = self._palette(image)

        map_h, map_w = max(1, h // 20), max(1, w // 20)
        color_map_small = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        for i in range(map_h):
            for j in range(map_w):
                color_map_small[i, j] = palette[self._rnd.randint(0, len(palette) - 1)]
        stain_color_map = cv2.resize(color_map_small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        opacity = cv2.resize(_np_rng(self._rnd).random((max(1, h // 10), max(1, w // 10))), (w, h), interpolation=cv2.INTER_CUBIC)
        alpha = (final_mask.astype(np.float32) / 255.0) * opacity * current_intensity
        out = out * (1 - alpha[..., None]) + stain_color_map * alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class TorchvisionAugmentation(BaseAugmentation):
    """Wrap any torchvision / PIL transform as a BNNR augmentation candidate.

    This allows standard transforms (``RandomHorizontalFlip``, ``ColorJitter``,
    ``RandAugment``, etc.) to participate in BNNR's iterative selection process.

    Example::

        from torchvision import transforms
        aug = TorchvisionAugmentation(
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            name_override="color_jitter",
            probability=0.5,
        )
    """

    name: str = "torchvision_aug"

    def __init__(self, transform: object, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._transform = transform

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        from PIL import Image  # local import to keep PIL optional at module level

        pil_image = Image.fromarray(image)
        result = self._transform(pil_image)  # type: ignore[operator]
        out = np.asarray(result)
        if out.ndim == 2:
            out = np.stack([out] * 3, axis=-1)
        return out.astype(np.uint8)


__all__ = [
    "BaseAugmentation",
    "AugmentationRegistry",
    "ChurchNoise",
    "BasicAugmentation",
    "DifPresets",
    "Drust",
    "LuxferGlass",
    "ProCAM",
    "Smugs",
    "TeaStains",
    "TorchvisionAugmentation",
]
