"""Core augmentation primitives, registry, and built-in augmentation operators."""

from __future__ import annotations

import abc
import math
import random
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch
from torch import Tensor

from bnnr.image_scale import BatchScale, detect_batch_scale, from_unit, to_unit
from bnnr.utils import lazy_cv2 as cv2

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

    def apply_tensor(self, images: Tensor, *, scale: BatchScale | None = None) -> Tensor:
        """Apply this augmentation to a BCHW float batch, whatever its convention.

        Implementations only ever see [0, 1] tensors (``apply_tensor_native``)
        or unnormalised uint8 arrays (``apply_batch``). This method adapts the
        incoming batch to that and converts the result back, so a normalised or
        [0, 255] batch is no longer silently truncated on the way in.

        Pass *scale* when the caller already detected the convention, which is
        also the only way to augment a normalised batch: detecting it here has
        no access to the denormalisation statistics and raises instead.
        """
        if scale is None:
            scale = detect_batch_scale(images)

        unit = to_unit(images, scale)

        if self.device_compatible:
            return from_unit(self.apply_tensor_native(unit), scale)

        # Default fallback path for augmentations that do not implement GPU-native variant.
        np_images = np.clip(
            unit.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0, 0.0, 255.0
        ).astype(np.uint8)
        aug = self.apply_batch(np_images)
        tensor = torch.as_tensor(aug, device=images.device, dtype=images.dtype).permute(0, 3, 1, 2)
        return from_unit(tensor / 255.0, scale)

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
            # Built-ins are registered under a canonical descriptive name plus a
            # legacy "augmentation_N" alias. Keep the first name a class registers
            # as its canonical name so logs, events, and reports show e.g.
            # "church_noise" instead of "augmentation_1"; later alias registrations
            # only add a lookup key without renaming the class.
            if "name" not in aug_cls.__dict__:
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


_NOISE_KINDS = ("white", "gaussian", "pink")
_NOISE_MODES = frozenset({"regional", "uniform"})

_DIF_KINDS = ("warm", "cold", "sharpen", "blur", "vivid", "fade")
#: Global mode drops the two spatial effects: a whole-image sharpen or blur is a
#: different augmentation, not a cheaper version of a localized one.
_DIF_GLOBAL_KINDS = ("warm", "cold", "vivid", "fade")
_DIF_EFFECT_MODES = frozenset({"circles", "global"})

_CAMERA_PROFILES = ("cheap", "smartphone", "pro", "webcam", "darkroom")
_PROCAM_MODES = frozenset({"profile", "wb_gamma"})


def _np_rng(rnd: random.Random) -> np.random.Generator:
    return np.random.default_rng(rnd.randrange(0, 2**32 - 1))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tensor-side colour helpers.
#
# cv2's uint8 HSV puts S and V on [0, 255], so scaling a channel by a factor and
# clipping at 255 is the same operation as scaling s or v on [0, 1] and clamping
# at 1.0. That is what lets the numpy and tensor paths of DifPresets and ProCAM
# run the same saturation and value adjustments.
# ---------------------------------------------------------------------------


def _rgb_to_hsv(images: Tensor) -> Tensor:
    """BCHW RGB in [0, 1] -> BCHW HSV with h in [0, 1), s and v in [0, 1]."""
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    maxc, _ = images[:, :3].max(dim=1)
    minc, _ = images[:, :3].min(dim=1)
    span = maxc - minc

    # A grey pixel has no hue; 0 is the convention cv2 uses too.
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    rc = (maxc - r) / safe_span
    gc = (maxc - g) / safe_span
    bc = (maxc - b) / safe_span

    h = torch.where(
        maxc == r,
        bc - gc,
        torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = torch.where(span > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    s = torch.where(maxc > 0, span / torch.where(maxc > 0, maxc, torch.ones_like(maxc)), torch.zeros_like(maxc))
    return torch.stack([h, s, maxc], dim=1)


def _hsv_to_rgb(hsv: Tensor) -> Tensor:
    """Inverse of :func:`_rgb_to_hsv`."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    idx = (i % 6).long()

    options = torch.stack(
        [
            torch.stack([v, t, p], dim=1),
            torch.stack([q, v, p], dim=1),
            torch.stack([p, v, t], dim=1),
            torch.stack([p, q, v], dim=1),
            torch.stack([t, p, v], dim=1),
            torch.stack([v, p, q], dim=1),
        ],
        dim=0,
    )
    gather_idx = idx.unsqueeze(0).unsqueeze(2).expand(1, -1, 3, -1, -1)
    return options.gather(0, gather_idx).squeeze(0)


def _scale_saturation_value(images: Tensor, sat: float, val: float) -> Tensor:
    """Multiply the S and V channels, the tensor twin of the cv2 HSV path."""
    hsv = _rgb_to_hsv(images[:, :3].clamp(0.0, 1.0))
    hsv[:, 1] = (hsv[:, 1] * sat).clamp(0.0, 1.0)
    hsv[:, 2] = (hsv[:, 2] * val).clamp(0.0, 1.0)
    out = _hsv_to_rgb(hsv)
    if images.shape[1] > 3:
        out = torch.cat([out, images[:, 3:]], dim=1)
    return out


def _shift_channels(images: Tensor, shifts: tuple[float, float, float]) -> Tensor:
    """Add a per-channel offset given in R, G, B order, in [0, 1] units."""
    out = images.clone()
    n = min(images.shape[1], 3)
    offset = torch.tensor(shifts[:n], device=images.device, dtype=images.dtype)
    out[:, :n] = out[:, :n] + offset.view(1, n, 1, 1)
    return out.clamp(0.0, 1.0)


_SHARPEN_KERNEL = ((0.0, -1.0, 0.0), (-1.0, 5.0, -1.0), (0.0, -1.0, 0.0))


def _sharpen(images: Tensor) -> Tensor:
    """The 3x3 sharpen kernel cv2.filter2D runs, with matching reflect padding."""
    c = images.shape[1]
    kernel = torch.tensor(_SHARPEN_KERNEL, device=images.device, dtype=images.dtype)
    kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="reflect")
    return torch.nn.functional.conv2d(padded, kernel, groups=c).clamp(0.0, 1.0)


def _gaussian_blur_t(images: Tensor, kernel_size: int) -> Tensor:
    """Odd-sized Gaussian blur with the sigma cv2 derives from the kernel."""
    import torchvision.transforms.functional as tv_functional

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    limit = 2 * min(images.shape[-2], images.shape[-1]) - 1
    if k > limit:
        k = max(3, limit if limit % 2 == 1 else limit - 1)
    return tv_functional.gaussian_blur(images, [k, k])


def _feather_kernel(feather: int, h: int, w: int) -> int:
    """Odd blur kernel for a feather radius, capped to fit inside the image.

    torch's reflect padding refuses a pad wider than the dimension it pads, and
    cv2 quietly widens its border instead. Capping here, on the shared path,
    keeps the numpy and tensor masks the same rather than letting the two
    disagree on small images.
    """
    k = max(3, int(feather) * 2 + 1)
    limit = 2 * min(h, w) - 1
    if k > limit:
        k = limit if limit % 2 == 1 else limit - 1
    return max(3, k)


def _feathered_circle(
    h: int, w: int, cx: int, cy: int, radius: int, feather: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """(1, 1, H, W) mask in [0, 1]: a filled circle, blurred by ``feather``.

    Same construction as the numpy path (``cv2.circle`` then a Gaussian blur of
    kernel :func:`_feather_kernel`), so the two produce the same footprint.
    """
    ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    hard = (((xs - cx) ** 2 + (ys - cy) ** 2) <= radius * radius).to(torch.float32)
    mask = _gaussian_blur_t(hard.view(1, 1, h, w), _feather_kernel(feather, h, w))
    return mask.to(dtype)


@AugmentationRegistry.register("augmentation_1")
@AugmentationRegistry.register("church_noise")
class ChurchNoise(BaseAugmentation):
    """Line-partitioned regional noise, identical on the numpy and tensor paths.

    ``num_lines`` random straight lines split the image into regions, and each
    region gets its own noise kind (white, gaussian or pink) and its own
    standard deviation drawn from ``noise_strength_range``.

    ``noise_mode`` selects the transform, and both paths implement both modes,
    so the device no longer decides which transform runs:

    ``"regional"`` (default)
        The transform described above. This is what the numpy path has always
        done and what the tensor path now does too.
    ``"uniform"``
        One Gaussian noise field over the whole image with a single standard
        deviation. Cheaper, and what the tensor path used to do
        unconditionally. ``num_lines`` has no effect in this mode.

    Regional noise on the tensor path costs one noise field per region rather
    than one per image. Pass ``noise_mode="uniform"`` to trade the regional
    structure back for that, or to reproduce runs made before this change.
    """

    device_compatible: bool = True

    def __init__(
        self,
        num_lines: int = 3,
        noise_strength_range: tuple[float, float] = (5.0, 14.0),
        noise_mode: str = "regional",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if noise_mode not in _NOISE_MODES:
            raise ValueError(
                f"noise_mode must be one of {sorted(_NOISE_MODES)}, got {noise_mode!r}"
            )
        self.num_lines = num_lines
        self.noise_strength_range = noise_strength_range
        self.noise_mode = noise_mode

    def __repr__(self) -> str:
        # Extend, do not replace: the base repr carries probability and intensity.
        return f"{super().__repr__()[:-1]}, noise_mode={self.noise_mode})"

    # ------------------------------------------------------------------
    # Shared plan: both paths draw the same regions and the same per-region
    # parameters from the same RNG, so they are the same transform.
    # ------------------------------------------------------------------

    def _regional_plan(self, h: int, w: int) -> tuple[np.ndarray, list[tuple[int, float, str]]]:
        """Draw the region map and the (region, std, kind) triples for one image."""
        regions = _line_partitions(h, w, max(1, int(self.num_lines)), self._rnd)
        plan = [
            (int(region), self._rnd.uniform(*self.noise_strength_range), self._rnd.choice(_NOISE_KINDS))
            for region in np.unique(regions)
        ]
        return regions, plan

    # ------------------------------------------------------------------
    # Tensor path
    # ------------------------------------------------------------------

    def _torch_noise(
        self,
        kind: str,
        std: float,
        h: int,
        w: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> Tensor:
        """One (H, W) noise field of the given kind, scaled to *std*."""
        if kind == "white":
            span = std * math.sqrt(3.0)
            uniform = torch.rand(h, w, device=device, dtype=torch.float32, generator=generator)
            return ((uniform * 2.0 - 1.0) * span).to(dtype)
        if kind == "gaussian":
            normal = torch.randn(h, w, device=device, dtype=torch.float32, generator=generator)
            return (normal * std).to(dtype)

        # Pink: 1/f spectrum, mirroring the numpy path.
        real = torch.randn(h, w, device=device, dtype=torch.float32, generator=generator)
        imag = torch.randn(h, w, device=device, dtype=torch.float32, generator=generator)
        fy = torch.fft.fftfreq(h, device=device).reshape(-1, 1)
        fx = torch.fft.fftfreq(w, device=device).reshape(1, -1)
        radius = torch.sqrt(fx * fx + fy * fy)
        radius[0, 0] = 1.0
        pink = torch.fft.ifft2(torch.complex(real, imag) / radius).real
        pink = (pink - pink.mean()) / (pink.std() + 1e-8)
        return (pink * std).to(dtype)

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """Tensor-native noise on BCHW float32 tensors in [0, 1]."""
        if self._rnd.random() > self.probability:
            return images
        b, _, h, w = images.shape
        generator = torch.Generator(device=images.device)
        generator.manual_seed(self._rnd.randrange(0, 2**63 - 1))

        if self.noise_mode == "uniform":
            std = self._rnd.uniform(*self.noise_strength_range) / 255.0
            noise = torch.randn(
                b, 1, h, w, device=images.device, dtype=images.dtype, generator=generator
            ) * std
        else:
            noise = torch.zeros(b, 1, h, w, device=images.device, dtype=images.dtype)
            for idx in range(b):
                regions, plan = self._regional_plan(h, w)
                region_map = torch.as_tensor(regions, device=images.device)
                for region, std, kind in plan:
                    sample = self._torch_noise(
                        kind,
                        std / 255.0,  # plan is in pixel units, tensors are in [0, 1]
                        h,
                        w,
                        device=images.device,
                        dtype=images.dtype,
                        generator=generator,
                    )
                    noise[idx, 0] = torch.where(region_map == region, sample, noise[idx, 0])

        result = (images + noise).clamp(0.0, 1.0)
        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    # ------------------------------------------------------------------
    # Numpy path
    # ------------------------------------------------------------------

    def _np_noise(self, kind: str, std: float, h: int, w: int, np_rng: np.random.Generator) -> np.ndarray:
        """One (H, W) noise field of the given kind, scaled to *std*."""
        if kind == "white":
            return np_rng.uniform(-std * math.sqrt(3), std * math.sqrt(3), size=(h, w)).astype(np.float32)
        if kind == "gaussian":
            return np_rng.normal(0.0, std, size=(h, w)).astype(np.float32)

        spectrum = np_rng.normal(size=(h, w)) + 1j * np_rng.normal(size=(h, w))
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        radius = np.sqrt(fx * fx + fy * fy)
        radius[0, 0] = 1.0
        pink = np.fft.ifft2(spectrum / radius).real
        pink = (pink - pink.mean()) / (pink.std() + 1e-8)
        return (pink * std).astype(np.float32)

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape
        out: np.ndarray = image.astype(np.float32).copy()

        if self.noise_mode == "uniform":
            std = self._rnd.uniform(*self.noise_strength_range)
            noise = self._np_noise("gaussian", std, h, w, _np_rng(self._rnd))
            out = np.clip(out + noise[:, :, None], 0, 255)
            return out.astype(np.uint8)

        regions, plan = self._regional_plan(h, w)
        for region, std, kind in plan:
            mask = regions == region
            noise = self._np_noise(kind, std, h, w, _np_rng(self._rnd))
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
    """Localized colour effects, identical on the numpy and tensor paths.

    ``effect_mode`` selects the transform, and both paths implement both modes,
    so the device no longer decides which transform runs:

    ``"circles"`` (default)
        ``num_circles_range`` feathered circles, each with its own effect drawn
        from warm, cold, sharpen, blur, vivid and fade, composited over the
        original. This is what the numpy path has always done and what the
        tensor path now does too.
    ``"global"``
        One effect applied to the whole image, drawn from warm, cold, vivid and
        fade. Cheaper, and close to what the tensor path used to do
        unconditionally. ``num_circles_range``, ``radius_range`` and ``feather``
        have no effect in this mode.

    Both paths draw their circles, their effects and every effect parameter from
    one shared ``_circle_plan`` / ``_global_plan``, so the two are the same
    transform by construction rather than by inspection.

    Circle mode on the tensor path costs one feathered mask and one full-image
    effect per circle. Pass ``effect_mode="global"`` to trade the locality back
    for that, or to approximate runs made before this change.

    The channel order of the warm and cold shifts was wrong on the numpy path,
    which added the blue offset to red and the red offset to blue: ``warm``
    cooled the image and ``cold`` warmed it. Both paths now apply the offsets in
    R, G, B order.
    """

    device_compatible: bool = True

    def __init__(
        self,
        num_circles_range: tuple[int, int] = (3, 6),
        radius_range: tuple[int, int] = (15, 60),
        feather: int = 35,
        effect_mode: str = "circles",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if effect_mode not in _DIF_EFFECT_MODES:
            raise ValueError(
                f"effect_mode must be one of {sorted(_DIF_EFFECT_MODES)}, got {effect_mode!r}"
            )
        self.num_circles_range = num_circles_range
        self.radius_range = radius_range
        self.feather = feather
        self.effect_mode = effect_mode

    def __repr__(self) -> str:
        # Extend, do not replace: the base repr carries probability and intensity.
        return f"{super().__repr__()[:-1]}, effect_mode={self.effect_mode})"

    # ------------------------------------------------------------------
    # Shared plan: both paths draw the same circles and the same per-effect
    # parameters from the same RNG, so they are the same transform.
    # ------------------------------------------------------------------

    def _effect_params(self, kind: str) -> dict[str, Any]:
        """Draw the parameters of one effect. Shifts are in pixel units."""
        if kind == "warm":
            return {
                "shifts": (
                    float(self._rnd.randint(15, 40)),
                    float(self._rnd.randint(5, 25)),
                    float(self._rnd.randint(-15, 5)),
                )
            }
        if kind == "cold":
            return {
                "shifts": (
                    float(self._rnd.randint(-5, 10)),
                    float(self._rnd.randint(-15, 5)),
                    float(self._rnd.randint(20, 45)),
                )
            }
        if kind == "vivid":
            return {"sat": self._rnd.uniform(1.2, 1.7), "val": self._rnd.uniform(1.1, 1.5)}
        if kind == "fade":
            return {"sat": self._rnd.uniform(0.4, 0.8), "val": self._rnd.uniform(0.7, 1.0)}
        if kind == "blur":
            return {"kernel": int(self._rnd.choice([5, 7, 9, 11]))}
        return {}

    def _circle_plan(self, h: int, w: int) -> list[tuple[int, int, int, str, dict[str, Any]]]:
        """Draw ``(cx, cy, radius, kind, params)`` for every circle of one image."""
        plan: list[tuple[int, int, int, str, dict[str, Any]]] = []
        for _ in range(self._rnd.randint(*self.num_circles_range)):
            radius = min(self._rnd.randint(*self.radius_range), min(h, w) // 2)
            cx = self._rnd.randint(radius, max(radius + 1, w - radius))
            cy = self._rnd.randint(radius, max(radius + 1, h - radius))
            kind = self._rnd.choice(_DIF_KINDS)
            plan.append((cx, cy, radius, kind, self._effect_params(kind)))
        return plan

    def _global_plan(self) -> tuple[str, dict[str, Any]]:
        """Draw the single effect used by ``effect_mode="global"``."""
        kind = self._rnd.choice(_DIF_GLOBAL_KINDS)
        return kind, self._effect_params(kind)

    # ------------------------------------------------------------------
    # Tensor path
    # ------------------------------------------------------------------

    def _effect_tensor(self, images: Tensor, kind: str, params: dict[str, Any]) -> Tensor:
        """Apply one effect to a whole BCHW batch in [0, 1]."""
        if kind in {"warm", "cold"}:
            r, g, b = params["shifts"]
            return _shift_channels(images, (r / 255.0, g / 255.0, b / 255.0))
        if kind in {"vivid", "fade"}:
            return _scale_saturation_value(images, params["sat"], params["val"])
        if kind == "sharpen":
            return _sharpen(images)
        if kind == "blur":
            return _gaussian_blur_t(images, params["kernel"])
        return images

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """Tensor-native DifPresets on BCHW float32 tensors in [0, 1]."""
        if self._rnd.random() > self.probability:
            return images
        _, _, h, w = images.shape

        if self.effect_mode == "global":
            kind, params = self._global_plan()
            result = self._effect_tensor(images, kind, params).clamp(0.0, 1.0)
        else:
            result = images.clone()
            for cx, cy, radius, kind, params in self._circle_plan(h, w):
                mask = _feathered_circle(
                    h, w, cx, cy, radius, self.feather,
                    device=images.device, dtype=images.dtype,
                )
                # The effect is computed from the untouched image, as on the
                # numpy path: circles layer over the original, not over each
                # other's output.
                effect = self._effect_tensor(images, kind, params)
                result = effect * mask + result * (1.0 - mask)
            result = result.clamp(0.0, 1.0)

        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    # ------------------------------------------------------------------
    # Numpy path
    # ------------------------------------------------------------------

    def _effect_numpy(self, img: np.ndarray, kind: str, params: dict[str, Any]) -> np.ndarray:
        """Apply one effect to a single HWC uint8 RGB image."""
        if kind in {"warm", "cold"}:
            r, g, b = params["shifts"]
            channels = cv2.split(img.astype(np.int16))
            merged = cv2.merge([
                np.clip(channels[0] + r, 0, 255),
                np.clip(channels[1] + g, 0, 255),
                np.clip(channels[2] + b, 0, 255),
            ])
            return np.asarray(merged, dtype=np.uint8)
        if kind in {"vivid", "fade"}:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["sat"], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * params["val"], 0, 255)
            return np.asarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        if kind == "sharpen":
            kernel = np.array(_SHARPEN_KERNEL, dtype=np.float32)
            return np.asarray(cv2.filter2D(img, -1, kernel))
        if kind == "blur":
            k = params["kernel"]
            return np.asarray(cv2.GaussianBlur(img, (k, k), 0))
        return img

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = self.validate_input(image)
        h, w, _ = image.shape

        if self.effect_mode == "global":
            kind, params = self._global_plan()
            return np.clip(self._effect_numpy(image, kind, params), 0, 255).astype(np.uint8)

        final: np.ndarray = image.copy().astype(np.float32)
        for cx, cy, radius, kind, params in self._circle_plan(h, w):
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_u8, (cx, cy), radius, 255, -1)
            k = _feather_kernel(self.feather, h, w)
            mask = cv2.GaussianBlur(mask_u8, (k, k), 0).astype(np.float32) / 255.0
            aug = self._effect_numpy(image, kind, params).astype(np.float32)
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
    """Camera-profile simulation, identical on the numpy and tensor paths.

    ``camera_mode`` selects the transform, and both paths implement both modes,
    so the device no longer decides which transform runs:

    ``"profile"`` (default)
        One of cheap, smartphone, pro, webcam or darkroom, each a white balance
        shift plus the contrast, saturation or gamma step that profile implies.
        This is what the numpy path has always done and what the tensor path now
        does too.
    ``"wb_gamma"``
        A white balance shift and a gamma correction, with no profile. Cheaper,
        and what the tensor path used to do unconditionally.

    Both paths draw the profile and every parameter from one shared
    ``_camera_plan``, so the two are the same transform by construction rather
    than by inspection.

    The saturation step runs through cv2's uint8 HSV on the numpy path and
    through the equivalent float HSV on the tensor path, so the two agree to
    within uint8 quantisation rather than exactly.

    White balance offsets were applied in reversed channel order on the numpy
    path, so a profile that meant to warm an image cooled it. Both paths now
    apply them in R, G, B order.
    """

    device_compatible: bool = True

    def __init__(self, camera_mode: str = "profile", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if camera_mode not in _PROCAM_MODES:
            raise ValueError(
                f"camera_mode must be one of {sorted(_PROCAM_MODES)}, got {camera_mode!r}"
            )
        self.camera_mode = camera_mode

    def __repr__(self) -> str:
        # Extend, do not replace: the base repr carries probability and intensity.
        return f"{super().__repr__()[:-1]}, camera_mode={self.camera_mode})"

    # ------------------------------------------------------------------
    # Shared plan
    # ------------------------------------------------------------------

    def _camera_plan(self) -> tuple[str, dict[str, Any]]:
        """Draw the profile and its parameters. Shifts are in pixel units."""
        if self.camera_mode == "wb_gamma":
            shifts = tuple(float(self._rnd.uniform(-5, 5)) for _ in range(3))
            return "wb_gamma", {"shifts": shifts, "gamma": self._rnd.uniform(0.9, 1.1)}

        profile = self._rnd.choice(_CAMERA_PROFILES)
        if profile == "cheap":
            return profile, {
                "shifts": (
                    float(self._rnd.randint(-5, 3)),
                    float(self._rnd.randint(-3, 3)),
                    float(self._rnd.randint(-3, 5)),
                ),
                "contrast": self._rnd.uniform(0.85, 1.0),
            }
        if profile == "smartphone":
            return profile, {
                "shifts": tuple(float(self._rnd.randint(-2, 5)) for _ in range(3)),
                "sat": self._rnd.uniform(1.05, 1.15),
            }
        if profile == "pro":
            return profile, {
                "shifts": tuple(float(self._rnd.randint(-2, 2)) for _ in range(3)),
                "gamma": self._rnd.uniform(0.95, 1.05),
            }
        if profile == "webcam":
            return profile, {
                "shifts": (
                    float(self._rnd.randint(-5, 3)),
                    float(self._rnd.randint(0, 5)),
                    float(self._rnd.randint(-3, 3)),
                )
            }
        return profile, {
            "shifts": (
                float(self._rnd.randint(0, 5)),
                float(self._rnd.randint(-2, 2)),
                float(self._rnd.randint(0, 5)),
            ),
            "gamma": self._rnd.uniform(1.0, 1.15),
        }

    # ------------------------------------------------------------------
    # Tensor path
    # ------------------------------------------------------------------

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """Tensor-native ProCAM on BCHW float32 tensors in [0, 1]."""
        if self._rnd.random() > self.probability:
            return images
        profile, params = self._camera_plan()

        r, g, b = params["shifts"]
        result = _shift_channels(images, (r / 255.0, g / 255.0, b / 255.0))

        if "contrast" in params:
            mean = result.mean()
            result = (mean + (result - mean) * params["contrast"]).clamp(0.0, 1.0)
        if "sat" in params:
            result = _scale_saturation_value(result, params["sat"], 1.0)
        if "gamma" in params:
            result = result.clamp(1e-8, 1.0).pow(1.0 / max(params["gamma"], 1e-6)).clamp(0.0, 1.0)

        if self.intensity < 1.0:
            result = images * (1.0 - self.intensity) + result * self.intensity
        return result

    # ------------------------------------------------------------------
    # Numpy path
    # ------------------------------------------------------------------

    def _adjust_wb(self, img: np.ndarray, shift: tuple[float, float, float]) -> np.ndarray:
        """Add a per-channel offset given in R, G, B order."""
        channels = cv2.split(img.astype(np.int16))
        merged = cv2.merge([
            np.clip(channels[0] + shift[0], 0, 255),
            np.clip(channels[1] + shift[1], 0, 255),
            np.clip(channels[2] + shift[2], 0, 255),
        ])
        return np.asarray(merged, dtype=np.uint8)

    def _gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return np.asarray(cv2.LUT(img, table))

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = self.validate_input(image)
        profile, params = self._camera_plan()

        img = self._adjust_wb(img, params["shifts"])

        if "contrast" in params:
            img = np.clip(
                (img - img.mean()) * params["contrast"] + img.mean(), 0, 255
            ).astype(np.uint8)
        if "sat" in params:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params["sat"], 0, 255)
            img = np.asarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        if "gamma" in params:
            img = self._gamma(img, params["gamma"])
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
        noise_low_small = _np_rng(self._rnd).integers(0, 255, size=(small_h, small_w), dtype=np.uint8)
        noise_low = cv2.resize(noise_low_small, (w, h), interpolation=cv2.INTER_CUBIC)
        _, mask_shape = cv2.threshold(noise_low, shape_threshold, 255, cv2.THRESH_BINARY)

        noise_high_u8 = _np_rng(self._rnd).integers(0, 255, size=(h, w), dtype=np.uint8)
        noise_high = cv2.GaussianBlur(noise_high_u8, (3, 3), 0)
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
