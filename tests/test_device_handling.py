"""Device-handling regression tests (issue #356).

These pin the contract that an explicit ``device="cpu"`` run is never
auto-promoted to CUDA, even on a host where a GPU is visible. The failure this
guards against is a saliency explainer moving the shared model to CUDA in place,
which later breaks CPU training with a FloatTensor/cuda.FloatTensor mismatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bnnr.xai import OptiCAMExplainer, generate_saliency_maps


class _TinyNet(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        return self.fc(self.pool(x).flatten(1))


# n_iters=0 skips the optimization loop. That keeps the test runnable on a
# CPU-only build (a patched-available CUDA otherwise trips torch's Adam graph
# capture health check) while still exercising the device selection at the top
# of explain(), which is where the promotion bug lived.


def test_opticam_does_not_promote_cpu_model_when_cuda_visible(monkeypatch):
    """OptiCAM must keep a CPU model on CPU even if a GPU appears available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = _TinyNet()
    images = torch.rand(2, 3, 32, 32)
    labels = torch.randint(0, 3, (2,))

    maps = OptiCAMExplainer(n_iters=0).explain(model, images, labels, [model.conv])

    assert next(model.parameters()).device.type == "cpu"
    assert maps.shape[0] == 2


def test_generate_saliency_maps_keeps_cpu_when_cuda_visible(monkeypatch):
    """The public entrypoint must not create CUDA tensors for a CPU run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = _TinyNet()
    images = torch.rand(2, 3, 32, 32)
    labels = torch.randint(0, 3, (2,))

    maps = generate_saliency_maps(
        model, images, labels, [model.conv], method="opticam", n_iters=0
    )

    assert next(model.parameters()).device.type == "cpu"
    assert maps.shape[0] == 2
