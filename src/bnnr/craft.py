"""CRAFT (Concept Recursive Activation FacTorization) for Explainability.

Implements the CRAFT method from Fel et al., 2023:
- NMF decomposition of feature activations into visual concepts
- Gradient-based concept sensitivity: how each concept influences the output
- Recursive multi-layer decomposition for hierarchical concept discovery

References
----------
Fel, T., Music, A., Boutin, V., Picard, D., & Lefort, M. (2023).
"CRAFT: Concept Recursive Activation FacTorization for Explainability."
CVPR 2023.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import torch
from sklearn.decomposition import NMF
from torch import Tensor, nn


class NMFConceptExplainer:
    """NMF-based concept decomposition of feature activations.

    Decomposes intermediate activations of a model into ``n_concepts``
    non-negative factors using NMF. Each factor corresponds to a
    visual concept.
    """

    name = "nmf_concepts"

    def __init__(self, n_concepts: int = 10, patch_size: int = 32, use_cuda: bool = True) -> None:
        self.n_concepts = n_concepts
        self.patch_size = patch_size
        self.use_cuda = use_cuda
        self._last_concepts: np.ndarray | None = None
        self._last_basis: np.ndarray | None = None

    def _extract_activations(self, model: nn.Module, images: Tensor, target_layers: list[nn.Module]) -> Tensor:
        activations: list[Tensor] = []

        def hook_fn(_module: nn.Module, _inp: tuple[Tensor], output: Tensor) -> None:
            activations.append(output.detach())

        handles = [layer.register_forward_hook(hook_fn) for layer in target_layers]
        try:
            with torch.no_grad():
                _ = model(images)
        finally:
            for handle in handles:
                handle.remove()

        if not activations:
            raise RuntimeError("No activations captured for concept decomposition")
        return activations[-1]

    def explain(self, model: nn.Module, images: Tensor, labels: Tensor, target_layers: list[nn.Module]) -> np.ndarray:
        _ = labels  # labels kept for API compatibility with BaseExplainer.
        acts = self._extract_activations(model, images, target_layers)  # [B, C, H, W]
        b, c, h, w = acts.shape
        flat = acts.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        flat = np.maximum(flat, 0)

        nmf = NMF(n_components=min(self.n_concepts, c), init="nndsvda", max_iter=200)
        w_mat = nmf.fit_transform(flat)  # [B*H*W, K]
        h_mat = nmf.components_  # [K, C]

        concept_strength = w_mat.reshape(b, h, w, -1).mean(axis=(1, 2))
        saliency = w_mat.reshape(b, h, w, -1).max(axis=-1).astype(np.float32)
        saliency -= saliency.min(axis=(1, 2), keepdims=True)
        saliency /= saliency.max(axis=(1, 2), keepdims=True) + 1e-8

        self._last_concepts = concept_strength
        self._last_basis = h_mat
        return cast(np.ndarray, saliency)

    def get_concept_importance(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> dict[int, float]:
        _ = self.explain(model, images, labels, target_layers)
        if self._last_concepts is None:
            return {}
        mean_scores = self._last_concepts.mean(axis=0)
        return {int(idx): float(score) for idx, score in enumerate(mean_scores)}


class RealCRAFTExplainer:
    """Full CRAFT implementation with gradient-based concept sensitivity.

    This is the "real" CRAFT method from Fel et al., 2023.

    Key differences from ``NMFConceptExplainer``:
    1. **Gradient-based sensitivity**: Uses backpropagation to compute
       how much each NMF concept contributes to the model's output.
    2. **Concept importance via Sobol indices or gradient norms**:
       Measures sensitivity of the output to each concept.
    3. **Recursive decomposition**: Can apply CRAFT at multiple layers
       to discover hierarchical concepts.

    Parameters
    ----------
    n_concepts : int
        Number of concepts to discover (NMF components).
    use_cuda : bool
        Whether to use GPU for forward/backward passes.
    sensitivity_method : str
        Method for computing concept sensitivity. Options:
        - ``"gradient"`` (default): gradient of output w.r.t. concept coefficients
        - ``"deletion"``: measure output change when concept is removed
    """

    name = "craft"

    def __init__(
        self,
        n_concepts: int = 10,
        use_cuda: bool = True,
        sensitivity_method: str = "gradient",
    ) -> None:
        self.n_concepts = n_concepts
        self.use_cuda = use_cuda
        self.sensitivity_method = sensitivity_method

        # Stored after explain()
        self._basis: np.ndarray | None = None  # [K, C] NMF basis
        self._coefficients: np.ndarray | None = None  # [B, K] mean concept strengths
        self._sensitivity: np.ndarray | None = None  # [B, K] concept sensitivities
        self._concept_saliency: np.ndarray | None = None  # [B, H, W] sensitivity-weighted saliency

    def _extract_activations_with_grad(
        self,
        model: nn.Module,
        images: Tensor,
        target_layers: list[nn.Module],
    ) -> tuple[Tensor, Tensor]:
        """Extract activations with gradient capability.

        Returns
        -------
        activations : Tensor
            [B, C, H, W] activations from the target layer.
        logits : Tensor
            [B, num_classes] model output logits.
        """
        captured: list[Tensor] = []

        def hook_fn(_module: nn.Module, _inp: tuple[Tensor], output: Tensor) -> None:
            captured.append(output)

        handles = [layer.register_forward_hook(hook_fn) for layer in target_layers]
        try:
            logits = model(images)
        finally:
            for handle in handles:
                handle.remove()

        if not captured:
            raise RuntimeError("No activations captured for CRAFT")
        return captured[-1], logits

    def _nmf_decompose(self, activations: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Apply NMF to flatten activations.

        Returns
        -------
        W : ndarray [B*H*W, K]
            Concept coefficients per spatial location.
        H : ndarray [K, C]
            Concept basis vectors.
        """
        b, c, h, w = activations.shape
        flat = activations.detach().permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        flat = np.maximum(flat, 0)

        n_components = min(self.n_concepts, c)
        nmf = NMF(n_components=n_components, init="nndsvda", max_iter=300, random_state=42)
        w_mat = nmf.fit_transform(flat)  # [B*H*W, K]
        h_mat = nmf.components_  # [K, C]
        return w_mat, h_mat

    def _compute_gradient_sensitivity(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
        h_mat: np.ndarray,
    ) -> np.ndarray:
        """Compute gradient-based concept sensitivity.

        For each concept k, compute how much the model output changes
        w.r.t. the concept's contribution to the activations.

        Returns
        -------
        sensitivity : ndarray [B, K]
            Gradient norm for each concept per image.
        """
        b = images.shape[0]
        k = h_mat.shape[0]
        sensitivity = np.zeros((b, k), dtype=np.float32)

        # Convert basis vectors to tensors
        h_tensor = torch.tensor(h_mat, dtype=images.dtype, device=images.device)  # [K, C]

        model.eval()

        # Capture activations and compute gradients
        captured: list[Tensor] = []

        def hook_fn(_module: nn.Module, _inp: tuple[Tensor], output: Tensor) -> None:
            output.retain_grad()
            captured.append(output)

        handles = [layer.register_forward_hook(hook_fn) for layer in target_layers]
        try:
            logits = model(images)
        finally:
            for handle in handles:
                handle.remove()

        if not captured:
            return sensitivity

        activations = captured[-1]  # [B, C, H, W]

        # Compute gradient of target class output w.r.t. activations
        target_scores = logits.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
        target_scores.sum().backward(retain_graph=False)

        if activations.grad is None:
            return sensitivity

        grad = activations.grad  # [B, C, H, W]

        # Project gradients onto concept basis vectors
        # grad: [B, C, H, W], h_tensor: [K, C]
        # For each concept k, sensitivity = ||sum_{hw} grad . h_k||
        grad_flat = grad.permute(0, 2, 3, 1)  # [B, H, W, C]
        for concept_k in range(k):
            concept_vec = h_tensor[concept_k]  # [C]
            # dot product: [B, H, W]
            dots = (grad_flat * concept_vec).sum(dim=-1)
            # Sensitivity: mean absolute projection
            sensitivity[:, concept_k] = dots.abs().mean(dim=(1, 2)).detach().cpu().numpy()

        return sensitivity

    def _compute_deletion_sensitivity(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
        h_mat: np.ndarray,
        activations: Tensor,
    ) -> np.ndarray:
        """Compute deletion-based concept sensitivity.

        For each concept k, measure how much the model output changes
        when concept k's contribution is removed from the activations.

        Returns
        -------
        sensitivity : ndarray [B, K]
        """
        b = images.shape[0]
        k = h_mat.shape[0]
        sensitivity = np.zeros((b, k), dtype=np.float32)

        h_tensor = torch.tensor(h_mat, dtype=images.dtype, device=images.device)

        # Get baseline output
        with torch.no_grad():
            baseline_logits = model(images)
            baseline_scores = baseline_logits.gather(1, labels.view(-1, 1)).squeeze(1)

        # For each concept, remove it and measure change
        act_flat = activations.detach().permute(0, 2, 3, 1)  # [B, H, W, C]

        for concept_k in range(k):
            concept_vec = h_tensor[concept_k]  # [C]
            # Remove concept k's contribution (project out)
            projection = ((act_flat * concept_vec).sum(dim=-1, keepdim=True) * concept_vec)
            modified_act = act_flat - projection  # [B, H, W, C]

            # Create modified hook
            modified_act_bchw = modified_act.permute(0, 3, 1, 2)  # [B, C, H, W]

            def make_hook(mod_act: Tensor) -> Any:
                def hook_fn(_module: nn.Module, _inp: tuple[Tensor], _output: Tensor) -> Tensor:
                    return mod_act
                return hook_fn

            handles = [layer.register_forward_hook(make_hook(modified_act_bchw)) for layer in target_layers]
            try:
                with torch.no_grad():
                    modified_logits = model(images)
                    modified_scores = modified_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            finally:
                for handle in handles:
                    handle.remove()

            # Sensitivity = absolute change in target score
            sensitivity[:, concept_k] = (baseline_scores - modified_scores).abs().cpu().numpy()

        return sensitivity

    def explain(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> np.ndarray:
        """Run full CRAFT explanation.

        Returns
        -------
        saliency : ndarray [B, H_act, W_act]
            Concept-sensitivity-weighted saliency maps.
        """
        model.eval()

        # Step 1: Extract activations
        activations, _ = self._extract_activations_with_grad(model, images, target_layers)
        b, c, h, w = activations.shape

        # Step 2: NMF decomposition
        w_mat, h_mat = self._nmf_decompose(activations)
        self._basis = h_mat
        k = h_mat.shape[0]

        # Step 3: Compute concept sensitivity
        if self.sensitivity_method == "gradient":
            sensitivity = self._compute_gradient_sensitivity(
                model, images, labels, target_layers, h_mat,
            )
        elif self.sensitivity_method == "deletion":
            sensitivity = self._compute_deletion_sensitivity(
                model, images, labels, target_layers, h_mat, activations,
            )
        else:
            raise ValueError(f"Unknown sensitivity method: {self.sensitivity_method}")

        self._sensitivity = sensitivity

        # Step 4: Compute sensitivity-weighted saliency
        w_spatial = w_mat.reshape(b, h, w, k)  # [B, H, W, K]
        self._coefficients = w_spatial.mean(axis=(1, 2))  # [B, K]

        # Weight each concept's spatial map by its sensitivity
        sensitivity_expanded = sensitivity.reshape(b, 1, 1, k)
        weighted = (w_spatial * sensitivity_expanded).sum(axis=-1)  # [B, H, W]

        # Normalize to [0, 1]
        weighted = weighted.astype(np.float32)
        for i in range(b):
            min_v = weighted[i].min()
            max_v = weighted[i].max()
            if max_v - min_v > 1e-8:
                weighted[i] = (weighted[i] - min_v) / (max_v - min_v)
            else:
                weighted[i] = 0.0

        self._concept_saliency = weighted
        return cast(np.ndarray, weighted)

    def get_concept_importance(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> dict[int, float]:
        """Get importance score for each concept.

        If ``explain()`` has already been called, uses cached sensitivity.
        Otherwise, runs ``explain()`` first.

        Returns
        -------
        dict[int, float]
            Concept index → mean sensitivity score.
        """
        if self._sensitivity is None:
            self.explain(model, images, labels, target_layers)

        if self._sensitivity is None:
            return {}

        # Mean sensitivity across batch
        mean_sens = self._sensitivity.mean(axis=0)
        return {int(idx): float(score) for idx, score in enumerate(mean_sens)}

    def get_concept_basis(self) -> np.ndarray | None:
        """Return the NMF basis matrix [K, C] from the last explain() call."""
        return self._basis

    def get_concept_sensitivity(self) -> np.ndarray | None:
        """Return the sensitivity matrix [B, K] from the last explain() call."""
        return self._sensitivity


class RecursiveCRAFTExplainer:
    """Recursive multi-layer CRAFT decomposition.

    Applies CRAFT at multiple layers of the model to discover
    hierarchical concept relationships.

    Parameters
    ----------
    n_concepts : int
        Number of concepts per layer.
    layer_names : list[str] | None
        Names of layers to analyze (for reporting purposes).
    use_cuda : bool
        Whether to use GPU.
    sensitivity_method : str
        Sensitivity method (``"gradient"`` or ``"deletion"``).
    """

    name = "recursive_craft"

    def __init__(
        self,
        n_concepts: int = 10,
        use_cuda: bool = True,
        sensitivity_method: str = "gradient",
        layer_names: list[str] | None = None,
    ) -> None:
        self.n_concepts = n_concepts
        self.use_cuda = use_cuda
        self.sensitivity_method = sensitivity_method
        self.layer_names = layer_names

        # Results per layer
        self._layer_results: list[dict[str, Any]] = []

    def explain_recursive(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers_list: list[list[nn.Module]],
    ) -> list[np.ndarray]:
        """Run CRAFT at each layer in ``target_layers_list``.

        Parameters
        ----------
        target_layers_list:
            List of target layer groups. CRAFT is applied to each group
            independently.

        Returns
        -------
        list[np.ndarray]
            Saliency maps from each layer group, ordered from shallow to deep.
        """
        self._layer_results = []
        all_saliency: list[np.ndarray] = []

        for layer_idx, layers in enumerate(target_layers_list):
            craft = RealCRAFTExplainer(
                n_concepts=self.n_concepts,
                use_cuda=self.use_cuda,
                sensitivity_method=self.sensitivity_method,
            )
            saliency = craft.explain(model, images, labels, layers)
            importance = craft.get_concept_importance(model, images, labels, layers)

            layer_name = (
                self.layer_names[layer_idx]
                if self.layer_names and layer_idx < len(self.layer_names)
                else f"layer_{layer_idx}"
            )

            self._layer_results.append({
                "layer_name": layer_name,
                "saliency": saliency,
                "importance": importance,
                "basis": craft.get_concept_basis(),
                "sensitivity": craft.get_concept_sensitivity(),
            })
            all_saliency.append(saliency)

        return all_saliency

    def get_layer_results(self) -> list[dict[str, Any]]:
        """Return per-layer CRAFT results from the last explain_recursive() call."""
        return self._layer_results


class CRAFTExplainer(NMFConceptExplainer):
    """Backward-compatible CRAFT alias.

    .. deprecated::
        Use :class:`RealCRAFTExplainer` for the full CRAFT implementation
        with gradient-based sensitivity, or :class:`NMFConceptExplainer`
        for the lightweight NMF-only approximation.
    """

    name = "craft"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "CRAFTExplainer is a lightweight NMF approximation. "
            "For the full CRAFT implementation (Fel et al., 2023) with "
            "gradient-based sensitivity, use RealCRAFTExplainer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "NMFConceptExplainer",
    "RealCRAFTExplainer",
    "RecursiveCRAFTExplainer",
    "CRAFTExplainer",
]
