# ICD plug-in — saliency-guided augmentation in your own training loop

Use **ICD** (Intelligent Coarse Dropout) and **AICD** (Anti-ICD) inside a standard PyTorch loop — without `BNNRTrainer`, branch search, or the full BNNR dashboard.

## When to use this

- You already have a custom `train()` / `fit()` loop (Lightning hooks, research code, etc.).
- You want **XAI-guided augmentations** on top of [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) saliency maps.
- You do **not** need automatic augmentation search — you will pick ICD/AICD yourself.

For automatic branch search and reporting, use [`BNNRTrainer`](golden_path.md) or [`bnnr analyze`](analyze.md) instead.

## Install

```bash
pip install bnnr
```

`bnnr` depends on **`grad-cam>=1.5.4`** (PyPI package `grad-cam`, import `pytorch_grad_cam`). No separate install is required for the default saliency path.

## Requirements

1. **Indexed batches** — your `DataLoader` should yield `(image, label, index)` so `XAICache` can key saliency maps per sample. Wrap any `(image, label)` dataset (see [minimal example](../examples/classification/icd_plugin_minimal.py)).
2. **Images** — tensors in **`[0, 1]`** float `BCHW`, or uint8 converted inside ICD. Do not apply ImageNet `Normalize` before ICD unless you convert back for saliency.
3. **`target_layers`** — list of `nn.Module` layers for Grad-CAM (typically the last `Conv2d` before the classifier).

## Minimal loop

1. Build `model` and `target_layers`.
2. Wrap the dataset with an index (third tuple element).
3. **Precompute** saliency once per training run:

   ```python
   from bnnr.xai_cache import XAICache
   from bnnr.icd import ICD

   cache = XAICache("./xai_cache")
   cache.precompute_cache(
       model=model,
       train_loader=train_loader,
       target_layers=target_layers,
       n_samples=len(train_dataset),
       method="gradcam",  # uses pytorch-grad-cam via BNNR
   )
   ```

4. Create augmentations:

   ```python
   icd = ICD(model=model, target_layers=target_layers, cache=cache, explainer="gradcam")
   aicd = AICD(model=model, target_layers=target_layers, cache=cache, explainer="gradcam")
   ```

5. In each training step, apply on the batch (uint8 `NHWC` path):

   ```python
   aug_np = icd.apply_batch_with_labels(imgs_uint8, labels_np, sample_indices=idx_np)
   ```

   Or per image: `icd.set_label(int(label))` then `icd.apply(image_uint8_hwc)`.

Runnable script: [`examples/classification/icd_plugin_minimal.py`](../examples/classification/icd_plugin_minimal.py).

## Built on pytorch-grad-cam

BNNR uses [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for saliency when `method="gradcam"` or `method="opticam"` (both route through `GradCAM` in [`src/bnnr/xai.py`](../src/bnnr/xai.py) today).

## Citation

If you use ICD/AICD from BNNR in research, cite the **method paper**, **BNNR software**, and **pytorch-grad-cam** (saliency). Full BibTeX blocks: [citation.md](citation.md).

```bibtex
@article{walo2026icd,
  author  = {Walo, Mateusz},
  title   = {Intelligent Coarse Dropout and Anti-ICD: Saliency-Guided Masking Augmentation for Visual Classifiers},
  year    = {2026},
  doi     = {10.5281/zenodo.20581077},
  url     = {https://doi.org/10.5281/zenodo.20581077},
  note    = {Preprint},
  publisher = {Zenodo}
}
```

```bibtex
@software{walo2026bnnr,
  author = {Walo, Mateusz and Morzhak, Diana and Zydorczyk, Dominika and Saczuk, Zuzanna},
  title = {{BNNR}: Bulletproof Neural Network Recipe},
  year = {2026},
  url = {https://github.com/bnnr-team/bnnr},
  version = {0.4.13},
  doi = {10.5281/zenodo.20581372},
  license = {MIT}
}
```

## Performance

| Approach | Cost |
|----------|------|
| **`XAICache.precompute_cache`** (recommended) | One forward + CAM pass per training sample before epochs |
| **No cache** | `RuntimeWarning` + online CAM every ICD call — very slow |

Precompute on a subset (`n_samples=...`) if you only need ICD on part of the dataset.

## See also

- **Grad-CAM → ICD bridge** (raw `GradCAM` vs BNNR `ICD` on the same batch): [`examples/integrations/gradcam_to_icd_loop.py`](../examples/integrations/gradcam_to_icd_loop.py)
- Full trainer + branch search: [`golden_path.md`](golden_path.md)
- Failure analysis without retraining: [`analyze.md`](analyze.md)
- Ecosystem integrations hub: [`integrations.md`](integrations.md)

## What this is not

- **Not** a replacement for [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — it consumes saliency to **augment** images.
- **Not** a guarantee of +X% accuracy — see benchmark protocol caveats in [`benchmarks.md`](benchmarks.md).
