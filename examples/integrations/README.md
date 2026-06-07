# BNNR integration examples (third-party stacks)

Runnable scripts referenced from [docs/integrations.md](../../docs/integrations.md) (hub: [integrations.md](../../docs/integrations.md)).

## pytorch-grad-cam

- [`gradcam_to_icd_loop.py`](gradcam_to_icd_loop.py) — same CIFAR-10 batch: raw `GradCAM` overlay vs BNNR `ICD` using `gradcam` saliency. Companion doc: [plugin_icd.md](../../docs/plugin_icd.md).

```bash
pip install bnnr
PYTHONPATH=src python examples/integrations/gradcam_to_icd_loop.py
```

## Ultralytics YOLO

- [`ultralytics_yolo_quickstart.py`](ultralytics_yolo_quickstart.py) — `UltralyticsDetectionAdapter` on COCO128 with DetectionICD/AICD (not the `yolo` CLI).

```bash
pip install "bnnr[ultralytics]"
PYTHONPATH=src:examples/integrations python examples/integrations/ultralytics_yolo_quickstart.py --quick
```

## Citation

If you use ICD/AICD examples in a paper or upstream docs, cite the [ICD/AICD method paper](https://doi.org/10.5281/zenodo.20581077) and [BNNR software](../../docs/citation.md#bnnr-software). With grad-cam saliency or Ultralytics, cite those projects as well ([details](../../docs/citation.md)).
