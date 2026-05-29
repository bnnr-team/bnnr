# BNNR integration examples (third-party stacks)

Runnable scripts referenced from [docs/integrations.md](../../docs/integrations.md) (hub lands in a follow-up PR).

## pytorch-grad-cam

- [`gradcam_to_icd_loop.py`](gradcam_to_icd_loop.py) — same CIFAR-10 batch: raw `GradCAM` overlay vs BNNR `ICD` using `gradcam` saliency. Companion doc: [plugin_icd.md](../../docs/plugin_icd.md).

```bash
pip install bnnr
PYTHONPATH=src python examples/integrations/gradcam_to_icd_loop.py
```
