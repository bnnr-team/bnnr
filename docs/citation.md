# Citing BNNR

If you use BNNR in research, a report, or a downstream integration guide, cite this repository. Pin a [release tag](https://github.com/bnnr-team/bnnr/releases) (for example `v0.4.10`) when you need a fixed version.

Authors: Mateusz Walo, Diana Morzhak, Dominika Zydorczyk, Zuzanna Saczuk ([team record](../AUTHORS.md)).

## BNNR

```bibtex
@software{walo2026bnnr,
  author = {Walo, Mateusz and Morzhak, Diana and Zydorczyk, Dominika and Saczuk, Zuzanna},
  title = {{BNNR}: Bulletproof Neural Network Recipe},
  year = {2026},
  url = {https://github.com/bnnr-team/bnnr},
  version = {0.4.10},
  license = {MIT}
}
```

Plain text (papers without BibTeX):

> Walo, M.; Morzhak, D.; Zydorczyk, D.; Saczuk, Z. BNNR (Bulletproof Neural Network Recipe). https://github.com/bnnr-team/bnnr (2026).

## BNNR with pytorch-grad-cam (ICD / `gradcam` saliency)

BNNR depends on [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for Grad-CAM saliency. Cite **both** BNNR and pytorch-grad-cam when you use ICD/AICD or the [grad-cam integration example](../examples/integrations/gradcam_to_icd_loop.py).

```bibtex
@misc{jacobgilpytorchcam,
  title = {PyTorch library for CAM methods},
  author = {Jacob Gildenblat and contributors},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/jacobgil/pytorch-grad-cam}},
}
```

## BNNR with Ultralytics YOLO

When you use [`UltralyticsDetectionAdapter`](../src/bnnr/detection_adapter.py) or the [Ultralytics quickstart](../examples/integrations/ultralytics_yolo_quickstart.py), cite **BNNR** and follow [Ultralytics](https://github.com/ultralytics/ultralytics) licensing and citation guidance for the YOLO stack you use.

## Integration docs and examples

Upstream docs (grad-cam, Ultralytics) that describe BNNR should include the BNNR entry above alongside their own project citation. Public links: [integrations.md](integrations.md), [plugin_icd.md](plugin_icd.md).
