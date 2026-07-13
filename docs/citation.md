# Citing BNNR

If you use BNNR in research, a report, or a downstream integration guide, cite the appropriate entry below. Pin a [release tag](https://github.com/bnnr-team/bnnr/releases) (for example `v0.6.5`) when you need a fixed software version. <!-- x-release-please-version -->

Authors (software): Mateusz Walo, Diana Morzhak, Dominika Zydorczyk, Zuzanna Saczuk ([team record](../AUTHORS.md)).

| You use | Cite |
|---------|------|
| BNNR library (any feature) | [BNNR software](#bnnr-software) |
| ICD or AICD augmentation | [ICD/AICD method paper](#icd-aicd-method-paper) + [BNNR software](#bnnr-software) |
| ICD/AICD with `gradcam` saliency | [ICD/AICD method paper](#icd-aicd-method-paper) + [BNNR software](#bnnr-software) + [pytorch-grad-cam](#bnnr-with-pytorch-grad-cam-icd-gradcam-saliency) |

## ICD/AICD method paper

Cite this when you use **ICD** or **AICD** (classification or detection), describe the method, or reproduce the tile-based masking construction. LaTeX sources: [bnnr-research](https://github.com/bnnr-team/bnnr-research).

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

Plain text (papers without BibTeX):

> Walo, M. Intelligent Coarse Dropout and Anti-ICD: Saliency-Guided Masking Augmentation for Visual Classifiers. https://doi.org/10.5281/zenodo.20581077 (2026).

## BNNR software

```bibtex
@software{walo2026bnnr,
  author = {Walo, Mateusz and Morzhak, Diana and Zydorczyk, Dominika and Saczuk, Zuzanna},
  title = {{BNNR}: Bulletproof Neural Network Recipe},
  year = {2026},
  url = {https://github.com/bnnr-team/bnnr},
  version = {0.4.14},
  doi = {10.5281/zenodo.20581372},
  license = {MIT}
}
```

Plain text (papers without BibTeX):

> Walo, M.; Morzhak, D.; Zydorczyk, D.; Saczuk, Z. BNNR (Bulletproof Neural Network Recipe). https://doi.org/10.5281/zenodo.20581372 (2026).

## BNNR with pytorch-grad-cam (ICD / `gradcam` saliency)

BNNR depends on [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for Grad-CAM saliency. Cite the [ICD/AICD method paper](#icd-aicd-method-paper), **BNNR software**, and pytorch-grad-cam when you use ICD/AICD or the [grad-cam integration example](../examples/integrations/gradcam_to_icd_loop.py).

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

When you use [`UltralyticsDetectionAdapter`](../src/bnnr/detection_adapter.py) or the [Ultralytics quickstart](../examples/integrations/ultralytics_yolo_quickstart.py), cite **BNNR software** and follow [Ultralytics](https://github.com/ultralytics/ultralytics) licensing and citation guidance for the YOLO stack you use.

## Integration docs and examples

Upstream docs (grad-cam, Ultralytics) that describe BNNR should include the entries above alongside their own project citation. Public links: [integrations.md](integrations.md), [plugin_icd.md](plugin_icd.md).
