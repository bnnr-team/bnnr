# Ecosystem outreach — grad-cam + Ultralytics (Faza 1)

**Prerequisite:** Faza 0 merged on `main` (plugin ICD, integration examples, [integrations.md](integrations.md)).

---

## Stable URLs (copy-paste)

| Asset | URL |
|-------|-----|
| ICD plug-in guide | https://github.com/bnnr-team/bnnr/blob/main/docs/plugin_icd.md |
| Grad-CAM → ICD example | https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/gradcam_to_icd_loop.py |
| Ultralytics quickstart | https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/ultralytics_yolo_quickstart.py |
| Integrations hub | https://github.com/bnnr-team/bnnr/blob/main/docs/integrations.md |
| Detection guide | https://github.com/bnnr-team/bnnr/blob/main/docs/detection.md |

---

## A) pytorch-grad-cam — GitHub issue

**Repo:** https://github.com/jacobgil/pytorch-grad-cam  
**When:** After Faza 0 is on `main`.

**Title:** `Tutorial idea: from Grad-CAM heatmaps to saliency-guided augmentation (external example)`

**Body (template):**

```markdown
Hi — BNNR (MIT, https://github.com/bnnr-team/bnnr) uses `grad-cam` as a core dependency for saliency-guided augmentations (ICD/AICD). We precompute Grad-CAM maps and apply tile-based masking during training — see our minimal plug-in doc:

https://github.com/bnnr-team/bnnr/blob/main/docs/plugin_icd.md

We'd like to contribute a **short tutorial link** (not vendoring code into your tree) showing the path from a familiar `GradCAM` call to actionable augmentation on the same batch:

https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/gradcam_to_icd_loop.py

Would you prefer:
- (a) a README bullet under Tutorials linking externally,
- (b) a small `tutorials/bnnr_saliency_guided_augmentation.md` in your repo with links, or
- (c) another format you recommend?

Happy to open a PR once you point at the preferred option. We'll cite the pytorch-grad-cam BibTeX in our docs regardless.

Thanks!
```

**Follow-up PR (after maintainer OK):** ~40-line `tutorials/bnnr_saliency_guided_augmentation.md` with disclaimer: third-party MIT project, not affiliated.

---

## B) Ultralytics — docs PR

**Repo:** https://github.com/bnnr/ultralytics (fork `ultralytics/ultralytics`)  
**Branch:** `docs/bnnr-integration`  
**CLA:** required — sign when opening PR.

**Title:** `Docs: Add BNNR integration guide (XAI-guided aug + UltralyticsDetectionAdapter)`

**New file:** `docs/en/integrations/bnnr.md`

**Outline:**

1. Frontmatter (`description`, `keywords`: YOLO, XAI, augmentation, open source)
2. What is BNNR — MIT, link repo, one paragraph
3. What it is not — does not replace `model.train()` / `yolo` CLI
4. Installation — `pip install ultralytics bnnr` or `pip install "bnnr[ultralytics]"`
5. Quickstart — condensed from `ultralytics_yolo_quickstart.py` (`--quick`, COCO128)
6. Detection augmentations — DetectionICD/AICD, bbox-aware
7. Reports / XAI — link BNNR detection.md; note classification `bnnr analyze` separately
8. Limitations — images `[0,1]` float; Ultralytics version; first-run weight download
9. FAQ — adapter vs raw model; can I use `yolo train` separately; license (MIT vs AGPL)
10. Summary + link https://github.com/bnnr-team/bnnr/blob/main/docs/integrations.md

**Also edit:** `docs/en/integrations/index.md` — add BNNR row/card like other integrations.

**PR body opener:**

```markdown
Adds an integration guide for BNNR's `UltralyticsDetectionAdapter`: bbox-aware XAI-guided augmentation search and training reports alongside Ultralytics YOLOv8, without replacing the native `yolo train` CLI.

Runnable example: https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/ultralytics_yolo_quickstart.py
```

---

## Checklist before sending

- [ ] Faza 0 examples run locally (`--quick` for YOLO script)
- [ ] No “official partnership” or SOTA claims
- [ ] Ultralytics CLA signed
- [ ] grad-cam: issue opened **before** unsolicited large PR
- [ ] Update [integrations.md](integrations.md) upstream status table with issue/PR numbers after submission

---

## After merge (BNNR repo)

- GitHub Discussions post: “Integration docs ready — feedback welcome” (technical tone, link hub)
- Track referrers: `github.com/jacobgil`, `ultralytics.com` in GitHub Insights
