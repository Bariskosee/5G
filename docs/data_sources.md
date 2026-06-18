# Data Source Tracking Table

Fill this in **as data is downloaded**. This file is the single source of
truth for what data went into each model — needed for the FTR report's
"Veri Hazırlama Süreci" section and for license compliance.

---

## Model A — Unified Detector

| Source                          | Used Classes                              | Skipped Classes      | License        | Target Count | Format       | Status | Notes |
| ------------------------------- | ----------------------------------------- | -------------------- | -------------- | ------------ | ------------ | ------ | ----- |
| VTID2 (Mendeley)                | sedan, suv, hatchback, pickup             | other                | open           | 3,000-3,800  | needs convert | TODO   | Verify on small sample first |
| Driver Behaviors (Roboflow Jui) | phone, cigarette, no_seatbelt             | seatbelt (positive)  | check          | 4,000-5,000  | YOLO         | TODO   | Inspect 100 images first |
| Abnormal Driver (Roboflow)      | drink, phone, no_seatbelt                 | duplicates of above  | check          | 1,500-2,500  | YOLO         | TODO   | Likely overlap with above |
| COCO subset                     | laptop, person                            | all other 78 classes | open           | ~1,500 each  | COCO→YOLO    | TODO   | Filter to in-cabin context where possible |
| Manual — TR vehicles            | minibus, panelvan, kamyon                 | n/a                  | self           | 1,000+       | YOLO         | TODO   | Roboflow project, team labels |
| Manual — teknocan               | teknocan                                  | n/a                  | self           | 100-200      | YOLO         | TODO   | Verify reference object first |

## Model B — Plate Detector

| Source                          | Used Classes   | License   | Target Count | Format | Status | Notes |
| ------------------------------- | -------------- | --------- | ------------ | ------ | ------ | ----- |
| Turkish Number Plates (Roboflow plakatanima) | license_plate | CC BY 4.0 | ~2,246 | YOLO | TODO | Primary TR source |
| License Plates of Vehicles in Turkey | license_plate | CC BY 4.0 | ~2,567 | check  | TODO | Secondary TR source |
| Low-light plates (small set)    | license_plate  | check     | ~335         | check  | TODO | Night augmentation |

---

## Sample-First Protocol

Before any full download, complete this checklist for every source:

- [ ] Downloaded ~50-100 sample images
- [ ] Verified class names match expectations (or remap needed is documented)
- [ ] Verified bounding box quality (random spot-check of 10 images)
- [ ] Verified license terms allow training + redistribution
- [ ] Estimated final size of full download
- [ ] Decided final target count for this source

ONLY after all six checks pass, proceed to full download.

---

## License Compliance

The FTR report must cite all data sources. Track license info here so it can
be copied directly into the report's references section.

| Source | License | Citation Format |
| ------ | ------- | --------------- |
| VTID2 | open (verify exact terms) | Author et al. (2020). VTID2 Vehicle Type Identification Dataset. Mendeley Data. |
| Turkish Number Plates | CC BY 4.0 | plakatanima. Turkish Number Plates Dataset. Roboflow Universe. |
| ... | ... | ... |
