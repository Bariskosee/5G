# Class Lists — TEKNOFEST 2026 Penta Tech Models

## Vehicle Model Classes
**Model:** `vehicle_best.pt` (YOLOv8m, COCO pretrained, remapped to competition labels)
**Internal class IDs used at runtime (COCO subset):**

| ID | COCO Name | Competition Label (arac_bilgisi.tip) |
|----|-----------|--------------------------------------|
| 2  | car       | sedan                                |
| 5  | bus       | minibus                              |
| 7  | truck     | kamyon                               |

**Full project vehicle taxonomy (model_a_classes.yaml — unified detector):**

| ID | Internal Name | Competition Label |
|----|---------------|-------------------|
| 0  | sedan         | sedan             |
| 1  | suv           | suv               |
| 2  | hatchback     | hatchback         |
| 3  | pickup        | pickup            |
| 4  | minibus       | minibus           |
| 5  | panelvan      | panelvan          |
| 6  | kamyon        | kamyon            |

**Vehicle-type-only standalone config (model_a_vehicle_type_classes.yaml):**

| ID | Name      |
|----|-----------|
| 0  | sedan     |
| 1  | suv       |
| 2  | hatchback |
| 3  | pickup    |
| 4  | minibus   |
| 5  | panelvan  |
| 6  | kamyon    |

---

## Behavior Model Classes
**Model:** No `.pt` file — behavior detection is handled by MediaPipe FaceLandmarker
and rule-based modules (no YOLO training required for this category).

**YOLO-detected behaviors (from model_a_classes.yaml — unified detector):**

| ID | Internal Name | Competition Label    | Kategori       |
|----|---------------|----------------------|----------------|
| 7  | phone         | telefonla_konusma    | sofor_eylemi   |
| 8  | cigarette     | sigara_icme          | sofor_eylemi   |
| 9  | drink         | su_icme              | sofor_eylemi   |
| 10 | no_seatbelt   | emniyet_kemeri_ihlali| sofor_eylemi   |
| 11 | teknocan      | teknocan             | nesneler       |
| 12 | laptop        | bilgisayar           | nesneler       |
| 13 | person        | (ROI assignment)     | yolcular       |

**Landmark/rule-based behaviors (no class ID — not YOLO):**

| Source Module        | Competition Label | Kategori     |
|----------------------|-------------------|--------------|
| MediaPipe mouth MAR  | esneme            | sofor_eylemi |
| MediaPipe head yaw   | arkaya_bakma      | sofor_eylemi |
| MediaPipe head yaw   | etrafa_bakinma    | sofor_eylemi |
| Vehicle bbox tracker | slalom            | sofor_eylemi |

---

## Plate Model Classes
**Model:** `plate_best.pt` (YOLOv8s, trained from scratch on Turkish plates dataset)

| ID | Name          | Competition Output       |
|----|---------------|--------------------------|
| 0  | license_plate | arac_bilgisi.plaka (OCR) |

Plate text is extracted via EasyOCR (tr + en) applied lazily when bbox
confidence > 0.70, then normalized via Turkish plate regex
(`src/ocr/plate_regex.py`).

---

## Passenger Labels (Rule-based ROI, not YOLO classes)

| Competition Label | Kategori  | Assignment Rule                     |
|-------------------|-----------|-------------------------------------|
| on_koltuk         | yolcular  | Person bbox in front-right ROI      |
| arka_koltuk_1     | yolcular  | Person bbox in rear-left ROI        |
| arka_koltuk_2     | yolcular  | Person bbox in rear-right ROI       |

---

## Vehicle Color Classes (HSV+Lab rule-based, no YOLO)

beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi
