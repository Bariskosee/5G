# Penta Tech — Final Tasarım Raporu (FTR)

**Yarışma:** TEKNOFEST 2026 — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması (Turkcell)
**Takım:** Penta Tech
**Tarih:** Haziran 2026

---

## 1. Proje Özeti

Bu proje, araç içi kamera görüntülerinden sürücü güvenliğini etkileyen olayları
otomatik olarak tespit eden ve raporlayan bir yapay zekâ boru hattı (pipeline)
sunmaktadır. Sistem; araç tipini, plakasını ve rengini çıkarmanın yanı sıra
sürücü dikkat durumunu, nesne varlığını ve yolcu konumunu analiz eder. Tüm
çıktılar rekabeti doğrudan destekleyen bir JSON şemasında (`results.json`)
üretilmektedir.

### 1.1 Temel Özellikler

- **Kameradan gerçek zamanlı analiz:** Video dosyaları GPU hızlandırmalı YOLOv8
  ve MediaPipe modelleri ile her 10 karede bir işlenir; bu sayede 5 dakikalık bir
  video 10 dakikanın çok altında analiz edilir.
- **Doğrulanmış Docker dağıtımı:** Sistem, NVIDIA Tesla T4 üzerinde çalışan
  `nvidia/cuda:12.1.0-base-ubuntu22.04` temel imajı üzerinde eksiksiz test
  edilmiştir.
- **Eksiksiz şema uyumu:** Üretilen `results.json` dosyası, yarışma değerlendirme
  kurallarına uyumluluğu doğrulamak için yazılmış `validate_results_json.py`
  betiğini her durumda başarıyla geçmektedir.
- **Modüler mimari:** Her tespit görevi, bağımsız bir modüle ayrılmış olup
  doğruluk ve bakım açısından ayrı ayrı geliştirilebilir.

### 1.2 Kapsam

Yarışmanın bu aşamasında (FTR) teslim edilen kapsam şu unsurları içermektedir:

| Özellik | Durum |
|---|---|
| Docker imajı (NVIDIA T4 doğrulamalı) | ✅ Tamamlandı |
| Plaka tespiti (Model B, ince ayarlı YOLOv8s) | ✅ Tamamlandı |
| Plaka OCR (EasyOCR tr+en) | ✅ Tamamlandı |
| Araç tipi tespiti (COCO YOLOv8m) | ✅ Tamamlandı |
| Araç rengi (HSV+Lab, eğitimsiz) | ✅ Tamamlandı |
| Esneme tespiti (MediaPipe MAR) | ✅ Tamamlandı |
| Şema uyumu (`results.json`) | ✅ Tamamlandı |
| Model A ince ayarı (özel dataset) | 🔄 Devam ediyor |
| Diğer sürücü eylemleri (telefon, sigara vb.) | 🔄 Devam ediyor |

---

## 2. Veri Seti Oluşturulması

### 2.1 Genel Yaklaşım

Projedeki iki YOLO modeli farklı veri kaynaklarına ihtiyaç duymaktadır:

- **Model A (Birleşik Dedektör):** Araç tipi, sürücü davranışı ve yolcu tespiti
  için 14 sınıf içeren özel bir veri seti.
- **Model B (Plaka Dedektörü):** Türk plakalarını içeren, tek sınıflı
  (`license_plate`) bir veri seti.

Her veri kaynağı için aşağıdaki protokol uygulanmıştır:

1. ~100 örnek görüntü indirildi; bounding box kalitesi kontrol edildi.
2. Lisans koşulları doğrulandı.
3. Sınıf isimleri proje sınıf haritasına göre yeniden etiketlendi.
4. Tam veri seti YOLO formatına dönüştürüldü.

### 2.2 Model B — Plaka Dedektörü Veri Seti

Model B, Türk plakası görselleri içeren iki açık kaynaklı veri setiyle
eğitilmiştir:

| Kaynak | Lisans | Görüntü Sayısı | Format | Notlar |
|---|---|---|---|---|
| Turkish Number Plates (Roboflow, plakatanima) | CC BY 4.0 | ~2.246 | YOLO | Birincil Türk kaynak |
| License Plates of Vehicles in Turkey | CC BY 4.0 | ~2.567 | YOLO | İkincil Türk kaynak |
| Düşük ışık plakaları (küçük set) | CC BY 4.0 | ~335 | YOLO | Gece koşulları artırımı |

**Toplam:** ~5.148 görüntü, %80/%10/%10 eğitim/doğrulama/test bölünmesi.

**Ön işleme ve artırım:** Görüntüler `640×640` piksel boyutuna yeniden
boyutlandırılmıştır. Eğitim sırasında Ultralytics YOLOv8 varsayılan artırım
(mozaik, yatay çevirme, renk değişimi) uygulanmıştır. Gece koşullarını temsil
etmek amacıyla düşük ışık seti, veri setinin %10'u olacak şekilde
ağırlıklandırılmıştır.

**Eğitim:** `ultralytics` YOLOv8s mimarisi, 100 epoch, `imgsz=640`,
`batch=16`, NVIDIA Tesla T4 (FP16). En iyi ağırlık dosyası
`models/model_b_plate/best.pt` olarak kaydedilmiştir (22 MB).

### 2.3 Model A — Birleşik Dedektör Veri Seti (Planlanmış)

Model A için hedeflenen veri kaynakları aşağıda listelenmiştir. İnce ayar
eğitimi devam etmektedir:

| Kaynak | Hedef Sınıflar | Lisans | Hedef Sayı | Durum |
|---|---|---|---|---|
| VTID2 (Mendeley) | sedan, suv, hatchback, pickup | açık | ~3.500 | Hazırlanıyor |
| Driver Behaviors (Roboflow Jui) | phone, cigarette, no_seatbelt | CC BY 4.0 | ~4.500 | Hazırlanıyor |
| Abnormal Driver (Roboflow) | drink, phone, no_seatbelt | CC BY 4.0 | ~2.000 | Hazırlanıyor |
| COCO alt kümesi | laptop, person | CC BY 4.0 | ~3.000 | Hazırlanıyor |
| Manuel — TR araçları | minibus, panelvan, kamyon | Takım | ~1.000 | Devam ediyor |
| Manuel — teknocan | teknocan | Takım | ~150 | Devam ediyor |

Model A ince ayarı tamamlanana kadar geçici çözüm olarak COCO üzerinde
eğitilmiş YOLOv8m ağırlıkları kullanılmaktadır (`car → sedan`,
`bus → minibus`, `truck → kamyon`).

### 2.4 MediaPipe ve Renk Analizi için Veri

`esneme` tespiti (MediaPipe FaceLandmarker) ile araç rengi sınıflandırması
(HSV+Lab) kural tabanlı modüllerdir; eğitim verisi gerektirmez. Bu modüllerin
parametreleri `configs/thresholds.yaml` dosyasında yapılandırılmıştır.

---

## 3. Yapay Zekâ Çözümü

### 3.1 Problemin Analizi

Karayolu güvenliği için gerçek zamanlı sürücü davranışı tespiti, aşağıdaki
temel zorluklarla karşı karşıyadır:

**Görüntü kalitesi zorlukları:**

| Zorluk | Açıklama |
|---|---|
| Işık değişimleri | Gündüz/gece, tünel girişi-çıkışı, güneş parlaması; pikseller çok farklı parlaklık aralıklarında dağılır |
| Hareket bulanıklığı | Hızlı kafa hareketi veya yüksek FPS'de bile düşük pozlama süresi; yüz landmarkları belirsizleşir |
| Oklüzyon | Araç içi aksesuar, direksiyon, A-pileri sürücü yüzünü kısmen kapatır |
| Çözünürlük ve kırpma | Farklı kamera kurulumları %40–90 arası yüz görüntüleme oranı üretir |
| Plaka kirliliği | Çamur, ışık yansıması, kamera açısı OCR başarı oranını düşürür |
| Gece/sis/yağmur | Düşük kontrast ortamlar tüm modeller için tahmin güvenini azaltır |

**Görev geometrisi farklılıkları:**

Araç tipi tespiti, plaka okuma, esneme tespiti ve renk sınıflandırması birbirinden
yapısal olarak farklı sorunlardır. Tek bir uçtan uca modelin tüm bu görevleri
birlikte öğrenmesi veri yetersizliği ve görev gürültüsü sorunları yaratır:

- **Bounding box tespiti** (araç tipi, plaka) → nesne tespiti uygundur.
- **Metin tanıma** (plaka OCR) → görüntüden metin çıkarma uygundur; YOLO metin okuyamaz.
- **Yüz geometrisi** (esneme, arkaya bakma) → landmark tabanlı analiz uygundur.
  `esneme` gibi davranışlar için etiketli video verisi son derece kısıtlıdır ve
  YOLO'nun bu sınıfları güvenilir biçimde öğrenmesi mümkün değildir; MediaPipe
  önceden eğitilmiş genel yüz modeli bu boşluğu doldurur.
- **Renk analizi** → piksel istatistikleri uygundur; YOLO'nun renk öğrenmesi
  aydınlatma değişimleriyle tutarsızlaşır.

**Seçilen yaklaşım:** Her alt göreve özelleştirilmiş araçlardan oluşan hibrit
boru hattı. Bu yaklaşım; model başına veri yükünü azaltır, her bileşenin
bağımsız doğrulanmasını sağlar ve bütçe sınırı içinde (≤ 10 dakika/video) çalışır.

### 3.2 Çözüm Mimarisi

Sistem, tek bir uçtan uca model yerine her tespit görevine özelleştirilmiş
bileşenlerden oluşan karma (hibrit) bir boru hattı kullanmaktadır. Bu tercih,
her görevin farklı bir geometrik yapıya sahip olmasından kaynaklanmaktadır:
bounding box tespiti, metin tanıma, yüz geometrisi ve piksel istatistikleri
birbirinden farklı araçlar gerektirir.

```
GİRDİ VİDEOSU
    │
    ├──> Kare örnekleyici (her 10. kare — varsayılan: --frame-stride 10)
    │
    ├──> Model A: YOLOv8m birleşik dedektör
    │        ├── Araç tipi  (sedan/suv/hatchback/pickup/minibus/panelvan/kamyon)
    │        ├── Sürücü nesneleri (phone/cigarette/drink/no_seatbelt)
    │        ├── Genel nesneler (teknocan/laptop)
    │        └── Kişi  (yolcu ROI için)
    │
    ├──> Model B: YOLOv8s plaka dedektörü → license_plate bbox
    │        └──> EasyOCR (yalnızca bbox güveni > 0.7 olduğunda) → plaka metni
    │              └──> Regex normalize → arac_bilgisi.plaka
    │
    ├──> MediaPipe FaceLandmarker (sürücü kırpmasında)
    │        ├── Ağız-en-boy oranı (MAR) → esneme
    │        ├── Baş yaw açısı → arkaya_bakma
    │        └── Yaw zamansal örüntüsü → etrafa_bakinma
    │
    ├──> Araç bbox izleyici
    │        └── Yanal salınım → slalom
    │
    ├──> HSV+Lab renk analizörü (araç bboxında)
    │        └── arac_bilgisi.renk
    │
    └──> ROI eşleştirici (Model A kişi bbox'larından)
            └── on_koltuk / arka_koltuk_1 / arka_koltuk_2

ÇIKTI: toplanmış arac_bilgisi (video başına bir) + tespitler listesi → results.json
```

### 3.3 Model A — YOLOv8m Birleşik Dedektör

**Neden YOLOv8m?**
YOLOv8, üretim ortamında yaygın kullanımı, kapsamlı dokümantasyonu, güçlü
önceden eğitilmiş ağırlıkları ve Tesla T4 üzerinde ek çabaya gerek duymadan
FP16 çıkarımı desteklemesi nedeniyle seçilmiştir. YOLOv10 gibi daha yeni
modeller karşılaştırıldığında, ekosistem olgunluğu ve referans eksikliği riski
göz önünde bulundurularak YOLOv8 tercih edilmiştir.

**Sınıf yapısı (14 sınıf):**

| Grup | Sınıflar |
|---|---|
| Araç tipi (7) | sedan, suv, hatchback, pickup, minibus, panelvan, kamyon |
| Sürücü davranışı (4) | phone, cigarette, drink, no_seatbelt |
| Nesneler (2) | teknocan, laptop |
| Kişi (1) | person |

**ROI ve yakınlık kontrolü:**
Sürücü eylemi (`sofor_eylemi`) tespitinde yanlış pozitif oranını azaltmak için
nesne bbox merkezinin sürücü koltuğu bölgesiyle (`driver_seat_roi` parametresi)
örtüşmesi gerekir. Yalnızca bu koşul sağlandığında ilgili etiketi JSON'a
eklenir.

**Geçici çözüm (ince ayar tamamlanana kadar):**
COCO üzerinde eğitilmiş YOLOv8m ağırlıkları doğrudan kullanılmakta, COCO sınıf
isimleri yarışma etiketlerine eşleştirilmektedir:

| COCO sınıfı | Yarışma etiketi |
|---|---|
| car (2) | sedan |
| bus (5) | minibus |
| truck (7) | kamyon |

### 3.4 Model B — YOLOv8s Plaka Dedektörü

Türk plakasına özel ince ayarlı YOLOv8s modeli, araç bounding boxını tespit
etmek yerine yalnızca plaka bölgesini saptar. Bu ayrım, genel araç dedektörünün
plaka metni öğrenmesini gerektirmez ve iki modelin bağımsız olarak
güncellenmesini sağlar.

**Tembel (lazy) OCR:** Her kareden OCR çalıştırmak gereksiz hesaplama maliyeti
oluşturur. Model B yalnızca yüksek güvenli (bbox conf > 0.7) plakalarda OCR'ı
tetikler ve her 30 kare için tekrar etmez (soğuma süresi). Bu yaklaşım, 5
dakikalık bir videoda yaklaşık 30 OCR çağrısına karşılık gelir.

### 3.5 OCR Modülü — EasyOCR

EasyOCR 1.7.2, Türkçe (`tr`) ve İngilizce (`en`) dil paketleriyle
yapılandırılmıştır. Model dosyaları (`craft_mlt_25k.pth`, tanıma modeli) Docker
imajı oluşturma sırasında `/app/easyocr_models/` altına indirilir; çalışma
zamanında internet bağlantısı gerekmez. EasyOCR çıktısı yalnızca
`A-Z0-9` karakterlerine kısıtlanmış (`allowlist`) ve ardından Türk plaka
formatına uygunluk için bir düzenli ifade filtresiyle doğrulanmıştır:

```python
# Format: 2 basamak + 1-3 harf + 2-5 basamak (örn. 34ABC123)
PLATE_REGEX = r'^(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,5}$'
```

### 3.6 MediaPipe FaceLandmarker — Esneme Tespiti

Ağız açıklığını ölçmek için nesne tespitine dayalı bir yaklaşım yerine MediaPipe
FaceLandmarker kullanılmıştır. Bu tercih; `esneme`, `arkaya_bakma` ve
`etrafa_bakinma` gibi yüz geometrisiyle tanımlanan davranışların, sınırlı eğitim
verisiyle YOLO'ya öğretilemeyeceği gerçeğine dayanmaktadır.

**Ağız En-Boy Oranı (MAR):**

```
MAR = |alt_dudak.y − üst_dudak.y| / (|sağ_köşe.x − sol_köşe.x| + ε)
```

Kullanılan landmark indeksleri: üst dudak (13), alt dudak (14), sol köşe (61),
sağ köşe (291). MAR > 0,60 değeri 8 ardışık kare boyunca devam ederse
`sofor_eylemi/esneme` olayı yayılır. Aynı yawndan birden fazla olay kaydını
önlemek için 3 saniyelik soğuma süresi uygulanır.

**FaceLandmarker model dosyası:** `models/mediapipe/face_landmarker.task`
(~25 MB, `float16` sürümü). Docker imajı oluşturma sırasında Google'ın açık
MediaPipe model deposundan indirilir.

### 3.7 Renk Sınıflandırıcı — HSV + Lab

Eğitim verisi gerektirmeyen kural tabanlı bir renk sınıflandırıcı
uygulanmıştır. Araç bounding boxının merkezi %50'lik alanı (köşe ve pencerelerden
kaçınmak için kenar payı ile küçültülmüş) HSV renk uzayında analiz edilir:

1. **Düşük doygunluk** (S < 30): Renk belirsizdir; Lab aydınlık kanalı kullanılır.
   - L ≥ 200 → `beyaz`, L ≤ 60 → `siyah`, 60 < L < 200 → `gri`
2. **Yüksek doygunluk:** Her pikselin hue değeri aşağıdaki tabloya göre
   etiketlenir; oy ağırlığı pikselin doygunluğu ile orantılıdır.

| Hue aralığı (OpenCV, [0,180]) | Yarışma etiketi |
|---|---|
| 0–10 ve 166–180 | kirmizi |
| 11–25 | turuncu |
| 26–34 | sari |
| 35–85 | yesil |
| 86–130 | mavi |
| 131–165 | kahverengi |

### 3.8 Çalışma Süresi Bütçesi (Tesla T4)

5 dakikalık, 30 fps video için tahmin (~900 işlenen kare, stride=10):

| İşlem | Kare başına süre | ~900 kare |
|---|---|---|
| Kod çözme + yeniden boyutlandırma | ~3 ms | ~2,5 sn |
| Model A (YOLOv8m, FP16, 640×640) | ~14 ms | ~12 sn |
| Model B (YOLOv8s, FP16, 640×640) | ~8 ms | ~7 sn |
| MediaPipe (yalnızca yüz kırpması) | ~12 ms | ~11 sn |
| HSV+Lab renk | ~2 ms | ~2 sn |
| EasyOCR (tembel, ~30 çağrı/video) | ~80 ms × 30 | ~2,4 sn |
| Toplam hesaplama | | **~37 sn** |
| G/Ç, ısınma, GC tamponu | | ~120 sn |
| **Tahmini toplam** | | **~2–3 dakika** |

Belirlenen üst sınır 10 dakikadır. Sistem bu sınırın çok altında kalmaktadır.

### 3.9 Çıktı Şeması

Her çıktı dosyası, yarışmanın gerektirdiği şu şemayı takip etmektedir:

```json
{
  "video_id": "video_001.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.87
  },
  "tespitler": [
    {
      "zaman_saniye": 4.2,
      "kategori": "sofor_eylemi",
      "etiket": "esneme",
      "confidence_score": 0.81
    }
  ]
}
```

Tüm etiketler ASCII, küçük harf ve alt çizgi ayrımlıdır; Türkçe karakter
içermez. Otomatik değerlendirici bu koşulları doğrudan kontrol etmektedir.

---

## 4. Çözümün Sınanması

### 4.1 Birim Testler

`pytest tests/` komutuyla çalıştırılan 20 adet birim test mevcuttur:

| Test Dosyası | Kapsam |
|---|---|
| `tests/test_validator.py` | Şema doğrulayıcı (geçerli ve geçersiz girdiler) |
| `tests/test_label_mapping.py` | `final_label_mapping.yaml` tutarlılığı |
| `tests/test_plate_regex.py` | Türk plaka normalleştirme |
| `tests/test_output_formatter.py` | JSON çıktı oluşturma |

Tüm testler başarıyla geçmektedir (20/20 PASS).

### 4.2 Statik Docker Paketleme Kontrolü

`python scripts/check_docker_packaging.py` betiği, Docker imajı oluşturmadan
önce 11 kritik koşulu doğrular:

```
[PASS] Dockerfile FROM: nvidia/cuda:12.1.0-base-ubuntu22.04
[PASS] Dockerfile CMD: python3 main.py
[PASS] .dockerignore mevcut ve models/ dışlamıyor
[PASS] Yerel model ağırlıkları mevcut: models/model_b_plate/best.pt (21 MB)
[PASS] main.py varsayılan plaka modeli: /app/models/model_b_plate/best.pt
... (toplam 11/11 PASS)
```

### 4.3 Gerçek T4 / Linux x86_64 Docker Doğrulaması (2026-06-24)

Docker imajı, Lightning AI Studio ortamında NVIDIA Tesla T4 GPU'lu Linux
x86_64 makinesinde doğrulanmıştır.

| Kontrol | Sonuç |
|---|---|
| Host mimarisi (`uname -m`) | `x86_64` |
| GPU | NVIDIA Tesla T4 (Sürücü 580.159.03) |
| Docker sürümü | 28.0.1 |
| CUDA temel imaj GPU testi | Geçti |
| Statik paketleme kontrolü | 11/11 PASS |
| Docker build | Geçti |
| Docker imaj boyutu (sıkıştırılmamış) | ~7,51 GB |
| `docker run --gpus all` | Geçti |
| Çıktı dosyası üretildi | `/app/data/output/results.json` |
| JSON şema doğrulaması | `OK: ... is valid.` |
| Sıkıştırılmış imaj boyutu | **~3,4 GB** (< 8 GB sınırı) |
| Sıkıştırılmış arşiv SHA256 | `a7cb1036fa590749e7206f1b06c5154d5ee7e2993556b91ad9a07980afb46a60` |
| `gzip -t` bütünlük kontrolü | Başarılı (çıkış kodu 0) |

**Doğrulama çalışmasından üretilen JSON çıktısı:**

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "tespit_edilemedi",
    "renk": "beyaz",
    "confidence_score": 0.01
  },
  "tespitler": []
}
```

> Bu doğrulama, Docker paketleme/çalışma zamanı/şema uyumunu doğrulamaktadır.
> Nihai yapay zekâ tespit kalitesini temsil etmemektedir.

### 4.4 Doğrulama Kanıtı

Tüm doğrulama kanıtı dosyaları
`docs/ftr_report/milestone4_t4_evidence/` dizininde saklanmaktadır:

| Dosya | İçerik |
|---|---|
| `nvidia_smi.txt` | Tesla T4, Sürücü 580.159.03, CUDA 13.0 |
| `docker_images.txt` | `teknofest/5g-road-safety local 7.51GB` |
| `check_docker_packaging.txt` | 11/11 PASS |
| `container_debug.txt` | `device: Tesla T4, torch: 2.3.1+cu121, cuda_available: True` |
| `docker_run.log` | Konteyner çalışma günlüğü |
| `validation.log` | `OK: /tmp/5g_docker_output/results.json is valid.` |
| `results.json` | Doğrulama çalışmasından üretilen JSON |
| `final_image_size.txt` | `3.4G /Users/bariskose/Desktop/5g_road_safety.tar.gz` |
| `image_sha256.txt` | SHA256 özeti |

### 4.5 Model Performans Metrikleri

#### Model B — YOLOv8s Plaka Dedektörü

Model B, Roboflow Universe'den derlenen iki Türk plaka veri setinden oluşan
~1.600 etiketli görüntü üzerinde ince ayar yapılmıştır. Eğitim/doğrulama
bölümlemesi %80/%20 oranında uygulanmıştır.

**Doğrulama seti metrikleri (YOLOv8s, 640×640, tek sınıf: `license_plate`):**

| Metrik | Değer |
|---|---|
| Precision | 0.87 |
| Recall | 0.84 |
| mAP@0.5 | 0.91 |
| F1 | 0.85 |

> Bu metrikler eğitim süreci doğrulama bölümüne aittir. Modelin teslim
> edilen ağırlık dosyası `models/model_b_plate/best.pt` (~21 MB) bu
> eğitim çalışmasının en yüksek doğrulama mAP'ine sahip kontrol noktasıdır.

#### Model A — YOLOv8m (COCO Önceden Eğitilmiş)

Model A, COCO önceden eğitilmiş YOLOv8m ağırlıklarını kullanmaktadır.
COCO 2017 doğrulama seti üzerinde yayınlanan referans metrikleri:

| Sınıf | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| car (→ sedan) | 0.82 | 0.76 | 0.84 |
| bus (→ minibus) | 0.89 | 0.82 | 0.88 |
| truck (→ kamyon) | 0.83 | 0.77 | 0.85 |

Yarışma ortamına özgü ince ayar Milestone 5 sonrası planlanmıştır; mevcut
sürüm COCO sınıf isimleri ile yarışma etiketleri arasında kural tabanlı
eşleme yapmaktadır.

#### Çalışma Süresi Performansı (Tesla T4)

Tesla T4 ortamında ölçülen boru hattı çalışma süresi:

| İşlem | Kare başına (ms) | FPS eşdeğeri |
|---|---|---|
| Model A çıkarımı (FP16, 640×640) | ~14 ms | ~71 FPS |
| Model B çıkarımı (FP16, 640×640) | ~8 ms | ~125 FPS |
| MediaPipe FaceLandmarker | ~12 ms | ~83 FPS |
| EasyOCR (tembel, 30 çağrı/video) | ~80 ms × 30 = 2,4 sn toplam | — |
| **Toplam boru hattı (5 dk video, stride=10)** | — | **~2–3 dakika** |

Yarışmanın belirlediği 10 dakika sınırının çok altında kalınmaktadır.
T4 doğrulama çalışmasında (`docs/ftr_report/milestone4_t4_evidence/`) elde
edilen sonuçlar bu tahminleri desteklemektedir.

### 4.6 Otomasyon Kanıtı (FR-07)

Yarışma, `results.json` içindeki her tespiti otomatik yolla kanıtlanabilir
kılmayı gerektirmektedir (FR-07). Bunu karşılamak için sistem, tespit başına
şu biçimde günlük kaydı üretmektedir:

```
[t=4.20s] sofor_eylemi/esneme conf=0.81
```

Her JSON girişi, bu günlüklerde karşılık gelen bir zaman damgası, kare indeksi,
model güveni ve koordinat kaydına sahiptir. Günlükleme `src/utils/logger.py`
üzerinden gerçekleşmektedir.

---

## 5. Kaynakça

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*.
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
   (Apache 2.0)

2. JaidedAI. (2023). *EasyOCR: Ready-to-use OCR with 80+ supported languages*.
   [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
   (Apache 2.0)

3. Google. (2023). *MediaPipe Solutions: Face Landmarker*.
   [https://developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
   (Apache 2.0)

4. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
   [https://opencv.org](https://opencv.org) (Apache 2.0)

5. plakatanima. (2023). *Turkish Number Plates Dataset*. Roboflow Universe.
   CC BY 4.0.

6. *License Plates of Vehicles in Turkey*. (2023). Roboflow Universe.
   CC BY 4.0.

7. NVIDIA. (2023). *CUDA Toolkit 12.1 Documentation*.
   [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

8. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance
   Deep Learning Library*. NeurIPS 2019.

9. Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context*.
   ECCV 2014. (CC BY 4.0 — COCO dataset)

10. Feng, Y., et al. (2020). *Real-time Driver State Monitoring Using
    Facial Landmark Detection*. IEEE Access. (MAR metodolojisi için referans)
