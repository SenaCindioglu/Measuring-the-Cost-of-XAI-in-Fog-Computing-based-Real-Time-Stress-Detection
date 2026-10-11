# results/ — hangi dosya ne, hangisi makaleye girer

Son düzenleme: 24 Eylül 2026.

Klasörler ölçüm aşamasına göre numaralı. `eski_ve_hatali/` içindekiler ya
geçersiz ya da daha sonraki bir koşuyla geçersizleşti; neden olduğu o
klasörün kendi README'sinde yazılı.

> **v1 / v2 ayrımı (11 Ekim 2026).** 01–13 klasörlerinin tamamı **v1**
> protokolüyle ölçüldü: istemci **kapalı döngü** (cevap gelmeden sonraki
> tick gönderilmiyordu), üç yerleşim **üç farklı yazılım yığınında**
> (torch 2.8.0 / 2.14.0 / 2.14.0+cu130), XAI hücreleri **tek koşu**, uzak
> sunucuda kaynak özeti oturum değil sunucu ömrü boyunca. Katman
> karşılaştırmaları (05, 07, 13) ve çok istemcili yük (10, 13/cloud_load_4)
> bu yüzden makalede **nihai sayı olarak kullanılmaz**; v2 koşuları
> `15_v2/` altına gelecek (protokol: `docs/RUNBOOK_v2.md`). Tek yerleşim
> içindeki bulgular (04 içi oranlar, 08 kalibrasyon, 09 seçici XAI, 11
> kalite) yöntemden bağımsız olduğu için geçerliliğini koruyor.
> Makale sayıları elle değil `tools/paper_numbers.py` ile üretilir.

Yeni koşular `results/` kökünde oluşur (script'ler oraya yazıyor). Koşu
bittikten sonra ilgili klasöre taşı, yoksa kök yeniden karışır.

---

## 01_streaming — tek süreçli replay, aşama profili

Model ve istemci aynı süreçte; ağ yok. Aşama bazlı hesaplama süresinin
kaynağı bu.

| Dosya | Ne | Anahtar sayılar |
|---|---|---|
| `streaming_summary_native_fixed.json` | S11+S14, native yol (yeniden örnekleme ölçüm içinde), 1175 pencere | resampling 1.57 ms, preprocessing 0.25 ms, inference 3.04 ms, toplam 5.50 ms |
| `streaming_raw_native_fixed.json` | Aynı koşunun pencere bazlı ham ölçümleri | Pareto ve histogram için |

---

## 02_benchmark — tekrarlı ve dönüşümlü ölçüm

Tek koşu bu makinede güvenilir değil: işlemci frekans ölçeklemesi değerleri
kaydırıyor. Bu dosyalar dönüşümlü tekrarla üretildi, koşu arası yayılımı da
taşıyor.

| Dosya | Ne | Sonuç |
|---|---|---|
| `benchmark_thread_sweep.json` | İş parçacığı 1 / 2 / 4, S11, 4 tur | Fark yayılımın altında; varsayılan 1 seçildi, gerekçe hız değil kararlılık (yayılım %2.9'a karşı %11–15) |
| `benchmark_pipeline_cmp.json` | `prepared` ve `native` yolların karşılaştırması | native 5.50 ms, prepared 3.97 ms; yeniden örnekleme toplamın ~%29'u |
| `benchmark_xai_cpu.json` | LIME'ın bu makinede (istemci tarafı CPU) tekrarlı ölçümü | LIME 500 örnek p50 710 ms, p95 1130 ms; 250 örnek p50 490 ms |

**Not:** `benchmark_xai_cpu.json` **istemci makinesinde** ölçüldü, fog
düğümünde değil. Fog düğümü sayıları `04_fog/` içinde ve daha yavaş.

---

## 03_loopback — istemci–sunucu ayrı, aynı makine

TCP var, ağ yok (loopback). Aşama 3'te raporlanan sayıların kaynağı.

| Dosya | Ne | Anahtar sayılar |
|---|---|---|
| `fog_realtime_1user.json` | Tek istemci, gerçek zamanlı, S11 ilk 10 dk | latency 0.81 ms, queuing 0.045 ms, execution 7.18 ms, response 8.82 ms, jitter 3.53 ms, 187 kbps |
| `fog_full_queue.json` | İki istemci, kuyruk gecikmesinin anlam kazandığı koşu | queuing p99 0.37 ms → 5.50 ms (açık istek kuyruğu eklendikten sonra) |

---

## 04_fog — ikinci bilgisayar, gerçek ağ (**makalenin ana tablosu**)

Sunucu 192.168.1.14'te (Intel 12 çekirdek, 16 GB), istemci bu makinede,
Wi-Fi üzerinden. Saat farkı 60 saniyede bir yeniden ölçülüyor, yani latency
geçerli. Hepsi S11, tek istemci, 600 tick, 44 ölçülen pencere.

| Dosya | Açıklayıcı | response p50 | response p95 | execution | açıklama (xai) | sunucu CPU | sunucu RSS |
|---|---|---|---|---|---|---|---|
| `fog_xai_none.json` | yok | **10.6 ms** | 42.2 ms | 3.5 ms | — | %0.0 | 210 MB |
| `fog_xai_shapucuz.json` | SHAP, nsamples 50 | **442.6 ms** | 574.0 ms | 394.1 ms | 390.4 ms | %5.0 | 422 MB |
| `fog_xai_limeucuz.json` | LIME, 250 örnek | **645.2 ms** | 802.5 ms | 588.1 ms | 584.0 ms | %10.8 | 442 MB |
| `fog_xai_lime.json` | LIME, 500 örnek | **1227.5 ms** | 1407.1 ms | 1158.1 ms | 1154.2 ms | %22.8 | 449 MB |
| `fog_xai_shap.json` | SHAP, nsamples 200 | **1456.2 ms** | 1548.5 ms | 1431.1 ms | 1426.2 ms | %25.9 | 464 MB |

Latency hepsinde 4.3–5.3 ms bandında (Wi-Fi), yani açıklayıcı seçimi ağı
etkilemiyor — beklenen davranış, ölçümün tutarlılık kontrolü olarak işe yarar.

**Bu tablodan çıkan üç sonuç:**

1. Açıklama, sistemin baskın maliyeti. Açıklayıcı kapalıyken yanıt 10.6 ms;
   en pahalı yapılandırmada 1456 ms, yani **137 kat**.
2. Ucuz çalışma noktaları ShaTS'in 1 s/pencere sınırının altında kalıyor
   (SHAP 443 ms, LIME 645 ms); pahalı olanlar p95'te aşıyor.
3. Bellek 210 MB'dan 464 MB'a çıkıyor. 2 GB'lık bir Raspberry Pi'de bu
   sıkışır; en az 4 GB gerekir.

**Eksik:** bu koşular tek maliyet noktasında ve eşik taraması yok
(hepsi "her pencerede açıkla" ya da "hiç açıklama"). Eşik taraması
(0.6 / 0.7 / 0.8 / 0.9) ve çok istemcili yük hâlâ yapılacak.

**Kayıt boşluğu:** hangi açıklayıcının kullanıldığı yalnızca dosya adında
duruyor; sunucu bunu istemciye göndermiyor, JSON içinde yok. Sunucu
istatistiklerine `xai_method`, `xai_threshold` ve açıklayıcı yapılandırması
eklenmeli, yoksa dosya adı değişirse bilgi kaybolur.

---

## 05_yerlesim — ortak yerleşim / fog karşılaştırması

| Dosya | Ne | Sonuç |
|---|---|---|
| `deployments_edge_vs_fog.json` | 3 tur dönüşümlü, S11, 600 tick, XAI yok | ortak yerleşim: latency 0.928, execution 6.344, response 7.880 ms · fog: latency 4.910, execution 3.372, response 10.151 ms · ikisinde de 187 kbps (tur medyanlarının medyanı) |

Fog düğümünün hesabı daha hızlı yaptığı buradan çıkıyor (3.37'ye karşı
6.34 ms): ortak yerleşimde istemci ve sunucu aynı işlemciyi paylaşıyor.
(Bu satırda önceden 0.81/6.19/7.44 ve 3.69/3.61/9.97 yazıyordu; JSON'daki
hiçbir tura karşılık gelmiyordu, daha eski bir koşudan kalmıştı.)

**Terim uyarısı:** "edge" burada *ortak yerleşim* anlamında, giyilebilir
cihaz anlamında değil. Makalede tanımı verilmeli.

---

## 06_xai_tarama — açıklayıcı hiperparametre taraması (Colab, GPU)

Nedensel normalizasyonla, yani sunucunun gördüğü girdi uzayında koşuldu.
Kalite metrikleri donanımdan bağımsız olduğu için bu dosyalardan alınır;
**süre değerleri GPU'ya ait**, makalenin süre tablosu `04_fog/`'dan gelir.

| Dosya | Ne |
|---|---|
| `xai_sweep_lime_causal.csv` | LIME OFAT taraması, 9 yapılandırma × 20 sütun |
| `xai_sweep_shap_causal.csv` | SHAP OFAT taraması, 8 yapılandırma |
| `xai_lime_vs_shap_recommended_causal.csv` | Seçilen iki yapılandırmanın karşılaştırması |

Seçilen yapılandırmalar (elle, kalite–gecikme tablosuna bakılarak):
LIME 500 örnek / 20 dilim / kw 0.5 · SHAP arka plan 100 (kmeans) / nsamples 200.
İkinci (ucuz) nokta: SHAP nsamples 50 — kaliteden %1.4 verip 8.6 kat hızlı.
Kayıtlı hali: `checkpoints/xai_config.json`.

---

## 07_edge_xai — ortak yerleşimde XAI, `04_fog` ile birebir eşleşen koşular

Aynı beş durum, aynı denek (S11), aynı süre (600 tick), aynı ölçüm yöntemi.
Değişen tek şey sunucunun yeri: burada istemciyle **aynı makinede**.
`04_fog/` ile yan yana okunmak üzere üretildi.

| Dosya | Açıklayıcı | xai | execution | response |
|---|---|---|---|---|
| `edge_xai_none.json` | yok | — | 5.8 ms | **7.3 ms** |
| `edge_xai_shapucuz.json` | SHAP, nsamples 50 | 417 ms | 460.4 ms | 462.0 ms |
| `edge_xai_limeucuz.json` | LIME, 250 örnek | 390 ms | 351.5 ms | 352.8 ms |
| `edge_xai_lime.json` | LIME, 500 örnek | 831 ms | 688.5 ms | 690.0 ms |
| `edge_xai_shap.json` | SHAP, nsamples 200 | 1224 ms | 1152.4 ms | 1154.5 ms |

Loopback olduğu için `response ≈ execution`; aradaki fark `04_fog/`'da gerçek
ağ maliyetidir.

### ⚠️ Bu tablo `04_fog/` ile doğrudan kıyaslanamaz

Sıralama yönteme göre değişiyor: LIME'da edge hızlı (390'a karşı 590 ms),
SHAP'ın ucuz noktasında fog hızlı (395'e karşı 417 ms), tabanda ise fog
hızlı (3.5'e karşı 5.8 ms). Donanım farkı böyle davranmaz.

Sebep **yazılım yığını farkı**: edge'de PyTorch 2.8.0 / Python 3.11, fog'da
2.14.0 / Python 3.14. LIME toplu ileri geçiş, SHAP gradyan hesabı yapıyor;
iki farklı kod yolu iki sürümden farklı etkileniyor. İkisi de her koşunun
`server_hardware` alanında kayıtlı.

**Makalede ya bu cümle yazılır, ya da fog düğümü Docker'la (python:3.11)
yeniden koşulup yığın eşitlenir.** Olduğu gibi "fog daha yavaş" diye
sunulamaz.

---

## 08_kalibrasyon — güven kalibrasyonu (Faz D1)

Seçici XAI'nin ön koşulu: `confidence` gerçek doğruluk olasılığını temsil
ediyor mu? Sunucunun gördüğü girdi uzayında ölçüldü (native yeniden
örnekleme + nedensel z-score), yani sayı dağıtıma taşınıyor.

| Dosya | Ne | Sonuç |
|---|---|---|
| `calibration_d1.json` | S11 ve S14, reliability kutuları, ECE/MCE/Brier, çapraz denek sıcaklık ölçekleme, aday eşik tablosu | aşağıda |

**İki denek taban tabana zıt:**

| | S11 | S14 |
|---|---|---|
| doğruluk | 0.9520 | 0.6902 |
| ortalama güven | 0.9093 | 0.9085 |
| fazla özgüven | −0.043 | **+0.218** |
| ECE | **0.0469** (iyi) | **0.2265** (bozuk) |

Model ikisine de aynı güvenle yaklaşıyor ama birinde %95, diğerinde %69
haklı. S14'te **kendinden emin biçimde yanılıyor**.

**Sıcaklık ölçekleme çalışmadı.** Çapraz denek, her iki yönde de kötüleştirdi
(S14→S11: ECE 0.047→0.256; S11→S14: 0.227→0.276). Bozukluk zıt yönlerde
olduğu için tek parametre ikisini birden düzeltemiyor. Bu bir hata değil,
raporlanacak bir bulgu.

**Aday eşikler (düzeltmesiz):**

| eşik | S11 açıklanan | S11 hata yakalama | S14 açıklanan | S14 hata yakalama |
|---|---|---|---|---|
| 0.70 | %10.4 | 0.621 | %4.4 | 0.060 |
| **0.80** | **%14.4** | **0.897** | %11.8 | 0.120 |
| 0.90 | %21.4 | 1.000 | %29.3 | 0.288 |

**Makaleye giren ayrım:** maliyet tasarrufu her iki denekte gerçekleşiyor
(eşik 0.80'de açıklama oranı %12–14, yani her pencereyi açıklamaya göre
~%86 daha az hesap). Fayda ise kalibrasyona bağlı: iyi kalibre denekte
hataların %90'ı yakalanıyor, kötü kalibre denekte %12'si. İki iddia ayrı
yazılmalı.

**Sınırlılık:** iki tutulmuş denek az; "denek bağımlı" sonucu n=2 üzerine
kurulu.

Üreten: `python -m tools.calibration --tag d1`

---

## 09_secici_xai — seçici açıklama (Faz D2 + D4 orta satır)

Makalenin ana iddiasının kanıtı: güven eşiğinin altındaki pencereleri
açıklamak, her pencereyi açıklamaya göre maliyeti kaça indiriyor?

### `fog_sel80_tam.json` — **ana sonuç**

S11'in **tam kaydı** (10 466 tick, 87 dakika), fog düğümü, `shap_ucuz`,
eşik 0.80. **1030 ölçülen pencere** — diğer tüm fog koşularının 23 katı,
p95 ve p99 burada gerçekten anlamlı.

| durum | ort response | p95 | pencere |
|---|---|---|---|
| hiç açıklama (`04_fog/fog_xai_none`) | 15.9 ms | 42.2 ms | 44 |
| **seçici (0.80)** | **80.1 ms** | 445.7 ms | **1030** |
| hep açıkla (`04_fog/fog_xai_shapucuz`) | 450.5 ms | 574.0 ms | 44 |

**Seçici açıklama, her pencereyi açıklamaya göre 5.6 kat ucuz**, ve D1'e
göre hataların %89.7'si yakalanıyor. Açıklama oranı %14.1 (145/1030).

**Uyarı:** üç satırın örneklem boyutu eşit değil (1030'a karşı 44).
Karşılaştırma ortalamalar üzerinden geçerli ama p95/p99'da yalnızca seçici
satır sağlam.

### `selective_xai_sweep_d2.json` — projeksiyon, **ölçümle doğrulandı**

D1'in açıklama oranını `04_fog`'un maliyetiyle birleştiren karışım modeli.
Ana koşu bunu test etti:

| | tahmin | ölçüm | hata |
|---|---|---|---|
| açıklama oranı | %14.4 | %14.1 | %2.3 |
| ort response | 78.5 ms | 80.1 ms | %2.0 |
| p95 response | 487.3 ms | 445.7 ms | %8.5 |

Ortalamada %2: beklenen, çünkü karışımın ortalaması matematiksel olarak
kesin. p95'te %8.5: özet çeyrekliklerden ara değerle tahmin edildiği için
yaklaşık olduğu baştan işaretliydi.

**Kuyrukta birikme YOK** (queuing ort 0.0, p99 0.1 ms). Patlamalı yükün
(%86 pencere 3.6 ms, %14 pencere ~365 ms) kuyruğu bozması beklenebilirdi;
bozmadı. Yani modelin kaçırabileceği etkileşim etkisi yok ve dosyadaki
diğer satırlar (diğer açıklayıcılar, S14) güvenle projeksiyon olarak
sunulabilir.

### `kisa_segment_sel{60,80,90}.json` — eşiğin hiç tetiklenmediği koşular

600 tick'lik (5 dk) koşular, **üçünde de açıklama oranı %0**.

**Bu bir hata değil, kayıt içi durağansızlığın kanıtı.** S11'in ilk 5
dakikasında en düşük güven **0.929**; en yüksek eşik (0.90) bile hiçbir
pencerenin altına inmiyor. Düşük güvenli ilk pencere **#100**, yani
kaydın 8. dakikasında. Kayıttaki 87 düşük güvenli pencerenin hepsi ilk
400 pencere içinde; sonrası tekrar yüksek güvene dönüyor.

Seçici XAI deneyi bu yüzden tam kayıt üzerinde koşuldu.

**Dosya adları yanıltıcı:** `sel60` gerçekte eşik **0.70** ile koşuldu
(sunucu künyesi öyle diyor). Etiket kayması koşu sırasında oldu. Gerçek
eşikler künyede: `server.xai.threshold`. Eşik 0.60 hiç koşulmadı.
Dosya adına değil künyeye bakın — künyenin kayda geçirilmesinin
gerekçesi tam olarak budur.

---

## 10_yuk — çok istemcili yük (A4)

1/2/4 istemci, fog düğümü, S11+S14 döngüsel, 600 tick. Tek istemci
satırları `04_fog/`'dan.

| durum | istemci | queuing ort | q p95 | execution | response ort | response p95 |
|---|---|---|---|---|---|---|
| XAI yok | 1 | 0.030 ms | 0.03 | 3.6 | 15.9 | 42.2 |
| XAI yok | 2 | 1.607 ms | 4.51 | 3.6 | 24.3 | 107.8 |
| XAI yok | 4 | 4.555 ms | 12.50 | 3.7 | 28.5 | 120.3 |
| hep açıkla | 1 | 0.032 ms | 0.05 | 400.2 | 450.5 | 574.0 |
| hep açıkla | 2 | **160.1 ms** | 381.6 | 371.1 | 601.3 | 853.6 |
| hep açıkla | 4 | **437.3 ms** | 1125.1 | 370.7 | 984.9 | 1643.1 |

**Mekanizma ortalama doluluk değil, eşzamanlı patlama.** 4 istemcide
teorik doluluk %32 iken ölçülen kuyruk p95 1125 ms. İstemciler birlikte
başlatıldığı için faz hizalı kalıyor, pencerelerini aynı anda üretiyor ve
tek worker'da açıklamalar arka arkaya diziliyor:

```
(N-1) × açıklama süresi = 3 × 370 = 1110 ms     ölçülen p95 1125 ms
```

XAI'siz tarafta da tutuyor: 3 × 3.7 = 11 ms, ölçülen 12.5 ms.

**Bu bir üst sınırdır** — gerçek dağıtımda istemciler rastgele fazlarda
başlar. Raporda belirtilmeli.

**Kapasite** (hop 5000 ms / execution): XAI yok 1394 · seçici 90 ·
hep açıkla 12 istemci. Referans makale tek worker için ~20 kullanıcı
veriyor. Seçici açıklama kapasiteyi **7.5 kat** artırıyor.

---

## 11_pareto — kalite / gecikme dengesi (C5)

Kalite **bu makinede** ölçüldü (donanımdan bağımsız), gecikme `04_fog/`'dan
(dağıtım donanımı). `lime_ucuz` yapılandırması Colab taramasında hiç yoktu;
Pareto grafiğinin o noktası kalite ekseninde boşluktaydı, burada kapatıldı.

| nokta | gecikme | faithfulness | monotonicity | sensitivity |
|---|---|---|---|---|
| `shap_ucuz` | 395 ms | 0.2499 ± 0.0426 | 0.5650 | 0.1228 |
| `lime_ucuz` | 590 ms | 0.2092 ± 0.0345 | 0.6038 | **0.0489** |
| `lime` | 1189 ms | 0.2324 ± 0.0373 | 0.5787 | **0.0446** |
| `shap` | 1422 ms | 0.2632 ± 0.0445 | 0.5487 | 0.1282 |

**Faithfulness'ta hiçbir nokta diğerini yenmiyor.** 40 pencerede standart
hata ~0.04, farkların hepsi onun içinde. Dolayısıyla seçim faithfulness'a
göre yapılamaz — Colab taramasının 0.003'lük farklarla kazanan seçmesi bu
yüzden yanlıştı.

**Ayrışma sensitivity'de ve kesin:** LIME 0.045–0.049, SHAP 0.123–0.128.
Fark hata payının 15 katı. **LIME 2.7 kat daha kararlı açıklama üretiyor.**

Gerçek denge şu: **SHAP 3 kat ucuz, LIME 2.7 kat kararlı, sadakat eşit.**

`pareto_c5.png` iki panelli: solda faithfulness (hata çubukları örtüşüyor),
sağda sensitivity (ayrışma net).

---

## 12_profil — aşama bazlı profil tablosu (B1)

Yeni ölçüm değil, dört kaynaktan derleme. Aynı aşamanın değeri kaynağa göre
değişiyor ve sebepleri farklı; dağılmış halde bırakılırsa makalede yanlış
satır alıntılanır.

| aşama | tek süreç | tekrarlı native | tekrarlı prepared | fog uçtan uca |
|---|---|---|---|---|
| resampling | 1.378 | 1.573 (±%3.3) | — | 1.044 |
| preprocessing | 0.218 | 0.252 (±%3.5) | 0.580 (±%7.9) | 0.172 |
| tensor_prep | 0.141 | 0.149 (±%2.7) | 0.135 (±%5.7) | 0.114 |
| inference | 2.682 | 3.039 (±%8.9) | 2.969 (±%9.1) | 1.948 |
| postprocessing | 0.086 | 0.087 (±%3.0) | 0.085 (±%5.5) | 0.046 |
| **total** | 4.689 | **5.502 (±%8.1)** | 3.971 (±%9.4) | **3.530** |

Yayılım taşıyan tek sütun `tekrarlı native` (4 tur dönüşümlü) — makalede
belirsizlik gerektiğinde oradan alınmalı. Fog sütunu düşük çünkü o makine
daha hızlı; tek süreç sütunu istemci makinesinde ölçüldü.

Üreten: `python -m tools.profile_table`

---

## 13_bulut — bulut yerleşimi (A6), Docker ile

Sunucu **DigitalOcean VPS**'te (167.71.43.64, "DO-Regular" CPU, 2 GB RAM,
1 vCPU), Docker konteynerinde, **torch 2.14.0+cu130** (CUDA tekerleği,
CPU-only makinede — v2'de CPU tekerleğiyle değişiyor). İstemci bu
makinede, internet üzerinden.

**Sütun istatistikleri:** aşağıdaki XAI tablosunda `xai` ortalama,
`execution` ve `response` medyan. Bulut için tek yön `latency` saat
belirsizliği (±25–51 ms) mertebesinde olduğundan makalede raporlanmaz;
yerine `(response − execution − queuing)/2` kullanılır.

**Dosya adları düzeltildi.** Ölçümler `fog_cloud_xai_*` ve
`fog_docker_load_4` adlarıyla gelmişti ama hepsi buluta gidiyor; `fog`
öneki yanıltıcıydı. Gerçek adres her koşunun `config.server` alanında.

| dosya | ne |
|---|---|
| `cloud_taban_3tur.json` | XAI'siz, 3 tur dönüşümlü, tam kayıt |
| `cloud_xai_{shapucuz,limeucuz,lime,shap}.json` | dört açıklayıcı, her pencere |
| `cloud_load_4.json` | 4 istemci, **4120 ölçülen pencere** |

### Bulutta XAI maliyeti

| açıklayıcı | xai | execution | response | p95 | latency |
|---|---|---|---|---|---|
| `shap_ucuz` | 337 ms | 343.6 | **429.8** | 509.0 | 60.5 |
| `lime_ucuz` | 361 ms | 344.3 | 463.3 | 608.3 | 53.1 |
| `lime` | 782 ms | 747.6 | 870.1 | 1091.8 | 54.0 |
| `shap` | 1220 ms | 1173.3 | **1300.5** | 1715.8 | 54.8 |

### Üç yerleşim yan yana (response medyan, ms)

| açıklayıcı | edge (ortak) | fog (LAN) | cloud (internet) |
|---|---|---|---|
| yok | 7.3 | 10.6 | **63.5** |
| `shap_ucuz` | 462.0 | 442.6 | 429.8 |
| `lime_ucuz` | 352.8 | 645.2 | 463.3 |
| `lime` | 690.0 | 1227.5 | 870.1 |
| `shap` | 1154.5 | 1456.2 | 1300.5 |

**Buluttaki fark ağdan geliyor, hesaptan değil.** Latency edge'de ~0,
fog'da 4.3–5.3 ms, bulutta **53–61 ms**. Hesap tarafında bulut VPS'i fog
düğümünden yavaş değil; `shap_ucuz`'da hatta biraz hızlı (337'ye karşı
395 ms).

**Asıl sonuç:** açıklama açıkken yerleşim farkı önemsizleşiyor. XAI'siz
durumda bulut edge'in 8.7 katı (63.5 / 7.3), ama `shap` ile yalnızca 1.13
katı (1300.5 / 1154.5) — çünkü 50 ms'lik ağ farkı 1200 ms'lik açıklamanın
yanında kayboluyor. Bu, referans makalenin "fog, cloud'dan üstündür"
sonucunu **XAI yükü altında geçersiz kılıyor**.

**Uyarı:** bu sütunlar farklı yazılım yığınlarında ölçüldü (bkz.
`07_edge_xai` uyarısı). Bulut Docker'da python:3.11 çalıştırıyor, fog
düğümü Python 3.14, edge Python 3.11. Yerleşim karşılaştırması bu farkı
içerir.

### 4 istemci yükü, bulutta

```
n=4120 pencere   queuing ort 4.74 ms  p95 16.08 ms   response medyan 62.1 ms (ort 67.5)  p95 117.8 ms
```

`10_yuk/`'teki fog koşusunun bulut karşılığı. Kuyruk fog'daki XAI'siz
4 istemci koşusuna (p95 12.50 ms) yakın — beklenen, çünkü ikisinde de
açıklama kapalı.

---

## eski_ve_hatali — makaleye girmez

Ayrıntılı gerekçeler: [eski_ve_hatali/README.md](eski_ve_hatali/README.md).
Özet: sızıntılı normalizasyonla ölçülen XAI taramaları, saat kayması
düzeltilmeden önce alınan fog koşusu, `prepared` yol dönemi streaming
sonuçları ve duman testleri.
