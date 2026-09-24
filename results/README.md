# results/ — hangi dosya ne, hangisi makaleye girer

Son düzenleme: 24 Eylül 2026.

Klasörler ölçüm aşamasına göre numaralı. **Makaleye girecek sayılar yalnızca
01–06 klasörlerinden alınır.** `eski_ve_hatali/` içindekiler ya geçersiz ya da
daha sonraki bir koşuyla geçersizleşti; neden olduğu o klasörün kendi
README'sinde yazılı.

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
| `deployments_edge_vs_fog.json` | 3 tur dönüşümlü, S11, 600 tick, XAI yok | ortak yerleşim: latency 0.81, execution 6.19, response 7.44 ms · fog: latency 3.69, execution 3.61, response 9.97 ms · ikisinde de 187 kbps |

Fog düğümünün hesabı daha hızlı yaptığı buradan çıkıyor (3.61'e karşı
6.19 ms): ortak yerleşimde istemci ve sunucu aynı işlemciyi paylaşıyor.

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

## eski_ve_hatali — makaleye girmez

Ayrıntılı gerekçeler: [eski_ve_hatali/README.md](eski_ve_hatali/README.md).
Özet: sızıntılı normalizasyonla ölçülen XAI taramaları, saat kayması
düzeltilmeden önce alınan fog koşusu, `prepared` yol dönemi streaming
sonuçları ve duman testleri.
