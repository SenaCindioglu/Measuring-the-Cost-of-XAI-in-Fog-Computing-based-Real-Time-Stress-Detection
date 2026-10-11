# ICAIIC 2027 — yapılacaklar, öncelik sırasıyla

Son güncelleme: **11 Ekim 2026.** Son tarih: **15 Ekim 2026** (EDAS, 6 sayfa).
Kalan: **4 gün.**

Bu dosya `ICAIIC_PLAN.md`'yi **tamamlar**, yerine geçmez. Plan "ne yazılacak"
sorusunu cevaplıyor; bu dosya "hangi sırayla ne yapılacak" sorusunu.

**KAPSAM DEĞİŞİKLİĞİ (11 Ekim):** Raspberry Pi 4 gerçek kaynak-kısıtlı fog
düğümü olarak **ICAIIC'e giriyor** (eski P0.3 kararı tersine döndü). Deney
bütçesi ~2 saat; Pi elde ve kurulabilir durumda. Tek engelleyici: Pi'ye SSH
erişimi + 64-bit OS + bağımlılıklar. Detay aşağıda **PX** bölümünde.

**Ölçüm penceresi kuralı güncellendi:** 1 Ekim yerine **12 Ekim akşamı**
kapanıyor (yalnızca PX Pi koşusu için uzatıldı). 12 Ekim'den sonra yeni deney
açılmaz, sadece yazım ve düzeltme. PX ve P1 dışındaki her şey masa başı iştir.

---

## Özet tablo

| Öncelik | İş | Süre | Son gün |
|---|---|---|---|
| **P0** | Üç karar (engelleyici) — verildi | — | tamam |
| **PX** | 🔴 **Pi'yi erişilebilir yap + kur + ~2 saat koş** | ~4 saat | **12 Eki** |
| **P1** | Yazılım yığınını eşitle ve yerleşimleri yeniden koş | 2–3 saat | **12 Eki** |
| **P2** | Veri hijyeni: tablo üreteci + beş sayı düzeltmesi | 3 saat | 12 Eki |
| **P3** | Şekil 1 ve LaTeX iskeleti | 3 saat | 12 Eki |
| **P4** | Makaleyi yaz (Pi sütunu dahil) | 2 gün | 14 Eki |
| **P5** | Revizyon, format, gönderim | 1 gün | 15 Eki |
| **P6** | IISEC'e ertelenenler (çok worker, broker, stat. test) | — | 29 Kas |

---

# P0 — Bugün verilecek üç karar (30 dakika)

Bunlar kilitlenmeden Tablo I ve Abstract yazılamaz.

### P0.1 Kanonik model sayısı — ✅ KARAR VERİLDİ (28 Eylül)

Aynı model için üç sayı dolaşıyordu: `config.py` val F1 0,9386 · 30 Ağustos
sweep LOSO F1 0,9307 · "makale için sonuçlar" Balanced CNN LOSO 0,920/0,920.

**Önemli tespit:** dağıtılan checkpoint de **tek bölünme**. `checkpoint_meta`:
13 denek eğitim, `val_subjects = [S11, S14]`, `best_val_f1 = 0.9386`. Yani
LOSO ensemble değil. Dolayısıyla "LOSO 0,92" demek, *dağıtmadığımız bir
eğitimin* sayısını raporlamak olurdu — transformer için eleştirdiğimiz hatanın
aynısı.

**Karar: LOSO sayısı kanoniktir, ama üç sayı üç ayrı adla verilecek ve asla
karıştırılmayacak.**

| # | Ne ölçer | Sayı | Makalede nasıl geçecek |
|---|---|---|---|
| 1 | **Mimarinin** başarımı | LOSO F1 **0,92** (15 kat) | *"the architecture attains 0.92 F1 under 15-fold LOSO"* |
| 2 | **Dağıtılan checkpoint** | val F1 **0,9386** (S11+S14 tutulmuş) | *"the deployed checkpoint was trained with S11 and S14 held out"* |
| 3 | Aynı checkpoint, **gerçek zamanlı** | doğruluk **0,8222** / stres F1 **0,6944** | *"…drops to 0.8222 under leak-free causal normalisation"* |

Üçüncü satır zaten makalenin bir bulgusu (düşüş modelin değil normalizasyon
kısıtının sonucu), yani bu kurgu hem dürüst hem işimize yarıyor.

**Yazım kuralları:**
- 1 numaraya "deployed model's accuracy" **deme**; 2 numaraya "LOSO" **deme**.
- Hangi seed ve hangi undersampling ile üretildiğini yaz.
- TranSensFusers'ın %91,13'ü de **tek bölünmeden** geliyor
  (`TEST_SUBJECT='S3'`, `VAL_SUBJECT='S16'`). Tablo I'de **"protocol"** ve
  **"sampling rate"** sütunları olacak; transformer satırında *"single
  held-out subject (S3)"*, 700 Hz / 7000 örnek. 28,6× çıkarım farkının bir
  kısmı örnekleme hızından geliyor, dürüstçe gösterilecek.

### P0.2 `Transformer.ipynb` depoya alınsın mı
Tekrar üretilebilirlik için alınması iyi; 3,7 MB. `.dockerignore` zaten
`*.ipynb`'yi imajdan dışlıyor, yani dağıtımı etkilemez.
**Önerim: al.**

### P0.3 Pi ICAIIC'e GİRİYOR — ✅ KARAR DEĞİŞTİ (11 Ekim)
Eski karar "Pi P6'ya ertelendi" idi. **Tersine döndü:** Pi elde, deney ~2 saat,
ve Pi'nin kattığı sonuç (kaynak-kısıtlı düğümde her pencereyi açıklamak
imkânsız) makalenin **ana tezini güçlendiren** en güçlü tek bulgu. Bu yüzden
6 sayfaya sığdırmaya değer: tek ek sütun (tablo) + tek ek çubuk (şekil 4).

**Çok worker ve broker hâlâ P6'da** — onlar ICAIIC'e girmiyor, 6 sayfaya ve
12 Ekim penceresine sığmaz. Sadece **tek Pi worker** giriyor.

Detaylı iş listesi: aşağıda **PX**.

---

# PX — Raspberry Pi 4 gerçek fog düğümü 🔴 (11–12 Ekim)

Makalenin "sis" düğümü şu ana kadar bir masaüstüydü; ölçülen **dağıtık
yerleşim**ti, kaynak kısıtı değildi. Pi 4 ile bu gerçek oluyor. Referans
makalelerin (FogETex, ICACT emotion) üçü de worker olarak Pi 4 kullanıyor —
yani bu, metodolojiyi onlarla **birebir** hizalayan parça.

**ÖLÇÜLDÜ (11 Ekim, Pi 4 Model B Rev 1.1, 4 GB, duman testi 2 pencere/durum):**

| Durum | Masaüstü (ref) | **Pi 4 ÖLÇÜLEN** `xai` | response | 5 s hop bütçesi | Pi/masaüstü |
|---|---|---|---|---|---|
| XAI yok | ~3,5 ms | — (inference 14,3 ms) | 27 ms | %0,5 | ~5× |
| LIME cheap (250/20) | ~346–588 ms | **769 ms** | 807 ms | %16 | ~2× |
| SHAP cheap (n=50) | ~394 ms | **1177 ms** | 1236 ms | %25 | ~3× |

⚠️ **Tahmin yanlış çıktı — tez düzeltildi.** Beklenen ~10× yavaşlama ve
"bütçeyi aşar" gerçekleşmedi: Pi yalnızca ~2–3× yavaş, her iki açıklayıcı da
tek kullanıcıda 5 s bütçesinin altında. Dolayısıyla **"her pencereyi açıklamak
imkânsız" iddiası DÜŞTÜ.** Yeni (dürüst) çerçeve:

> Kaynak-kısıtlı bir fog düğümünde (Pi 4) açıklamalı çıkarım tek kullanıcıda
> gerçek zamanlıdır (LIME ~0,77 s, SHAP ~1,18 s). Seçici XAI imkânsızı mümkün
> kılmaz; **çok-kullanıcı kapasitesini** artırır: worker başına ~5 s/1,18 s ≈
> **4 SHAP** veya ~5 s/0,77 s ≈ **6 LIME** kullanıcı; seçici açıklama (yalnız
> düşük-güvenli pencereler) bu sayıyı katlar.

ICAIIC'te bu, Pi'nin per-pencere maliyetini ölçüp **kapasite argümanını**
sayısallaştırır; çok-worker doğrulaması IISEC'te (P6.3). Bu düzeltme makalenin
Discussion ve Conclusion'ına aynen girecek — tahmine değil ölçüme dayandığı
için hakeme karşı daha güçlü.

**Not — stack:** Pi torch 2.14.1+cpu, masaüstü 2.8.0+cpu. Pi/masaüstü oranı
~2–3× olduğundan yığın farkı (1,1–1,7×) bu sonucu gömmez ama artık ihmal
edilebilir de değil; Pi sütunu "gösterim" olarak sunulur (bkz. PX.6), kesin
oran iddiası edilmez.

### PX.1 Pi'yi erişilebilir yap — ENGELLEYİCİ
İlk iş SSH. Pi ağda ve SSH portu açık (ölçüldü: 192.168.1.12, port 22 açık),
ama **kullanıcı adı/şifre** gerekiyor.
- Kartı kuran kişiden kullanıcı adı + şifre al; yoksa kartı 64-bit Raspberry
  Pi OS ile yeniden yaz (kurulumda SSH + kullanıcı baştan ayarlanır).
- Giriş sonrası **ilk komut `uname -m`** → **`aarch64` görmeli.** `armv7l`
  ise 32-bit OS; torch/shap düzgün kurulmaz, kartı 64-bit ile yeniden yaz.

### PX.2 Depoyu ve bağımlılıkları kur
```
# Pi'de
git clone <depo> && cd files      # ya da laptoptan scp ile kopyala
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Model dosyası ve checkpoint'ler de gitmeli (`.pt`). `.gitignore`'u kontrol et;
`wesad_best_model_f32-128_k9.pt` depoda değilse scp ile ayrıca kopyala.

### PX.3 Duman testi (akşam, 15 dk)
```
# Pi'de
python -m serving.server --host 0.0.0.0 --port 9099
# Laptopta (Pi IP = 192.168.1.12)
python run_fog_experiment.py --server-host 192.168.1.12 --subject S11 \
    --realtime --max-ticks 30 --tag pi_smoke
```
Bitti ölçütü: JSON düşüyor, `hardware` alanında Pi modeli görünüyor, response
medyanı masaüstü sisin ~birkaç katı. **Aktif soğutucu tak** — throttling
ölçümü bozar (`vcgencmd measure_temp` ile izle, 80 °C altı kalsın).

### PX.4 Asıl koşu (~2 saat, 12 Ekim)

⚠️ **XAI SUNUCU TARAFINDA seçilir** (ölçüldü 11 Ekim). `run_fog_experiment`'e
`--xai-method` vermek **uzak Pi sunucusunu etkilemez** — Pi sunucusu hangi
`--xai-method` ile başlatıldıysa onu kullanır. `compare_deployments.py`'nin
`--xai-method` bayrağı **yok** ve zaten uzak sunucuya XAI geçiremez. Bu yüzden
XAI karşılaştırması **eşli `run_fog_experiment` koşularıyla** yapılır:
her XAI durumu için (a) Pi sunucusunu o yöntemle yeniden başlat, (b) Pi'yi
`--server-host` ile ölç, (c) edge'i yerel sunucuyla ölç.

Üç XAI durumu: yok / LIME cheap / SHAP n=50.

**Her durum için Pi'de sunucuyu yeniden başlat** (Ctrl+C sonra):
```bash
# Pi'de — durum = yok
python -m serving.server --host 0.0.0.0 --port 9099 --xai-method none
# durum = LIME (her pencere açıkla)
python -m serving.server --host 0.0.0.0 --port 9099 --xai-method lime_cheap --xai-threshold 1.0
# durum = SHAP
python -m serving.server --host 0.0.0.0 --port 9099 --xai-method shap_cheap --xai-threshold 1.0
```

**Koşu uzunluğu kararı (11 Ekim):** `--max-ticks 2000` → (185+15 ısınma)×10
tick ≈ **17 dk realtime ≈ ~185 ölçülü pencere.** Gerekçe: (1) ICACT'in deney
başına **197 penceresine** denk; (2) kendi P2.1 kuralın "~44 pencereden p95/p99
çekme" diyor — tail istatistiği için yüzlerce pencere şart. Durum başına **1
uzun koşu** yeterli; tekrar birimi koşudur, kısa koşuları birleştirmekten iyi.

```powershell
# Laptopta — her durumda Pi (fog) + edge ölçümü, örnek: LIME
python run_fog_experiment.py --server-host 192.168.1.12 --subject S11 `
    --realtime --max-ticks 2000 --tag pi_lime
python run_fog_experiment.py --subject S11 --realtime --max-ticks 2000 `
    --xai-method lime_cheap --xai-threshold 1.0 --tag edge_lime
```
Aynısını `none` ve `shap_cheap` için (Pi sunucusunu her durumda yeniden
başlatarak). 3 durum × 2 yerleşim × 1 uzun koşu = 6 koşu × 17 dk ≈
**~100 dk**, sunucu yeniden başlatma ile ~2 saat. Vaktin kalırsa durum başına
bir 5 dk'lık kısa tekrar tutarlılık için eklenir (asıl değer uzun koşuda).

Sonuç dosyalarını yeni `results/15_pi/` altına topla.

### PX.5 `shap` aarch64'te kurulamazsa — yedek
`pip install shap` Pi'de derleme isteyip patlayabilir. O zaman:
- SHAP durumunu **atla**, LIME + "XAI yok" ile koş.
- Makalede: *"SHAP could not be installed on the aarch64 worker; LIME, being
  model-agnostic and pure-Python in the forward path, ran without issue."*
- Bu bile bir bulgu: model-agnostik yöntemin dağıtım kolaylığı.

### PX.6 Stack eşitleme kaygısı Pi'de GEÇERSİZ
P1'de masaüstü yerleşimler arası 1,1–1,7× fark yığın farkından geliyordu ve
bu yüzden iddia edilemiyordu. Pi'de etki **~10×** — yani torch tekerleği
aarch64'te farklı olsa bile sonucu değiştirmez. Pi sütunu bu yüzden P1.4
anlamında bir **gözlem/gösterim** olarak sunulur, mikro-saniye hassas bir
benchmark olarak değil. Makalede: *"the Pi demonstrates the order-of-magnitude
cost of a resource-constrained node, not a stack-matched micro-benchmark."*

### PX.7 Bitti ölçütü
`results/15_pi/` altında üç JSON; Pi'de SHAP n=50 ve/veya LIME 250 medyanı
5 s hop bütçesini aşıyor (ya da en az %50'sini tutuyor). Şekil 4'e Pi çubuğu,
Tablo (yerleşim) üçüncü sütun olarak Pi girer.

---

# P1 — En kritik ölçüm: yazılım yığınını eşitle 🔴

**Neden en öncelikli:** şu an üç yerleşimin sonucu karşılaştırılamaz ve
Bölüm V-C'deki yerleşim iddiası (8,7× → 1,13×) bu yüzden savunulamaz durumda.
Bu, hakemin reddedebileceği tek teknik açık.

### Kanıt — yığınlar farklı

| Yerleşim | Python | PyTorch |
|---|---|---|
| Uç (co-located) | 3.11.9 | 2.8.0**+cpu** |
| Sis | 3.14.7 | 2.14.0+cpu |
| Bulut | 3.11.16 | 2.14.0**+cu130** (CUDA yapısı) |

### Kanıt — sonuç fiziksel olarak tutarsız

Açıklama süresi medyanları, sis/uç oranı:

| Durum | Uç (ms) | Sis (ms) | Sis / Uç |
|---|---|---|---|
| LIME 500 örnek | 682,0 | 1154,2 | 1,69 |
| LIME 250 örnek | 345,9 | 584,0 | 1,69 |
| SHAP n=200 | 1143,3 | 1426,2 | 1,25 |
| SHAP n=50 | 445,5 | 390,4 | **0,88** |
| XAI yok | 5,8 | 3,5 | **0,60** |

LIME'da oran iki noktada da tam 1,69. SHAP'ta 1,25 ve 0,88 — **işaret
değiştiriyor.** Donanım farkı bunu yapmaz. Gradyan yolu (SHAP
GradientExplainer) sürüm farkından etkileniyor, ileri-geçiş yolu (LIME)
etkilenmiyor. Bu haliyle "sis daha yavaş" cümlesi kurulamaz.

### P1.1 Tek imaj hazırla
`python:3.11` tabanlı, `torch==2.14.0+cpu` sabitlenmiş tek Docker imajı.
Mevcut Docker kurulumu üzerine gidilecek; sürümler `requirements.txt`'te
**pinlenecek** (`==`, `>=` değil).

### P1.2 Üç yerleşimde de aynı imajla koş
5 XAI durumu × 3 yerleşim × 5 dk ≈ **75 dk saf koşu.** Docker kurulumu ve
imaj çekme ile birlikte 2–3 saat ayır.

> ⚠️ `ICAIIC_PLAN.md` F4'ü "sadece fog, ~25 dk" diye yazmıştı. **Yetersiz:**
> uç hâlâ torch 2.8'de, bulut hâlâ cu130 tekerleğinde kalırdı. Üçü de
> yeniden koşulacak.

### P1.3 Bitti ölçütü
Sis/uç oranı **beş durumda da aynı yönde** ve birbirine yakın büyüklükte
olmalı. Hâlâ işaret değiştiriyorsa donanım dışı bir etken daha var demektir.

### P1.4 Fallback — eşitlenemezse
Yerleşim karşılaştırmasını makalenin **iddiası olmaktan çıkar**, gözlem
olarak bırak: "XAI açıkken yerleşimler arası fark, yığın farkının
mertebesine iniyor" de ve 8,7× / 1,13× sayılarını verme. Makale bu olmadan
da ayakta durur — ana bulgu açıklama maliyeti, yerleşim değil.

**Son gün: 1 Ekim.** Bu tarihe kadar bitmezse P1.4 uygulanır.

---

# P2 — Veri hijyeni (30 Eylül, ~3 saat)

Bunlar ölçüm değil, makaleye yanlış sayı taşımayı önleyen işler.

### P2.1 `tools/paper_tables.py` yaz — en yüksek getirili tek iş
Tablo III–VI'yı JSON'dan LaTeX `tabular` olarak üretsin, `\input{}` ile
girsin. **Elle sayı kopyalanmayacak.** İçine üç koruma gömülecek:

- **Sınıflandırma kilidi:** `04_fog`, `07_edge_xai`, `13_bulut` dosyalarındaki
  `classification` alanı okunursa **hata fırlat**. Kanonik kaynak
  `01_streaming`: 2123 ölçülen / 1198 etiketli, doğruluk 0,8222, stres F1
  0,6944. (Fog koşularında S11'in ilk 5 dakikası var, 42 pencerenin hepsi
  stres-dışı, doğruluk 1,00 — anlamsız.)
- **`n` sütunu zorunlu:** her tabloda pencere sayısı görünecek.
- **p95/p99 kilidi:** 44 pencerelik koşulardan p95/p99 çekilirse uyarı ver.
  Tek sağlam uç-değer kaynağı `09_secici_xai/fog_sel80_tam.json` (1030
  pencere).

### P2.2 Ad çakışmasını gider 🔴
`02_benchmark/benchmark_xai_cpu.json` içindeki **`lime_ucuz` = 500 örnek /
5 dilim**. Sunucudaki **`lime_cheap` = 250 örnek / 20 dilim**. Aynı ad,
farklı yapılandırma, farklı maliyet. İkisi makalede yan yana gelirse
tutarsızlık çıkar.

**Yapılacak:** dosyadaki kaydın adını `lime_seg5` yap, ya da o dosyayı
makale dışı ilan et ve `results/README.md`'ye öyle yaz.

### P2.3 `results/README.md` — beş sayı düzeltmesi
Üçü doğrulandı:

| Yer | README'de yazan | Dosyada olan |
|---|---|---|
| `benchmark_xai_cpu.json` | "250 örnek p50 490 ms" | **Böyle bir kayıt yok.** `lime` = 500/20/kw0,5 → p50 710,2 ms; `lime_ucuz` = 500/**5**/kw0,25 → p50 677,6 ms |
| `05_yerlesim` | uç 0,81 / 6,19 / 7,44 · sis 3,69 / 3,61 / 9,97 | uç **0,928 / 6,344 / 7,880** · sis **4,910 / 3,372 / 10,151** (tur medyanlarının medyanı) |
| `01_streaming` | "1175 pencere" | **2123 ölçülen / 1198 etiketli** |

Kalan iki farkı ayrıntılı denetim raporunun 12. bölümünden al ve aynı şekilde
düzelt.

### P2.4 Kök `README.md` — eski tablo
Satır 215–224'teki "Ölçülen sonuçlar" tablosu `prepared` dönemine ait
(preprocessing 0,189 · inference 1,15 · toplam 1,270 ms). Güncel `native`
medyanları: yeniden örnekleme 1,378 · çıkarım 2,682 · **toplam 4,689 ms**.
Ya düzelt ya "eski — bkz. `results/01_streaming`" notuyla sil.

### P2.5 İki dil düzeltmesi (plana not olarak)

- **Eşik taraması ölçüm değil, projeksiyon.** `kisa_segment_sel60/80/90`
  koşularında açıklama oranı **0,0** — eşik hiç tetiklenmemiş. Uçtan uca
  ölçülmüş tek eşik **τ=0,80, 1030 pencere**. Diğer eşikler karışım
  modelinden. Makalede: *"projected, validated at τ=0.80."*
- **Ortalama/medyan ayrımı.** `fog_sel80_tam`: ortalama response **80,1 ms**,
  medyan **11,0 ms** (dağılım iki tepeli: %86 pencere 3,6 ms, %14 pencere
  ~365 ms). "Mean" olduğu açıkça yazılacak, medyan da verilecek.

---

# P3 — Şekil ve iskelet (1 Ekim, ~3 saat)

### P3.1 Şekil 1: sistem mimarisi — **zorunlu**
Sistem bildirisinde mimari şekli olmadan hakem sistemi anlayamaz. TikZ ile
`main_en.tex` içinde çizilecek. Kutular:

```
Sensor client (replay, 500 ms tick)
  → TCP → Reader task (t_recv damgası) → Queue
  → Worker: resample 64 Hz → causal z-score → CNN → softmax
           → conf < τ ? → explainer (LIME / SHAP-GE)
  → Reply
```
Üç yerleşim küçük etiketlerle gösterilecek.

> Not: 24 Eylül tarihli Türkçe `main.tex` içinde çalışan bir TikZ diyagramı
> zaten var, başlangıç noktası olarak kullanılabilir.

### P3.2 `main_en.tex` iskeleti
IEEEtran, İngilizce, `babel` Türkçe **kalkacak**, ondalık **nokta**.
Bölüm başlıkları ve `\input{}` çağrıları hazır dursun, metin sonra dolsun.

---

# P4 — Yazım (1–6 Ekim)

`ICAIIC_PLAN.md` Bölüm C'deki iskeleti izle. Sıra bilinçli: sonuçlar önce,
giriş en son.

| Gün | Bölüm |
|---|---|
| 13 Eki | **V. Results** (Pi sütunu + şekil 4 dahil) + **III. System** + **IV. Setup** + Tablo I, II |
| 14 Eki | **I. Intro** + **II. Related Work** + Abstract + **VI, VII, VIII** + kaynakça → hocaya taslak |

> Takvim sıkıştı (4 gün). Hoca dönüşü için ayrı gün yok; 14 Ekim akşamı taslağı
> gönder, 15 sabahı dönüşü işle ve yükle. Mümkünse 13 Ekim akşamı erken bir
> taslak at ki 14'te geri bildirim gelsin.

**Yazarken uyulacak üç kural:**
1. Her sayının yanında hangi koşudan geldiği ve pencere sayısı.
2. "SHAP" değil **"SHAP (GradientExplainer)"** ilk geçişte; "LIME" değil
   **"region-based LIME"**. Literatürdeki "SHAP yavaş" iddiası KernelSHAP'a
   ait, farklı maliyet yapısı — bunu Discussion'da açıkla.
3. "edge" değil **"co-located"**; "ucuz" değil **"cheap"**; "clinical" ve
   "live system" hiç geçmeyecek — bu bir **emülasyon**.

**Bitirme projesinden ne girmeyecek:** XAI kalite tablosu (farklı model,
farklı normalizasyon uzayı, farklı LIME uygulaması — çelişki daveti), kanal
önem analizi, vaka çalışmaları, Integrated Gradients, veri seti
karşılaştırmaları, fizibilite/Gantt.

---

# P5 — Revizyon ve gönderim (7–15 Ekim)

| Gün | İş |
|---|---|
| 15 Eki sabah | Hoca dönüşünü işle, tam okuma, **6 sayfa sınırı**, şekil kalitesi, EDAS formatı |
| **15 Eki öğleden önce** | **EDAS'a yükle.** Son saatlerde EDAS yavaşlar — öğleye kadar bitir. |

Boş saat çıkarsa: Pareto şekli (`14_figures/fig3`) ya da S14 seçici koşusu
(5 dk). **12 Ekim'den sonra yeni deney açılmaz** (PX dahil).

---

# P6 — IISEC'e ertelenenler (29 Kasım, genişletilmiş sürüm)

ICAIIC'ten sonra, hakem yorumları beklenirken yapılacaklar.

### P6.1 Raspberry Pi 4 — ✅ ICAIIC'E TAŞINDI (11 Ekim)
Bu madde artık **PX** altında ana kapsamda. Buraya not olarak kalsın; detay
yukarıda. P6'da kalan tek Pi işi: **çok worker + broker** (PX tek worker).

### P6.2 Kaynak kısıtını doğru yöntemle taklit et
⚠️ **Tek çekirdeğe kilitleme (affinity) işe yaramıyor** — ölçüldü: süreç
zaten tek iplik, tepe CPU %105–111, affinity hiçbir şeyi değiştirmiyor.
Bağlayıcı tek yöntem **kesirli CPU kotası**: `docker run --cpus=0.1`.

### P6.3 Çok worker ve broker
Şu an tek worker + açık kuyruk. XAI açıkken tek worker 5 s hop bütçesinde
**3–12 kullanıcı** taşıyor (`10_yuk`: 4 istemci × `shap_cheap` → response
medyanı 963 ms). Yani dağıtılacak gerçek bir yük artık var — XAI'siz yoktu.

- İş parçacığı değil **süreç** kullan (Python GIL).
- IPC: `multiprocessing.Queue` ile başla; Linux'ta FIFO da ölç.
- Bellek bağlayıcı olacak: worker başına ~465 MB (SHAP açıkken).
  4 worker ≈ 1,9 GB → Pi 4'te 4 GB sınırda, 8 GB rahat.
- Broker'ı **en-az-yüklü worker'a** yönlendirsin, round-robin değil: iş
  süreleri 394–1431 ms arasında değişiyor.

### P6.4 Tekrar sayısını 6'ya çıkar ve istatistik testi ekle
İki yönlü Wilcoxon işaretli sıra testinde ulaşılabilecek en küçük p değeri
n=5'te 0,0625, **n=6'da 0,031**. Anlamlılık iddia edilecekse tur sayısı
en az 6 olmalı. Cohen's *d* her durumda raporlanabilir.

**Tekrarlama birimi koşudur, pencere değil.** Pencere bazında test sahte
tekrarlama olur (aynı koşudaki pencereler ısınma kayması yüzünden bağımsız
değil) ve güveni olduğundan büyük gösterir. Bu cümle makaleye de girmeli.

### P6.5 Diğer
S14 seçici koşusu · kişiye özel kalibrasyon · kullanıcı çalışması ·
model karmaşıklığı ablasyonu (F1–gecikme Pareto).

---

# Riskler

| Risk | Belirti | Yedek |
|---|---|---|
| Docker kurulumu uzuyor | 29 Eylül akşamı imaj hazır değil | P1.4: yerleşim iddiasını gözleme indir |
| P1 sonrası oran hâlâ tutarsız | işaret değiştirmeye devam ediyor | P1.4 + sınırlılıklara açık madde |
| 6 sayfaya sığmıyor | 11 Ekim'de 7+ sayfa | Bölüm V-E (açıklayıcı kalitesi) ve Pareto şekli çıkar |
| Hoca dönüşü gecikiyor | 10 Ekim'de yanıt yok | 11–12 Ekim'i kendi okumana ayır, 13'te tek tur al |
| Pi gelmiyor | — | Sorun değil, P6'da |

---

# Hemen şimdi (11 Ekim)

1. 🔴 **PX.1 — Pi'ye SSH gir.** Kullanıcı adı/şifreyi bul; yoksa kartı 64-bit
   OS ile yeniden yaz. Girince `uname -m` → `aarch64` doğrula. Her şey buna
   bağlı; önce bu.
2. Pi erişilir olunca **PX.2 kurulum** + **PX.3 duman testi** (bu akşam). Aktif
   soğutucuyu tak.
3. Kurulum/koşu arka planda sürerken masa başı işler: **P2.2, P2.3** (sayı
   düzeltmeleri) ve **P1.1** (tek Docker imajı) paralel ilerlesin.
4. Yarın (12 Ekim): **PX.4 asıl Pi koşusu (~2 saat)** + P1 yerleşim koşuları.
   Akşam ölçüm penceresi kapanır.
