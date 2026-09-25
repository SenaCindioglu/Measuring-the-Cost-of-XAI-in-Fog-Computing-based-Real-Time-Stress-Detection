# ICAIIC bildirisi — hakem gözüyle değerlendirme ve yazım planı

Hazırlanma: 25 Eylül 2026. Bu dosya `WRITING_PLAN.md`'nin yerine geçer
(o ICACT içindi; yapı büyük ölçüde aynı, kapsam kararları burada güncel).

**CFP'den doğrulandı (25 Eylül):** son tarih **15 Ekim 2026**, EDAS
https://edas.info/N35066, IEEE çift sütun, en az 10 pt, A4/letter,
**en fazla 6 sayfa** (en az 3). Yani 20 gün var, 6 değil — çizelge ve F4
kararı buna göre güncellendi (Bölüm F, G).

---

## BÖLÜM A — Hakem raporu (bildiriyi gönderilmiş gibi okudum)

### Ne var elimde, hakem ne görecek

Elimdeki malzeme üç ayrı çalışmadan geliyor ve **üçü aynı bildiriye sığmaz**:

| kaynak | ne | model | normalizasyon | XAI |
|---|---|---|---|---|
| Bitirme projesi (Ocak 2026) | CNN / CNN-LSTM / TranSensFusers pencere taraması, LOSO | TranSensFusers en iyi (91.13%) | denek bazlı (sızıntılı) | SHAP + LIME + IG, `lime` paketi |
| 30 Ağustos sweep | Mimari OFAT, dengeli CNN | depth2 [32,128] k9, 49.9K param | denek bazlı | — |
| Bu depo (Eylül) | gerçek zamanlı sistem, fog/edge/cloud, seçici XAI | **aynı 49.9K CNN, dondurulmuş** | **nedensel (sızıntısız)** | SHAP GradientExplainer + bölge tabanlı özel LIME |

**Hakem olarak ilk sorum:** "Bitirmede en iyi model TranSensFusers'tı, neden
bildiride 50K parametrelik CNN var?" Bu sorunun cevabı bildiride **açıkça**
yazılmazsa red sebebi olur. Cevap var ve iyi: dağıtım için gecikme odaklı
seçim. Aşağıda nasıl yazılacağı var.

### Güçlü yanlar (hakem bunları takdir eder)

1. **Gerçekten ölçülmemiş bir soru.** Nauta ve ark. Co-12 çerçevesi çalışma
   süresini kapsam dışı bırakıyor; WESAD üzerinde XAI uygulayan hiçbir çalışma
   pencere başına açıklama gecikmesi vermiyor. Boşluk gerçek ve belgelenmiş.
2. **Simülasyon değil, uçtan uca gerçek ölçüm.** Gerçek TCP, üç fiziksel
   yerleşim (aynı makine / LAN'daki ikinci bilgisayar / DigitalOcean), yedi
   sistem metriği, saat senkronizasyonu. ICAIIC'nin *information &
   communication* kimliğine tam oturuyor.
3. **Ana bulgu net ve büyük:** açıklama, çıkarımın 118–421 katı. Yerleşim
   kararı XAI'siz belirleyici (bulut/edge 8.7×), XAI'li önemsiz (1.13×).
4. **Çözüm önerisi ölçülmüş:** güven eşikli seçici açıklama, 1030 pencerede
   doğrulanmış, projeksiyon ile ölçüm %2 uyumlu. 5.6× ucuzlama, hataların
   %90'ı hâlâ açıklanıyor, kapasite 12 → 90 istemci.
5. **Dürüst sınırlılık:** kalibrasyon bulgusu (S11 ECE 0.047, S14 0.227)
   kendini zayıflatan bir sonuç ve bunu raporlamak güven verir.
6. **Doğrudan karşılaştırılabilir referans:** Erzurumluoğlu ve ark. ICACT 2025,
   aynı yedi metrik, benzer kurulum.

### Zayıf yanlar (hakem bunları yazacak — hepsine ön cevap hazır olmalı)

| # | Hakem itirazı | Şiddet | Ön cevap / yapılacak |
|---|---|---|---|
| Z1 | "Neden en iyi model değil?" | **yüksek** | III.A'da gecikme odaklı model seçimi tablosu. **Ölçüldü (25 Eyl, aynı CPU, tek iplik):** TranSensFusers 1,859,026 param / 103.7 ms; CNN 49,890 / 3.63 ms → **37.3× parametre, 28.6× çıkarım**. Dengeli CNN LOSO F1 0.92, transformer 0.904. Cevap kapanmış sayılır. |
| Z2 | "n=2 tutulmuş denek" | yüksek | Sınırlılıklarda açıkça. Kalibrasyon bulgusu n=2 üzerine kurulu, "denek bağımlı" iddiası genelleme değil gözlem. |
| Z3 | "SHAP dediğiniz KernelSHAP mi?" | yüksek | Her yerde **SHAP GradientExplainer** yaz. Literatürdeki "SHAP yavaş" iddiası KernelSHAP'a ait; farklı maliyet yapısı. Bunu Discussion'da açıkla — bulguları literatürden ayıran şey bu. |
| Z4 | "Bitirmede SHAP daha kararlıydı, burada LIME. Hangisi doğru?" | **yüksek** | **Bitirmenin XAI kalite tablosunu bu bildiriye KOYMA.** Farklı model, farklı normalizasyon uzayı, farklı LIME uygulaması (standart paket vs bölge tabanlı), metrik ölçeği 10× farklı. Karşılaştırılamaz; ikisini yan yana koymak hakemi bu soruya davet eder. |
| Z5 | "Yerleşim karşılaştırması adil mi?" | orta | Edge PyTorch 2.8/Py3.11, fog 2.14/Py3.14, cloud Docker py3.11. `server_hardware` alanında kayıtlı. Sınırlılıklarda tek cümle; ya da fog'u Docker'da tekrar koş (25 dk) ve yığını eşitle. |
| Z6 | "Faithfulness'ta fark yok, o zaman kalite bölümü ne için?" | orta | Tam olarak bu bulgu: **fark yok, seçim gecikme ve kararlılıkla yapılır.** Bunu iddia olarak yaz, zayıflık olarak değil. |
| Z7 | "Fog düğümü masaüstü, Pi değil" | orta | Sınırlılık + hesaplı çıkarım: referans Pi 4 execution 34.7 ms / bizde 3.6 → ~10×; aynı oran `shap_cheap`'i ~4 s'ye taşır, 5 s hop bütçesini doldurur. Seçici açıklama Pi'de zorunlu hale gelir. |
| Z8 | "Fog koşuları 44 pencere, tek tekrar" | orta | Ana seçici koşu 1030 pencere. Diğerleri ortalama karşılaştırması için yeterli; p95/p99 yalnızca 1030'luk koşudan. Yayılım gereken yerde `02_benchmark` (4 tur dönüşümlü). |
| Z9 | "Kuyruk ölçümü gerçekçi mi?" | düşük | İstemciler eşzamanlı başlatıldı → üst sınır. (N−1)×açıklama modeli ölçümle uyuşuyor. |
| Z10 | "Kullanıcı çalışması yok" | düşük | Standart XAI sınırlılığı; bir cümle. |

### Hakem kararım

**Gönderilebilir — ama ancak tek bir hikâye anlatırsa.** Üç çalışmayı
birleştirmeye çalışan bir bildiri 6 sayfada dağılır ve Z1+Z4'ten red yer.
Tek hikâye şu:

> **Açıklama, gerçek zamanlı stres tespitinde darboğazdır; nerede hesapladığın
> değil, ne zaman açıkladığın belirleyicidir.**

Bitirme projesi bu hikâyede **arka plan**dır (modelin nereden geldiği),
**sonuç** değil.

---

## BÖLÜM B — Kapsam kararları (yazmaya başlamadan önce kesinleşmeli)

### B1. Bitirme projesinden ne girer, ne girmez

| bitirme içeriği | karar | nerede / nasıl |
|---|---|---|
| Üç mimari + pencere taraması | **girer, tek tablo** | III.A "Model selection", 3 satır: her mimarinin en iyi penceresi, acc, F1, parametre. Gerekçe cümlesi: dağıtım için CNN. |
| TranSensFusers 91.13% | girer, tabloda | ama "en iyi model" değil, "en yüksek doğruluk, en pahalı" |
| CNN 30 s / CNN-LSTM 30 s sonuçları | tabloda birer satır | ayrı anlatım yok |
| Kanal önem analizi (Şekil 5.1–5.8) | **girmez** | farklı model; bildiri bu soruyu sormuyor |
| Vaka çalışmaları (TP/TN/FP/FN) | **girmez** | 6 sayfaya sığmaz, konu dışı |
| XAI kalite tablosu (Tablo 5.10) | **girmez** | Z4 — karşılaştırılamaz, çelişki daveti |
| Integrated Gradients | **girmez** (senin kararın) | hiçbir yerde geçmesin |
| Literatür taraması (Bölüm 2) | girer, **ağır kesilmiş** | II'de iki paragraf; DEAP/SWELL/AMIGOS/AffectiveROAD karşılaştırması **girmez** |
| Fizibilite, Gantt, gereksinimler | **girmez** | bitirme formalitesi |
| Focal loss, AdamW, LOSO protokolü | girer, iki cümle | III.A |

**Ek istek:** bitirme XAI sonuçlarına tek cümlelik atıf yapılabilir:
*"In the offline study [thesis], SHAP and LIME were compared on the
transformer model; here we restrict the comparison to the deployed CNN in the
server's causal input space, which is the only setting in which explanation
cost is meaningful."* Bu, Z4'ü açmadan bitirmeyi anmış olur.

### B2. Hangi sayılar kanonik

Aynı model için üç farklı sayı dolaşıyor. **Birini seç, her yerde onu kullan:**

| kaynak | sayı |
|---|---|
| `config.py` | val F1 0.9386 (S11+S14, seed 42) |
| 30 Ağustos sweep | LOSO F1 0.9307 (depth2 k9) |
| "makale için sonuçlar" s.1 | Balanced CNN LOSO Acc 0.920 / F1 0.920 |

Önerim: **LOSO** sayısı (0.92 / 0.92 ya da 0.9307) — çünkü TranSensFusers de
LOSO ile raporlanmış, karşılaştırma adil olur. Hangisinin son eğitim olduğunu
sen biliyorsun; kayıtta hangi seed ve hangi undersampling ile üretildiğini de
yaz. Val F1 0.9386'yı **kullanma**, iki denek üzerinden ve LOSO değil.

**TranSensFusers sayıları (25 Eyl, `Transformer.ipynb` hücre 10'dan
çıkarılıp bu makinede ölçüldü):** 1,859,026 parametre (not defteri çıktısıyla
birebir), 103.7 ms medyan / 118.1 ms p95 çıkarım, girdi 14×7000 @ 700 Hz.
Aynı koşulda CNN 3.63 ms. Bu kapandı.

### ⚠️ B2-ek: %91.13 tek bölünmeden geliyor, LOSO değil

Hücre 10'daki `Config`: `TEST_SUBJECT = 'S3'`, `VAL_SUBJECT = 'S16'`, geri
kalan 13 denek eğitim. Yani bitirmenin "LOSO" dediği sayı **tek bir test
deneğinin** (S3) sonucu; 15 katlı ortalama değil. Doğrulama F1'i 1.0000 (S16)
— tek denekte olağan ama kırılgan.

Bildiride bu sayıya **"LOSO" deme.** Doğru ifade:

> *…reached 91.13% accuracy on a single held-out subject (S3) in the offline
> study [thesis]…*

CNN'in 0.92'si ise gerçek 15 katlı LOSO. Tablo I'de ikisinin protokolü ayrı
sütunda yazılmalı; aksi halde hakem kodu isterse tutarsızlık çıkar. Bu ayrıca
Z1'i güçlendiriyor: CNN daha sıkı protokolde aynı seviyeyi tutuyor.

**Dikkat — pencere hızı da farklı:** transformer 700 Hz ham sinyalde (7000
örnek/pencere), CNN 64 Hz'de (640). Tablo I'de "sampling rate" sütunu olsun;
28.6× çıkarım farkının bir kısmı buradan geliyor ve dürüstçe gösterilmeli.

### B3. Adlandırma — tutarlılık

- **SHAP → "SHAP (GradientExplainer)"** ilk geçtiği yerde, sonra "SHAP-GE" ya
  da sadece SHAP ama Methods'ta açık tanım. Z3'ün tek panzehiri bu.
- **LIME → "region-based LIME"** ilk geçtiği yerde; 20 zaman dilimi × 14 kanal
  bölge, kanal ortalamasıyla kapatma. Standart `lime` paketi değil, açıkça yaz.
- **cheap** — `shap_cheap`, `lime_cheap`. "ucuz" hiçbir yerde geçmesin.
- **placement** — edge yerine **co-located** kullan. Bitirme ve referans
  makalede "edge" farklı anlamda; karışıklık hakemi yorar.
- **explanation rate** — "trigger rate" değil, tutarlı kal.

---

## BÖLÜM C — Bildiri iskeleti (6 sayfa)

```
Title + Abstract (200–230 kelime)                       0.3
I.   Introduction                                        0.6
II.  Related Work                                        0.5
III. System                                              1.3
     A. Model selection and training
     B. Real-time pipeline: replay, causal normalisation, client–server
     C. Explainers and selective triggering
IV.  Experimental Setup                                  0.5
V.   Results                                             2.0
     A. Explanation dominates the pipeline
     B. Selective explanation recovers the cost
     C. Placement stops mattering under explanation
     D. Capacity under load
     E. Explainer quality: equal faithfulness, different stability
     F. The calibration condition
VI.  Discussion                                          0.4
VII. Limitations and Future Work                         0.3
VIII. Conclusion                                         0.1
References                                               ~0.4 (yaklaşık 20)
                                                         ------
                                                         ~6.4 → sıkıştırılacak
```

### I. Introduction — dört paragraf, tek sayfa değil, yarım

1. Giyilebilir stres tespiti olgun; WESAD'da %90+ rutin. XAI de yaygın.
2. **Ama açıklamanın maliyeti ölçülmüyor.** Co-12 alıntısı. Gerçek zamanlı bir
   sistemde bu boşluk pencere bütçesine dokunuyor.
3. Bu çalışma: dondurulmuş bir sınıflandırıcıyı gerçek istemci–sunucu
   dağıtımına yerleştirip açıklama maliyetini yedi sistem metriğiyle ölçüyor,
   sonra güven eşikli seçici açıklama öneriyor.
4. Katkılar — **dört madde, sıralı, ölçülebilir:**
   - (i) ilk uçtan uca ölçüm: açıklama çıkarımın 118–421×'i, üç yerleşimde
   - (ii) yerleşim kararı açıklama yükü altında anlamsızlaşıyor (8.7× → 1.13×)
   - (iii) seçici açıklama: 5.6× ucuz, hataların %90'ı hâlâ açıklanıyor,
     kapasite 7.5×; projeksiyon ile ölçüm %2 uyumlu
   - (iv) faydanın kalibrasyon koşulu ve dört açıklayıcı noktasında
     sadakat–kararlılık–maliyet dengesi

### II. Related Work — iki paragraf

- *Fog-based physiological monitoring.* Referans makale burada; broker'ın
  gereksizliği (43.3 vs 43.9 ms) tek cümle. Chen (I-Fabric), Ejaz & Shah,
  Meng — bitirme literatüründen üç atıf yeter.
- *The cost of explanation.* Co-12, Quantus/OpenXAI süre sütunu yok, ShaTS
  1 s/pencere sınırı, gömülü XAI ölçen iki çalışma. `docs/literature.md`
  bölüm 3 ve 6'dan.

**Kes:** veri seti karşılaştırmaları, mimari aileler taraması, kullanıcı
çalışması boşluğu (bir cümle Future Work'e).

### III. System

**A. Model selection and training.** Tablo I: üç mimari. Metin: LOSO, focal
loss (γ=2), 10 s pencere / 5 s kayma (%50 örtüşme), çoğunluk etiketi, sınıf
dengeleme. Seçim cümlesi:

> *Although the transformer reached the highest LOSO accuracy, a compact CNN
> (depth 2, filters [32,128], kernel 9; 49,890 parameters) matched its F1 under
> class-balanced training at [X]× fewer parameters and [Y] ms inference, and was
> frozen for deployment. All subsequent measurements use this model unchanged.*

**B. Real-time pipeline.** Üç alt-nokta, her biri bir paragraf:
- *Replay at native rates.* Göğüs 700 Hz, bilek 64/32/4 Hz, 500 ms tick,
  sunucuda 64 Hz'e indirgeme (maliyeti ölçülüyor).
- *Leak-free causal normalisation.* Welford, yalnızca o ana kadarki veri.
  Offline denek-bazlı z-score'un gerçek zamanlı karşılığı. Doğruluk
  düşüşünün sebebi model değil bu kısıt (bir cümle).
- *Client–server and clocks.* Açık kuyruk + tek worker (referans makaleyle
  aynı, queuing ölçülebilmesinin ön koşulu). `perf_counter`, NTP mantığıyla
  fark, **60 s'de bir yeniden senkronizasyon** ve doğrusal ara değer.
  Belirsizlik = RTT/2, raporlanıyor.

**C. Explainers and selective triggering.**
- Region-based LIME: 20 dilim × 14 kanal = 280 bölge, kanal ortalamasıyla
  maskeleme, ağırlıklı Ridge. İki çalışma noktası: 500 ve 250 örnek.
- SHAP GradientExplainer: 100 pencerelik kmeans arka plan (eğitim
  deneklerinden, test denekleriyle kesişmiyor), nsamples 200 ve 50. Seed sabit
  (aksi halde sensitivity yöntemin kendi gürültüsünü sayıyor — 0.139 → 0.075).
- Kalite ölçütleri: faithfulness = insertion − deletion AUC, monotonicity,
  sensitivity. Taban çizgi kanal ortalaması (fizyolojik gerekçe).
- **Selective triggering:** açıklama yalnızca `confidence < τ` ise, tahminle
  aynı worker'da, execution time'ın içinde. τ=0 hiç, τ=1 her pencere.
  Açıklayıcı açılışta ısıtılıyor (SHAP ilk çağrı 12.4 s).

### IV. Experimental Setup

Tablo II: üç yerleşim — CPU, RAM, Python/PyTorch, ağ. **Yazılım yığını
farkını tabloya koy**, saklama. Ölçüm kuralları: ısınma 15 pencere hariç,
tek iplik (gerekçe: yayılım %2.9 vs %11–15, dönüşümlü 4 tur), S11 tam kayıt
seçici koşuda (1030 pencere), diğerleri 600 tick (44 pencere).

Referans makale karşılaştırma protokolü: onların 3 s pencere / 1 s kayma /
32 istek/s; bizim 10 s / 5 s / 2 tick/s. Kapasite hesabı hop/execution.

### V. Results — altı alt bölüm, her biri "ölçüm → sayı → anlam"

**A. Explanation dominates.** Tablo III (fog, beş durum, aşama dökümü) +
Şekil 2 (fig1). Cümle: inference 3.5 ms, xai 395–1422 ms, oran 118–421×.
Bellek 210 → 464 MB.

**B. Selective explanation.** Tablo IV (hiç / seçici 0.80 / hep) + Şekil 3
(fig2). 1030 pencere, %14.1 açıklama, response 450.5 → 80.1 ms, D1'e göre
hataların %89.7'si. Projeksiyon vs ölçüm: %2.3 / %2.0 / %8.5. Kuyrukta
birikme yok (patlamalı yük etkisi yok).

**C. Placement.** Şekil 4 (fig5). XAI'siz bulut/co-located 8.7×, `shap` ile
1.13×. Latency 0 / 4.3–5.3 / 53–61 ms. Bulut VPS hesapta fog'dan yavaş
değil. **Yığın farkı uyarısı burada bir cümle.**

**D. Capacity.** Tablo V (1/2/4 istemci × XAI yok/var) ya da metin +
Şekil (fig4). (N−1)×açıklama modeli: 3×370=1110 tahmin, 1125 ölçüm. Kapasite
1394 / 90 / 12. Referans: ~20.

**E. Explainer quality.** Tablo VI (dört nokta: gecikme, faithfulness ± SEM,
monotonicity, sensitivity). **Baskınlık yok** cümlesi. LIME 2.7× kararlı,
SHAP-GE 3× ucuz. Örnek başına: LIME 2.4 ms, SHAP 7–8 ms → literatürle uyum
ama toplam tersine dönüyor. Şekil (fig3) yer varsa.

**F. Calibration.** Şekil 5 (fig6). ECE 0.047 vs 0.227, aynı ortalama güven.
Eşik 0.80'de S11 %90 / S14 %12 hata yakalama. Sıcaklık ölçekleme çapraz
denekte başarısız (zıt yön). **Maliyet tasarrufu her ikisinde, fayda
kalibrasyona bağlı.**

### VI. Discussion — üç paragraf

1. Mimari kararın yer değiştirmesi: yerleşim → tetikleme politikası.
2. Referans makaleyle karşılaştırma: **yalnızca bant genişliği** doğrudan
   (187 vs 374 kbps, ham sinyal vs 24 öznitelik). Zaman metrikleri donanım
   farkı taşıyor. Onların tablosu da edge'i gösteriyor, fog'u seçme gerekçesi
   zaman değil.
3. GradientExplainer ≠ KernelSHAP: literatürün "SHAP yavaş ama doğru"
   beklentisinin neden tutmadığı.

### VII. Limitations — beş madde, her biri bir cümle

n=2 denek · fog masaüstü (Pi hesabı) · yazılım yığını farkı · kuyruk üst
sınır · kullanıcı çalışması yok. Future: Pi 4 (IISEC), çok worker, kişiye
özel kalibrasyon.

---

## BÖLÜM D — Şekil ve tablo listesi (nihai)

| # | içerik | kaynak | durum |
|---|---|---|---|
| Fig 1 | **Sistem mimarisi diyagramı** (istemci → TCP → kuyruk → worker → model → [τ?] → açıklayıcı) | **yok, çizilecek** | ⚠️ |
| Fig 2 | maliyet dökümü, doğrusal+log | `14_figures/fig1` | ✅ |
| Fig 3 | seçici: oran+yakalama / maliyet | `14_figures/fig2` | ✅ |
| Fig 4 | yerleşim × açıklayıcı | `14_figures/fig5` | ✅ |
| Fig 5 | kalibrasyon reliability | `14_figures/fig6` | ✅ |
| (Fig 6) | Pareto, yer kalırsa | `14_figures/fig3` | ✅ |
| Tab I | model seçimi: CNN-30s, CNN-LSTM-30s, TranSensFusers-10s, **deployed CNN-10s**; sütunlar: window · rate · protocol · acc · F1 · params · inference ms | bitirme Tablo 5.4 + 25 Eyl ölçümü | ✅ transformer 1.86M / 103.7 ms; CNN 49.9K / 3.63 ms. CNN-LSTM parametresi yok → "—" |
| Tab II | yerleşim donanımı | JSON `server_hardware` | ✅ |
| Tab III | fog aşama dökümü × 5 | `04_fog` | ✅ |
| Tab IV | hiç/seçici/hep | `04_fog` + `09_secici_xai` | ✅ |
| Tab V | kapasite | `10_yuk` | ✅ |
| Tab VI | açıklayıcı kalitesi | `11_pareto` | ✅ |

**Fig 1 zorunlu.** Sistem bildirisinde mimari şekli olmadan hakem sistemi
anlayamaz. TikZ ile `main.tex` içinde çizilebilir; kutular: Sensor client
(replay) → TCP → Reader task (t_recv) → Queue → Worker (resample → causal
z-score → CNN → conf<τ? → explainer) → Reply. Üç yerleşim küçük etiketlerle.

`tools/paper_tables.py` yazılmalı: Tablo III–VI'yı JSON'dan LaTeX `tabular`
olarak üretsin, `\input{}` ile girsin. Elle sayı kopyalanmasın.

---

## BÖLÜM E — Başlık ve özet taslağı

### Başlık adayları

1. *Explanation Is the Bottleneck: Measuring and Gating XAI Cost in Fog-Based
   Real-Time Stress Detection*
2. *When to Explain, Not Where to Compute: Confidence-Gated Explanations for
   Real-Time Physiological Stress Detection*
3. *The Runtime Cost of Explaining Stress: A Fog Deployment Study with
   Selective LIME and SHAP*

Önerim 2 — bulguyu cümle olarak taşıyor.

### Abstract (İngilizce, ~220 kelime)

> Explainable AI is increasingly attached to physiological stress classifiers,
> yet the computational cost of producing an explanation is rarely measured and,
> to our knowledge, never inside a deployed real-time pipeline. We instrument an
> end-to-end stress detection system — a 49.9K-parameter 1D-CNN trained on WESAD
> under leave-one-subject-out validation — as a genuine TCP client–server
> deployment across co-located, fog (LAN) and cloud placements, and measure
> latency, queuing delay, execution time, response time, jitter, bandwidth and
> resource usage under periodic clock resynchronisation. On the fog node,
> inference costs 3.6 ms per 10-second window, while a single region-based LIME
> or SHAP (GradientExplainer) explanation costs 395–1,422 ms: explanation, not
> inference, dominates the pipeline by 118–421×, and it — not placement —
> determines response time (cloud is 8.7× slower than co-located without
> explanation, 1.13× with it). We propose confidence-gated selective explanation.
> At a threshold of 0.80, measured over 1,030 windows, mean response time falls
> from 450 ms to 80 ms (5.6×) while 90% of misclassified windows still receive an
> explanation, and single-worker capacity rises from 12 to 90 concurrent clients;
> a mixture-model projection agrees with the measurement to within 2%. Across
> four explainer operating points faithfulness is statistically
> indistinguishable, whereas LIME is 2.7× more stable to input noise and SHAP is
> 3× cheaper per explanation. Finally, the benefit of selective explanation is
> conditional on calibration: expected calibration error differs by 5× between
> the two held-out subjects, which we report as a limitation.

Bu özetin her sayısı `results/` altındaki bir dosyaya işaret ediyor.

---

## BÖLÜM F — Yazmadan önce kapatılacak dört boşluk

| # | boşluk | süre | durum |
|---|---|---|---|
| F1 | TranSensFusers parametre + çıkarım süresi | — | ✅ **kapandı** (37.3× / 28.6×) |
| F2 | Kanonik LOSO sayısı seçimi (B2) + transformer için "tek denek" ifadesi (B2-ek) | 10 dk | **sen karar ver** |
| F3 | Fig 1 mimari diyagramı | 1–2 saat | **evet** |
| F4 | Fog'u Docker'da (py3.11) tekrar koşup yığını eşitlemek — 5 durum × 5 dk | ~25 dk + Docker kurulumu | **artık evet** — 20 gün var, Z5 tamamen kapanır |
| F5 | `Transformer.ipynb` (3.7 MB) ve XAI not defterini depoya alıp almama kararı | 5 dk | senin kararın — tekrar üretilebilirlik için alınması iyi olur; `.dockerignore` zaten `*.ipynb`'yi imajdan dışlıyor |

---

## BÖLÜM G — Çizelge (15 Ekim, 20 gün)

| tarih | iş |
|---|---|
| 25–26 Eyl | F2, F5 kararları. F3 Fig 1. `main_en.tex` iskeleti, `tools/paper_tables.py`. |
| 27–28 Eyl | **F4:** fog düğümünde Docker (python:3.11) ile 5 XAI durumu yeniden; `04_fog` ve `07_edge_xai` tabloları aynı yığında. |
| 29 Eyl–1 Eki | V. Results — altı alt bölüm, tablolar JSON'dan. |
| 2–3 Eki | III. System + IV. Setup. Tablo I, II. |
| 4–5 Eki | I. Intro + II. Related + Abstract. |
| 6 Eki | VI, VII, VIII. Kaynakça. **Hocaya ilk tam taslak.** |
| 7–10 Eki | Hoca dönüşü, düzeltmeler, dil. |
| 11–12 Eki | Tam okuma, sayfa sınırı (6), şekil kalitesi, EDAS formatı. |
| 13 Eki | Hocaya son sürüm. |
| 14 Eki | Son düzeltme. |
| **15 Eki** | **EDAS'a yükle.** Son güne bırakma; EDAS sunucusu son saatlerde yavaşlar. |

**Kural:** F4 dışında yeni ölçüm yok. 1 Ekim'den sonra hiç yok.
Boş kalan gün varsa Pareto şekline ya da S14 seçici koşusuna (5 dk) gider,
yeni deney açılmaz.

---

## BÖLÜM H — Yazım notları (WRITING_PLAN.md'den, geçerli)

- Cümle cümle çevirme, yeniden yaz.
- Ondalık nokta. `babel` Türkçe kalkacak.
- Terim sözlüğü `WRITING_PLAN.md`'de; ek: co-located, explanation rate,
  confidence-gated, region-based LIME, SHAP (GradientExplainer).
- Her sayının yanında hangi koşudan geldiği (pencere sayısı) — hakem
  istatistiksel gücü görebilsin.
- "Our study shows…" değil, doğrudan bulgu.
