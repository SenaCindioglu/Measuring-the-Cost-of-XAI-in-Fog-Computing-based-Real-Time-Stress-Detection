# Çalışma planı: gerçek zamanlı, seçici açıklamalı stres tespiti

Son güncelleme: 20 Eylül 2026.

Kapsam kararı: **tek model (TemporalCNN) ve iki XAI yöntemi (SHAP, LIME).**
Transensefusers, CNN-LSTM ve Integrated Gradients bu makalenin dışında.

Bu belge, hocanın verdiği rotayı (offline sonuçları dondur → replay emülasyonu →
aşama bazlı profil → XAI'yi hesaplama maliyeti olarak ölç → güven eşikli seçici
XAI → Pareto analizleri → makale) mevcut depo durumuna ve referans makaleye
(Erzurumluoğlu ve ark., ICACT 2025, DREAMER üzerinde fog mimarisinde duygu
tanıma) oturtur. Her görevin çıktısı, komutu ve "bitti" ölçütü var.

---

## 0. Takvim

| Hedef | Son gün | Bugünden | Durum |
|---|---|---|---|
| ICEET 2026 | 12 Eylül 2026 | geçti | kaçtı |
| **ICACT 2027** (birincil) | **1 Ekim 2026** | **11 gün** | hedef |
| ICAIIC 2027 (ikincil) | 1 Ekim 2026 (güvenli tarih) | 11 gün | ICACT ile aynı metin |
| **IISEC 2027** (yedek) | **29 Kasım 2026** | **10 hafta** | yedek, genişletilmiş hali |

Tek model ve iki XAI ile 1 Ekim seti tam plandır; Kasım'a yalnızca model
karmaşıklığı ablasyonu (Faz E) ve donanım ayakları (Pi, VPS) kalır.

**Karar noktası: 25 Eylül akşamı.** A, C ve D fazları bitmemişse ICACT
bırakılır, tüm güç IISEC'e döner. Yarım makale göndermektense tam makaleyi iki
ay sonra göndermek daha iyi.

---

## 1. Envanter

**Hazır (dondurulmuş, tekrar dokunulmaz):**

| Parça | Nerede | Not |
|---|---|---|
| TemporalCNN, depth 2, [32,128], k9, 49 890 parametre | `models/temporal_cnn.py`, `wesad_best_model_f32-128_k9.pt` | offline val F1 0.9386 (S11, S14); seed 42 |
| Replay emülasyonu, ham hızda tick tick | `streaming/`, `run_streaming_emulation.py` | window latency ile computation latency ayrı |
| Aşama profili: resampling / preprocessing / tensor_prep / inference / postprocessing | `profiling/latency_profiler.py` | mean, median, p95, p99, CPU, RSS |
| Tekrarlı, dönüşümlü ölçüm | `benchmark_runs.py` | koşu içi ve koşu arası dağılım ayrı |
| İstemci–sunucu, TCP, yedi metrik | `serving/`, `run_fog_experiment.py` | latency, queuing, execution, response, jitter, bandwidth, resource |
| Yerleşim karşılaştırması | `compare_deployments.py` | edge / fog / cloud, dönüşümlü, yayılımlı |
| Fog tek istemci koşusu (87 dk, Wi-Fi) | `results/fog_fog_1user.json` | **latency sütunu geçersiz** (saat kayması), diğerleri geçerli |
| Sızıntısız normalizasyon (`running`, `fixed`) | `data/normalization.py` | offline subject-bazlı z-score'un nedensel karşılığı |
| OFAT mimari taraması (depth, filtre, kernel) | mezuniyet raporu | Faz E için hazır tablo |

**Eksik (mezuniyet projesinden taşınacak ya da yazılacak):**

| Parça | Kaynak | İş |
|---|---|---|
| SHAP ve LIME açıklayıcılar | mezuniyet projesi | ham 640×14 pencere üzerinde, tek pencere için çalışacak arayüz; LIME için bölge tanımı |
| Açıklama kalitesi metrikleri (monotonicity, sensitivity, faithfulness) | mezuniyet projesi | taşıma |
| Güven kalibrasyonu ve eşik mekanizması | yazılacak | küçük |
| Seçici XAI'nin sunucuya entegrasyonu | yazılacak | orta |
| Grafik üretim script'i | yazılacak | küçük |

**İki ön görev:**

- **Ö1.** Mezuniyet projesindeki SHAP/LIME kodunu ve kalite metriklerini bul
  (Colab defteri, klasör ya da başka depo). `legacy/` altına olduğu gibi koy,
  sonra `xai/` paketine taşı. Özellikle LIME'ın ham sinyalde nasıl
  uygulandığına bak: hangi bölge tanımı, kaç örnek. Bulunamazsa sıfırdan
  yazılır; `shap` (GradientExplainer) ve `lime` ile bir buçuk günlük iş.
- **Ö2.** Terim: makale boyunca "real-time emulation / online replay
  experiment", "live system" ya da "clinical" değil. ICACT makalesi de
  DREAMER'ı kayıttan akıtmıştı; yöntemin savunulabilirliği buna dayanır.

---

## 2. Ölçüm kuralları (her fazda geçerli)

1. **Batch boyutu 1.** Gerçek kullanıcı tek pencere gönderir.
2. **Isınma hariç.** İlk 15 pencere istatistiğe girmez.
3. **Yüzlerce pencere.** S11 tam kaydı 587 pencere, S11+S14 1175. XAI
   deneylerinde tam kayıt; yalnızca yerleşim karşılaştırmasında `--max-ticks 600`.
4. **Mean, median, p95, p99 birlikte.** "En hızlı çalışma" raporlanmaz.
5. **Tekrarlı ve dönüşümlü.** Karşılaştırmalarda `--repeat 3`, sıra A, B, A, B.
   Fark, tekrarlar arası yayılımın altındaysa "fark yok" yazılır.
6. **Tek iş parçacığı.** Gerekçe kararlılık: yayılım %2.9'a karşı %11–15.
7. **Sızıntısız normalizasyon.** `running` varsayılan, `fixed` ikinci sütun.
8. **Kalibrasyon ve eşik seçimi test deneğinde yapılmaz.** İki tutulmuş denek
   var: S11'de kalibre et, S14'te ölç; sonra tersi. Ortalama + ayrı ayrı.
9. **Window latency ≠ computation latency.** 10 s pencere ve 5 s güncelleme
   aralığı her tabloda ayrı satır.
10. **Her sonuç JSON'a**, grafikler JSON'dan üretilir, elle sayı taşınmaz.
11. **XAI yapılandırması her sayının yanında.** "SHAP X ms" yetmez: açıklayıcı
    türü, arka plan boyutu; LIME için örnek (perturbation) sayısı, bölge
    tanımı ve çekirdek genişliği.

---

## 3. Fazlar

### Faz A: sistem ölçümünü tamamla

Referans makalenin karşılığı. Yedi metriğin tamamı, edge ve fog.

| # | Görev | Çıktı | Bitti ölçütü |
|---|---|---|---|
| A1 | **Saat kaymasını düzelt.** Saat farkı her 60 s'de yeniden ölçülür, anlar kaydedilir, latency en yakın ölçümle düzeltilir. | `serving/client.py`, `serving/protocol.py` | uzun koşuda latency medyanı pozitif ve (response − execution − queuing)/2 ile tutarlı |
| A2 | **Edge tabanı.** Edge ve fog, dönüşümlü, 3 tekrar. | `results/deployments_edge_vs_fog.json` | yayılımlı tablo |
| A3 | **Kısıtlı fog düğümü.** İkinci bilgisayar tek çekirdek (`--threads 1` + affinity). Normal ve kısıtlı iki ayrı `compare_deployments` koşusu. | `results/deployments_fog_tam.json`, `..._fog_kisitli.json` | execution ve CPU % farkı; "taklit" ibaresi |
| A4 | **Çok istemcili yük.** 1, 2, 4 istemci fog'da. | `results/fog_load_{1,2,4}.json` | queuing p95'in istemci sayısıyla büyümesi |
| A5 | **Wi-Fi ve Ethernet.** | `results/deployments_ag.json` | jitter ve response p95 farkı |
| A6 | **Bulut kararı.** VPS alınırsa `cloud=<ip>`; alınmazsa makalede tek cümle. | karar | 22 Eylül |
| A7 | README'deki iki hata: `start /affinity` PowerShell'de çalışmıyor; `fog_tam=... fog_kisitli=...` tek komutta olmaz. | `README.md` | düzeltildi |

```bash
# A2
python compare_deployments.py --subject S11 --realtime --max-ticks 600 --repeat 3 \
    --deployment edge=local fog=192.168.1.14 --tag edge_vs_fog

# A3 (ikinci bilgisayarda, PowerShell)
$p = Start-Process python -ArgumentList "-m serving.server --host 0.0.0.0 --threads 1" -PassThru
$p.ProcessorAffinity = 1

# A4
python run_fog_experiment.py --server-host 192.168.1.14 --subject S11 S14 --clients 4 --realtime --max-ticks 600 --tag fog_load_4
```

A1 yarım gün kod; A2–A5 toplam ~3 saat koşu, ikinci bilgisayar açıkken diğer
işlerle paralel.

### Faz B: modelin dondurulmuş profili

Model taşıma yok; TemporalCNN'in mevcut sayıları makaleye girer. Tek küçük iş:

| # | Görev | Çıktı |
|---|---|---|
| B1 | Mevcut profil sonuçlarını (`benchmark_thread_sweep`, `benchmark_pipeline_cmp`, `streaming_summary_native_fixed`) tek tabloya topla: aşama bazlı p50/p95, RSS, parametre sayısı, offline ve streaming doğruluk (`running`, `fixed`). | `results/model_profile.json` |

Bu, makaledeki "sistem hesabı ne kadar ucuz" satırının dayanağı: 640×14
pencere, ~3.6 ms execution, 193 MB RSS.

### Faz C: XAI'yi hesaplama maliyeti olarak ölç

Hocanın vurgusu: offline'da en iyi bulunan yöntem gerçek zamanda en iyi
olmayabilir; bu başarısızlık değil, muhtemel ana sonuç.

| # | Görev | Çıktı |
|---|---|---|
| C1 | `xai/` paketi: ortak arayüz `explain(model, window) -> attribution (640×14)`. `xai/shap_.py` (GradientExplainer; DeepExplainer BatchNorm ile sorun çıkarırsa not düşülür), `xai/lime_.py` (bölge bazlı, aşağıya bak) | `xai/` |
| C2 | Profiler'a `xai` aşaması: süre, CPU %, tepe RSS ayrı | `profiling/latency_profiler.py` |
| C3 | Kalite metrikleri: faithfulness (deletion/insertion), monotonicity, sensitivity | `xai/quality.py` |
| C4 | **Parametre taraması.** LIME örnek sayısı 100 / 250 / 500 / 1000 ve bölge sayısı (zaman dilimi 5 / 10 / 20 × 14 kanal). SHAP arka plan 10 / 50 / 100 (ve `nsamples` varsa 50 / 200). Her nokta: üç kalite metriği + süre p50/p95 + RSS, S11+S14 tam kayıt, 3 tekrar dönüşümlü. | `results/xai_sweep_lime.json`, `results/xai_sweep_shap.json` |
| C5 | Pareto: kalite'ye karşı açıklama süresi, iki yöntem iki eğri, parametre noktaları işaretli. Aranan nokta: kalite az düşer, süre çok düşer. | şekil 2 |

**Seçilen yapılandırmalar (21 Eylül 2026, GPU taraması, elle seçim).** Aşama
sonlarında kalite–gecikme tablosu basılıp karar elle verildi; otomatik seçim
yalnızca faithfulness'a baktığı için kullanılmadı.

| Yöntem | Yapılandırma | Gecikme (GPU) | Faithfulness | En iyi kaliteye göre |
|---|---|---|---|---|
| LIME | 500 örnek, 5 dilim, kw 0.25 | 55.5 ms | 0.2370 | −%7, 4.3× hızlı |
| SHAP | arka plan 100 (kmeans), nsamples 50 | 50.5 ms | 0.2287 | −%2.3, 7.7× hızlı |

İkisi de artık aynı maliyette (~50 ms), yani karşılaştırma eşit bütçede
yapılabiliyor. Eşit maliyette LIME daha sadık (0.2370'e karşı 0.2287) ve çok
daha kararlı (sensitivity 0.0135'e karşı ~0.08).

**Bunun Faz D'ye etkisi.** Açıklama 300 ms mertebesine inince (CPU tahmini),
5 saniyelik hop bütçesinin yalnızca %6'sını tutuyor. Yani tek kullanıcıda
seçici XAI'nin kazancı küçülür. Kazanç üç yerde ortaya çıkar: çok kullanıcıda
(4 × 300 ms = 1.2 s), zayıf donanımda (Pi sınıfı düğüm birkaç kat yavaş) ve
pahalı yapılandırmada. Bu yüzden D2 eşik taraması **iki maliyet noktasında**
koşulmalı: seçilen ucuz yapılandırma ve kalite-maksimum yapılandırma. Tablo o
zaman "seçici açıklama ne zaman gerekir" sorusunu da cevaplar.

**LIME'ın ham sinyalde bölge tanımı.** Girdi 640×14 = 8 960 değişken; LIME'ı
her örneği ayrı değişken sayarak koşmak hem çok pahalı hem anlamsız (tek bir
örneğin katkısı yorumlanamaz). Görüntüdeki "süper piksel" karşılığı burada
**zaman dilimi × kanal** bölgeleridir: örneğin 10 dilim × 14 kanal = 140 bölge.
Bir bölge "kapatıldığında" o kanalın o dilimi arka plan değeriyle (sıfır ya da
o kanalın ortalaması) değiştirilir. Bölge sayısı bir tasarım değişkenidir ve
C4'te taranır; az bölge hızlı ama kaba, çok bölge yavaş ama ince. LIME'ın
maliyeti örnek sayısı × model çağrısı olduğu için (1000 örnek = 1000 ileri
geçiş, pencere başına ~1 s mertebesi) bu, "açıklama 2 s altı" gereksinimini
doğrudan test eden yöntem.

`requirements.txt`'e `shap`, `lime`, `scikit-learn` eklenir; sunucu için
opsiyonel tutulur.

Mezuniyet sonuçları (dondurulmuş): SHAP monotonicity ve sensitivity'de en iyi,
en dengeli; LIME seyrek ama daha az kararlı. Yeni tablo bu satırlara süre ve
bellek ekler. Beklenen gerilim: SHAP kaliteli ama gradyan gerektirir (modelin
içine erişim), LIME model-agnostik ama örnek sayısıyla maliyeti doğrusal
büyür. "Hangisi gerçek zamanlı kullanıma daha uygun?" sorusunun cevabı bu
tablodan çıkar ve mezuniyetteki "en iyi"den farklı olabilir.

### Faz D: güven eşikli seçici XAI (ana yenilik)

```
akış → ön işleme → pencere → CNN → güven (softmax max)
    güven ≥ eşik  → sadece tahmin
    güven <  eşik → tahmin + açıklama (Faz C'de seçilen yöntem ve yapılandırma)
```

| # | Görev | Çıktı |
|---|---|---|
| D1 | **Kalibrasyon.** Reliability diyagramı, ECE. Gerekirse temperature scaling (tek parametre, kalibrasyon deneğinde öğrenilir, test deneğinde uygulanır). | `results/calibration_{S11,S14}.json`, şekil |
| D2 | **Eşik taraması** 0.60 / 0.70 / 0.80 / 0.90, çapraz denek. Her eşik: açıklanan pencere %, yakalanan hata % (yanlış sınıflanan pencerelerin kaçında açıklama tetiklendi), pencere başına ortalama hesap (ms), p95 gecikme, CPU/RSS, dakikada açıklama sayısı. | `results/selective_xai_sweep.json` |
| D3 | **Sunucuya entegrasyon.** Yanıt zaten `confidence` taşıyor; `--xai-method`, `--xai-threshold` bayrakları; eşik altında açıklama üretilir, `stages` içine `xai` yazılır. Eşik 0 = hiç, 1.0 = her pencere. | `serving/server.py` |
| D4 | **Fog'da uçtan uca.** Üç durum: hep açıkla (1.0), seçilen eşik, hiç (0.0). Response time, execution ve sunucu CPU'su. | `results/fog_selective_{always,thr,never}.json` |
| D5 | Kontrol: sınıflandırma doğruluğu eşikten bağımsız aynı (olmalı). | tabloda tek satır |

D4, A ile D'yi birleştiren deney ve makalenin başlığına girecek sonuç:
kaynağı kısıtlı bir fog düğümünde seçici açıklama, her pencerede açıklamaya göre
hesabı ne kadar azaltıyor, açıklamalar hataların ne kadarını yakalıyor.

Beklenen tablo:

| eşik | açıklanan % | yakalanan hata % | ort. hesap / pencere | p95 response | CPU % |
|---|---|---|---|---|---|
| 1.00 (hep) | 100 | 100 | | | |
| 0.90 | | | | | |
| 0.80 | | | | | |
| 0.70 | | | | | |
| 0.60 | | | | | |
| 0.00 (hiç) | 0 | 0 | | | |

### Faz E: model karmaşıklığı ablasyonu (29 Kasım seti)

Tek model kararı ablasyonu ortadan kaldırmıyor; hocanın 4. sorusu "aynı
mimarinin küçük/büyük varyantları" ile cevaplanır ve OFAT taraması zaten var.

| # | Görev |
|---|---|
| E1 | OFAT tablosunu makale formatına al: depth 1/2/3, filtre [16,64] / [32,128] / [32,64,128], kernel 5/9/13 için F1 ve stres recall. |
| E2 | Bu varyantlardan 4–5'ini aynı seed ve koşullarla eğit (`training/train_final_model.py`, küçük model, CPU'da saatler). |
| E3 | `benchmark_runs.py --model` listesiyle dönüşümlü profil: inference p50/p95, RSS, parametre. |
| E4 | Pareto: F1'e karşı gecikme. ICACT makalesindeki "104 öznitelik %84 / 78 ms yerine 24 öznitelik %80 / 45 ms" mantığıyla çalışma noktası seç. |

1 Ekim için yalnızca E1 (mevcut sayılar) tabloya girer; E2–E4 Kasım.

### Faz F: şekiller ve makale

Dört şekil:

1. **F1'e karşı inference latency**: TemporalCNN varyantları (1 Ekim'de OFAT'tan F1, gecikme yalnızca seçili model; Kasım'da tam).
2. **XAI kalitesine karşı açıklama süresi**: SHAP ve LIME eğrileri, parametre noktaları (Faz C).
3. **Güven eşiğine karşı açıklanan pencere % ve ortalama sistem gecikmesi**: iki y ekseni (Faz D).
4. **Yığılmış gecikme çubuğu**: resampling + preprocessing + inference + XAI; edge ve fog; "hep" ve "seçici" sütunları.

Ek: referans makaleyle metrik metrik tablo (README'deki, gerçek fog sütunuyla).

**6 sayfalık yapı:**

| Bölüm | Sayfa | İçerik |
|---|---|---|
| I. Giriş | ½ | gerçek zamanlı stres tespiti, XAI'nin hesap maliyeti, dört katkı |
| II. İlgili çalışmalar | ½ | ICACT 2025 fog makalesi, WESAD üzerinde gerçek zamanlı çalışmalar, XAI maliyeti üzerine az sayıda iş. Geniş literatür ve GUI çıkar. |
| III. Sistem | 1 | replay emülasyonu, aşamalar, istemci–sunucu, yedi metrik, seçici XAI (akış şeması) |
| IV. Deney düzeneği | ¾ | donanım, ağ, yerleşimler, ölçüm kuralları, sızıntısız normalizasyon, çapraz kalibrasyon, XAI yapılandırmaları |
| V. Sonuçlar | 2½ | Faz A tablosu, dört şekil, seçici XAI tablosu, referans karşılaştırması |
| VI. Tartışma ve sınırlılıklar | ½ | emülasyon (canlı değil), taklit fog düğümü, iki tutulmuş denek, cloud yok, saat belirsizliği, tek model |
| VII. Sonuç | ¼ | |

**Dört katkı:**

1. Denek bağımsız WESAD stres CNN'i, sızıntısız çevrimiçi normalizasyonla replay altında değerlendirilmiş.
2. Aşama bazlı gerçek zamanlı profil; edge ve fog yerleşiminde yedi metrik.
3. SHAP ve LIME'ın açıklama kalitesine karşı hesaplama maliyeti, parametre taramasıyla.
4. Güven eşikli seçici açıklama: hesap düşer, açıklamalar belirsizliğin olduğu yerde kalır.

Mezuniyet raporundaki "model yanıtı 100 ms altı, açıklama 2 s altı" hedefleri
**test edilmiş gereksinim** olarak sunulur: hangi yapılandırmalar altında, hangileri değil.

---

## 4. Gün gün (20 Eylül – 1 Ekim)

| Gün | Yapılacak | Paralelde koşan |
|---|---|---|
| 20 Eyl (Cmt) | Ö1: XAI kodunu bul. A1 kod. | — |
| 21 Eyl (Paz) | A1 kısa test. A7. C1: SHAP + LIME arayüzü, bölge tanımı. | A2 (~30 dk) |
| 22 Eyl (Pzt) | C2, C3. A6 bulut kararı. | A3, A5 |
| 23 Eyl (Sal) | C4 tarama başlat (uzun, arka planda). D1 kalibrasyon. | A4, C4 |
| 24 Eyl (Çar) | D2 eşik taraması. B1 profil tablosu. | C4 devam |
| 25 Eyl (Per) | D3 sunucu entegrasyonu. **Karar noktası:** A, C, D tamam mı? | — |
| 26 Eyl (Cum) | D4 fog koşuları. `tools/make_figures.py`, dört şekil. | — |
| 27 Eyl (Cmt) | Makale III, IV, V. Tablolar JSON'dan. | eksik koşu tekrarı |
| 28 Eyl (Paz) | Makale I, II, VI, VII. Şekiller son hali. | — |
| 29 Eyl (Pzt) | Hocaya gönder. | — |
| 30 Eyl (Sal) | Geri bildirim, düzeltme. | — |
| 1 Ekim (Çar) | Gönderim. | — |

## 5. 29 Kasım seti (2 Ekim – 29 Kasım)

| Hafta | İş |
|---|---|
| 1–2 | E2 varyant eğitimleri; C4'ün kalan noktaları (SHAP nsamples, LIME çekirdek genişliği ve arka plan seçimi) |
| 3–4 | E3, E4; şekil 1 tam hali |
| 5–6 | varsa Raspberry Pi ya da VPS koşuları; daha fazla tutulmuş denek için LOSO fold'larından kalibrasyon tekrarı |
| 7–8 | makale genişletme, ICACT hakem yorumları varsa işlenir |
| 27–29 Kasım | gönderim |

---

## 6. Riskler ve yedekler

| Risk | Belirti | Yedek |
|---|---|---|
| XAI kodu bulunamıyor | Ö1 boş | shap + lime ile sıfırdan, bir buçuk gün |
| LIME pencere başına saniyeler sürüyor | 1000 örnek × ileri geçiş | bulgu olarak yazılır; seçici XAI'nin gerekçesi güçlenir; örnek ve bölge sayısı taramada zaten var |
| SHAP GradientExplainer BatchNorm ile uyumsuz | hata ya da tutarsız atıf | model eval modunda, `shap.GradientExplainer` yerine `KernelExplainer` bölge bazlı; hangisi kullanıldığı yazılır |
| Saat kayması düzelmiyor | latency hâlâ negatif | (response − execution − queuing)/2 raporlanır, yöntem yazılır |
| İkinci bilgisayar güçlü | CPU %0.04 | A3 kısıtlı mod, "taklit" ibaresiyle |
| Model kötü kalibre | ECE yüksek, eşikler anlamsız | temperature scaling; yine kötüyse eşik yerine entropi denenir, ikisi de yazılır |
| İki tutulmuş denek az | eşik sonuçları denekler arası oynuyor | çapraz kalibrasyon + ayrı ayrı; sınırlılıklarda; Kasım'da LOSO fold'ları |
| Zaman | 25 Eylül'de A, C veya D eksik | ICACT bırakılır, IISEC |

---

## 7. Referans makaleyle hizalama

Erzurumluoğlu ve ark., ICACT 2025.

| Onlarda | Bizde | Makalede nasıl söylenir |
|---|---|---|
| DREAMER, duygu | WESAD, stres | farklı görev, aynı sistem metodolojisi |
| 3 s pencere, 24 öznitelik, saniyede bir | 10 s pencere, ham sinyal, 5 s'de bir | pencere gecikmesi bizde uzun, hesap ucuz; ayrı raporlanır |
| Worker + broker Pi 4, Wi-Fi | tek fog düğümü, Wi-Fi ve Ethernet | broker tek worker'da gereksiz (onların sonucu: 43.9'a karşı 43.3 ms) |
| Edge / fog / cloud | edge / fog (+ A6) | eksikse açıkça |
| Yedi metrik | aynı yedi metrik, aynı tanımlar | yan yana tablo |
| XAI yok | SHAP ve LIME maliyeti + seçici XAI | bizim eklediğimiz boyut |
| Bant genişliği 374 kbps | 187 kbps | ham sinyal gönderdiğimiz halde yarısı |
| Doğruluk–gecikme çalışma noktası (104 → 24 öznitelik) | XAI parametre taraması (C) ve CNN ablasyonu (E) | aynı felsefe |
