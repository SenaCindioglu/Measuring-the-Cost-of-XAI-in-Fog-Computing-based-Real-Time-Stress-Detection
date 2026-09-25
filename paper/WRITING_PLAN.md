# İngilizce makale yol planı

Son güncelleme: 25 Eylül 2026. Hedef: ICACT, 1 Ekim.

---

## Teşhis: elindeki şey bir rapor, bildiri değil

`main.tex` şu an 1404 satır, 9 bölüm, 25 alt bölüm. Karşılaştırma: referans
makale (Erzurumluoğlu ve ark.) 9 sayfa ve **7 bölüm, 12 alt bölüm**.

İki sorun var:

1. **Fazla ayrıntı.** Bir raporda her ölçüm anlatılır. Bir bildiride yalnızca
   iddiayı taşıyan ölçümler anlatılır, gerisi tek cümleye iner ya da çıkar.
2. **En güçlü bulgular taslakta yok.** Taslak 21 Eylül'den önce yazıldı;
   seçici XAI (D1+D2+D4), kalibrasyon, çok istemcili yük ve bulut koşuları
   sonradan geldi. Bunlar makalenin *asıl katkısı* ve şu an eksikler.

Yani yapılacak iş "çeviri" değil, **yeniden kurgu + çeviri**.

---

## Temel kural: Sonuçlar bölümü İDDİAYA göre bölünür, deneye göre değil

Elinde 14 numaralı sonuç klasörü var. Bir bildiride 5 sonuç alt bölümü olur.
Klasör başına bir alt bölüm yazarsan rapor olur, bildiri olmaz.

Her alt bölüm **tek bir cümleyi kanıtlar**. Aşağıdaki eşleme bunu yapıyor.

| # | İddia (alt bölüm başlığı olur) | Kanıt | Şekil/Tablo |
|---|---|---|---|
| 1 | Explanation, not inference, dominates the real-time pipeline | `04_fog`, `12_profil` | fig1 + Tablo I |
| 2 | Selective explanation recovers most of that cost | `08_kalibrasyon`, `09_secici_xai` | fig2 + Tablo II |
| 3 | The benefit depends on confidence calibration | `08_kalibrasyon` | fig6 |
| 4 | Under explanation, placement stops mattering | `05_yerlesim`, `07_edge_xai`, `13_bulut` | fig5 |
| 5 | Explanation sets the multi-user capacity limit | `10_yuk` | fig4 |
| 6 | Equal faithfulness, different stability (LIME vs SHAP) | `11_pareto`, `06_xai_tarama` | fig3 |

Altısı da ölçülmüş, hepsinin sayısı elinde. Bu yeter; daha fazlası bildiriyi
şişirir.

---

## Hedef yapı

```
Abstract                                    ~200 kelime
I.   Introduction                           ~0.75 sayfa
II.  Related Work                           ~0.75 sayfa
III. System Design                          ~1.5 sayfa
IV.  Experimental Setup                     ~0.75 sayfa
V.   Results                                ~2.5 sayfa   <- ağırlık burada
VI.  Discussion                             ~0.5 sayfa
VII. Limitations and Future Work            ~0.5 sayfa
VIII. Conclusion                            ~0.25 sayfa
                                            ------------
                                            ~7.5 sayfa + kaynaklar
```

### I. Introduction

Dört paragraf, şu sırayla:

1. Giyilebilir stres tespiti olgunlaştı, doğruluk doyuma ulaştı.
2. XAI yaygınlaştı — **ama hesaplama maliyeti ölçülmüyor.** Nauta ve ark.
   Co-12 alıntısı burada kalsın, güçlü bir dayanak.
3. Gerçek zamanlı bir sistemde bu boşluk neden önemli: açıklama pencere
   bütçesini yiyor.
4. Katkı listesi — **dört madde, daha fazlası değil:**
   - uçtan uca açıklama maliyetinin yedi sistem metriğiyle birlikte ölçümü
   - güven eşikli seçici açıklama ve ölçülmüş kazancı
   - açıklama yükü altında yerleşim kararının anlamsızlaştığının gösterilmesi
   - kalibrasyonun seçici mekanizma üzerindeki belirleyici rolü

Mevcut taslağın girişi iyi; kısaltıp katkı listesini güncellemek yeterli.

### II. Related Work

Mevcut dört alt bölümü **ikiye indir**:
- *Fog-based physiological monitoring* (referans makale buraya)
- *The cost of explanation* (Co-12, Quantus, ShaTS sınırı, gömülü XAI)

"Seçici ve koşullu açıklama" ve "Boşluk" alt bölümlerini bu ikisinin içine
eritip ayrı başlık verme.

### III. System Design

Mevcut sekiz alt bölümü **dörde indir**:

| tut | birleştir / kes |
|---|---|
| Model and windowing | "Önceki aşama" bölümünü buraya tek paragraf olarak göm |
| Leak-free online normalisation | — |
| Client–server separation and clock layer | "İki ardışık düzen yolu"nu tek paragrafa indir |
| Explainers and selective triggering | kalite ölçütlerini buraya al |

**Saat katmanı bölümünü kesme.** `perf_counter` gerekçesi ve periyodik
yeniden senkronizasyon, ölçümün geçerliliğinin dayanağı; hakem bunu sorar.

### IV. Experimental Setup

Donanım tablosu (üç yerleşim), ağ, ölçüm kuralları (ısınma, dönüşümlü tekrar,
tek iplik gerekçesi), koşu envanteri — hepsi **tek sayfanın altında**.

`02_benchmark` (iplik taraması, ardışık düzen karşılaştırması) buraya
**metodoloji gerekçesi** olarak girer, sonuç olarak değil:
> "Thread count was fixed at one: across four interleaved repetitions the
> difference between 1, 2 and 4 threads (15.8%) stayed below the
> run-to-run spread, while single-threaded measurement varied five times
> less (2.9% against 11–15%)."

### V. Results — yukarıdaki altı iddia, o sırayla

Her alt bölüm üç parçadan oluşsun: **ne ölçüldü → sayı → ne anlama geliyor.**
Yorum paragraf sonunda, tek cümle.

### VI. Discussion

Üç nokta, her biri bir paragraf:
1. Açıklama maliyeti mimari kararı değiştiriyor (yerleşim → tetikleme politikası)
2. Referans makaleyle karşılaştırma — **yalnızca bant genişliği** doğrudan
   karşılaştırılabilir; zaman metrikleri donanım ve yazılım yığını farkı içeriyor
3. Tek worker kapasitesi mimari bir seçim, modelin sınırı değil

### VII. Limitations and Future Work

Dürüstlük burada kazandırır. Beşi de yazılmalı:
- İki tutulmuş denek — "denek bağımlı" sonucu n=2 üzerine kurulu
- Sıcaklık ölçekleme çapraz denekte başarısız oldu
- Fog düğümü kaynak kısıtlı değil (Pi 4 ~10 kat yavaş olurdu)
- Yerleşim sütunları farklı yazılım yığınlarında ölçüldü (PyTorch 2.8/2.14)
- Kuyruk ölçümü üst sınır: istemciler eşzamanlı başlatıldı

---

## Kesilecekler (rapordan bildiriye)

| şu an | ne yapılacak |
|---|---|
| Mikrodenetleyici fizibilitesi (§ 1113) | Future Work'te **iki cümle**. Ayrı alt bölüm olmasın. |
| Yeniden örneklemenin yeri / bant genişliği (§ 698) | Discussion'da bant genişliği karşılaştırmasının içine eritilsin |
| Ölçümün tekrarlanabilirliği (§ 662) | Setup'a taşınsın, sonuç değil yöntem |
| Sınıflandırma başarımının yoldan bağımsızlığı (§ 1074) | Results'ta **bir cümle** + tek satır tablo |
| Ortak yerleşimde yedi metrik (§ 735) | İddia 4'ün içine eritilsin, ayrı alt bölüm olmasın |
| "Önceki Aşama" bölümü (§ 148) | System Design'a tek paragraf |

Bu kesintiler ~1.5 sayfa kazandırır ve o sayfa yeni sonuçlara gider.

---

## Çeviri notları

**Cümle cümle çevirme, yeniden yaz.** Türkçe akademik cümle yapısı uzun ve
çok yan cümleli; birebir çeviri İngilizcede ağır okunur. Her paragrafın ne
dediğini bul, İngilizce olarak baştan kur.

### Terim sözlüğü — tutarlı kullan

| Türkçe | İngilizce |
|---|---|
| açıklama (XAI çıktısı) | explanation |
| açıklayıcı | explainer |
| sadakat | faithfulness |
| kararlılık / duyarlılık | stability / sensitivity |
| sızıntısız | leak-free |
| nedensel normalizasyon | causal normalisation |
| pencere / kayma | window / hop |
| ısınma | warm-up |
| koşular arası yayılım | run-to-run spread |
| eşik | threshold |
| tetiklenme oranı | trigger rate (veya explanation rate) |
| yerleşim | placement |
| kuyruk gecikmesi | queuing delay |
| ucuz çalışma noktası | cheap operating point |
| kaynak kısıtlı | resource-constrained |

**Kod kimliklerinde artık `cheap` kullanılıyor** (`shap_cheap`, `lime_cheap`);
şekillerde "SHAP (cheap)" yazıyor. Metinde de böyle geçsin.

### Sık yapılan hatalar

- "Our study shows that..." yerine doğrudan bulguyu yaz: "Explanation
  accounted for 98.7% of server-side computation."
- Türkçe pasif yapıyı İngilizceye taşıma; İngilizcede etken cümle daha net.
- Sayıları ondalık **noktayla** yaz (`10.6 ms`), virgülle değil. Mevcut
  taslakta `10{,}6` biçimi var — hepsi değişecek.
- `babel` Türkçe ayarı kalkacak, `\kapatKisayol` ve Türkçe adlandırma
  satırları silinecek.

---

## Altı günlük plan

| gün | iş |
|---|---|
| **25 Eyl** | İskeleti kur: yeni `main_en.tex`, bölüm başlıkları, şekil yerleri. Sonuçlar bölümünün altı alt başlığını **boş** aç. |
| **26 Eyl** | Results V — altı alt bölümü sayılarla doldur. Tablolar JSON'dan, elle yazma. En zor gün, en değerli gün. |
| **27 Eyl** | System Design III + Experimental Setup IV. Mevcut Türkçe metinden yeniden yaz. |
| **28 Eyl** | Introduction I + Related Work II + Abstract. Sonuçlar yazıldıktan sonra yazmak daha kolay. |
| **29 Eyl** | Discussion VI + Limitations VII + Conclusion VIII. Kaynakça kontrolü. |
| **30 Eyl** | Baştan sona okuma, şekil/tablo numaraları, sayfa sınırı, dil düzeltmesi. |
| **1 Eki** | Gönder. |

**Kural: 26 Eylül'den sonra yeni ölçüm yapma.** Elindeki altı bulgu bir
bildiriye fazlasıyla yeter; yeni koşu eklemek yazma zamanını yer.

---

## Rakamları elle kopyalama

Tablo değerleri `results/` altındaki JSON'lardan gelmeli. Elle yazılan her
sayı, sonradan bir koşu tekrarlandığında sessizce yanlış kalır.

`tools/profile_table.py` bunun örneği. Makale tabloları için benzer bir
`tools/paper_tables.py` yazılabilir — LaTeX `tabular` gövdesini doğrudan
üretir, `main_en.tex` içine `\input{}` ile girer.

---

## Şekiller hazır

`results/14_figures/` — altısı da İngilizce etiketli, teslimde çeviri
gerektirmiyor.

| dosya | bölüm |
|---|---|
| `fig1_cost_breakdown.png` | Results, iddia 1 |
| `fig2_selective.png` | Results, iddia 2 |
| `fig6_calibration.png` | Results, iddia 3 |
| `fig5_placement.png` | Results, iddia 4 |
| `fig4_capacity.png` | Results, iddia 5 |
| `fig3_pareto.png` | Results, iddia 6 |

Not: mevcut `main.tex` şekilleri TikZ/pgfplots ile çiziyor. PNG'ye geçmek
derlemeyi hızlandırır ama Overleaf'te dosya yüklemek gerekir. İkisi de olur;
karar senin.
