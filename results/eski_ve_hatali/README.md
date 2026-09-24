# eski_ve_hatali — bu dosyalardaki sayılar makaleye girmez

Her dosya için nedenini yazdım, çünkü "neden kullanmıyoruz" sorusu makale
yazarken ve hakem cevabında tekrar tekrar karşımıza çıkacak. Silmedim:
bir ölçümün neden geçersiz olduğunu gösteren kayıt da bir kanıttır.

---

## Saat kayması düzeltilmeden önce alınan fog koşusu

| Dosya | Sorun |
|---|---|
| `fog_fog_1user.json` | **latency medyanı −8.75 ms.** Paket gönderilmeden önce varamayacağı için bu değer fiziksel olarak imkânsız. |

**Neden böyle oldu:** saat farkı yalnızca bağlantı başında bir kez
ölçülüyordu. İki makinenin kristali aynı hızda tiklemediği için fark zamanla
kayıyor; 87 dakikalık bu koşuda kayma onlarca milisaniyeye ulaştı ve latency'yi
negatife düşürdü.

**Düzeltme:** saat farkı artık 60 saniyede bir yeniden ölçülüyor ve her
paketin latency'si iki ölçüm arasında doğrusal ara değerle düzeltiliyor
(`--clock-resync-s`).

**Kısmen geçerli:** aynı saatten ölçülen metrikler etkilenmedi — execution
3.62 ms, response 8.35 ms, bant genişliği 187 kbps, sınıflandırma
(doğruluk %95.2, stres F1 0.881). Bunlar alıntılanabilir ama koşu A1
sonrasında yine tekrarlanacağı için buraya kaldırıldı; karışıklık riski
kazançtan büyük.

---

## Sızıntılı normalizasyonla ölçülen XAI taramaları

| Dosya | Sorun |
|---|---|
| `xai_sweep_lime.csv` | Pencereler tüm kaydın istatistiğiyle normalize edildi |
| `xai_sweep_shap.csv` | Aynı + SHAP'ın seed'i sabitlenmemişti |
| `xai_lime_vs_shap_recommended.csv` | Yukarıdakilerden türetilmiş karşılaştırma |

**Birinci sorun — yanlış girdi uzayı.** Tarama pencereleri deneğin tüm
kaydının ortalama ve standart sapmasıyla normalize ediyordu; bu geleceğe
sızmak demek ve gerçek zamanlı sistemde imkânsız. Sunucu nedensel
normalizasyon yapıyor. İki uzay ölçüldü: bağıl L2 farkı 0.787 (S11) ve
0.773 (S2), standart sapma 1.00'e karşı 1.25. Yani XAI, dağıtımda
görmeyeceği bir girdi dağılımı üzerinde ölçülmüş oldu.

**İkinci sorun — SHAP kendi rastgeleliğini kararlılık metriğine karıştırdı.**
`GradientExplainer` her çağrıda rastgele örnekleme yapıyor ve seed sabit
değildi. Seed sabitlendikten sonra SHAP'ın sensitivity değeri 0.139'dan
0.075'e düştü; yani ölçülen "girdi duyarlılığı"nın neredeyse yarısı yöntemin
kendi gürültüsüymüş.

**Üçüncü sorun — ısınma yok.** Ölçüm döngüsü ilk pencereden başlıyordu.

**Güncel hali:** `results/06_xai_tarama/` (nedensel uzay, seed sabit, ısınma
var, ayrıca kendi-kararsızlık metriği eklendi).

**Yine de bir işe yarıyorlar:** iki uzayı yan yana koyunca sızıntılı ölçümün
XAI'yi olduğundan iyi gösterdiği görülüyor (LIME faithfulness 0.2476 → 0.2333,
sensitivity 0.0174 → 0.0264). Bu, makalede metodolojik bir not olarak
kullanılabilir — ama sonuç olarak değil, karşı örnek olarak.

---

## `prepared` yol dönemi streaming sonuçları

| Dosya | Sorun |
|---|---|
| `streaming_raw_val_running.json`, `streaming_summary_val_running.json` | Yeniden örnekleme ölçüm dışında |
| `streaming_raw_val_fixed.json`, `streaming_summary_val_fixed.json` | Aynı |
| `streaming_raw_deneme.json`, `streaming_summary_deneme.json` | Aynı, ayrıca deneme koşusu |
| `report_data.json` | Yukarıdaki koşulardan türetilmiş rapor verisi |

**Birinci sorun.** Bu koşularda sinyal, replay başlamadan önce 64 Hz'e
indirilmişti, yani yeniden örnekleme maliyeti ölçümün dışında kaldı. Gerçek
bir sistemde sensör ham hızında gönderir ve indirgeme sunucuda yapılır.
Yeniden örnekleme tek başına toplam hesabın ~%29'u, yani küçük bir fark değil.

**İkinci sorun.** Bu dönemde çoğunluk etiketi geçersiz olan pencereler
tamamen atlanıyordu; bu, normalizasyon istatistiğinin o pencerelerde hiç
güncellenmemesi demekti. Gerçek bir sunucu etiketi görmediği için böyle bir
atlama yapamaz. Düzeltildikten sonra doğruluk 0.7674'ten 0.8222'ye çıktı.

**Güncel hali:** `results/01_streaming/streaming_summary_native_fixed.json`.

**Not:** README.md'nin "Ölçülen sonuçlar (S11 + S14, tick=500ms, 1175
pencere)" tablosundaki sayılar bu eski koşulardan geliyor (preprocessing
0.189 ms, inference 1.15 ms, toplam 1.27 ms). Makaleye native yolun sayıları
girmeli.

---

## Duman testleri — ölçüm değil, kodun çalıştığını doğrulama

| Dosya | Ne |
|---|---|
| `fog_statstest.json` | Sunucu tarafı kaynak ölçümü eklenirken yapılan test; `--realtime` yok, 4.59 ms execution |
| `deployments_selftest.json` | `compare_deployments.py` yazılırken yapılan test; gerçek zamanlı değil, 300 tick |

Gerçek zamanlı olmadıkları için kuyruk gecikmesi ve jitter anlamsız; koşu
uzunlukları da rapor için yetersiz.
