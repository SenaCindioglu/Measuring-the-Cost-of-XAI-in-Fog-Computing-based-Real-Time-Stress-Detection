# Literatür taraması v2 — 12 Ekim 2026 (birincil kaynaklardan doğrulandı)

Her satır makalenin kendisinden (tam metin/özet) çekildi; doğrulama düzeyi
belirtildi. Eski `literature.md` notlarıyla çelişen bulgular en altta.

| # | Makale | Model | Veri seti | XAI | Süre ölçümü | Donanım |
|---|---|---|---|---|---|---|
| 1 | Erzurumluoğlu 2025, ICACT (tam metin) | Random Forest, 24 öznitelik | DREAMER (EEG+ECG) | yok | 7 iletim metriği; exec 34.7 ms worker, yanıt 43.3/73.3 ms (fog/bulut); XAI yok | Pi 4 8GB 1.8GHz worker+broker, DO bulut, 802.11ac |
| 2 | ShaTS — Franco de la Peña 2025, arXiv:2506.01450 (tam metin) | Stacked BiLSTM, 10×69 pencere | SWaT (endüstriyel IoT) | ShaTS (Shapley, zaman gruplamalı) | Evet: pencere başına; **"1 s/pencereden uzun xAI gerçek zamana uygun değildir"**; KernelSHAP sınırı aşıyor, ShaTS altında | i7-14700K + RTX 4080 |
| 3 | Taguta 2026, arXiv:2609.14036 (tam metin) | LSTM-GRU sıcaklık tahmini | Saha verisi (soğuk zincir, LoRaWAN) | Koşullu SHAP (yalnız kırılım tahmininde) | Evet: SHAP **+195 ms**, toplam 555 ms, +%28 CPU; taban 283 ms | Pi 4, gerçek dağıtım, buluttan bağımsız |
| 4 | Warnecke 2020, EuroS&P (tam metin) | MLP/CNN/RNN (Drebin+, Mimicus+, DAMD, VulDeePecker) | Güvenlik veri setleri | 6 yöntem: Grad, IG, LRP, LIME, SHAP, LEMNA | Evet: örnek başına LIME 0.031–0.74 s, KernelSHAP ≥9.1 s (CPU); LRP 8.4 ms | Xeon E5 v3 2.6 GHz |
| 5 | FastSHAP — Jethani 2022, ICLR (özet + önceki doğrulama) | Ayrı açıklayıcı ağ (amortize) | Tablosal + görüntü | Shapley (öğrenilmiş) | Evet: "orders of magnitude" hızlanma; tek ileri geçiş; bedeli uzun eğitim | V100 (önceki nottan) |
| 6 | Paes, Wei, Calmon 2024, NeurIPS 37 (proceedings) | Amortize açıklayıcılar üzerine | Çeşitli | Seçici açıklama (örneklerin bir kısmına pahalı açıklama) | Çıkarım sayısı üzerinden maliyet; ms ölçümü odak değil | — |
| 7 | HealthFog — Tuli 2020, FGCS (özet) | Topluluk DL, kalp hastalığı | Kalp hastası verisi | yok | Evet: güç, bant, gecikme, jitter, doğruluk, execution | FogBus; edge/fog/bulut |
| 8 | Almadhor 2025, Front. Bioeng. (tam metin) | 1D-CNN (%98, 3 sınıf) + temporal conformer | **WESAD** + SCIENTISST-MOVE + DREAMER | Var (açıklama tekniği uygulanmış) | **Hayır: hiçbir çıkarım/XAI süresi yok** | belirtilmemiş |
| 9 | El Rashidy 2025, Front. Comput. Sci. (tam metin) | PCA_CNN + YOLOv8+SA | Yüz görüntüleri (48×48, 35,888) | LIME + Grad-CAM | Kısmen: 81 FPS uç cihaz; **XAI süresi yok** | uç cihaz (model belirsiz), fog mimarisi |
| 10 | Ghose 2025, Front. Digit. Health (tam metin) | kNN, DT, LR, GNB (HRV öznitelikleri) | SWELL-KW | LIME | **Hayır: yalnız eğitim süreleri (ör. kNN 31 s); çıkarım/LIME süresi yok** | Jetson Orin Nano 8GB |
| 11 | Das 2021/23, ACM TiiS (özet) | LSTM, akıllı ev aktivite tanıma | Akıllı ev sensörleri | SHAP vb. + doğal dil | Kullanıcı çalışması odaklı; sürelere özet erişilemedi | — |
| 12 | Gia 2015, IEEE CIT (ICACT makalesi içi doğrulama + UTU kaydı) | Dalgacık tabanlı ECG öznitelik çıkarımı | ECG | yok | Evet: öznitelik çıkarımı 96.3 ms; bant verimi >%90 | Akıllı ağ geçidi (fog) |
| 13 | Nauta 2023, ACM CSUR Co-12 (önceki birincil alıntı) | — (değerlendirme çerçevesi) | — | 12 kalite boyutu | **Çalışma süresini kapsam dışı bırakıyor** (açık cümleyle) | — |

## Eski notlarla çelişen / düzeltilen bulgular

- **Basaran & Dressler arXiv:2511.17514:** Özet "xAI-Native hibrit model +
  matematiksel ödünleşim çerçevesi" anlatıyor; eski nottaki "taban 5.3 ms,
  SHAP +15 ms O-RAN test yatağı" özetten doğrulanamadı. Makalede atıf YOK ✓.
- **Das 2021:** "giyilebilir, 33 sensör LSTM" değil; akıllı ev aktivite
  tanıma. Makalede atıf YOK ✓.
- **Prabhu & Sunitha 2026** ve **EvoApneaFormer**: aramayla bulunamadı.
  Makalede atıf YOK ✓.
- PMC11359526 (wearable stres XAI): CAPTCHA engeli, erişilemedi.

## Makalenin boşluk iddiasına etkisi (teyit)

Üç küme ayrımı taramayla tutuyor: (i) sistem metriği var–XAI yok [1,7,12];
(ii) XAI süresi var–dağıtım/ağ yok [2,4,5]; (iii) fizyolojik/uygulamada XAI
var–süre yok [8,9,10]. Tek gerçek dağıtımda koşullu SHAP ölçen çalışma [3]
(soğuk zincir, fizyolojik değil). Bizim makale (i)+(ii)+(iii) kesişimi +
seçici açıklamanın uçtan uca ölçümü.
