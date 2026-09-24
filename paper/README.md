# paper/ — IEEE biçiminde iç teknik rapor

`main.tex`, bitirme projesinden sonra yapılan gerçek zamanlı / sis mimarisi /
XAI maliyeti çalışmasının tamamını IEEE konferans biçiminde, Türkçe olarak
raporlar. Bitirme projesi Bölüm II'de kısaca özetlenir; geri kalan her şey bu
depodaki ölçümlerden gelir.

## Derleme

```bash
pdflatex main
pdflatex main        # referanslar ve şekil numaraları için ikinci geçiş
```

Overleaf'e yüklemek için `main.tex` tek başına yeterlidir; harici şekil dosyası
yoktur (üç şekil de TikZ/pgfplots ile çizilir).

Kullanılan paketler: `IEEEtran`, `babel` (turkish), `booktabs`, `pgfplots`,
`tikz`. Hepsi TeX Live ve Overleaf'te standarttır. MiKTeX ilk derlemede
eksik paketleri indirmek isteyebilir.

**Türkçe babel notu.** `babel`'in Türkçe dili `=`, `:` ve `!` karakterlerini
etkin hale getirir; bu matematik kipini ve tablo kurulumunu bozar. Bu yüzden
`\begin{document}` hemen sonrasında `\kapatKisayol` (yani `\shorthandoff{=:!}`)
çağrılır. Tireleme ve "Şekil / TABLO / Kaynaklar" adlandırması korunur.

## Sayılar nereden geliyor

Rapordaki her tablo, `results/` altındaki JSON/CSV dosyalarından okunmuştur.
Elle taşınan sayı yoktur.

| Rapor bölümü | Kaynak dosya |
|---|---|
| Tablo II, III (bitirme özeti) | `main (7).pdf` (bitirme raporu) |
| Tablo VI — aşama profili | `results/01_streaming/streaming_summary_native_fixed.json` |
| Tablo VII — iplik taraması | `results/02_benchmark/benchmark_thread_sweep.json` |
| Tablo VIII — ardışık düzen | `results/02_benchmark/benchmark_pipeline_cmp.json` |
| Tablo IX — geri döngü yedi metrik | `results/03_loopback/fog_realtime_1user.json` |
| Tablo X — uç/sis karşılaştırması | `results/05_yerlesim/deployments_edge_vs_fog.json` |
| Tablo XI, XII + Şekil 2 — XAI taraması | `results/06_xai_tarama/*.csv` |
| Tablo XIV + Şekil 3 — **ana tablo** | `results/04_fog/fog_xai_*.json` |
| Tablo XV — sınıflandırma | `01_streaming` + `03_loopback` + README |
| Tablo XVI — MCU | `python -m tools.mcu_feasibility` |

## Raporda açıkça eksik olarak işaretlenenler

Bölüm IX (Devam Eden İşler) bunları sayar; metinde "yapıldı" gibi
sunulmamıştır:

- Güven kalibrasyonu (güvenilirlik diyagramı, ECE, sıcaklık ölçekleme)
- Eşik taraması (0.60 / 0.70 / 0.80 / 0.90) ve kapsam–hata yakalama eğrisi
- Çok istemcili yük (1 / 2 / 4 istemci)
- Düşük güvenli pencerelerde açıklama kararsızlığı
- Model karmaşıklığı ablasyonu (F1–gecikme Pareto)
- Bulut ayağı, Ethernet tekrarı, kaynak kısıtlı sis düğümü koşusu

## Dikkat edilmesi gereken iki nokta

1. **Sis koşularındaki sınıflandırma sayıları anlamsızdır.** `04_fog/` koşuları
   S11'in ilk beş dakikasını kapsıyor ve o aralıkta hiç stres etiketi yok (42
   pencerenin hepsi stres dışı, doğruluk 1.00). Raporda bu Bölüm VIII-4'te
   açıkça yazılı; sınıflandırma başarımı Tablo XV'ten okunmalı.

2. **Referans makale (Erzurumluoğlu ve ark., ICACT 2025) doğrulanamadı.**
   `docs/literature.md` bunu işaretliyor. Rapordaki karşılaştırma metni
   künyeye bağımlı bırakılmadı, ama kaynakçaya girmeden önce birincil kaynak
   hocadan istenmeli.

## Konferans sürümü

Bu dosya tam iç rapordur (~14 sayfa). ICACT/IISEC için 6 sayfalık sürüm
`PLAN.md` Faz F'deki yapıya göre kısaltılmalı: Bölüm II bir paragrafa,
Bölüm IV alt başlıkları üç paragrafa iner; Tablo VII, VIII, XVI ve Şekil 1
çıkar.
