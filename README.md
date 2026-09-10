# WESAD Streaming / Real-Time Replay Emulasyonu

Offline LOSO pipeline'ından sonraki adım: eğitilmiş modeli, WESAD sinyalini
gerçek zamanlı bir sensörden geliyormuş gibi kronolojik olarak "replay" eden
bir simülasyona sokup, aşama bazlı (preprocessing / inference / vs.) gecikmeyi
ölçmek.

## Klasör yapısı

```
wesad_realtime/
├── config.py                      # tek merkezi ayar dosyası (yollar, pencere, model, tick)
├── data/
│   ├── __init__.py
│   ├── loader.py                  # WESAD .pkl -> ham 14 kanal sinyal (wrist+chest)
│   ├── windowing.py                # offline eğitim için pencereleme + undersampling
│   └── normalization.py            # SABİT (pre-computed) mean/std hesapla/kaydet/yükle/uygula
├── models/
│   ├── __init__.py
│   └── temporal_cnn.py             # TemporalCNN mimarisi (offline ile birebir aynı)
├── training/
│   ├── __init__.py
│   └── train_final_model.py        # TEK SEFERLİK: checkpoint + sabit norm stats üretir
├── streaming/
│   ├── __init__.py
│   ├── buffer.py                    # RollingBuffer: tick tick veri biriktirip pencere üretir
│   └── replay_simulator.py          # bir subject'i replay eder, her pencereyi profiler'a sokar
├── profiling/
│   ├── __init__.py
│   └── latency_profiler.py          # aşama bazlı süre + CPU/RAM/GPU + mean/median/p95 raporu
├── run_streaming_emulation.py       # ANA SCRIPT — buradan çalıştırılır
├── checkpoints/                     # (ilk çalıştırmada otomatik oluşur) model.pth + norm_stats.npz
└── results/                         # (ilk çalıştırmada otomatik oluşur) JSON latency raporları
```

## Kurulum notu

`config.py` içindeki `DATA_PATH`, `CHECKPOINT_DIR`, `RESULTS_DIR` yollarını
kendi ortamına göre düzenle (varsayılan Google Drive/Colab yolu).

`psutil` CPU/RAM ölçümü için kullanılıyor; Colab'da genelde hazır gelir,
yoksa: `pip install psutil`

## Kullanım — iki adım

### 1) Checkpoint + sabit normalizasyon istatistiklerini üret (bir kere)

Proje kök klasöründen (`wesad_realtime/` içinden), `-m` ile modül olarak çalıştır
(alt paketlerin doğru import edilmesi için önemli):

```bash
cd wesad_realtime

# Örnek: S17'yi test olarak dışarıda bırakan bir LOSO modeli
python -m training.train_final_model --leave-out S17 --tag loso_s17

# veya: gerçek deployment senaryosu için tüm subjectlerle eğit
python -m training.train_final_model --all-subjects --tag deployment_final
```

Bu, `checkpoints/` altına şunları yazar:
- `model_<tag>.pth`
- `norm_stats_<tag>.npz`

### 2) Streaming/replay emülasyonunu çalıştır

`run_streaming_emulation.py` kök klasörde olduğu için doğrudan çalıştırılabilir
(`-m` gerekmez):

```bash
python run_streaming_emulation.py \
    --model checkpoints/model_loso_s17.pth \
    --norm-stats checkpoints/norm_stats_loso_s17.npz \
    --subject S17 \
    --tick-ms 500 \
    --warmup-windows 15 \
    --tag s17_test
```

Birden fazla subject art arda replay edilebilir (`--subject S2 S3 S4`), bu
durumda hepsinin ölçümleri aynı özet raporda birleşir.

Çıktı olarak `results/` altına iki JSON dosyası yazılır:
- `streaming_raw_<tag>.json` — her pencerenin ham ölçümleri (Pareto grafikleri
  için bir sonraki adımda kullanılacak)
- `streaming_summary_<tag>.json` — mean/median/p95/std özet istatistikleri

## Önemli metodolojik notlar

- **Normalizasyon SABİT**: her subject kendi istatistikleriyle değil, sadece
  train subjectlerinden hesaplanmış tek bir mean/std ile normalize edilir —
  gerçek zamanlı sistemde gelecekteki verinin istatistiğini bilemeyeceğin için
  bu şart.
- **Window latency ≠ computation latency**: 10s pencere + 5s hop, verinin
  *toplanma* süresidir; bu script sadece *hesaplama* (preprocessing+inference)
  süresini ölçer. İkisini raporda ayrı ayrı belirt.
- **Warm-up hariç tutulur**: ilk `warmup_windows` pencere (varsayılan 15),
  CUDA/cache ısınması nedeniyle yanıltıcı olduğu için özet istatistiklere dahil
  edilmez.
- **`--realtime` bayrağı** sadece demo/gösterim amaçlıdır (tick'ler arasına
  gerçek `time.sleep()` ekler); latency ölçümünü etkilemez, sadece scripti
  gerçek zaman hızında çalıştırır.
