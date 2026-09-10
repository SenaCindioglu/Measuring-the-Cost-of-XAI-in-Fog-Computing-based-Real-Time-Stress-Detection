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

## Seçili model

Streaming emülasyonu `wesad_best_model_f32-128_k9.pt` checkpoint'i ile çalışır:

| | |
|---|---|
| Mimari | depth=2, filters=[32, 128], kernel=9 |
| Parametre | 49.890 |
| Eğitim seed'i | 42 (`wesad_save_best_model.py`) |
| Train subject (13) | S2 S16 S7 S10 S4 S3 S17 S6 S9 S13 S15 S5 S8 |
| Val subject (2) | **S11, S14** — replay için doğru seçim bunlar |
| Offline val F1 (weighted) | 0.9386 |

Checkpoint kendi mimarisini içinde taşıdığı için (`filters`, `kernel_size`,
`num_channels`, dropout'lar, `meta`) script mimariyi doğrudan oradan okur;
`config.py` değerleri sadece yeniden eğitim ve eski düz `state_dict`
checkpoint'leri için geçerlidir.

## KRİTİK: kanal sırası

14 kanalın sırası eğitim scriptiyle **birebir** aynı olmak zorundadır. Sıra
bozulursa model karışık kanal alır ve doğruluk çöker (accuracy 1.00 yerine
0.52 gözlendi). Doğru sıra **önce chest (8), sonra wrist (6)**:

```
0-2  chest ACC x,y,z   3-  chest ECG    4-  chest EMG
5    chest EDA         6-  chest Temp   7-  chest Resp
8-10 wrist ACC x,y,z   11- wrist BVP    12- wrist EDA    13- wrist TEMP
```

`data/loader.py` bu sırayı üretir, `config.CHANNEL_NAMES` ile belgelenmiştir.

## Kurulum notu

`config.py` içindeki `DATA_PATH`, `CHECKPOINT_DIR`, `RESULTS_DIR` yollarını
kendi ortamına göre düzenle. Bunlar `WESAD_DATA_PATH`, `WESAD_CHECKPOINT_DIR`,
`WESAD_RESULTS_DIR` ortam değişkenleriyle de ezilebilir. `DATA_PATH`, içinde
`S2/S2.pkl` gibi alt klasörler bulunan WESAD kök klasörünü göstermelidir.

`psutil` CPU/RAM ölçümü için kullanılıyor; Colab'da genelde hazır gelir,
yoksa: `pip install psutil`

## Kullanım

Elinde eğitilmiş checkpoint varsa (normal durum) doğrudan 2. adıma geç;
1. adım yalnızca sıfırdan yeni bir model eğitmek istersen gerekir.

### 1) (Opsiyonel) Sıfırdan yeni checkpoint üret

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

## Normalizasyon modları

Model offline eğitimde **subject bazlı** `StandardScaler` ile normalize edilmiş
veri gördü. Bu gerçek zamanlı sistemde doğrudan kullanılamaz, çünkü subject'in
tüm kaydının istatistiğini bilmek geleceğe sızmaktır. İki sızıntısız alternatif
ölçülüyor:

| `--norm-mode` | Ne yapar | Kime karşılık gelir |
|---|---|---|
| `running` (varsayılan) | Welford ile, sadece o ana kadar görülen veriden nedensel mean/std | Eğitimdeki subject-bazlı z-score'un sızıntısız hali |
| `fixed` | Yalnızca train subjectlerinden hesaplanmış tek bir sabit mean/std | Katı deployment senaryosu: yeni kullanıcı hakkında hiçbir şey bilinmiyor |

`running` modda istatistik güncelleme maliyeti her tick'te ödenir ve dürüstlük
için pencerenin `preprocessing` süresine eklenir.

Raporda ikisini de ver: aradaki fark, kişiselleştirmenin (subject'e uyarlanan
normalizasyon) gerçek zamanlı sistemde ne kadar değer taşıdığını gösterir.

## Önemli metodolojik notlar
- **Window latency ≠ computation latency**: 10s pencere + 5s hop, verinin
  *toplanma* süresidir; bu script sadece *hesaplama* (preprocessing+inference)
  süresini ölçer. İkisini raporda ayrı ayrı belirt.
- **Warm-up hariç tutulur**: ilk `warmup_windows` pencere (varsayılan 15),
  CUDA/cache ısınması nedeniyle yanıltıcı olduğu için özet istatistiklere dahil
  edilmez.
- **`--realtime` bayrağı** sadece demo/gösterim amaçlıdır (tick'ler arasına
  gerçek `time.sleep()` ekler); latency ölçümünü etkilemez, sadece scripti
  gerçek zaman hızında çalıştırır.
- **Offline accuracy ile streaming accuracy karşılaştırılırken**: offline sayı
  subject-bazlı normalizasyonla üretilir, streaming sayı sızıntısız
  normalizasyonla. Düşüş modelin değil, normalizasyon kısıtının sonucudur.

## Ölçülen sonuçlar (S11 + S14, CPU, tick=500ms, 1175 pencere)

Hesaplama gecikmesi (warm-up hariç):

| Aşama | mean | median | p95 |
|---|---|---|---|
| preprocessing (`running`) | 0.189 ms | 0.182 ms | 0.206 ms |
| preprocessing (`fixed`) | 0.032 ms | 0.029 ms | 0.036 ms |
| inference | 1.15 ms | 1.11 ms | 1.33 ms |
| **total** (`running`) | **1.270 ms** | 1.225 ms | 1.460 ms |
| **total** (`fixed`) | **1.240 ms** | 1.215 ms | 1.446 ms |

Sınıflandırma:

| Koşul | accuracy | stress F1 |
|---|---|---|
| Offline (subject-bazlı norm, sızıntılı) | 0.9361 | — (weighted F1 = 0.9386) |
| Streaming `running` | 0.7677 | 0.5741 |
| Streaming `fixed` | 0.6996 | 0.4676 |
