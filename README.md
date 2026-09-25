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

### Docker ile sunucuyu deploy et

Sunucu, varsayılan olarak `running` normalizasyonu kullanır; bu modda WESAD
ham veri klasörü gerekmez. Docker ve Docker Compose kurulu sunucuda proje kök
klasöründen:

```bash
docker compose up -d --build
docker compose logs -f stress-server
```

TCP sunucusu `0.0.0.0:9099` adresinde dinler. İstemciyi çalıştırırken sunucunun
IP adresini kullan:

```bash
python compare_deployments.py --subject S11 --realtime \
  --deployment fog=SERVER_IP:9099 --tag docker_test
```

Konteyneri durdurmak için:

```bash
docker compose down
```

`fixed` normalizasyon kullanılacaksa WESAD veri klasörünü sunucuya koyup
`WESAD_DATA_PATH` ile konteynere mount etmen veya önceden üretilmiş norm stats
dosyasını `checkpoints/` altında image'a dahil etmen gerekir.

XAI ile Docker sunucusunu başlatmak için VM üzerinde:

```bash
WESAD_XAI_METHOD=lime_ucuz WESAD_XAI_THRESHOLD=1.0 \
docker compose up -d --build
```

Ardından istemci makinesinde ayrı bir etiketle çalıştır:

```bash
python run_fog_experiment.py --server-host SERVER_IP --subject S11 \
  --realtime --max-ticks 600 --live --tag xai_limeucuz
```

`--live` ve `--realtime` aynı seçenektir. XAI açıklama süresi sunucu cevabındaki
`stages.xai` alanına yazılır; `threshold=1.0` her tahmin penceresinde açıklama
çalıştırır. SHAP karşılaştırması için VM'yi şu şekilde başlat:

```bash
WESAD_XAI_METHOD=shap WESAD_XAI_THRESHOLD=1.0 \
docker compose up -d --build
```

Daha hızlı SHAP yapılandırması için `WESAD_XAI_METHOD=shap_ucuz` kullan.

Tam LIME yapılandırması için `WESAD_XAI_METHOD=lime` kullan; bu, seçili
artifaktan gelen 500 perturbasyon örneğini çalıştırır. `lime_ucuz` 250 örnek
kullanır.

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

---

## Ölçüm metodolojisi (Aşama 1)

Tek bir koşunun sonucu bu makinede güvenilir değil. Ölçülen davranış: aynı
yapılandırma arka arkaya koşulduğunda değerler yukarı kayıyor (işlemci frekans
ölçeklemesi), ve iş parçacığı sayısı 1/2/4 arasında dönüşümlü koşulduğunda
aradaki fark koşudan koşuya olan yayılımın altında kalıyor.

Bu yüzden gecikme sayıları **tekrarlı ve dönüşümlü** ölçülür:

```bash
python benchmark_runs.py --subject S11 --repeat 4 --threads 1 2 4 --tag thread_sweep
python benchmark_runs.py --subject S11 --repeat 4 --pipeline prepared native --tag pipeline_cmp
```

Birden fazla yapılandırma verilirse koşular dönüşümlü sırada yapılır (tur 1:
A, B / tur 2: A, B ...), böylece ısınma kayması hepsine eşit dağılır.

**İki ayrı dağılım vardır, karıştırılmamalı:**

| | Ne ölçer | Nereye yazılır |
|---|---|---|
| Koşu içi dağılım | bir koşudaki 1000+ pencerenin p50/p95/p99'u | sistemin en kötü durum davranışı |
| Koşu arası yayılım | aynı deneyi tekrarlayınca medyanın oynaması | ölçümün güven aralığı |

İş parçacığı sayısı bu iş yükünde belirleyici değil. Varsayılanın 1 olmasının
gerekçesi hız değil kararlılık: tek iplikte ölçüm beş kat daha az oynuyor
(yayılım %2.9, diğerlerinde %11-15). İzole mikro-benchmark yanıltıcıdır, karar
her zaman tam pipeline üzerinde verilmelidir.

## İki pipeline yolu (Aşama 2)

`--pipeline` bayrağı, yeniden örneklemenin nerede yapıldığını belirler.

| Yol | Sinyal nasıl gelir | Yeniden örnekleme |
|---|---|---|
| `prepared` | tüm kayıt replay'den önce 64 Hz'e indirilmiş | ölçüm **dışında** |
| `native` (varsayılan) | her sensör kendi hızında (göğüs 700 Hz, bilek 32/64/4 Hz) | her pencerede, sunucuda, **ölçülür** |

Gerçek bir sistemde sensör ham hızında veri gönderir ve indirgeme sunucuda
yapılır, yani `native` gerçekçi olan yoldur. `prepared` eski sonuçlarla
karşılaştırma için duruyor.

**Ölçülen fark** (S11, 4 tur dönüşümlü, tek iplik, koşu içi medyanların medyanı):

| Yol | resampling | preprocessing | inference | toplam |
|---|---|---|---|---|
| `prepared` | — | 0.580 ms | 2.969 ms | 3.971 ms |
| `native` | 1.573 ms | 0.252 ms | 3.039 ms | 5.502 ms |

Yeniden örnekleme tek başına toplamın yaklaşık %29'u, yani çıkarımdan sonraki
en pahalı aşama. Aradaki %38.6'lık fark, koşular arası yayılımın (%9.4) çok
üstünde olduğu için gerçek.

**Doğruluk korunuyor.** Pencere başına yeniden örnekleme, tüm sinyali bir kerede
indirmekle birebir aynı sonucu vermez (kenar etkisi), ama fark ölçülemez
düzeyde (S11 + S14, nedensel normalizasyon):

| Yol | Doğruluk | Stres F1 | TN / FP / FN / TP |
|---|---|---|---|
| `prepared` | 0.7677 | 0.5741 | 718 / 188 / 85 / 184 |
| `native` | 0.7674 | 0.5732 | 720 / 187 / 87 / 184 |

**Etiketleme farkı.** `prepared` yolu geçersiz etiketli örnekleri sinyalden
atıp kalanları birleştiriyor; bu zamanda atlama demek ve gerçek zamanlı bir
sistemde mümkün değil. `native` yolunda zaman kesintisiz akar, çoğunluk etiketi
geçersiz olan pencereler yalnızca raporlama dışında bırakılır.

**Bant genişliği.** 500 ms'lik bir tick'te ham hızda 11 536 bayt taşınır, 64 Hz'e
indirilmiş halde 1 792 bayt. Yani indirgemeyi sunucuda yapmak ağdan 6.4 kat
fazla veri geçirmek demektir. Bu, Aşama 3'teki edge/fog yerleşim kararının
sayısal dayanağıdır.

## İstemci–sunucu ayrımı (Aşama 3)

Önceki aşamalara kadar istemci ve sunucu aynı süreçte çalışıyordu. Bu, ağ
kaynaklı metriklerin tanım gereği sıfır olması demekti. `serving/` paketi ikisini
gerçek bir TCP bağlantısıyla ayırır.

```
serving/protocol.py   tel formatı, çerçeveleme, saat senkronizasyonu
serving/server.py     açık istek kuyruğu + tek worker, model burada çalışır
serving/client.py     sensör istemcisi, ham hızda gönderir, zaman damgalarını kaydeder
run_fog_experiment.py sunucuyu başlatır, N istemciyi koşturur, raporu toplar
```

Tek komutla uçtan uca deney:

```bash
python run_fog_experiment.py --subject S11 --realtime --tag realtime_1user
python run_fog_experiment.py --subject S11 S14 --clients 4 --realtime --tag load_4
```

Uzaktaki bir sunucuya bağlanmak için `--server-host 192.168.1.42` yeterli;
o zaman sunucu yerelde başlatılmaz.

### Ölçülen yedi metrik

| Metrik | Tanım | Hangi saatler |
|---|---|---|
| latency | paketin istemciden sunucuya ulaşması | iki saat, fark düzeltmesi gerekir |
| queuing delay | worker meşgulken kuyrukta bekleme | sunucu saati |
| execution time | pipeline'ın çalışma süresi | sunucu saati |
| response time | istemcinin gördüğü toplam süre | istemci saati |
| jitter | response time'ın ardışık değişimi | istemci saati |
| bandwidth | gönderilen + alınan bayt / süre | — |
| resource usage | CPU ve bellek | sunucu tarafı |

### İki tasarım kararı ve nedenleri

**Saat katmanı `perf_counter_ns`, `time_ns` değil.** İlk sürüm `time.time_ns()`
kullanıyordu. Windows'ta çözünürlüğü ölçüldü ve 15.6 ms çıktı
(`GetSystemTimeAsFileTime`). Loopback gecikmesi 0.1 ms mertebesinde olduğu için
latency medyanı tam sıfır okuyordu, yani metrik tamamen anlamsızdı.
`perf_counter` 100 ns çözünürlük verir ama referans noktası sürece özeldir;
bu yüzden bağlantı başında NTP mantığıyla saat farkı ölçülür (PING/PONG, en
düşük gidiş-dönüş süresine sahip tur seçilir). Aynı makinede ölçülen fark
0.001 ms, belirsizlik ±0.13 ms. Bu düzeltme uzak makine durumunu da çözer.

**Sunucuda açık istek kuyruğu var.** İlk sürümde paket okunur okunmaz aynı
yerde işleniyordu; `t_start` ile `t_recv` bitişik olduğu için queuing delay her
zaman sıfıra yakın okuyordu ve bekleme süresi latency'nin içinde gizleniyordu.
Şimdi okuyucu görevi paketi alıp varış zamanını damgalar ve ortak kuyruğa
koyar, tek worker kuyruğu sırayla boşaltır. İki istemciyle p99 kuyruk gecikmesi
0.37 ms'den 5.50 ms'ye çıktı, yani metrik artık gerçekten bekleme ölçüyor.

### Sonuçlar

Gerçek sensör hızında, tek istemci, S11'in ilk 10 dakikası, loopback:

| Metrik | ortalama | medyan | p95 | p99 |
|---|---|---|---|---|
| latency | 0.909 ms | 0.809 ms | 1.466 ms | 2.707 ms |
| queuing delay | 0.062 ms | 0.045 ms | 0.085 ms | 0.622 ms |
| execution time | 8.273 ms | 7.179 ms | 10.524 ms | 11.268 ms |
| response time | 9.884 ms | 8.821 ms | 12.698 ms | 13.927 ms |

Jitter 3.529 ms. Bant genişliği 184.8 kbps gönderilen, 2.3 kbps alınan,
toplam 187.2 kbps.

Sunucu içi aşama dökümü (ortalama): resampling 2.211, preprocessing 0.449,
tensor_prep 0.324, inference 4.648, postprocessing 0.135 ms.

**Doğrulama.** Tek süreçli `--pipeline native` yolu ile istemci–sunucu yolu
aynı veride birebir aynı sınıflandırma sonucunu veriyor: doğruluk 0.8222,
stres F1 0.6944, karışıklık matrisi TN=743 FP=184 FN=29 TP=242. Bu, ağ
katmanının hesabı bozmadığının kanıtı.

### Bir hata daha yakalandı

Tek süreçli native yolda, çoğunluk etiketi geçersiz olan pencereler tamamen
atlanıyordu. Bu, normalizasyon istatistiğinin o pencerelerde hiç güncellenmemesi
demekti. Gerçek bir sunucu etiketi görmez, dolayısıyla böyle bir atlama yapamaz.
Düzeltildikten sonra doğruluk 0.7674'ten 0.8222'ye çıktı ve iki yol birbirine
eşitlendi. Geçersiz etiketli pencereler artık baştan sona işlenir, yalnızca
sınıflandırma raporundan çıkarılır.

### Makaleyle karşılaştırma

Erzurumluoğlu ve ark., ICACT 2025, DREAMER veri setiyle fog mimarisinde duygu
tanıma. Worker düğümü Raspberry Pi 4, istemci Wi-Fi üzerinden bağlı.

| Metrik | Makale (worker) | Bu çalışma (loopback) |
|---|---|---|
| latency | 4.7 ms | 0.81 ms |
| queuing delay | < 1 ms | 0.045 ms |
| execution time | 34.7 ms | 7.18 ms |
| response time | 43.3 ms | 8.82 ms |
| jitter | 17.4 ms | 3.53 ms |
| bandwidth | 374 kbps | 187 kbps |

**Bu tablo dikkatli okunmalı.** Ağ metrikleri karşılaştırılabilir değil: onların
ölçümü Wi-Fi üzerinden ayrı bir cihaza, bizimki aynı makinede loopback üzerinden.
Execution time farkı da donanım farkını içeriyor, Raspberry Pi ile masaüstü
işlemcisi kıyaslanıyor. Anlamlı olan tek karşılaştırma **bant genişliği**:
onlar 3 saniyelik pencereden 24 öznitelik çıkarıp gönderiyor, biz 10 saniyelik
pencerenin ham sinyalini gönderiyoruz, buna rağmen yarısı kadar veri geçiyor.

Asıl fark mimari: onların sisteminde öznitelik çıkarımı hesaplamanın çoğunu
oluşturuyor, bizde ham sinyal doğrudan modele giriyor ve hesap çok daha ucuz.
Buna karşılık bizim uçtan uca gecikmemizi hesap değil pencere uzunluğu
belirliyor: 10 saniyelik pencere, 8.8 ms yanıt süresinin bin katından fazla.

### İki makineyle çalıştırma

Sunucu WESAD veri setine **ihtiyaç duymaz**. Veri tamamen istemcide kalır,
sunucuya yalnızca ham sinyal parçaları akar. İkinci makinede gereken:

```bash
git clone https://github.com/SenaCindioglu/files.git
cd files
pip install torch numpy psutil
python -m serving.server --host 0.0.0.0 --port 9099
```

`--host 0.0.0.0` şart; varsayılan `127.0.0.1` yalnızca aynı makineden bağlantı
kabul eder. Windows güvenlik duvarı ilk çalıştırmada izin soracaktır.

İstemci tarafında (veri setinin olduğu makine):

```bash
python run_fog_experiment.py --server-host 192.168.1.42 --subject S11 --realtime --tag wifi_1user
```

`--server-host` verildiğinde sunucu yerelde başlatılmaz, sadece istemciler
koşar.

**Saat farkı ve ölçüm belirsizliği.** İki makinenin yüksek çözünürlüklü
sayaçları birbirinden bağımsızdır. Bağlantı başında NTP mantığıyla fark
ölçülür (12 tur, en düşük gidiş-dönüş süresine sahip tur seçilir) ve tek yönlü
gecikmeye uygulanır. Belirsizlik gidiş-dönüş süresinin yarısı kadardır ve
rapora yazılır:

**Kayma (drift) ve periyodik yeniden ölçüm.** Tek bir ölçüm uzun koşularda
yetmez: iki kristal birebir aynı hızda tiklemediği için fark zamanla kayar. 87
dakikalık bir Wi-Fi koşusunda kayma latency medyanını **negatife** düşürdü, yani
metrik geçersiz oldu. Bu yüzden saat farkı varsayılan olarak her 60 saniyede bir
yeniden ölçülür (`--clock-resync-s`, 0 = kapalı) ve her paketin latency'si iki
ölçüm arasında doğrusal ara değerle düzeltilir. Ölçüm, son cevap alındıktan
sonraki sessiz anda yapılır ve dört turdan oluşur, böylece akışı neredeyse hiç
kesmez. Özette toplam kayma ve kayma hızı (ppm) yazılır, ayrıca latency medyanı
`(response − execution − queuing) / 2` ile karşılaştırılıp tutarlılık kontrolü
basılır. Negatif medyan görülürse uyarı verilir.

| Bağlantı | tipik RTT | latency belirsizliği |
|---|---|---|
| loopback | 0.26 ms | ±0.13 ms |
| Ethernet | 0.5–2 ms | ±0.25–1 ms |
| Wi-Fi | 2–20 ms | ±1–10 ms |

Wi-Fi'da tek yönlü gecikmenin hata payı, ölçülen değerin kendisiyle aynı
mertebeye çıkabilir. **Bundan etkilenmeyen metrikler**: execution time ve
queuing delay (ikisi de sunucu saati), response time ve jitter (ikisi de
istemci saati). Ölçüm kalitesi önemliyse Ethernet tercih edilmelidir.

**Ne ölçmüş olursun, ne ölçmemiş olursun.** İki normal bilgisayar arasındaki
deney gerçek ağ gecikmesini, jitter'ı ve bant genişliğini verir. Ama bu bir
*dağıtık yerleşim* ölçümüdür, *fog düğümü* ölçümü değildir. Fog'un tanımındaki
kaynak kısıtı yoktur. Raporda bu ayrım açıkça belirtilmelidir. Kaynak kısıtını
donanımsız taklit etmek gerekirse sunucu Docker altında `--cpus=1 --memory=1g`
ile çalıştırılabilir; bu da taklit olduğu belirtilerek raporlanır.

## STM32 üzerinde çalışır mı? (analiz)

`tools/mcu_feasibility.py` modelin bir mikrodenetleyicide çalışıp
çalışmayacağını donanım olmadan hesaplar: katman katman parametre, aktivasyon
belleği, tepe RAM ihtiyacı ve çarpma-toplama sayısı.

```bash
python -m tools.mcu_feasibility
python -m tools.mcu_feasibility --window 320   # 5 saniyelik pencere
```

### Mimarideki yeri

STM32 bir **mikrodenetleyicidir**, işletim sistemi çalıştırmaz. Fog
taksonomisinde Raspberry Pi'nin yerini alamaz, çünkü farklı bir katmanda durur:

```
sensör / uç düğüm   ->  STM32 sınıfı MCU (giyilebilir cihazın içi)
edge                ->  telefon, gateway
fog                 ->  Raspberry Pi, küçük sunucu
cloud               ->  veri merkezi
```

Ama STM32 daha ilginç bir şeyin yerini alabilir: **ağın tamamının**. Çıkarım
giyilebilir cihazın içinde yapılırsa fog ile edge arasında seçim yapma sorunu
ortadan kalkar. Gecikme, mahremiyet ve bant genişliği açısından üçünü de yener.

### Ölçülen gereksinimler (10 s pencere, depth2 / [32,128] / k9)

| | Ağırlık (Flash) | Tepe RAM | Giriş tamponu |
|---|---|---|---|
| float32 | 194.9 KB | 320.0 KB | 35.0 KB |
| float16 | 97.4 KB | 160.0 KB | 17.5 KB |
| int8 | 48.7 KB | 80.0 KB | 8.8 KB |

Tepe aktivasyon `bn2` katmanında, 81 920 eleman. İkinci evrişim bloğu
128 kanal × 320 uzunluk ürettiği için darboğaz orada.

| STM32 ailesi | RAM | Flash | float32 | int8 |
|---|---|---|---|---|
| F103 (M3) | 20 KB | 128 KB | sığmaz | sığmaz |
| F407 (M4) | 192 KB | 1 MB | sığmaz | **sığar** |
| F746 (M7) | 320 KB | 1 MB | sığmaz | **sığar** |
| H743 (M7) | 1 MB | 2 MB | **sığar** | **sığar** |
| U575 (M33) | 786 KB | 2 MB | **sığar** | **sığar** |

### Hız sorun değil, bellek sorun

Pencere başına 14.50 milyon çarpma-toplama gerekiyor. Bütçe ise 5 000 ms,
yani hop süresi.

| Donanım | tahmini süre | bütçe kullanımı |
|---|---|---|
| Cortex-M4 @168 MHz, float32 | 287.7 ms | %5.8 |
| Cortex-M7 @480 MHz, float32 | 37.8 ms | %0.8 |
| Cortex-M7 @480 MHz, int8 | 12.1 ms | %0.2 |

En yavaş senaryo bile bütçenin yirmide birini kullanıyor. MCU'da darboğaz
kesinlikle bellek.

### Gerçekten yapmak için gereken iş

Bu analiz kağıt üstünde. Cihaza indirmek için:

1. Modeli ONNX'e aktar, STM32Cube.AI (X-CUBE-AI) ile C koduna çevir. Alternatif:
   TensorFlow Lite Micro veya elle CMSIS-NN.
2. int8 nicemleme yap (F407/F746 için zorunlu). Doğruluk kaybı ölçülmeli.
3. Ön işlemeyi porta et: 700 Hz'den 64 Hz'e indirgeme ve nedensel z-score.
   İkisi de basit, taşınabilir kod.
4. Ölçümü cihaz üstünde yap: DWT çevrim sayacı ile süre, UART ile rapor.
5. Veriyi cihaza besle: SD karttan replay ya da PC'den UART üzerinden akış.

Bu, birkaç günlük değil birkaç haftalık bir iş ve donanım gerektirir. Tez
kapsamında **fizibilite bölümü** olarak yukarıdaki sayılarla raporlanması,
tam port yapmaktan çok daha iyi bir maliyet-fayda dengesi sunar.

## Fog düğümü kurulumu (Raspberry Pi veya ikinci bilgisayar)

Sunucu Pi'de, istemci veri setinin olduğu makinede çalışır. **Pi'ye WESAD
verisi kopyalamaya gerek yok**, sadece depo ve PyTorch yeterli.

### 1. İşletim sistemi

**64-bit şart.** PyTorch'un 32-bit ARM için hazır paketi yok. Raspberry Pi
Imager'da "Raspberry Pi OS (64-bit)" seç. Kontrol:

```bash
uname -m        # aarch64 yazmalı, armv7l değil
```

Pi 4 veya Pi 5 önerilir, en az 2 GB bellek. PyTorch içe aktarıldığında tek
başına yaklaşık 200 MB tutuyor.

### 2. Kurulum

```bash
sudo apt update && sudo apt install -y git python3-venv
git clone https://github.com/SenaCindioglu/files.git
cd files
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch indirmesi büyük, bağlantıya göre birkaç dakika sürebilir.

### 3. Sunucuyu başlat

```bash
python -m serving.server --host 0.0.0.0 --port 9099
```

`--host 0.0.0.0` şart. Açılışta kendi donanımını yazar; `Raspberry Pi 4
Model B` gibi bir satır görmelisin. Pi'nin adresi:

```bash
hostname -I
```

### 4. İstemciyi çalıştır (kendi makinende)

```bash
python run_fog_experiment.py --server-host 192.168.1.42 --subject S11 --realtime --tag pi_1user
```

### Pi'ye özgü üç nokta

**İş parçacığı sayısını Pi üzerinde yeniden ölç.** Varsayılan 1, ama bu değer
20 çekirdekli bir x86'da ölçüldü. Pi 4'te dört Cortex-A72 çekirdek var ve
sonuç farklı çıkabilir. Sunucuyu sırayla `--threads 1`, `--threads 2`,
`--threads 4` ile çalıştırıp karşılaştır.

**Ethernet kullan.** Wi-Fi'da tek yönlü gecikmenin hata payı ölçülen değerle
aynı mertebeye çıkıyor (yukarıdaki belirsizlik tablosuna bak). Kablo takmak
ölçüm kalitesini birkaç kat artırır.

**Isınmayı kontrol et.** Pi 4 soğutucusuz çalışırken sıcaklık nedeniyle hız
düşürür ve bu gecikme ölçümünü bozar. Deney sırasında izle:

```bash
watch -n 2 vcgencmd measure_temp
```

`throttled=0x0` dışında bir değer görürsen soğutucu ekle ve ölçümü tekrarla.

### Ne ölçülmüş olur

Sunucu, bağlantı kapanırken kendi donanımını ve kaynak kullanımını istemciye
gönderir. Böylece makaledeki yedi metriğin hepsi elde edilir:

| Metrik | Nereden gelir |
|---|---|
| latency | istemci, saat farkı düzeltmeli |
| queuing delay | sunucu saati |
| execution time | sunucu saati |
| response time | istemci saati |
| jitter | istemci saati |
| bandwidth | istemci bayt sayacı |
| resource usage | **sunucu tarafı**, CPU % ve bellek MB |

### Karşılaştırma planı

Aynı deneyi üç yerleşimde koşup yan yana koy:

| Yerleşim | Sunucu nerede | Komut |
|---|---|---|
| edge | istemcinin kendi makinesi | `--subject S11 --realtime` |
| fog | Raspberry Pi, aynı ağ | `--server-host <pi-ip> --subject S11 --realtime` |
| cloud | uzak VPS | `--server-host <vps-ip> --subject S11 --realtime` |

Makalenin de yaptığı karşılaştırma bu. Bulut ayağı için küçük bir VPS aylık
birkaç dolar; bütçe yoksa o satır atlanıp edge ile fog karşılaştırılabilir,
eksik olduğu raporda belirtilir.

## Yerleşim karşılaştırması (edge / fog / cloud)

`compare_deployments.py` aynı deneyi birden fazla yerleşimde koşar ve tek
karşılaştırma tablosu üretir.

```bash
# Pi'de sunucuyu başlat, açık bırak
python -m serving.server --host 0.0.0.0 --port 9099

# Kendi makinende
python compare_deployments.py --subject S11 --realtime --max-ticks 600 \
    --repeat 3 --deployment edge=local fog=192.168.1.42 --tag edge_vs_fog
```

`local` yazılan yerleşimde sunucu otomatik başlatılıp koşu sonunda kapatılır.
Diğer adreslerde sunucunun o makinede zaten çalışıyor olması gerekir; script
başlamadan önce erişimi kontrol eder ve ulaşamıyorsa hemen hata verir.

Bulut eklemek istersen tek satır yeter: `--deployment edge=local
fog=192.168.1.42 cloud=203.0.113.7`

**Adil ölçüm için iki kural script'in içinde uygulanır.** Tüm yerleşimler aynı
denek, aynı tick, aynı süre ve aynı ısınma sayısıyla koşar; değişen tek şey
sunucunun yeri. Koşular dönüşümlü sırada yapılır (tur 1: edge, fog / tur 2:
edge, fog), böylece işlemci ısınmasından kaynaklanan kayma tüm ortamlara eşit
dağılır. Ardışık koşulsaydı en son ölçülen ortam sistematik olarak yavaş
görünürdü.

Çıktıda her metrik medyan ve tekrarlar arası yayılımla birlikte verilir.
Ortamlar arası fark yayılımın altında kalıyorsa script bunu açıkça söyler.

### Makalenin donanımı ve bizim karşılığımız

| Makale | Rol | Bizde |
|---|---|---|
| Asus ROG (Ryzen 7, 16 GB, Win 11) | istemci | kendi dizüstün |
| Raspberry Pi 4 (8 GB) | **worker** | senin Pi'n |
| Raspberry Pi 4 (8 GB) | broker | **yok, gerekmiyor** |
| DigitalOcean VPS (Frankfurt) | cloud | şimdilik atlandı |
| Wi-Fi modem (100/5 Mbps) | ağ | ev yönlendiricin |

**Broker neden gerekmiyor.** Makalenin mimarisinde broker, gelen istekleri en
az yüklü worker'a dağıtan bir yönlendirici. Tek worker varken dağıtacak bir
şey yok, sadece fazladan bir sıçrama eklemiş olursun. Makalenin kendi
sonuçlarında da broker ile worker sayıları neredeyse aynı çıkmış (43.9 ms'ye
karşı 43.3 ms), çünkü ikisi de aynı donanım ve broker hesabı kendisi yapıyor.
Broker ancak iki veya daha fazla worker'ın olduğu, yük dağıtımının ölçüldüğü
bir deneyde anlam kazanır.

**Ağ tercihi: kablo mu Wi-Fi mı.** İkisinin de gerekçesi var. Wi-Fi makaleye
sadık kalır ve gerçek dağıtım koşullarını yansıtır. Kablo ise saat farkı
belirsizliğini birkaç kat azaltır, yani latency ölçümü çok daha güvenilir olur.
En iyisi ikisini de koşup ayrı raporlamak; `--deployment fog_wifi=<ip>
fog_eth=<ip>` şeklinde iki ayrı tur yeterli.

### Fog düğümü olarak ikinci bir bilgisayar

Raspberry Pi yerine elindeki başka bir bilgisayarı da kullanabilirsin. Tek şart
ayrı bir fiziksel makine olması; aynı makinedeki sanal makine ya da konteyner
gerçek ağ üretmez.

İkinci bilgisayarda (Windows ya da Linux fark etmez):

```bash
git clone https://github.com/SenaCindioglu/files.git
cd files
pip install -r requirements.txt
python -m serving.server --host 0.0.0.0 --port 9099
```

Sunucu açılışta kendi IP adreslerini ve istemci tarafında çalıştırman gereken
komutu yazar, ayrıca `ipconfig` çalıştırmana gerek yok. Birden fazla adres
varsa kullanılabilir olanı işaretler; `169.254.` ile başlayanlar bağlantı-yerel
adreslerdir ve başka cihazdan erişilemez.

**Windows güvenlik duvarı.** İlk çalıştırmada izin penceresi çıkar, "Özel ağlar"
için izin vermelisin. Pencere çıkmazsa ya da yanlışlıkla reddedildiyse:

```powershell
New-NetFirewallRule -DisplayName "WESAD fog server" -Direction Inbound `
    -Protocol TCP -LocalPort 9099 -Action Allow -Profile Private
```

**İki makine aynı ağda olmalı.** Aynı Wi-Fi ya da aynı yönlendirici yeterli.
Misafir ağı (guest network) cihazları birbirinden yalıtır, orada çalışmaz.

### Güçlü makineyi fog düğümü gibi kısıtlamak

İkinci bilgisayar güçlüyse ölçtüğün şey *dağıtık yerleşim* olur, *kaynak
kısıtlı fog düğümü* değil. Makalenin argümanının yarısı düğümün zayıf olması.
Bu eksiği kapatmak için sunucuyu tek çekirdeğe hapsedebilirsin:

```powershell
# Windows (PowerShell)
$p = Start-Process python -ArgumentList "-m serving.server --host 0.0.0.0 --threads 1" -PassThru
$p.ProcessorAffinity = 1        # yalnizca 0. cekirdek
```

```bat
REM Windows (cmd). PowerShell'de calismaz, "start" orada Start-Process'e
REM cevrilir ve /affinity diye bir parametresi yoktur.
start /affinity 1 python -m serving.server --host 0.0.0.0 --threads 1
```

```bash
# Linux
taskset -c 0 python -m serving.server --host 0.0.0.0 --threads 1
```

### ⚠️ ÖLÇÜLDÜ: tek çekirdeğe kilitlemek bu iş yükünde hiçbir şey değiştirmiyor

Yukarıdaki yöntem **bağlayıcı değil.** İki sebebi de ölçüldü:

1. `--threads 1` zaten varsayılan (`config.DEFAULT_THREADS`), yani "normal"
   koşu da tek iplikle çalışıyor. İki mod arasındaki tek fark affinity kalıyor.
2. Süreç tek iplik olduğu için **zaten bir çekirdeği aşmıyor**. Fog
   düğümünde ölçülen süreç CPU'su (bir çekirdeğe göre yüzde):

   | koşu | ortalama | tepe |
   |---|---|---|
   | XAI yok | %0.0 | %6.2 |
   | `shap_ucuz`, her pencere | %5.0 | %105.7 |
   | `shap`, her pencere | %25.9 | %106.2 |
   | 4 istemci + `shap_ucuz` | %15.1 | %111.1 |

   Tepe değerler %100'ün hemen üstünde. Bir çekirdeğe kilitlemek ölçülebilir
   hiçbir fark yaratmaz.

**Bağlayıcı olan tek yöntem kesirli CPU kotasıdır** (Docker `--cpus=0.1`).
O, `shap_ucuz`'un 400 ms'sini ~4 saniyeye çıkarır ve 5 saniyelik hop
bütçesini neredeyse doldurur. Bedeli: Docker Desktop kurulumu ve torch'lu
imaj.

**Alternatif — ölçmeden yazmak.** Kısıt deneyi yapmadan da aynı sonuca
ölçülmüş sayılardan ulaşılabilir: referans çalışmanın Raspberry Pi 4
worker'ında execution 34.7 ms, bu çalışmada 3.6 ms, yani ~10 kat fark.
Aynı oran açıklamaya uygulanırsa `shap_ucuz` bir Pi 4'te ~4 saniye eder.
Bu, "taklit" ibaresine gerek bırakmadan seçici açıklamanın gerçek fog
donanımında neden zorunlu olduğunu gösterir.

Aşağıdaki iki koşulu yine de yapmak istersen (taklit olduğu belirtilerek):

```bash
# 1) Sunucu normal modda acikken
python compare_deployments.py --subject S11 --realtime --max-ticks 600 --repeat 3 \
    --deployment edge=local fog=192.168.1.8 --tag fog_tam

# 2) Sunucuyu kapat, tek cekirdege kilitleyip yeniden baslat, sonra
python compare_deployments.py --subject S11 --realtime --max-ticks 600 --repeat 3 \
    --deployment edge=local fog=192.168.1.8 --tag fog_kisitli
```

**İki ölçüm tek komuta sığmaz.** Script koşuları dönüşümlü sırada yapar (tur 1:
edge, fog / tur 2: edge, fog), yani aynı adres iki yerleşim adıyla verilirse her
turda sunucuyu normal ve kısıtlı mod arasında elle değiştirmen gerekirdi. Bu
yüzden iki ayrı çalıştırma yapılır; edge her ikisinde tekrarlandığı için iki
tablo yine karşılaştırılabilir kalır.
