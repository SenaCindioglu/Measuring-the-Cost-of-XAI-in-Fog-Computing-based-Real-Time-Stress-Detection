# v2 ölçüm protokolü — Raspberry Pi fog, tek imaj, açık döngü

Hazırlanma: 11 Ekim 2026. Son tarih: **15 Ekim 2026** (ICAIIC, EDAS).

Bu belge v1'deki dört metodolojik açığı kapatan yeniden koşunun adım adım
tarifidir. Sonuçlar `results/15_v2/` altına gelir, makale sayıları
`python -m tools.paper_numbers --v2 results/15_v2` ile yeniden üretilir.

| v1 açığı | v2 çözümü |
|---|---|
| Kapalı döngü istemci: açıklama sırasında sensör akışı duruyordu (yerel testte p95 661 ms) | `--loop open` (realtime'da varsayılan). `send_lag` ve `age` metrikleri kaydediliyor |
| Üç yerleşim üç yığın (torch 2.8.0 / 2.14.0 / 2.14.0+cu130) | Fog + cloud aynı imaj (`python:3.11.9-slim`, `torch==2.14.0` CPU); co-located aynı sürümler, yerel venv |
| XAI hücreleri tek koşu | Her hücre 3 tur, dönüşümlü sıra (`compare_deployments --repeat 3`) |
| Uzak sunucuda kaynak özeti sunucu ömrü boyunca | Oturum bazlı özet (sunucu kodu düzeltildi) |
| Fog = masaüstü PC | Fog = Raspberry Pi 4/5, Ethernet |

---

> **12 Ekim notu.** Pi koşuları (`results/15_pi/`) executor düzeltmesinden
> önceki sunucu sürümüyle alındı. Tek istemcili sonuçlar geçerli; çok
> istemcili koşularda yalnızca `response`/`age` geçerli. Pi'de depoyu
> güncelleyip (`server_build: v2.1-executor` JSON'da görünmeli) ve governor'ı
> `performance` yapıp `fog_pi_load{1,2,4}` tekrar koşulursa (3 × 5 dk)
> latency/queuing ayrımı da makaleye girebilir.

## 0. Öncelik (zaman yetmezse buradan kes)

1. **Pi seçici koşu, tam kayıt** (~90 dk, gözetimsiz) — makalenin en güçlü yeni sonucu.
2. Üç yerleşim matrisi: `none`, `shap_cheap`, `lime_cheap` (~45 dk/açıklayıcı).
3. Matris: `lime`, `shap` (Pi'de açıklama hop'u aşabilir — koşu uzar, bkz. §4).
4. (İsteğe bağlı) Pi'de 1/2/4 istemci, açık döngü.

## 1. Raspberry Pi hazırlığı

- **64-bit** Raspberry Pi OS (Bookworm): `uname -m` → `aarch64`.
- 4 GB+ RAM (SHAP açıkken süreç ~465 MB).
- **Aktif soğutucu.** Kısma olursa ölçüm sessizce bozulur.
- **Ethernet** (Wi-Fi değil). v1'de kablosuz yollarda açıklama açıkken ağ payı
  büyüdü (fog 3.5 → 24 ms); güç tasarrufu şüphesini ayırmak için kablo şart.
  İstemci makine de mümkünse kabloyla bağlansın; bağlantı türünü not et.
- Governor:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  vcgencmd get_throttled     # 0x0 olmalı
  ```
  Sunucu açılışta sıcaklık, governor ve `throttled` değerini basar ve her
  oturumun sonunda JSON'a yazar (`server.thermal`). `0x0` dışındaki her değer
  makalede belirtilmeli.
- Docker: `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER`

## 2. İmaj (Pi ve bulut VM'de aynı)

Depoyu iki makineye de çek (aynı commit!), sonra:
```bash
docker compose build
docker run --rm wesad-stress-server:v2 python -c "import torch; print(torch.__version__)"
#   -> 2.14.0+cpu olmalı (+cu... ise imaj yanlış)
```
Bulut VM'de eski konteyneri kaldır: `docker compose down && docker image prune`.
`shap` aarch64'te derleme isterse kurulum uzar; başarısız olursa LIME ile
devam et ve SHAP satırlarını "kurulamadı" diye not düş.

## 3. Co-located (istemci makinesi, Windows)

Aynı sürümler, yerel sanal ortamda (Docker Desktop WSL2'de NAT'lı ağ
kullandığı için "loopback" tanımı bozulur — co-located konteynerde koşulmaz):
```powershell
py -3.11 -m venv .venv311; .venv311\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
pip install -r requirements.txt
python -c "import torch, shap; print(torch.__version__, shap.__version__)"
```
Güç planı: **Yüksek performans**, şarj kablosu takılı, diğer uygulamalar kapalı.
Windows'ta zamanlayıcı çözünürlüğü yüzünden `send_lag` ~10–20 ms çıkar; normal.

## 4. Üç yerleşim matrisi

Her açıklayıcı için: önce **Pi ve bulutta** sunucuyu o açıklayıcıyla yeniden
başlat, sonra istemciden tek komut. `compare_deployments` uzak sunucuların
künyesini kontrol eder; biri yanlış açıklayıcıyla açıksa koşuyu durdurur.

```bash
# Pi'de ve bulutta (m = none | shap_cheap | lime_cheap | lime | shap)
WESAD_XAI_METHOD=shap_cheap WESAD_XAI_THRESHOLD=1.0 docker compose up -d --force-recreate
docker logs -f wesad-stress-server     # "Hazir, istemci bekleniyor" gorunene kadar
# none icin: WESAD_XAI_METHOD=none WESAD_XAI_THRESHOLD=0
```
```powershell
# istemcide
python compare_deployments.py --subject S11 --realtime --max-ticks 600 --repeat 3 `
    --deployment coloc=local fog=<PI_IP> cloud=<VM_IP> `
    --xai-method shap_cheap --xai-threshold 1.0 --tag tiers_shap_cheap
# none icin: --xai-method none --tag tiers_none
```
Çıktı: `results/deployments_tiers_<m>.json` → `results/15_v2/` altına taşı.
Süre: 5 dk × 3 yerleşim × 3 tur ≈ 45 dk/açıklayıcı.

**Pi'de açıklama hop'tan (5 s) uzunsa** kuyruk doğrusal büyür; istemci
gönderimi bitirdikten sonra cevapları bekler (koşu 5 dk yerine 15+ dk
sürebilir). Bu bir hata değil, makalenin bulgusudur: "her pencereyi
açıklamak bu düğümde sürdürülemez". `queuing_all` ve `age` metrikleri bunu
gösterir.

## 5. Pi seçici koşu (en önemli)

```bash
# Pi'de
WESAD_XAI_METHOD=shap_cheap WESAD_XAI_THRESHOLD=0.8 docker compose up -d --force-recreate
```
```powershell
python run_fog_experiment.py --server-host <PI_IP> --subject S11 --realtime --tag pi_sel80_full
```
Tam kayıt ~87 dk. Çıktı: `results/fog_pi_sel80_full.json` → `results/15_v2/`.
Ardından aynı şekilde `THRESHOLD=1.0` ile 600 tick'lik "hep açıkla" ve `none`
ile taban (bunlar §4'teki fog hücrelerinden de alınabilir).

## 6. Bitti ölçütü ve makale

1. `python -m tools.paper_numbers --v2 results/15_v2` — eksik hücre uyarısı olmamalı.
2. Fog/co-located açıklama süresi oranı **beş açıklayıcıda da aynı yönde**
   olmalı (v1'de işaret değiştiriyordu). Değiştiriyorsa yığın dışı bir etken var.
3. `paper/icaiic/main.tex` içindeki turuncu (`\vtwo`) cümleleri yeni sayılarla
   yeniden oku, gerekirse yeniden yaz; sonra `\renewcommand\vtwo[1]{#1}` ve
   v1 kutusu otomatik kalkar (`\ifvone`).
4. Seçici koşu Pi'den geldiyse `paper_numbers.py` içindeki `sel`/`alw`/`non`
   dosya yollarını `15_v2`'ye çevir.

## Ekte: v1 → v2 metrik değişiklikleri

- `send_lag` — gönderimin planlanan andan sapması. Kapalı döngüde akışın
  durma süresi, açık döngüde zamanlayıcı hassasiyeti.
- `age` — planlanan gönderim anından karara: coordinated omission'a karşı
  bağışık "response".
- `queuing_all` — tahmin üretmeyen ara paketler dahil kuyruk gecikmesi.
  Açıklama sürerken gelen paketlerin beklemesi yalnızca burada görünür.
- `pooled` (çok istemci) — tüm istemcilerin pencereleri tek dağılım; eski
  `metrics` istemci özetlerinin özeti (en kötü istemcinin p95'i).
- `raw` — pencere bazlı ham değerler (CDF, Wilcoxon için).
