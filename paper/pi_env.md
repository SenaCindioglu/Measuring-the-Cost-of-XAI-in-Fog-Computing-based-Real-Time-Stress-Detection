# Raspberry Pi fog düğümü — ortam künyesi (Setup bölümü için)

Kayıt: 11 Ekim 2026. Deney JSON'u donanımı otomatik yazar; bu dosya yazılım
sürümlerini ve kurulum notunu belgeler (tekrar-üretilebilirlik).

## Donanım
- Raspberry Pi 4 Model B, Broadcom BCM2711 (Quad-core Cortex-A72 @1.5 GHz (Rev 1.1; JSON cpu_mhz_max=1500))
- **4 GB** LPDDR4 (+2 GB swap)
- 32 GB microSD (29 GB bölüm, ~20 GB boş)
- Ağ: Ethernet, LAN3, 192.168.1.12
- Aktif soğutma; idle ~46 °C

## İşletim sistemi
- Raspberry Pi OS (Debian 13 "trixie")
- Çekirdek: 6.18.34+rpt-rpi-v8, **aarch64 (64-bit)**, glibc 2.41

## Yazılım
| Paket | Sürüm |
|---|---|
| Python | 3.13.5 |
| torch | **2.14.1+cpu** |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| shap | 0.53.0 |
| psutil | 7.2.2 |

## Tekrar-üretilebilirlik notu (makaleye)
Linux/aarch64'te varsayılan `torch` tekerleği CUDA çalışma zamanını (cuDNN,
cuBLAS, nccl ... ~2.5 GB) **zorunlu bağımlılık** olarak çeker ve GPU'suz Pi'de
hem gereksizdir hem `/tmp` (tmpfs) taşmasına yol açar. CPU-only yapı gereklidir:

```
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt    # torch zaten kurulu; CUDA çekilmez
```

shap, aarch64 için hazır tekerlekle derlemesiz kuruldu — LIME yedeğine
gerek kalmadı; SHAP deneyleri de Pi'de koşabiliyor.
