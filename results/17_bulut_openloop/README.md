# 17_bulut_openloop — kontrollü cloud XAI ölçümleri (Mariam, 12 Ekim 2026)

Kaynak: `origin/deploy` dalı, commit `23a1521` ("Add controlled cloud XAI results").
Dosyalar oradan aynen kopyalandı; `deploy` dalındaki kod değişiklikleri
(`serving/server.py`, `serving/client.py`, Dockerfile) `main`'e birleştirilmedi.

Koşullar: DigitalOcean 1 vCPU / 2 GB (x86_64), Docker, Python 3.11.9,
torch 2.14.1+cpu, shap 0.51.0, numpy 2.2.6; sunucu commit `c39487e`.
İstemci açık döngülü (`--open-loop`), S11, 2000 tick, 15 ısınma + 184 ölçülen pencere.

| Durum | latency med | execution med | execution p95 | response med | response p95 |
|---|---|---|---|---|---|
| none (baseline_retry1) | 39.0 | 4.2 | 5.5 | 71.2 | 151.3 |
| LIME-cheap | 44.7 | 349.3 | 446.4 | 456.9 | 527.4 |
| SHAP-cheap | 50.9 | 271.3 | 361.0 | 375.2 | 458.2 |
| LIME | 49.3 | 706.2 | 856.7 | 812.3 | 984.5 |
| SHAP | 46.1 | 1094.0 | 1403.9 | 1154.4 | 1497.8 |

Uyarı: saat senkronu yalnız 2 örnekle yapılmış, RTT ~85–100 ms, belirsizlik
±50 ms. Tek yön latency bu yüzden güvenilir değil; response süresi istemci
tarafında ölçüldüğü için güvenilir.
