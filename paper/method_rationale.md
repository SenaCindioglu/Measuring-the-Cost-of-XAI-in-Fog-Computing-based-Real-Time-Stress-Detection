# Yöntem gerekçeleri — makale metni (Setup / Measurement / Discussion)

Kime: ICAIIC hakemleri ve okur. Aşağıdaki İngilizce paragraflar doğrudan
makaleye girecek; her blokun üstünde hangi bölüme ait olduğu yazılı. Sayılar
11 Ekim ön (duman) koşusundan; 185-pencerelik asıl koşu bunları netleştirecek,
büyüklük mertebesi değişmez.

---

## IV. Experimental Setup — Windowing and update cadence

> The model consumes a fixed 10 s window (the length it was trained with) and a
> new prediction is emitted every 5 s (the update hop). We distinguish the
> *window latency* (the 10 s of signal a decision integrates) from the
> *computation latency* (the time to process one window) and report them
> separately. The 5 s hop is adequate for stress, whose physiological
> correlates (electrodermal activity, heart rate, respiration) vary over tens of
> seconds rather than sub-second intervals; a faster hop would add computation
> without finer temporal information. This differs from the emotion-recognition
> reference [ICACT], which uses a 3 s window and a 1 s hop; the two cadences are
> therefore not directly comparable and are stated explicitly. The hop is a
> design parameter, not a hardware limit: it could be shortened at the cost of a
> tighter per-second compute budget, which matters once explanation is enabled.

## IV. Experimental Setup — Fog worker and software stack

> The fog worker is a Raspberry Pi 4 Model B (Broadcom BCM2711, quad-core
> Cortex-A72 @1.8 GHz, 4 GB LPDDR4), running 64-bit Raspberry Pi OS (Debian 13,
> kernel 6.18, aarch64) and PyTorch 2.14.1 (CPU build). On aarch64 the default
> PyTorch wheel declares the CUDA runtime as a hard dependency; the CPU-only
> build is required on a GPU-less node. The co-located baseline runs the
> identical server on the client host (PyTorch 2.8.0, CPU). Because the Pi/host
> ratio reported below is of order 2–3x, we present the Pi results as a
> demonstration of explained inference on a resource-constrained node rather
> than as a stack-matched micro-benchmark.

## IV. Experimental Setup — Measurement protocol and sample size

> Each configuration is measured in a single realtime run of ~185 windows
> (~17 min of streaming); the first 15 windows are discarded as warm-up. This
> matches the per-experiment sample of the reference (197 windows) and is
> dictated by the tail statistics: stable p95/p99 estimates require hundreds of
> windows, so runs of a few dozen windows are not used for tail reporting. The
> unit of repetition is the run, not the window: windows within a run are not
> independent (warm-up drift), so pooling them would overstate confidence.
> Batch size is one (a user sends a single window); measurements are single-
> threaded for stability.

## VI. Discussion — Cost of per-window explanation on the fog node

> On the Raspberry Pi worker, a single prediction costs ~14 ms (inference only).
> Adding an explanation to every window raises the per-window cost to a median
> 744 ms (p95 830 ms) for region-based LIME (250 samples, 20 segments) and
> 1181 ms (p95 1197 ms) for SHAP (GradientExplainer, 50 samples, 100-sample
> background) — 15% and 24% of the 5 s hop budget, respectively (184-window
> realtime runs on S11). Contrary to the expectation that a
> resource-constrained node would make per-window explanation infeasible, both
> explainers run comfortably within budget for a single user, and the Pi is only
> ~2–3x slower than the co-located host, not an order of magnitude. The value of
> confidence-thresholded (selective) explanation is therefore not to make the
> impossible possible for one user, but to extend multi-user capacity: at these
> costs a single worker sustains roughly 5 s / 1.18 s ~ 4 concurrent SHAP users
> or 5 s / 0.77 s ~ 6 LIME users, and explaining only low-confidence windows
> multiplies this figure. This correction follows from measurement rather than
> estimation, which we regard as a strength. (Multi-worker validation is left to
> future work.)

## VI. Discussion — Selective (confidence-thresholded) explanation on the Pi

> We measured selective explanation directly on the Pi over a 504-window run
> spanning baseline and stress (S11, ~43 min, same slice for both explainers).
> At a confidence threshold of 0.80, 20.0% of windows fell below the threshold
> and were explained (101 of 504, identical for LIME and SHAP since the
> explained fraction is a property of the model's confidence, not of the
> explainer). The mean per-window execution dropped to 172 ms for LIME and
> 254 ms for SHAP (medians ~22 ms, since 80% of windows are not explained),
> against 755 ms and 1193 ms when every window is explained — a 4.4x and 4.7x
> reduction in mean cost on the worker. The reduction scales with how often the
> model is uncertain: over the longer full recording (1030 windows, including
> the easier meditation and amusement segments) the explained fraction is 14.1%,
> which lowers the mean further. Because selective LIME is both the cheapest
> explained configuration (172 ms) and, at equal cost, the most faithful and
> most stable explainer, it is the recommended operating point for real-time
> explained inference on a resource-constrained fog node. Selective explanation
> preserves explanations where the model is uncertain while recovering most of
> the headroom the worker needs to serve additional users.

---

## Özet — makaleye giren beş gerekçe (Türkçe hatırlatma)

1. **Pencere 10 s / hop 5 s:** pencere modelden sabit; hop tasarım tercihi,
   stres yavaş değiştiği için 5 s yeterli. ICACT 3 s / 1 s'ten farklı, ayrı
   raporlanır.
2. **Window latency ≠ computation latency:** ikisi ayrı satır.
3. **Koşu uzunluğu ~185 pencere:** ICACT'in 197'sine denk + p95/p99 için
   yüzlerce pencere şart (P2.1).
4. **Tekrar birimi koşu, pencere değil:** ısınma kayması bağımsızlığı bozar.
5. **Pi CPU-only torch + "gösterim" çerçevesi:** oran ~2–3× olduğu için kesin
   benchmark değil, kaynak-kısıtlı düğümde gösterim. Ana bulgu: tek kullanıcıda
   XAI mümkün, kazanç çok-kullanıcı kapasitesinde.
