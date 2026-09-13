"""
streaming/replay_simulator.py
==============================
Bir subject'in kaydedilmis WESAD sinyalini, sanki gercek zamanli bir
sensorden geliyormus gibi kronolojik ve kucuk parcalar (tick) halinde
RollingBuffer'a besler. Her pencere hazir oldugunda:
    preprocessing (normalizasyon) -> tensore cevirme -> model.forward()
    -> postprocessing
adimlarini LatencyProfiler ile ayri ayri olcer.

NOT: Bu bir "online replay / real-time emulasyon" deneyidir, gercek
klinik/canli bir sistem degildir. Ham veri zaten diskten okunur; sadece
modele bir kerede degil, zaman ekseninde parca parca verilir.
"""

import time

import numpy as np
import torch

from data.loader import (load_wesad_subject, load_wesad_subject_native,
                         NATIVE_STREAMS, LABEL_RATE)
from data.windowing import filter_valid_labels, labels_to_binary
from data.normalization import apply_normalization, RunningNormalizer
from streaming.buffer import RollingBuffer, MultiRateBuffer


def replay_subject(subject_id, model, mean, std, cfg, profiler,
                    tick_ms=500, warmup_windows=15, realtime=False,
                    norm_mode='fixed'):
    """
    Parameters
    ----------
    model : eval moduna alinmis, cfg.DEVICE'a tasinmis TemporalCNN
    mean, std : sabit normalizasyon istatistikleri (data/normalization.py)
    profiler : profiling.latency_profiler.LatencyProfiler ornegi
    tick_ms : her adimda buffer'a kac ms'lik veri eklenecegi
    warmup_windows : olcume dahil edilmeyecek ilk pencere sayisi
    realtime : True ise tick'ler arasina gercek time.sleep() eklenir
               (demo/gercek-zamanli gosterim icin; latency olcumunu
               etkilemez cunku sleep suresi ayri tutulur)
    norm_mode : 'fixed'   -> verilen sabit mean/std kullanilir (deployment gercegi)
                'running' -> her tick'te guncellenen NEDENSEL mean/std kullanilir;
                             offline egitimdeki subject-bazli z-score'un gelecege
                             sizmayan karsiligi. Guncelleme maliyeti pencerenin
                             'preprocessing' suresine eklenir.
    """
    if norm_mode not in ('fixed', 'running'):
        raise ValueError(f"Bilinmeyen norm_mode: {norm_mode} ('fixed' veya 'running')")

    print(f"\n>> Replay basliyor: {subject_id} (tick={tick_ms}ms, "
          f"warmup={warmup_windows} pencere, norm={norm_mode})")

    X_raw, y_raw = load_wesad_subject(subject_id, cfg.DATA_PATH, cfg.TARGET_RATE)
    if X_raw is None:
        print(f"   x {subject_id} yuklenemedi, atlaniyor.")
        return 0

    X_raw, y_raw = filter_valid_labels(X_raw, y_raw, cfg.VALID_LABELS)
    y_bin = labels_to_binary(y_raw) if cfg.BINARY else y_raw

    tick_samples = max(1, int(cfg.TARGET_RATE * tick_ms / 1000))
    buffer = RollingBuffer(cfg.NUM_CHANNELS, cfg.WINDOW_SIZE, cfg.HOP_SIZE)

    runner = RunningNormalizer(cfg.NUM_CHANNELS) if norm_mode == 'running' else None
    pending_norm_ms = 0.0  # tick basina odenen istatistik guncelleme maliyeti

    n_total = len(X_raw)
    window_count = 0

    for start in range(0, n_total, tick_samples):
        chunk = X_raw[start:start + tick_samples]
        if len(chunk) == 0:
            continue

        if runner is not None:
            _t0 = time.perf_counter()
            runner.update(chunk)
            pending_norm_ms += (time.perf_counter() - _t0) * 1000.0

        window_ready = buffer.push(chunk)

        if realtime:
            time.sleep(tick_ms / 1000.0)

        if not window_ready:
            continue

        raw_window = buffer.get_window()
        if raw_window is None:
            continue

        window_count += 1
        is_warmup = window_count <= warmup_windows

        # Bu pencereye karsilik gelen etiket (cogunluk oylamasi, sadece raporlama icin)
        end_idx = min(start + tick_samples, n_total)
        window_start_idx = max(0, end_idx - cfg.WINDOW_SIZE)
        true_label = int(np.bincount(y_bin[window_start_idx:end_idx].astype(int)).argmax()) \
            if end_idx > window_start_idx else -1

        win_mean, win_std = (mean, std) if runner is None else (runner.mean, runner.std)

        with profiler.record_window(is_warmup=is_warmup, subject=subject_id) as rec:
            # 'running' modda tick basina odenen guncelleme maliyetini de
            # preprocessing'e yaz ki latency raporu durustce tam maliyeti gostersin
            if pending_norm_ms:
                rec.add_stage_ms('preprocessing', pending_norm_ms)
                pending_norm_ms = 0.0

            with rec.stage('preprocessing'):
                # (T, C) -> (T, C), sabit ya da nedensel istatistiklerle
                norm_window = apply_normalization(raw_window, win_mean, win_std)

            with rec.stage('tensor_prep'):
                x = norm_window.transpose(1, 0)  # (C, T)
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(cfg.DEVICE)  # (1, C, T)

            with rec.stage('inference'):
                with torch.no_grad():
                    logits = model(x_tensor)
                    if cfg.DEVICE.type == 'cuda':
                        torch.cuda.synchronize()

            with rec.stage('postprocessing'):
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                confidence = float(probs.max())

            rec.set_result(true_label=true_label, pred_label=pred, confidence=confidence)

    print(f"   OK {subject_id}: {window_count} pencere islendi "
          f"({min(warmup_windows, window_count)} tanesi warm-up olarak haric tutuldu)")
    return window_count


def replay_subject_native(subject_id, model, mean, std, cfg, profiler,
                          tick_ms=500, warmup_windows=15, realtime=False,
                          norm_mode='running'):
    """
    GERCEKCI yol: sensor kendi ham hizinda veri gonderir, yeniden ornekleme
    sunucuda, HER PENCEREDE yapilir ve olculur.

    replay_subject'ten farki
    ------------------------
    replay_subject, 700 Hz gogus sinyalini daha replay baslamadan once bir
    kerede 64 Hz'e indiriyor. Bu, gercek bir sistemde bedava olmayan bir isi
    olcum disinda birakiyor. Burada her akis kendi hizinda birikir ve pencere
    hazir oldugunda 64 Hz'e indirilir; maliyet 'resampling' asamasinda gorunur.

    Etiketleme farki
    ----------------
    Eski yol, gecersiz etiketli ornekleri sinyalden ATIYOR ve kalanlari
    birlestiriyordu; bu, zamanda atlama demek ve gercek zamanli bir sistemde
    mumkun degil. Burada zaman kesintisiz akar.

    ONEMLI: gecersiz etiketli pencere de BASTAN SONA islenir (yeniden
    ornekleme, normalizasyon istatistigi guncellemesi, cikarim dahil) ve
    yalnizca siniflandirma raporundan cikarilir (true_label = -1). Gercek
    bir sunucu etiketi hic gormez, dolayisiyla "bu pencereyi atlayayim"
    diyemez. Onceki surum bu pencerelerde normalizasyon istatistigini
    guncellemiyordu; bu, sunucu yoluna gore daha dusuk dogruluk veriyordu.
    """
    if norm_mode not in ('fixed', 'running'):
        raise ValueError(f"Bilinmeyen norm_mode: {norm_mode} ('fixed' veya 'running')")

    print(f"\n>> Replay basliyor [native]: {subject_id} (tick={tick_ms}ms, "
          f"warmup={warmup_windows} pencere, norm={norm_mode})")

    streams, raw_labels = load_wesad_subject_native(subject_id, cfg.DATA_PATH)
    if streams is None:
        print(f"   x {subject_id} yuklenemedi, atlaniyor.")
        return 0

    buf = MultiRateBuffer(NATIVE_STREAMS, cfg.WINDOW_DURATION,
                          cfg.HOP_DURATION, cfg.WINDOW_SIZE)
    runner = RunningNormalizer(cfg.NUM_CHANNELS) if norm_mode == 'running' else None

    tick_s = tick_ms / 1000.0
    # Her akis icin tick basina kac ornek gonderilecegi ve okuma imleci
    per_tick = {name: int(round(rate * tick_s)) for name, rate, _ in NATIVE_STREAMS}
    cursor = {name: 0 for name, _, _ in NATIVE_STREAMS}
    n_ticks = min(len(streams[name]) // per_tick[name]
                  for name, _, _ in NATIVE_STREAMS if per_tick[name] > 0)

    window_count = 0
    skipped = 0

    for _ in range(n_ticks):
        chunks = {}
        for name, _, _ in NATIVE_STREAMS:
            k = per_tick[name]
            c = cursor[name]
            chunks[name] = streams[name][c:c + k]
            cursor[name] = c + k

        ready = buf.push(chunks, tick_s)

        if realtime:
            time.sleep(tick_s)

        if not ready:
            continue

        # Pencerenin kapsadigi zaman araligina karsilik gelen 700 Hz etiketleri.
        # Gecersizse pencere yine islenir, sadece true_label = -1 olur ve
        # profiler onu siniflandirma ozetine katmaz.
        end_i = int(round(buf.elapsed_s * LABEL_RATE))
        start_i = int(round((buf.elapsed_s - cfg.WINDOW_DURATION) * LABEL_RATE))
        span = raw_labels[max(0, start_i):min(end_i, len(raw_labels))]
        if len(span) == 0:
            continue
        majority = int(np.bincount(span.astype(int)).argmax())
        if majority in cfg.VALID_LABELS:
            true_label = 1 if majority == 2 else 0
        else:
            true_label = -1
            skipped += 1

        window_count += 1
        is_warmup = window_count <= warmup_windows

        with profiler.record_window(is_warmup=is_warmup, subject=subject_id) as rec:
            with rec.stage('resampling'):
                # HAM hizdan 64 Hz'e indirgeme -- eski yolda bu is replay'den
                # once bir kere yapiliyordu, burada her pencerede yapiliyor
                raw_window = buf.get_window()          # (640, 14)

            if raw_window is None:
                continue

            with rec.stage('preprocessing'):
                if runner is not None:
                    # Yalnizca yeni gelen kismi istatistige kat (cift saymamak icin)
                    new_part = raw_window if window_count == 1 else raw_window[-cfg.HOP_SIZE:]
                    runner.update(new_part)
                    w_mean, w_std = runner.mean, runner.std
                else:
                    w_mean, w_std = mean, std
                norm_window = apply_normalization(raw_window, w_mean, w_std)

            with rec.stage('tensor_prep'):
                x = norm_window.transpose(1, 0)
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(cfg.DEVICE)

            with rec.stage('inference'):
                with torch.no_grad():
                    logits = model(x_tensor)
                    if cfg.DEVICE.type == 'cuda':
                        torch.cuda.synchronize()

            with rec.stage('postprocessing'):
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                confidence = float(probs.max())

            rec.set_result(true_label=true_label, pred_label=pred, confidence=confidence)

    print(f"   OK {subject_id}: {window_count} pencere islendi "
          f"({skipped} tanesi gecersiz etiketli, islendi ama raporlanmadi; "
          f"{min(warmup_windows, window_count)} warm-up haric)")
    return window_count
