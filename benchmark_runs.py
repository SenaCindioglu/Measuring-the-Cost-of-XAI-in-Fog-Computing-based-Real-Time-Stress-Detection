"""
benchmark_runs.py
==================
Ayni replay'i N kere tekrarlayip gecikme sayilarini BELIRSIZLIGIYLE BIRLIKTE
raporlar.

NEDEN GEREKLI
-------------
Tek bir kosunun sonucu bu makinede guvenilir degil. Olculen davranis:

  - Ayni yapilandirma arka arkaya kosuldugunda degerler yukari kayiyor
    (islemci frekans olceklemesi / isinma).
  - Is parcacigi sayisi 1, 2, 4 arasinda donusumlu kosuldugunda aradaki fark
    kosudan kosuya olan yayilimin altinda kaliyor, yani ayirt edilemiyor.

Bu yuzden "toplam gecikme 3.4 ms" gibi tek bir sayi yaniltici olur. Dogru
raporlama: N tekrarin MEDYANI + tekrarlar arasi yayilim.

IKI AYRI DAGILIM VAR, KARISTIRILMAMALI
--------------------------------------
  1) Kosu ICI dagilim  : bir kosudaki 1000+ pencerenin p50/p95/p99'u.
                          Gercek zamanli sistemin en kotu durum davranisini
                          bu belirler. Rapora giren asil sayi budur.
  2) Kosu ARASI yayilim: ayni deneyi tekrarlayinca medyanin ne kadar oynadigi.
                          Olcumun guven araligini bu belirler.

Kullanim:
    python benchmark_runs.py --subject S11 S14 --repeat 5 --tag bench_running
    python benchmark_runs.py --subject S11 --repeat 5 --threads 1 2 4 --tag thread_sweep

Birden fazla --threads degeri verilirse kosular DONUSUMLU sirada yapilir
(tur 1: t1, t2, t4 / tur 2: t1, t2, t4 ...). Boylece isinma kaymasi tum
yapilandirmalara esit dagilir ve aralarindaki fark gercekten karsilastirilabilir.
"""

import argparse
import json
import os
import statistics

import numpy as np
import torch

from config import config
from models.temporal_cnn import load_checkpoint
from data.normalization import ensure_fixed_stats
from streaming.replay_simulator import replay_subject
from profiling.latency_profiler import LatencyProfiler

STAGES = ['preprocessing', 'tensor_prep', 'inference', 'postprocessing', 'total']


def one_run(model, mean, std, cfg, subjects, tick_ms, warmup, norm_mode):
    """Tek bir replay kosusu yapar, ozet istatistikleri dondurur."""
    profiler = LatencyProfiler(device_type=cfg.DEVICE.type)
    profiler.start_sampling()
    for subj in subjects:
        replay_subject(subj, model, mean, std, cfg, profiler,
                       tick_ms=tick_ms, warmup_windows=warmup,
                       realtime=False, norm_mode=norm_mode)
    profiler.stop_sampling()
    return profiler.summary(exclude_warmup=True)


def spread(values):
    """Tekrarlar arasi yayilim olculeri."""
    v = np.array(values, dtype=float)
    med = float(np.median(v))
    return {
        'median': med,
        'min': float(v.min()),
        'max': float(v.max()),
        'range_pct': float((v.max() - v.min()) / v.mean() * 100.0) if v.mean() else 0.0,
        'cv_pct': float(v.std(ddof=1) / v.mean() * 100.0) if len(v) > 1 and v.mean() else 0.0,
        'runs': [float(x) for x in v],
    }


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--subject', nargs='+', required=True)
    parser.add_argument('--repeat', type=int, default=5, help="Tekrar sayisi (tur)")
    parser.add_argument('--threads', nargs='+', type=int, default=[cfg.DEFAULT_THREADS],
                        help="Karsilastirilacak is parcacigi sayilari; birden fazlaysa donusumlu kosulur")
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE)
    parser.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS)
    parser.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS)
    parser.add_argument('--tag', default='bench')
    args = parser.parse_args()

    model, arch, meta = load_checkpoint(args.model, cfg.DEVICE, fallback_cfg=cfg)

    if args.norm_mode == 'fixed':
        tag = os.path.splitext(os.path.basename(args.model))[0]
        cache = os.path.join(cfg.CHECKPOINT_DIR, f'norm_stats_{tag}.npz')
        mean, std = ensure_fixed_stats(cache, meta.get('train_subjects') or cfg.TRAIN_SUBJECTS, cfg)
    else:
        mean, std = None, None

    print("=" * 70)
    print("TEKRARLI GECIKME OLCUMU")
    print("=" * 70)
    print(f"Model      : {os.path.basename(args.model)} "
          f"(depth={arch['depth']}, filters={arch['filters']}, k={arch['kernel_size']})")
    print(f"Subjects   : {args.subject}   Normalizasyon: {args.norm_mode}")
    print(f"Tekrar     : {args.repeat} tur x {len(args.threads)} yapilandirma "
          f"= {args.repeat * len(args.threads)} kosu")
    print(f"Is parcacigi: {args.threads}" +
          ("   (donusumlu sirada)" if len(args.threads) > 1 else ""))
    print("=" * 70)

    # {threads: [summary, ...]}
    runs = {t: [] for t in args.threads}

    for r in range(args.repeat):
        for t in args.threads:                      # donusumlu: isinma kaymasi esit dagilir
            torch.set_num_threads(t)
            s = one_run(model, mean, std, cfg, args.subject,
                        args.tick_ms, args.warmup_windows, args.norm_mode)
            runs[t].append(s)
            print(f"  tur {r+1}/{args.repeat} | {t} iplik -> "
                  f"medyan {s['total']['median_ms']:.3f} ms, "
                  f"p95 {s['total']['p95_ms']:.3f} ms, "
                  f"jitter {s['jitter']['mean_abs_delta_ms']:.3f} ms")

    # ---------------- ozet ----------------
    report = {'config': {
        'model': os.path.basename(args.model), 'architecture': arch,
        'subjects': args.subject, 'norm_mode': args.norm_mode,
        'tick_ms': args.tick_ms, 'warmup_windows': args.warmup_windows,
        'repeat': args.repeat, 'threads': args.threads,
        'device': str(cfg.DEVICE), 'window_size': cfg.WINDOW_SIZE, 'hop_size': cfg.HOP_SIZE,
    }, 'by_threads': {}}

    for t in args.threads:
        ss = runs[t]
        entry = {'n_runs': len(ss), 'n_windows_per_run': ss[0]['n_windows'], 'stages': {}}
        for st in STAGES:
            entry['stages'][st] = {
                'median_of_run_medians': spread([s[st]['median_ms'] for s in ss]),
                'median_of_run_p95':     spread([s[st]['p95_ms'] for s in ss]),
            }
        entry['jitter'] = spread([s['jitter']['mean_abs_delta_ms'] for s in ss])
        entry['p99'] = spread([s['total']['p99_ms'] for s in ss])
        if 'cpu_percent' in ss[0]:
            entry['cpu_percent'] = spread([s['cpu_percent']['mean'] for s in ss])
        if 'classification' in ss[0]:
            entry['classification'] = ss[0]['classification']   # deterministik, tek kopya yeter
        report['by_threads'][str(t)] = entry

    print("\n" + "=" * 70)
    print(f"SONUC  ({args.repeat} tekrarin medyani +/- tekrarlar arasi yayilim)")
    print("=" * 70)
    hdr = f"{'iplik':>6}{'p50':>9}{'p95':>9}{'p99':>9}{'jitter':>9}{'CPU%':>8}{'yayilim':>10}"
    print(hdr)
    for t in args.threads:
        e = report['by_threads'][str(t)]
        p50 = e['stages']['total']['median_of_run_medians']
        p95 = e['stages']['total']['median_of_run_p95']['median']
        print(f"{t:>6}{p50['median']:>9.3f}{p95:>9.3f}{e['p99']['median']:>9.3f}"
              f"{e['jitter']['median']:>9.3f}"
              f"{e.get('cpu_percent', {}).get('median', float('nan')):>8.1f}"
              f"{p50['range_pct']:>9.1f}%")
    print("\nSutunlar: p50/p95/p99 kosu ICI dagilimdan (ms), 'yayilim' ise ayni")
    print("yapilandirmanin tekrarlari arasindaki fark. Yayilim buyukse tek kosunun")
    print("sonucu tek basina raporlanamaz.")

    if len(args.threads) > 1:
        meds = {t: report['by_threads'][str(t)]['stages']['total']['median_of_run_medians']
                for t in args.threads}
        best = min(meds, key=lambda t: meds[t]['median'])
        worst_spread = max(meds[t]['range_pct'] for t in args.threads)
        gap = (max(meds[t]['median'] for t in args.threads) -
               min(meds[t]['median'] for t in args.threads))
        rel_gap = gap / meds[best]['median'] * 100
        print(f"\nEn dusuk medyan: {best} iplik ({meds[best]['median']:.3f} ms)")
        print(f"Yapilandirmalar arasi fark: %{rel_gap:.1f} | "
              f"tekrarlar arasi en buyuk yayilim: %{worst_spread:.1f}")
        if rel_gap < worst_spread:
            print(">> Fark, olcum gurultusunun altinda. Is parcacigi sayisinin bu "
                  "is yukunde belirleyici olmadigi sonucuna varilir.")

    path = os.path.join(cfg.RESULTS_DIR, f'benchmark_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[i] Rapor kaydedildi: {path}")


if __name__ == "__main__":
    main()
