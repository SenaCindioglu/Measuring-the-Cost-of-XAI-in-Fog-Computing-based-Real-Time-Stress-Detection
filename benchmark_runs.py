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
from streaming.replay_simulator import replay_subject, replay_subject_native
from profiling.latency_profiler import LatencyProfiler

STAGES = ['resampling', 'preprocessing', 'tensor_prep', 'inference', 'postprocessing', 'total']


def one_run(model, mean, std, cfg, subjects, tick_ms, warmup, norm_mode, pipeline):
    """Tek bir replay kosusu yapar, ozet istatistikleri dondurur."""
    fn = replay_subject_native if pipeline == 'native' else replay_subject
    profiler = LatencyProfiler(device_type=cfg.DEVICE.type)
    profiler.start_sampling()
    for subj in subjects:
        fn(subj, model, mean, std, cfg, profiler,
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
    parser.add_argument('--pipeline', nargs='+', choices=['prepared', 'native'],
                        default=[cfg.DEFAULT_PIPELINE],
                        help="Karsilastirilacak pipeline yollari; birden fazlaysa donusumlu kosulur")
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
    print(f"Is parcacigi: {args.threads}   Pipeline: {args.pipeline}")
    print("=" * 70)

    # Karsilastirilan yapilandirmalar: (pipeline, iplik). Donusumlu kosulur.
    configs = [(p, t) for p in args.pipeline for t in args.threads]
    runs = {c: [] for c in configs}

    for r in range(args.repeat):
        for c in configs:
            pipe, t = c
            torch.set_num_threads(t)
            s = one_run(model, mean, std, cfg, args.subject,
                        args.tick_ms, args.warmup_windows, args.norm_mode, pipe)
            runs[c].append(s)
            print(f"  tur {r+1}/{args.repeat} | {pipe}/{t} iplik -> "
                  f"medyan {s['total']['median_ms']:.3f} ms, "
                  f"p95 {s['total']['p95_ms']:.3f} ms, "
                  f"jitter {s['jitter']['mean_abs_delta_ms']:.3f} ms")

    # ---------------- ozet ----------------
    report = {'config': {
        'model': os.path.basename(args.model), 'architecture': arch,
        'subjects': args.subject, 'norm_mode': args.norm_mode,
        'tick_ms': args.tick_ms, 'warmup_windows': args.warmup_windows,
        'repeat': args.repeat, 'threads': args.threads, 'pipeline': args.pipeline,
        'device': str(cfg.DEVICE), 'window_size': cfg.WINDOW_SIZE, 'hop_size': cfg.HOP_SIZE,
    }, 'by_config': {}}

    for c in configs:
        pipe, t = c
        ss = runs[c]
        entry = {'pipeline': pipe, 'threads': t, 'n_runs': len(ss),
                 'n_windows_per_run': ss[0]['n_windows'], 'stages': {}}
        for st in STAGES:
            if st not in ss[0]:
                continue                       # 'resampling' yalnizca native yolda var
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
        report['by_config'][pipe + '_t' + str(t)] = entry

    print("\n" + "=" * 70)
    print(f"SONUC  ({args.repeat} tekrarin medyani +/- tekrarlar arasi yayilim)")
    print("=" * 70)
    print(f"{'yapilandirma':<18}{'p50':>9}{'p95':>9}{'p99':>9}{'jitter':>9}{'CPU%':>8}{'yayilim':>10}")
    for c in configs:
        pipe, t = c
        e = report['by_config'][pipe + '_t' + str(t)]
        p50 = e['stages']['total']['median_of_run_medians']
        p95 = e['stages']['total']['median_of_run_p95']['median']
        name = pipe + '/' + str(t) + ' iplik'
        print(f"{name:<18}{p50['median']:>9.3f}{p95:>9.3f}"
              f"{e['p99']['median']:>9.3f}{e['jitter']['median']:>9.3f}"
              f"{e.get('cpu_percent', {}).get('median', float('nan')):>8.1f}"
              f"{p50['range_pct']:>9.1f}%")

    print()
    print("ASAMA DOKUMU (kosu ici medyanlarin medyani, ms)")
    print(f"{'yapilandirma':<18}" + ''.join(f"{st[:9]:>12}" for st in STAGES))
    for c in configs:
        pipe, t = c
        e = report['by_config'][pipe + '_t' + str(t)]
        row = ''
        for st in STAGES:
            if st in e['stages']:
                row += f"{e['stages'][st]['median_of_run_medians']['median']:>12.3f}"
            else:
                row += f"{'-':>12}"
        name = pipe + '/' + str(t) + ' iplik'
        print(f"{name:<18}{row}")
    print("\nSutunlar: p50/p95/p99 kosu ICI dagilimdan (ms), 'yayilim' ise ayni")
    print("yapilandirmanin tekrarlari arasindaki fark. Yayilim buyukse tek kosunun")
    print("sonucu tek basina raporlanamaz.")

    if len(configs) > 1:
        meds = {c: report['by_config'][c[0] + '_t' + str(c[1])]['stages']['total']['median_of_run_medians']
                for c in configs}
        best = min(meds, key=lambda c: meds[c]['median'])
        worst_spread = max(meds[c]['range_pct'] for c in configs)
        gap = (max(meds[c]['median'] for c in configs) -
               min(meds[c]['median'] for c in configs))
        rel_gap = gap / meds[best]['median'] * 100
        print(f"\nEn dusuk medyan: {best[0]}/{best[1]} iplik ({meds[best]['median']:.3f} ms)")
        print(f"Yapilandirmalar arasi fark: %{rel_gap:.1f} | "
              f"tekrarlar arasi en buyuk yayilim: %{worst_spread:.1f}")
        if rel_gap < worst_spread:
            print(">> Fark, olcum gurultusunun altinda; yapilandirmalar ayirt edilemiyor.")

    path = os.path.join(cfg.RESULTS_DIR, f'benchmark_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[i] Rapor kaydedildi: {path}")


if __name__ == "__main__":
    main()
