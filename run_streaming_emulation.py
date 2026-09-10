"""
run_streaming_emulation.py
===========================
WESAD real-time streaming/replay emulasyonunu calistiran ana script.
Proje kok klasorunden calistirilir (import'lar buna gore kuruldu).

Secili offline model: wesad_best_model_f32-128_k9.pt
    depth=2, filters=[32, 128], kernel=9, seed=0, val F1=0.9386
    train subjects (13): S2 S16 S7 S10 S4 S3 S17 S6 S9 S13 S15 S5 S8
    val   subjects (2) : S11, S14   <-- replay icin dogru secim bunlar

En kisa kullanim (varsayilanlar config.py'den gelir):
    python run_streaming_emulation.py --subject S11 --tag s11_fixed

Tam hali:
    python run_streaming_emulation.py \
        --model wesad_best_model_f32-128_k9.pt \
        --subject S11 S14 \
        --norm-mode fixed \
        --tick-ms 500 \
        --warmup-windows 15 \
        --tag val_subjects
"""

import argparse
import json
import os

from config import config
from models.temporal_cnn import load_checkpoint
from data.normalization import load_stats, ensure_fixed_stats
from streaming.replay_simulator import replay_subject
from profiling.latency_profiler import LatencyProfiler


def resolve_norm_stats(args, meta, cfg):
    """
    Sabit normalizasyon istatistiklerini bulur ya da uretir.

    Oncelik sirasi:
      1) --norm-stats ile acikca verilen dosya
      2) checkpoint meta'sindaki train_subjects'ten hesaplanip cache'lenen dosya
      3) config.TRAIN_SUBJECTS'ten hesaplanip cache'lenen dosya
    """
    if args.norm_stats:
        return load_stats(args.norm_stats)

    train_subjects = meta.get('train_subjects') or cfg.TRAIN_SUBJECTS
    tag = os.path.splitext(os.path.basename(args.model))[0]
    cache_path = os.path.join(cfg.CHECKPOINT_DIR, f'norm_stats_{tag}.npz')
    return ensure_fixed_stats(cache_path, train_subjects, cfg)


def warn_on_leakage(subjects, meta):
    """Replay edilen subject modelin egitim setindeyse uyar."""
    train_subjects = set(meta.get('train_subjects') or [])
    val_subjects = set(meta.get('val_subjects') or [])
    if not train_subjects:
        return
    leaked = [s for s in subjects if s in train_subjects]
    if leaked:
        print(f"\n[!] UYARI: {', '.join(leaked)} bu modelin EGITIM setinde. "
              f"Latency olcumu gecerli, ama accuracy iyimser cikar.")
        if val_subjects:
            print(f"[!] Gorulmemis subject icin: --subject {' '.join(sorted(val_subjects))}")


def main():
    cfg = config

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=cfg.MODEL_PATH,
                        help=f"Egitilmis model checkpoint (varsayilan: {os.path.basename(cfg.MODEL_PATH)})")
    parser.add_argument('--norm-stats', default=None,
                        help="Sabit normalizasyon istatistikleri (.npz). "
                             "Verilmezse checkpoint'in train subjectlerinden hesaplanip cache'lenir.")
    parser.add_argument('--subject', nargs='+', required=True, help="Replay edilecek subject(ler)")
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE,
                        help="fixed: sabit train istatistikleri | running: nedensel online z-score")
    parser.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS, help="Tick suresi (ms)")
    parser.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS,
                        help="Haric tutulacak ilk pencere sayisi")
    parser.add_argument('--realtime', action='store_true',
                        help="Tick'ler arasina gercek time.sleep() ekle (demo amacli)")
    parser.add_argument('--tag', type=str, default='run', help="Cikti dosya isimlerine eklenecek etiket")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Checkpoint bulunamadi: {args.model}")

    model, arch, meta = load_checkpoint(args.model, cfg.DEVICE, fallback_cfg=cfg)

    print("=" * 70)
    print("WESAD STREAMING / REPLAY EMULASYONU")
    print("=" * 70)
    print(f"Device       : {cfg.DEVICE}")
    print(f"Model        : {args.model}")
    print(f"Mimari       : depth={arch['depth']} filters={arch['filters']} "
          f"k={arch['kernel_size']} -> {arch['num_params']:,} parametre")
    if meta.get('best_val_f1') is not None:
        print(f"Offline val  : F1={meta['best_val_f1']:.4f} on {meta.get('val_subjects')}")
    print(f"Subjects     : {args.subject}")
    print(f"Normalizasyon: {args.norm_mode}")
    print(f"Tick         : {args.tick_ms}ms | Warmup: {args.warmup_windows} pencere | "
          f"Realtime sleep: {args.realtime}")
    print("=" * 70)

    warn_on_leakage(args.subject, meta)

    # 'running' modda sabit istatistiklere hic ihtiyac yok -> 13 subject'i
    # bosuna yuklemeyelim.
    if args.norm_mode == 'fixed':
        mean, std = resolve_norm_stats(args, meta, cfg)
    else:
        mean, std = None, None

    profiler = LatencyProfiler(device_type=cfg.DEVICE.type)

    for subj in args.subject:
        replay_subject(subj, model, mean, std, cfg, profiler,
                       tick_ms=args.tick_ms, warmup_windows=args.warmup_windows,
                       realtime=args.realtime, norm_mode=args.norm_mode)

    profiler.print_summary(exclude_warmup=True)

    # Calistirma kosullarini ozete gom ki JSON tek basina yorumlanabilsin
    summary = profiler.summary(exclude_warmup=True)
    summary['run_config'] = {
        'model': os.path.basename(args.model),
        'architecture': arch,
        'checkpoint_meta': meta,
        'subjects': args.subject,
        'norm_mode': args.norm_mode,
        'tick_ms': args.tick_ms,
        'warmup_windows': args.warmup_windows,
        'device': str(cfg.DEVICE),
        'window_size': cfg.WINDOW_SIZE,
        'hop_size': cfg.HOP_SIZE,
        'window_latency_s': cfg.WINDOW_DURATION,
        'hop_latency_s': cfg.HOP_DURATION,
        'seed': cfg.SEED,
    }

    raw_path = os.path.join(cfg.RESULTS_DIR, f'streaming_raw_{args.tag}.json')
    summary_path = os.path.join(cfg.RESULTS_DIR, f'streaming_summary_{args.tag}.json')
    profiler.save_raw_records(raw_path)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[i] Ozet istatistikler kaydedildi: {summary_path}")

    print(f"\n[NOT] Yukaridaki sureler yalnizca HESAPLAMA gecikmesidir. "
          f"Verinin toplanmasi ayrica {cfg.WINDOW_DURATION:.0f}s pencere + "
          f"{cfg.HOP_DURATION:.0f}s hop gerektirir; raporda ikisini ayri belirt.")


if __name__ == "__main__":
    main()
