"""
run_streaming_emulation.py
===========================
WESAD real-time streaming/replay emulasyonunu calistiran ana script.
Proje kok klasorunden calistirilir (import'lar buna gore kuruldu).

Kullanim ornegi:
    python run_streaming_emulation.py \
        --model /content/drive/MyDrive/WESAD/checkpoints/model_loso_s17.pth \
        --norm-stats /content/drive/MyDrive/WESAD/checkpoints/norm_stats_loso_s17.npz \
        --subject S17 \
        --tick-ms 500 \
        --warmup-windows 15

Birden fazla subject icin:
    python run_streaming_emulation.py --model ... --norm-stats ... --subject S2 S3 S4
"""

import argparse
import os

import torch

from config import config
from models.temporal_cnn import TemporalCNN
from data.normalization import load_stats
from streaming.replay_simulator import replay_subject
from profiling.latency_profiler import LatencyProfiler


def load_model(checkpoint_path, cfg):
    model = TemporalCNN(num_channels=cfg.NUM_CHANNELS, num_classes=2,
                         filters=cfg.CONV_FILTERS, kernel_size=cfg.KERNEL_SIZE,
                         conv_dropout=cfg.CONV_DROPOUT, fc_dropout=cfg.FC_DROPOUT)
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Egitilmis model checkpoint (.pth)")
    parser.add_argument('--norm-stats', required=True, help="Sabit normalizasyon istatistikleri (.npz)")
    parser.add_argument('--subject', nargs='+', required=True, help="Replay edilecek subject(ler)")
    parser.add_argument('--tick-ms', type=int, default=None, help="Tick suresi (ms)")
    parser.add_argument('--warmup-windows', type=int, default=None,
                         help="Haric tutulacak ilk pencere sayisi")
    parser.add_argument('--realtime', action='store_true',
                         help="Tick'ler arasina gercek time.sleep() ekle (demo amacli)")
    parser.add_argument('--tag', type=str, default='run', help="Cikti dosya isimlerine eklenecek etiket")
    args = parser.parse_args()

    cfg = config
    tick_ms = args.tick_ms or cfg.DEFAULT_TICK_MS
    warmup_windows = args.warmup_windows if args.warmup_windows is not None else cfg.DEFAULT_WARMUP_WINDOWS

    print("=" * 70)
    print("WESAD STREAMING / REPLAY EMULASYONU")
    print("=" * 70)
    print(f"Device: {cfg.DEVICE}")
    print(f"Model: {args.model}")
    print(f"Norm stats: {args.norm_stats}")
    print(f"Subjects: {args.subject}")
    print(f"Tick: {tick_ms}ms | Warmup windows: {warmup_windows} | Realtime sleep: {args.realtime}")
    print("=" * 70)

    model = load_model(args.model, cfg)
    mean, std = load_stats(args.norm_stats)

    profiler = LatencyProfiler(device_type=cfg.DEVICE.type)

    for subj in args.subject:
        replay_subject(subj, model, mean, std, cfg, profiler,
                        tick_ms=tick_ms, warmup_windows=warmup_windows,
                        realtime=args.realtime)

    profiler.print_summary(exclude_warmup=True)

    raw_path = os.path.join(cfg.RESULTS_DIR, f'streaming_raw_{args.tag}.json')
    summary_path = os.path.join(cfg.RESULTS_DIR, f'streaming_summary_{args.tag}.json')
    profiler.save_raw_records(raw_path)
    profiler.save_summary(summary_path)


if __name__ == "__main__":
    main()
