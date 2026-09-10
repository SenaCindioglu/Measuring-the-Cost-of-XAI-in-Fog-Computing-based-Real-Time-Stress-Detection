"""
profiling/latency_profiler.py
==============================
Streaming pipeline'inin her asamasini (preprocessing, inference, vs.)
ayri ayri zamanlayan, warm-up pencerelerini otomatik haric tutan, ve
mean/median/p95/std + sistem kaynak kullanimi (CPU%, RAM, GPU bellek)
raporlayan hafif bir profiler.

Bagimlilik: psutil (Colab'da genelde zaten kurulu; degilse
`pip install psutil`). psutil yoksa CPU/RAM olcumleri sessizce atlanir.
"""

import time
import json
import os
from contextlib import contextmanager

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _WindowRecorder:
    """Tek bir pencerenin asama surelerini toplayan yardimci obje."""

    def __init__(self, profiler, is_warmup):
        self.profiler = profiler
        self.is_warmup = is_warmup
        self.stage_times = {}
        self.true_label = None
        self.pred_label = None
        self.confidence = None
        self._total_start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed_ms

    def set_result(self, true_label, pred_label, confidence):
        self.true_label = true_label
        self.pred_label = pred_label
        self.confidence = confidence

    def _finalize(self):
        total_ms = (time.perf_counter() - self._total_start) * 1000.0
        self.stage_times['total'] = total_ms
        self.profiler._commit(self)


class LatencyProfiler:
    def __init__(self, device_type='cpu'):
        self.device_type = device_type
        self._records = []  # her pencere icin dict
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    @contextmanager
    def record_window(self, is_warmup=False):
        rec = _WindowRecorder(self, is_warmup)
        try:
            yield rec
        finally:
            rec._finalize()

    def _commit(self, rec):
        entry = {
            'is_warmup': rec.is_warmup,
            'true_label': rec.true_label,
            'pred_label': rec.pred_label,
            'confidence': rec.confidence,
            **rec.stage_times,
        }
        if self._proc is not None:
            entry['cpu_percent'] = self._proc.cpu_percent(interval=None)
            entry['rss_mb'] = self._proc.memory_info().rss / (1024 ** 2)
        if _HAS_TORCH and self.device_type == 'cuda' and torch.cuda.is_available():
            entry['gpu_mem_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
        self._records.append(entry)

    # ------------------------------------------------------------------
    # Ozet istatistikler
    # ------------------------------------------------------------------

    def summary(self, exclude_warmup=True):
        records = [r for r in self._records if not (exclude_warmup and r['is_warmup'])]
        if not records:
            return {}

        skip_keys = ('is_warmup', 'true_label', 'pred_label',
                     'confidence', 'cpu_percent', 'rss_mb', 'gpu_mem_mb')
        stage_names = [k for k in records[0].keys() if k not in skip_keys]

        summary = {}
        for stage in stage_names:
            values = np.array([r[stage] for r in records if stage in r])
            if len(values) == 0:
                continue
            summary[stage] = {
                'mean_ms': float(values.mean()),
                'median_ms': float(np.median(values)),
                'p95_ms': float(np.percentile(values, 95)),
                'std_ms': float(values.std()),
                'min_ms': float(values.min()),
                'max_ms': float(values.max()),
            }

        if 'cpu_percent' in records[0]:
            cpu_vals = np.array([r['cpu_percent'] for r in records])
            summary['cpu_percent'] = {'mean': float(cpu_vals.mean()), 'max': float(cpu_vals.max())}
        if 'rss_mb' in records[0]:
            rss_vals = np.array([r['rss_mb'] for r in records])
            summary['rss_mb'] = {'mean': float(rss_vals.mean()), 'max': float(rss_vals.max())}
        if 'gpu_mem_mb' in records[0]:
            gpu_vals = np.array([r['gpu_mem_mb'] for r in records])
            summary['gpu_mem_mb'] = {'mean': float(gpu_vals.mean()), 'max': float(gpu_vals.max())}

        summary['n_windows'] = len(records)
        summary['n_warmup_excluded'] = len(self._records) - len(records)
        return summary

    def print_summary(self, exclude_warmup=True):
        s = self.summary(exclude_warmup)
        if not s:
            print("Kayit yok.")
            return
        print(f"\n[SUMMARY] Latency ozeti ({s['n_windows']} pencere, "
              f"{s['n_warmup_excluded']} warm-up haric tutuldu):")
        for stage in ['preprocessing', 'tensor_prep', 'inference', 'postprocessing', 'total']:
            if stage in s:
                st = s[stage]
                print(f"   {stage:>15}: mean={st['mean_ms']:.3f}ms  "
                      f"median={st['median_ms']:.3f}ms  p95={st['p95_ms']:.3f}ms  "
                      f"std={st['std_ms']:.3f}ms")
        if 'cpu_percent' in s:
            print(f"   {'CPU %':>15}: mean={s['cpu_percent']['mean']:.1f}  max={s['cpu_percent']['max']:.1f}")
        if 'rss_mb' in s:
            print(f"   {'RAM (MB)':>15}: mean={s['rss_mb']['mean']:.1f}  max={s['rss_mb']['max']:.1f}")
        if 'gpu_mem_mb' in s:
            print(f"   {'GPU mem (MB)':>15}: mean={s['gpu_mem_mb']['mean']:.1f}  max={s['gpu_mem_mb']['max']:.1f}")

    # ------------------------------------------------------------------
    # Kayit / disa aktarma
    # ------------------------------------------------------------------

    def save_raw_records(self, path):
        """Her pencerenin ham kayitlarini JSON olarak kaydeder (Pareto analizi icin)."""
        with open(path, 'w') as f:
            json.dump(self._records, f, indent=2)
        print(f"[i] Ham kayitlar kaydedildi: {path}")

    def save_summary(self, path, exclude_warmup=True):
        with open(path, 'w') as f:
            json.dump(self.summary(exclude_warmup), f, indent=2)
        print(f"[i] Ozet istatistikler kaydedildi: {path}")
