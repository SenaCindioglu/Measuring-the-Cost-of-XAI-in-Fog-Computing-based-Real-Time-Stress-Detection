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
import threading
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

    def __init__(self, profiler, is_warmup, subject=None):
        self.profiler = profiler
        self.is_warmup = is_warmup
        self.subject = subject
        self.stage_times = {}
        self.true_label = None
        self.pred_label = None
        self.confidence = None
        self._total_start = time.perf_counter()
        # Pencere blogunun DISINDA (orn. tick basina) olculup bir asamaya
        # eklenen sureler. 'total' bunlari da icermek zorunda, yoksa asama
        # toplami total'i asar ve rapor tutarsiz olur.
        self._external_ms = 0.0

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed_ms

    def add_stage_ms(self, name, elapsed_ms):
        """Bu pencerenin disinda (orn. tick basina) olculmus sureyi bir asamaya ekler."""
        elapsed_ms = float(elapsed_ms)
        self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed_ms
        self._external_ms += elapsed_ms

    def set_result(self, true_label, pred_label, confidence):
        self.true_label = true_label
        self.pred_label = pred_label
        self.confidence = confidence

    def _finalize(self):
        total_ms = (time.perf_counter() - self._total_start) * 1000.0 + self._external_ms
        self.stage_times['total'] = total_ms
        self.profiler._commit(self)


class _ResourceSampler(threading.Thread):
    """
    CPU ve bellek kullanimini SABIT ARALIKLARLA, ayri bir iplikte orneklemeyi
    yapar.

    Neden ayri iplik: psutil'in cpu_percent'i iki cagri arasinda gecen sureye
    bakar. Pencere basina, aralik vermeden cagrildiginda arka arkaya gelen
    cagrilar arasinda neredeyse hic zaman gecmedigi icin surekli 0 okur.
    Sabit araliklarla orneklemek gercek yuku verir.
    """

    def __init__(self, proc, interval=0.1):
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.cpu = []
        self.rss = []
        self._stop_evt = threading.Event()

    def run(self):
        self.proc.cpu_percent(interval=None)  # ilk cagri referans noktasi
        while not self._stop_evt.wait(self.interval):
            try:
                self.cpu.append(self.proc.cpu_percent(interval=None))
                self.rss.append(self.proc.memory_info().rss / (1024 ** 2))
            except Exception:
                break

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)


class LatencyProfiler:
    def __init__(self, device_type='cpu', sample_interval=0.1):
        self.device_type = device_type
        self._records = []  # her pencere icin dict
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._sampler = None
        self._sample_interval = sample_interval

    # ------------------------------------------------------------------
    # Kaynak orneklemesi
    # ------------------------------------------------------------------

    def start_sampling(self):
        if self._proc is not None and self._sampler is None:
            self._sampler = _ResourceSampler(self._proc, self._sample_interval)
            self._sampler.start()

    def stop_sampling(self):
        if self._sampler is not None:
            self._sampler.stop()

    @contextmanager
    def record_window(self, is_warmup=False, subject=None):
        rec = _WindowRecorder(self, is_warmup, subject)
        try:
            yield rec
        finally:
            rec._finalize()

    def _commit(self, rec):
        entry = {
            'subject': rec.subject,
            'is_warmup': rec.is_warmup,
            'true_label': rec.true_label,
            'pred_label': rec.pred_label,
            'confidence': rec.confidence,
            **rec.stage_times,
        }
        if self._proc is not None:
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

        skip_keys = ('subject', 'is_warmup', 'true_label', 'pred_label',
                     'confidence', 'rss_mb', 'gpu_mem_mb')
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
                'p99_ms': float(np.percentile(values, 99)),
                'std_ms': float(values.std()),
                'min_ms': float(values.min()),
                'max_ms': float(values.max()),
            }

        # --- jitter: ardisik pencereler arasindaki sure degisimi ---
        # RFC 3550'deki tanimla ayni mantik: ardisik gecikmelerin mutlak farki.
        # std ile birlikte verilir cunku makalelerde ikisi de "jitter" diye gecer.
        tot = np.array([r['total'] for r in records if 'total' in r])
        if len(tot) > 1:
            summary['jitter'] = {
                'mean_abs_delta_ms': float(np.abs(np.diff(tot)).mean()),
                'std_ms': float(tot.std()),
                'p95_abs_delta_ms': float(np.percentile(np.abs(np.diff(tot)), 95)),
            }

        if self._sampler is not None and self._sampler.cpu:
            cpu_vals = np.array(self._sampler.cpu)
            summary['cpu_percent'] = {
                'mean': float(cpu_vals.mean()),
                'max': float(cpu_vals.max()),
                'p95': float(np.percentile(cpu_vals, 95)),
                'n_samples': int(len(cpu_vals)),
                'interval_s': self._sample_interval,
            }
        if 'rss_mb' in records[0]:
            rss_vals = np.array([r['rss_mb'] for r in records])
            summary['rss_mb'] = {'mean': float(rss_vals.mean()), 'max': float(rss_vals.max())}
        if 'gpu_mem_mb' in records[0]:
            gpu_vals = np.array([r['gpu_mem_mb'] for r in records])
            summary['gpu_mem_mb'] = {'mean': float(gpu_vals.mean()), 'max': float(gpu_vals.max())}

        summary['n_windows'] = len(records)
        summary['n_warmup_excluded'] = len(self._records) - len(records)

        # --- Streaming siniflandirma performansi (etiketi bilinen pencereler) ---
        pairs = [(r['true_label'], r['pred_label']) for r in records
                 if r.get('true_label') is not None and r['true_label'] >= 0
                 and r.get('pred_label') is not None]
        if pairs:
            y_true = np.array([a for a, _ in pairs])
            y_pred = np.array([b for _, b in pairs])
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_ = tp / (tp + fn) if (tp + fn) else 0.0
            summary['classification'] = {
                'n_labeled_windows': len(pairs),
                'accuracy': float((y_true == y_pred).mean()),
                'stress_precision': float(prec),
                'stress_recall': float(rec_),
                'stress_f1': float(2 * prec * rec_ / (prec + rec_)) if (prec + rec_) else 0.0,
                'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                'stress_ratio_true': float((y_true == 1).mean()),
                'stress_ratio_pred': float((y_pred == 1).mean()),
            }

        subjects = sorted({r['subject'] for r in records if r.get('subject')})
        if subjects:
            summary['subjects'] = subjects
        return summary

    def print_summary(self, exclude_warmup=True):
        s = self.summary(exclude_warmup)
        if not s:
            print("Kayit yok.")
            return
        print(f"\n[SUMMARY] Latency ozeti ({s['n_windows']} pencere, "
              f"{s['n_warmup_excluded']} warm-up haric tutuldu):")
        for stage in ['resampling', 'preprocessing', 'tensor_prep', 'inference', 'postprocessing', 'total']:
            if stage in s:
                st = s[stage]
                print(f"   {stage:>15}: mean={st['mean_ms']:.3f}ms  "
                      f"median={st['median_ms']:.3f}ms  p95={st['p95_ms']:.3f}ms  "
                      f"std={st['std_ms']:.3f}ms")
        if 'jitter' in s:
            j = s['jitter']
            print(f"   {'jitter':>15}: mean|delta|={j['mean_abs_delta_ms']:.3f}ms  "
                  f"std={j['std_ms']:.3f}ms  p95|delta|={j['p95_abs_delta_ms']:.3f}ms")
        if 'cpu_percent' in s:
            print(f"   {'CPU %':>15}: mean={s['cpu_percent']['mean']:.1f}  "
                  f"max={s['cpu_percent']['max']:.1f}  ({s['cpu_percent']['n_samples']} ornek)")
        if 'rss_mb' in s:
            print(f"   {'RAM (MB)':>15}: mean={s['rss_mb']['mean']:.1f}  max={s['rss_mb']['max']:.1f}")
        if 'gpu_mem_mb' in s:
            print(f"   {'GPU mem (MB)':>15}: mean={s['gpu_mem_mb']['mean']:.1f}  max={s['gpu_mem_mb']['max']:.1f}")

        if 'classification' in s:
            c = s['classification']
            cm = c['confusion']
            print(f"\n[SUMMARY] Streaming siniflandirma ({c['n_labeled_windows']} etiketli pencere):")
            print(f"   {'accuracy':>15}: {c['accuracy']:.4f}")
            print(f"   {'stress F1':>15}: {c['stress_f1']:.4f} "
                  f"(precision={c['stress_precision']:.4f}, recall={c['stress_recall']:.4f})")
            print(f"   {'confusion':>15}: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
            print(f"   {'stress orani':>15}: gercek={c['stress_ratio_true']:.3f}  "
                  f"tahmin={c['stress_ratio_pred']:.3f}")

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
