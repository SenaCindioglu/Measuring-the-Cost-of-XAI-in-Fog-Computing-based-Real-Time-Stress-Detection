"""
xai/benchmark.py
=================
Aciklama maliyetini BU makinelerde, tekrarli ve donusumlu olcer.

NEDEN YENIDEN OLCUYORUZ
------------------------
1) Tarama Colab'da, GPU'da kosuldu. Makalenin edge/fog tablosu ise bizim iki
   bilgisayarimiza ait ve ikisi de CPU. Iki donanimin sayilari ayni tabloya
   konulamaz: ilk karsilastirmada GPU, CPU'ya gore 4-6 kat hizli cikti.
2) Taramada her yapilandirma BIR kez olculdu. Ayni hesabi yapan
   yapilandirmalar arasinda %20'ye varan fark gorduk (ornek: LIME 1000/20
   icin 275, 217, 265, 237 ms). Tek olcumle "A, B'den hizli" demek bu
   gurultunun altinda kalir.

Depodaki olcum disiplini burada da gecerli (benchmark_runs.py ile ayni):
tekrarli ve DONUSUMLU kosu (tur 1: A, B / tur 2: A, B ...), boylece islemci
isinmasindan gelen kayma tum yapilandirmalara esit dagilir; ardisik kosulsa
en son olculen sistematik olarak yavas gorunurdu. Iki ayri dagilim ayri
raporlanir: kosu ICI yuzdelikler (p50/p95/p99) ve kosular ARASI yayilim.

Kullanim:
    python -m xai.benchmark --repeat 3 --windows 20
    python -m xai.benchmark --repeat 3 --windows 20 --subject S14 --threads 1
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config import config
from models.temporal_cnn import load_checkpoint
from xai.artifacts import load_artifacts
from xai import lime_ as lime_mod
from xai import shap_ as shap_mod


def _eval_windows(cfg, subject, n, seed=42):
    """Degerlendirme pencereleri: sunucunun gordugu girdi uzayinda (nedensel).

    Tarama script'indeki hazirlama fonksiyonunun kendisi kullanilir; kod
    kopyalanirsa iki taraf zamanla ayrisir.
    """
    from wesad_xai_ofat_sweep_colab import prepare_subject_data

    class _Cfg:
        TARGET_RATE = cfg.TARGET_RATE
        WINDOW_SIZE = cfg.WINDOW_SIZE
        STRIDE = cfg.HOP_SIZE
        VALID_LABELS = list(cfg.VALID_LABELS)
        BINARY = cfg.BINARY
        NORM_MODE = 'causal'

    X, y = prepare_subject_data(subject, cfg.DATA_PATH, _Cfg)
    if X is None:
        raise SystemExit(subject + " yuklenemedi")

    # Sinif dengeli altkume: stres pencereleri sinyal bakimindan farkli,
    # yalnizca birinden ornekleme gecikmeyi yanli olcebilir.
    rng = np.random.RandomState(seed)
    idx = []
    for cls in (1, 0):
        pool = np.where(y == cls)[0]
        take = min(n // 2, len(pool))
        if take:
            idx.append(rng.choice(pool, size=take, replace=False))
    chosen = np.concatenate(idx)
    return X[chosen]


def _stat(vals):
    a = np.asarray(vals, dtype=float)
    return {'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
            'min_ms': float(a.min()), 'max_ms': float(a.max()),
            'std_ms': float(a.std()), 'n': int(a.size)}


def _spread(medians):
    """Kosular arasi yayilim: olcumun guven araligi."""
    a = np.asarray(medians, dtype=float)
    med = float(np.median(a))
    return {'median_of_run_medians': med,
            'min': float(a.min()), 'max': float(a.max()),
            'range_pct': float(100.0 * (a.max() - a.min()) / med) if med else 0.0,
            'runs': [float(x) for x in a]}


def build_explainers(model, record, device='cpu', include_shap=True):
    """Olculecek yapilandirmalar.

    Iki maliyet noktasi kasitli: Faz D'de secici XAI'nin kazanci aciklama
    pahalilastikca buyuyor, o yuzden hem secilen hem ucuz nokta olculur.
    """
    out = [('lime', lime_mod.from_artifacts(model, record, device=device))]

    # LIME'in UCUZ noktasi CPU'da ornek sayisindan gecer, dilim sayisindan
    # degil. Olculdu (tek iplik, bu makine, dilim 20 sabit):
    #     100 ornek -> 205 ms | 250 -> 490 ms | 500 -> 955 ms
    # yani maliyet ornek sayisiyla dogrusal (~1.9 ms/ornek). Dilim sayisi
    # ise bedava: 500 ornekte dilim 20 ile 955 ms, dilim 5 ile 942 ms.
    # GPU'da tersi gorunuyordu (dilim 5 -> 67 ms, dilim 20 -> 173 ms), cunku
    # orada 500 ileri gecis toplu yapiliyor ve bolge kurulumu agirlik
    # kazaniyor. Dagitim CPU'da oldugu icin karar CPU olcumune gore verildi.
    cheap_lime = dict(record['lime'])
    cheap_lime.update(num_samples=250)
    out.append(('lime_ucuz', lime_mod.LimeExplainer(
        model, num_channels=record['window']['num_channels'],
        target_class=record.get('target_class', 1), device=device, **cheap_lime)))

    if include_shap:
        try:
            shap_full = shap_mod.from_artifacts(model, record, device=device)
            shap_cheap = shap_mod.from_artifacts(model, record, device=device, cheap=True)
            # Aciklayiciyi BURADA kur: shap importu tembel oldugu icin
            # eksik paket ancak ilk kullanimda patliyordu ve hata olcum
            # dongusunun ortasinda cikiyordu.
            _ = shap_full.explainer, shap_cheap.explainer
            out.append(('shap', shap_full))
            out.append(('shap_ucuz', shap_cheap))
        except ImportError as e:
            print("[!] SHAP atlandi: " + str(e).splitlines()[0])
            print("    (LIME olculmeye devam ediyor; SHAP icin: pip install shap)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='S11')
    ap.add_argument('--windows', type=int, default=20, help="olculecek pencere sayisi")
    ap.add_argument('--repeat', type=int, default=3, help="donusumlu tur sayisi")
    ap.add_argument('--threads', type=int, default=config.DEFAULT_THREADS)
    ap.add_argument('--no-shap', action='store_true')
    ap.add_argument('--tag', default='xai_cpu')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    record = load_artifacts()
    model, arch, _ = load_checkpoint(config.MODEL_PATH, 'cpu', fallback_cfg=config)
    windows = _eval_windows(config, args.subject, args.windows)
    explainers = build_explainers(model, record, 'cpu', include_shap=not args.no_shap)

    print("=" * 74)
    print("XAI MALIYETI -- CPU, tekrarli ve donusumlu")
    print("=" * 74)
    print("  denek %s | %d pencere | %d tur | %d iplik | %d yapilandirma"
          % (args.subject, len(windows), args.repeat, torch.get_num_threads(),
             len(explainers)))
    print("  pencere uzayi: nedensel normalizasyon (sunucudakiyle ayni)")
    print("=" * 74)

    # Isinma: ilk cagri her zaman yavas (tembel yuklemeler, BLAS kurulumu).
    for name, ex in explainers:
        ex.explain(windows[0])

    per_run = {name: [] for name, _ in explainers}
    for r in range(args.repeat):
        for name, ex in explainers:
            lat = []
            for w in windows:
                t0 = time.perf_counter()
                ex.explain(w)
                lat.append((time.perf_counter() - t0) * 1000.0)
            per_run[name].append(lat)
            print("  tur %d/%d | %-10s p50 %8.1f ms  p95 %8.1f ms"
                  % (r + 1, args.repeat, name, float(np.median(lat)),
                     float(np.percentile(lat, 95))))

    report = {'config': {'subject': args.subject, 'n_windows': len(windows),
                         'repeat': args.repeat, 'threads': torch.get_num_threads(),
                         'device': 'cpu', 'norm_mode': 'causal',
                         'model': os.path.basename(config.MODEL_PATH),
                         'model_params': arch['num_params']},
              'by_method': {}}

    for name, ex in explainers:
        runs = per_run[name]
        flat = [v for run in runs for v in run]
        report['by_method'][name] = {
            'explainer': ex.config(),
            'within_run': _stat(flat),
            'across_runs': _spread([float(np.median(run)) for run in runs]),
        }

    print("\n" + "=" * 74)
    print("SONUC  (kosu ici dagilim; parantez icinde kosular arasi yayilim)")
    print("=" * 74)
    print("  %-11s %9s %9s %9s %9s   %s" % ("yontem", "ort", "medyan", "p95", "p99",
                                            "yayilim"))
    for name in report['by_method']:
        w = report['by_method'][name]['within_run']
        a = report['by_method'][name]['across_runs']
        print("  %-11s %9.1f %9.1f %9.1f %9.1f   (%%%.1f)"
              % (name, w['mean_ms'], w['median_ms'], w['p95_ms'], w['p99_ms'],
                 a['range_pct']))

    hop_ms = config.HOP_DURATION * 1000
    print("\n  Pencere butcesi: %d ms (hop). Her pencerede aciklama uretilirse:" % hop_ms)
    for name in report['by_method']:
        med = report['by_method'][name]['within_run']['median_ms']
        print("    %-11s butcenin %%%.1f'i | tek dugum ~%.1f kullaniciya yetisir"
              % (name, 100 * med / hop_ms, hop_ms / med))

    path = os.path.join(config.RESULTS_DIR, 'benchmark_%s.json' % args.tag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("\n[i] Kaydedildi: " + path)


if __name__ == "__main__":
    main()
