"""
tools/profile_table.py
=======================
FAZ B1: dagilmis profil sonuclarini tek tabloya toplar.

NE YAPAR
--------
Uc ayri kaynaktaki asama bazli sureleri birlestirip makaleye girecek
tek tabloyu uretir. Yeni olcum YAPMAZ, yalnizca derler:

  01_streaming/streaming_summary_native_fixed.json
      tek surecli replay, 1175 pencere, asama bazli taban profil
  02_benchmark/benchmark_thread_sweep.json
      is parcacigi 1/2/4, 4 tur donusumlu -- kosular arasi yayilim burada
  02_benchmark/benchmark_pipeline_cmp.json
      prepared / native yol karsilastirmasi
  04_fog/fog_xai_none.json
      ayni pipeline'in fog dugumunde, uctan uca olculmus hali

NEDEN TEK TABLO GEREKIYOR
-------------------------
Ayni asamanin (orn. inference) degeri dort dosyada farkli cikiyor ve
sebepleri farkli: hangi makine, tek surec mi istemci-sunucu mu, kac
tekrar. Dagilmis halde birakilirsa makalede yanlis satir alintilanir.
Tablo her sayinin YANINDA kaynagini tasir.

Kullanim:
    python -m tools.profile_table
    python -m tools.profile_table --tag b1
"""

import argparse
import json
import os

from config import config

STAGES = ['resampling', 'preprocessing', 'tensor_prep', 'inference',
          'postprocessing', 'total']


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def from_streaming(d):
    """Tek surecli replay ozeti: dogrudan asama sozlukleri."""
    if not d:
        return None
    out = {}
    for s in STAGES:
        if s in d:
            out[s] = {'median_ms': d[s].get('median_ms'),
                      'p95_ms': d[s].get('p95_ms'),
                      'yayilim_pct': None}
    return out


def from_benchmark(d, key):
    """Tekrarli benchmark: medyanlarin medyani + kosular arasi yayilim."""
    if not d:
        return None
    block = (d.get('by_threads') or d.get('by_config') or {}).get(key)
    if not block:
        return None
    out = {}
    for s, v in (block.get('stages') or {}).items():
        med = v.get('median_of_run_medians') or {}
        p95 = v.get('median_of_run_p95') or {}
        out[s] = {'median_ms': med.get('median'), 'p95_ms': p95.get('median'),
                  'yayilim_pct': med.get('range_pct')}
    return out


def from_fog(d):
    """Fog kosusu: sunucu ici asama dokumu + uctan uca response."""
    if not d:
        return None
    s = d.get('summary', d)
    out = {}
    for name, v in (s.get('server_stages') or {}).items():
        out[name] = {'median_ms': v.get('median_ms'), 'p95_ms': v.get('p95_ms'),
                     'yayilim_pct': None}
    ex = s['metrics']['execution']
    out['total'] = {'median_ms': ex['median_ms'], 'p95_ms': ex['p95_ms'],
                    'yayilim_pct': None}
    return out


def build(cfg):
    r = cfg.RESULTS_DIR
    cols = [
        ('tek surec (native)',
         from_streaming(_load(os.path.join(r, '01_streaming',
                                           'streaming_summary_native_fixed.json'))),
         '01_streaming/streaming_summary_native_fixed.json',
         'tek surec, ag yok, 1175 pencere, tek kosu'),
        ('tekrarli (native, 1 iplik)',
         from_benchmark(_load(os.path.join(r, '02_benchmark',
                                           'benchmark_pipeline_cmp.json')), 'native_t1'),
         '02_benchmark/benchmark_pipeline_cmp.json',
         '4 tur donusumlu, yayilim tasiyan tek sutun'),
        ('tekrarli (prepared, 1 iplik)',
         from_benchmark(_load(os.path.join(r, '02_benchmark',
                                           'benchmark_pipeline_cmp.json')), 'prepared_t1'),
         '02_benchmark/benchmark_pipeline_cmp.json',
         'yeniden ornekleme olcum DISINDA'),
        ('fog dugumu (uctan uca)',
         from_fog(_load(os.path.join(r, '04_fog', 'fog_xai_none.json'))),
         '04_fog/fog_xai_none.json',
         'istemci-sunucu ayri, gercek ag, 44 pencere'),
    ]
    return [c for c in cols if c[1]]


def print_table(cols):
    print("=" * 96)
    print("B1 - ASAMA BAZLI PROFIL (medyan ms; parantezde kosular arasi yayilim)")
    print("=" * 96)
    head = f"{'asama':>16}"
    for name, _, _, _ in cols:
        head += f"{name:>29}"
    print(head)
    print("-" * 96)
    for s in STAGES:
        row = f"{s:>16}"
        for _, data, _, _ in cols:
            v = (data or {}).get(s)
            if not v or v.get('median_ms') is None:
                row += f"{'-':>29}"
            elif v.get('yayilim_pct') is not None:
                row += f"{v['median_ms']:>21.3f} (%{v['yayilim_pct']:.1f})"
            else:
                row += f"{v['median_ms']:>29.3f}"
        print(row)
    print()
    print("Kaynaklar:")
    for name, _, src, note in cols:
        print(f"  {name:<30} {src}")
        print(f"  {'':<30} {note}")


def main():
    cfg = config
    ap = argparse.ArgumentParser(description="B1: asama bazli profil tablosu")
    ap.add_argument('--tag', default='b1')
    args = ap.parse_args()

    cols = build(cfg)
    if not cols:
        raise SystemExit("Hicbir kaynak dosya bulunamadi.")
    print_table(cols)

    out = {'aciklama': ('Yeni olcum degil, mevcut dosyalardan derleme. Her '
                        'sutunun kaynagi ve olcum kosulu ayrica yazili.'),
           'sutunlar': [{'ad': n, 'kaynak': s, 'kosul': note, 'asamalar': d}
                        for n, d, s, note in cols]}
    path = os.path.join(cfg.RESULTS_DIR, f'profile_table_{args.tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[i] Kaydedildi: {path}")
    return out


if __name__ == "__main__":
    main()
