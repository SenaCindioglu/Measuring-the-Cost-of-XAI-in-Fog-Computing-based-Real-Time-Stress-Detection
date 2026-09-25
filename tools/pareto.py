"""
tools/pareto.py
================
FAZ C5: kalite-gecikme Pareto egrisi.

NE YAPAR
--------
Dondurulmus dort calisma noktasinin (lime, lime_cheap, shap, shap_cheap)
KALITESINI olcer ve fog dugumunde olculen GECIKMEYLE eslestirir, sonra
Pareto grafigini cizer.

NEDEN GEREKLI: tarama Colab'da GPU'da kosuldu, oradaki sureler dagitim
donanimini yansitmiyor. Ustelik `lime_cheap` (250 ornek, 20 dilim,
kernel 0.5) taramada HIC olculmedi -- o yapilandirma tarama izgarasinda
yoktu. Yani Pareto grafiginin dort noktasindan biri kalite ekseninde
bosluktaydi.

Bu script iki ekseni de dogru kaynaktan alir:
  kalite   -> burada olculur (donanimdan bagimsiz)
  gecikme  -> results/04_fog/ (dagitim donanimi, uctan uca)

STANDART HATA. Taramanin ilk halinde yoktu ve kazananlar 0.003
mertebesindeki faithfulness farklariyla seciliyordu; farkin gurultu olup
olmadigi soylenemiyordu. Burada her olcu ortalamasiyla birlikte standart
hatasini da tasir.

Kullanim:
    python -m tools.pareto
    python -m tools.pareto --windows 40 --subject S11 --tag c5
"""

import argparse
import json
import os

import numpy as np
import torch

from config import config
from models.temporal_cnn import load_checkpoint
from xai.artifacts import load_artifacts
from xai.benchmark import _eval_windows
from xai import lime_ as lime_mod, shap_ as shap_mod
from xai.quality import evaluate


#: (ad, kurucu, 04_fog dosyasi) -- sunucudaki --xai-method adlariyla ayni
POINTS = [
    ('shap_cheap', lambda m, r, d: shap_mod.from_artifacts(m, r, device=d, cheap=True),
     'fog_xai_shapucuz.json'),
    ('lime_cheap', lambda m, r, d: lime_mod.from_artifacts(m, r, device=d, num_samples=250),
     'fog_xai_limeucuz.json'),
    ('lime', lambda m, r, d: lime_mod.from_artifacts(m, r, device=d),
     'fog_xai_lime.json'),
    ('shap', lambda m, r, d: shap_mod.from_artifacts(m, r, device=d),
     'fog_xai_shap.json'),
]


def fog_latency(fog_dir, fname):
    """Fog dugumunde olculen aciklama suresi (ms)."""
    path = os.path.join(fog_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    s = d.get('summary', d)
    x = s.get('xai') or {}
    st = (s.get('server_stages') or {}).get('xai') or {}
    return {'mean_ms': x.get('mean_ms_per_explanation') or st.get('mean_ms'),
            'p95_ms': st.get('p95_ms'),
            'response_mean_ms': s['metrics']['response']['mean_ms'],
            'kaynak': fname}


def plot(rows, path):
    """Kalite (y) - gecikme (x) Pareto grafigi."""
    import matplotlib
    matplotlib.use('Agg')          # basliksiz ortamda da calissin
    import matplotlib.pyplot as plt

    # IKI PANEL. Tek panelde yalnizca faithfulness gosterilirse yaniltici
    # olur: o eksende dort nokta da hata payi icinde ayni. Asil ayrisma
    # sensitivity'de ve orada hata paylari cok kucuk.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fams = {'lime': ('#1f77b4', 'o', 'LIME'), 'shap': ('#d62728', 's', 'SHAP')}

    specs = [(axes[0], 'faithfulness', 'faithfulness_sem',
              'Faithfulness  (insertion - deletion AUC)', 'yuksek = iyi'),
             (axes[1], 'sensitivity', 'sensitivity_sem',
              'Sensitivity  (gurultuye karsi degisim)', 'dusuk = kararli')]

    for ax, key, semk, ylabel, note in specs:
        seen = set()
        for r in rows:
            fam = 'lime' if r['nokta'].startswith('lime') else 'shap'
            color, marker, label = fams[fam]
            ax.errorbar(r['gecikme_ms'], r[key], yerr=r.get(semk) or 0.0,
                        fmt=marker, color=color, markersize=9, capsize=4,
                        label=(label if fam not in seen else None))
            seen.add(fam)
            ax.annotate(r['nokta'], (r['gecikme_ms'], r[key]),
                        textcoords='offset points', xytext=(8, 6), fontsize=9)
        for fam in ('lime', 'shap'):
            pts = sorted([r for r in rows if r['nokta'].startswith(fam)],
                         key=lambda r: r['gecikme_ms'])
            if len(pts) > 1:
                ax.plot([p['gecikme_ms'] for p in pts], [p[key] for p in pts],
                        color=fams[fam][0], alpha=0.35, linewidth=1.5, zorder=0)
        ax.set_xlabel('Fog dugumunde aciklama suresi (ms)')
        ax.set_ylabel(ylabel)
        ax.set_title(note)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')

    fig.suptitle('Aciklama kalitesi - gecikme dengesi  '
                 '(hata cubuklari: standart hata)', y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    cfg = config
    ap = argparse.ArgumentParser(description="C5: kalite-gecikme Pareto")
    ap.add_argument('--subject', default='S11')
    ap.add_argument('--windows', type=int, default=40,
                    help="Kalite olcumunde kullanilacak pencere sayisi")
    ap.add_argument('--sensitivity-windows', type=int, default=10)
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--fog-dir', default=os.path.join(cfg.RESULTS_DIR, '04_fog'))
    ap.add_argument('--threads', type=int, default=cfg.DEFAULT_THREADS)
    ap.add_argument('--tag', default='c5')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    model, arch, _ = load_checkpoint(cfg.MODEL_PATH, torch.device('cpu'), fallback_cfg=cfg)
    record = load_artifacts()
    windows = _eval_windows(cfg, args.subject, args.windows)

    print("=" * 74)
    print("C5 - KALITE / GECIKME PARETO")
    print("=" * 74)
    print(f"Denek {args.subject} | {len(windows)} pencere (sinif dengeli) | "
          f"nedensel girdi uzayi")
    print(f"Kalite BURADA olculuyor, gecikme {os.path.basename(args.fog_dir)}'dan")
    print("=" * 74)

    rows = []
    for name, ctor, fname in POINTS:
        print(f"\n[{name}]")
        expl = ctor(model, record, 'cpu')
        q = evaluate(expl, windows, model=model, device='cpu',
                     n_sensitivity=args.sensitivity_windows,
                     n_trials=args.trials, verbose=True)
        lat = fog_latency(args.fog_dir, fname)
        if lat is None or lat['mean_ms'] is None:
            print(f"    [!] {fname} bulunamadi, gecikme eksik")
            continue
        rows.append({
            'nokta': name,
            'config': q['config'],
            'faithfulness': q['faithfulness']['mean'],
            'faithfulness_sem': q['faithfulness']['sem'],
            'monotonicity': q['monotonicity']['mean'],
            'sensitivity': q['sensitivity']['mean'],
            'sensitivity_sem': q['sensitivity']['sem'],
            'gecikme_ms': lat['mean_ms'],
            'gecikme_p95_ms': lat['p95_ms'],
            'response_mean_ms': lat['response_mean_ms'],
            'gecikme_kaynagi': lat['kaynak'],
            'n_pencere': q['faithfulness']['n'],
        })

    print("\n" + "=" * 74)
    print(f"{'nokta':>11}{'gecikme':>10}{'faithfulness':>20}{'monoton':>10}{'sensitivity':>14}")
    for r in sorted(rows, key=lambda x: x['gecikme_ms']):
        print(f"{r['nokta']:>11}{r['gecikme_ms']:9.0f}ms"
              f"{r['faithfulness']:13.4f} +/-{r['faithfulness_sem']:.4f}"
              f"{r['monotonicity']:10.4f}{r['sensitivity']:14.4f}")

    # Baskinlik HATA PAYIYLA test edilir. Nokta tahminine bakip "bu digerini
    # yeniyor" demek, taramanin ilk halindeki hatanin aynisi olur: 40 pencerede
    # faithfulness'in standart hatasi ~0.04, yani o mertebedeki farklar
    # gurultudur.
    print()
    print("Baskinlik testi (fark > iki standart hata toplami mi?)")
    any_dom = False
    for r in rows:
        for o in rows:
            if o is r or o['gecikme_ms'] >= r['gecikme_ms']:
                continue
            d = o['faithfulness'] - r['faithfulness']
            e = o['faithfulness_sem'] + r['faithfulness_sem']
            if d > e:
                print(f"  {r['nokta']} BASKILANIYOR -> {o['nokta']} "
                      f"(hem daha ucuz hem daha sadik; fark {d:.4f} > {e:.4f})")
                any_dom = True
    if not any_dom:
        print("  Hicbir nokta digerini BASKILAMIYOR: faithfulness farklarinin")
        print("  hepsi olcum hatasinin icinde kaliyor. Secim faithfulness'a")
        print("  gore degil, gecikme ve sensitivity uzerinden yapilmali.")

    print()
    print("Ayirt edilebilen farklar:")
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            for key, semk, lbl in (('faithfulness', 'faithfulness_sem', 'faithfulness'),
                                   ('sensitivity', 'sensitivity_sem', 'sensitivity')):
                d = abs(a[key] - b[key])
                e = a[semk] + b[semk]
                if d > e:
                    better = a['nokta'] if (a[key] > b[key]) == (key == 'faithfulness') \
                        else b['nokta']
                    print(f"  {lbl:13} {a['nokta']} vs {b['nokta']}: "
                          f"fark {d:.4f} > {e:.4f}  ({better} daha iyi)")
    print()

    out = {'config': {'subject': args.subject, 'n_windows': len(windows),
                      'sensitivity_windows': args.sensitivity_windows,
                      'trials': args.trials, 'threads': args.threads,
                      'kalite': 'bu makinede olculdu (donanimdan bagimsiz)',
                      'gecikme': f'{os.path.basename(args.fog_dir)} (fog dugumu)'},
           'noktalar': rows}
    jpath = os.path.join(cfg.RESULTS_DIR, f'pareto_{args.tag}.json')
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[i] Veri: {jpath}")

    if rows:
        ppath = os.path.join(cfg.RESULTS_DIR, f'pareto_{args.tag}.png')
        plot(rows, ppath)
        print(f"[i] Sekil: {ppath}")
    return out


if __name__ == "__main__":
    main()
