"""
tools/figures.py
=================
Paper figures. All labels in ENGLISH -- these go straight into the
manuscript, so nothing here is translated at submission time.

Every figure is built from committed result files; no measurement happens
here. If a source file is missing the figure is skipped with a warning
rather than silently drawing a wrong plot.

Usage:
    python -m tools.figures                # all figures
    python -m tools.figures --only 2 3     # selected ones
"""

import argparse
import json
import os

import numpy as np

from config import config

import matplotlib
matplotlib.use('Agg')                 # works headless
import matplotlib.pyplot as plt       # noqa: E402

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 160,
})

#: Internal method ids -> labels used in the paper.
LABEL = {
    'none': 'no XAI',
    'shap_ucuz': 'SHAP (cheap)',
    'shap_cheap': 'SHAP (cheap)',
    'lime_ucuz': 'LIME (cheap)',
    'lime_cheap': 'LIME (cheap)',
    'lime': 'LIME',
    'shap': 'SHAP',
}
LIME_C, SHAP_C, BASE_C = '#1f77b4', '#d62728', '#7f7f7f'


def color_of(name):
    if name.startswith('lime'):
        return LIME_C
    if name.startswith('shap'):
        return SHAP_C
    return BASE_C


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('summary', d)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")
    return path


# ----------------------------------------------------------------------
# Figure 1: where the time goes
# ----------------------------------------------------------------------

def fig_cost_breakdown(r, out_dir):
    """Stacked pipeline stages per explainer.

    The point of the figure: explanation dwarfs every other stage,
    including inference itself.
    """
    order = ['none', 'shap_ucuz', 'lime_ucuz', 'lime', 'shap']
    files = {'none': 'fog_xai_none.json', 'shap_ucuz': 'fog_xai_shapucuz.json',
             'lime_ucuz': 'fog_xai_limeucuz.json', 'lime': 'fog_xai_lime.json',
             'shap': 'fog_xai_shap.json'}
    stages = ['resampling', 'preprocessing', 'tensor_prep', 'inference',
              'postprocessing', 'xai']
    nice = {'resampling': 'resampling', 'preprocessing': 'normalisation',
            'tensor_prep': 'tensor prep', 'inference': 'inference',
            'postprocessing': 'post-processing', 'xai': 'explanation'}
    cmap = plt.get_cmap('tab20c')
    colors = [cmap(i) for i in (4, 5, 6, 0, 7)] + ['#d62728']

    rows, labels = [], []
    for k in order:
        s = load(os.path.join(r, '04_fog', files[k]))
        if s is None:
            continue
        st = s.get('server_stages') or {}
        rows.append([(st.get(x) or {}).get('mean_ms', 0.0) for x in stages])
        labels.append(LABEL[k])
    if not rows:
        return None

    rows = np.asarray(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for ax, logscale in ((ax1, False), (ax2, True)):
        bottom = np.zeros(len(rows))
        for j, st in enumerate(stages):
            vals = rows[:, j]
            ax.bar(labels, vals, bottom=bottom, color=colors[j],
                   label=nice[st] if not logscale else None,
                   edgecolor='white', linewidth=0.5)
            bottom += vals
        ax.set_ylabel('server-side time per window (ms)')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)
        if logscale:
            ax.set_yscale('log')
            ax.set_title('log scale — reveals the non-XAI stages')
        else:
            ax.set_title('linear scale — explanation dominates')

    # Annotate the ratio that carries the finding
    base = rows[0].sum() if labels and labels[0] == 'no XAI' else None
    if base:
        for i, lab in enumerate(labels[1:], start=1):
            ax1.text(i, rows[i].sum(), f"{rows[i].sum() / base:.0f}x",
                     ha='center', va='bottom', fontsize=9)

    ax1.legend(loc='upper left', frameon=True)
    fig.suptitle('Pipeline cost breakdown on the fog node (S11, per window)')
    fig.tight_layout()
    return save(fig, out_dir, 'fig1_cost_breakdown.png')


# ----------------------------------------------------------------------
# Figure 2: selective explanation
# ----------------------------------------------------------------------

def fig_selective(r, out_dir):
    """Threshold sweep: what you pay and what you catch."""
    cal = load(os.path.join(r, '08_kalibrasyon', 'calibration_d1.json'))
    swp = load(os.path.join(r, '09_secici_xai', 'selective_xai_sweep_d2.json'))
    meas = load(os.path.join(r, '09_secici_xai', 'fog_sel80_tam.json'))
    if cal is None or swp is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- left: explain rate and error recall, both MEASURED in D1 --
    for subj, style in (('S11', '-o'), ('S14', '--s')):
        block = ((cal.get('per_subject') or {}).get(subj) or {}).get('uncalibrated')
        if not block:
            continue
        t = [x['threshold'] for x in block['thresholds']]
        ax1.plot(t, [x['explain_rate'] * 100 for x in block['thresholds']],
                 style, color='#1f77b4', label=f'{subj} — windows explained')
        ax1.plot(t, [x['error_recall'] * 100 for x in block['thresholds']],
                 style, color='#2ca02c', label=f'{subj} — errors caught')
    ax1.set_xlabel('confidence threshold')
    ax1.set_ylabel('percentage of windows (%)')
    ax1.set_title('What is explained, what is caught')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # -- right: end-to-end response, projection vs measurement --
    # Kaynak dosya eski ('shap_ucuz') ya da yeni ('shap_cheap') adi
    # tasiyabilir; ikisi de kabul edilir.
    rows = [x for x in swp['satirlar']
            if x['aciklayici'] in ('shap_cheap', 'shap_ucuz') and x['denek'] == 'S11']
    rows.sort(key=lambda x: x['esik'])
    ax2.plot([x['esik'] for x in rows], [x['response_mean_ms'] for x in rows],
             '-o', color='#d62728', label='projected (mixture model)')
    if meas:
        x = meas.get('xai') or {}
        ax2.plot([0.80], [meas['metrics']['response']['mean_ms']], '*',
                 color='black', markersize=16, zorder=5,
                 label=f"measured at 0.80 ({x.get('n_windows', 0)} windows)")
    ax2.set_xlabel('confidence threshold')
    ax2.set_ylabel('mean response time (ms)')
    ax2.set_title('Cost of selective explanation, SHAP (cheap)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle('Selective explanation: explain only when the model is unsure')
    fig.tight_layout()
    return save(fig, out_dir, 'fig2_selective.png')


# ----------------------------------------------------------------------
# Figure 3: quality vs latency
# ----------------------------------------------------------------------

def fig_pareto(r, out_dir):
    d = load(os.path.join(r, '11_pareto', 'pareto_c5.json'))
    if d is None:
        return None
    rows = d['noktalar']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    specs = [(axes[0], 'faithfulness', 'faithfulness_sem',
              'Faithfulness (insertion - deletion AUC)', 'higher is better'),
             (axes[1], 'sensitivity', 'sensitivity_sem',
              'Sensitivity to input noise', 'lower is more stable')]

    for ax, key, semk, ylabel, title in specs:
        seen = set()
        for x in rows:
            fam = 'LIME' if x['nokta'].startswith('lime') else 'SHAP'
            ax.errorbar(x['gecikme_ms'], x[key], yerr=x.get(semk) or 0.0,
                        fmt='o' if fam == 'LIME' else 's',
                        color=color_of(x['nokta']), markersize=9, capsize=4,
                        label=fam if fam not in seen else None)
            seen.add(fam)
            ax.annotate(LABEL.get(x['nokta'], x['nokta']),
                        (x['gecikme_ms'], x[key]),
                        textcoords='offset points', xytext=(9, 5), fontsize=9)
        for fam in ('lime', 'shap'):
            pts = sorted([x for x in rows if x['nokta'].startswith(fam)],
                         key=lambda v: v['gecikme_ms'])
            if len(pts) > 1:
                ax.plot([p['gecikme_ms'] for p in pts], [p[key] for p in pts],
                        color=color_of(fam), alpha=0.35, linewidth=1.5, zorder=0)
        ax.set_xlabel('explanation time on the fog node (ms)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')

    fig.suptitle('Explanation quality against latency  (error bars: standard error)')
    fig.tight_layout()
    return save(fig, out_dir, 'fig3_pareto.png')


# ----------------------------------------------------------------------
# Figure 4: capacity under load
# ----------------------------------------------------------------------

def fig_capacity(r, out_dir):
    series = {
        'no XAI': [('04_fog/fog_xai_none.json', 1), ('10_yuk/fog_load_none_2.json', 2),
                   ('10_yuk/fog_load_none_4.json', 4)],
        'SHAP (cheap), every window': [
            ('04_fog/fog_xai_shapucuz.json', 1), ('10_yuk/fog_load_shapucuz_2.json', 2),
            ('10_yuk/fog_load_shapucuz_4.json', 4)],
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {'no XAI': BASE_C, 'SHAP (cheap), every window': SHAP_C}

    for label, items in series.items():
        n, q95, resp = [], [], []
        for rel, k in items:
            s = load(os.path.join(r, *rel.split('/')))
            if s is None:
                continue
            n.append(k)
            q95.append(s['metrics']['queuing']['p95_ms'])
            resp.append(s['metrics']['response']['p95_ms'])
        if not n:
            continue
        ax1.plot(n, q95, '-o', color=colors[label], label=label)
        ax2.plot(n, resp, '-o', color=colors[label], label=label)

    for ax, ylab, title in ((ax1, 'queuing delay, p95 (ms)', 'Queue builds up only when explaining'),
                            (ax2, 'response time, p95 (ms)', 'End-to-end worst case')):
        ax.set_xlabel('concurrent clients')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks([1, 2, 4])
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)

    fig.suptitle('Single-worker capacity: clients arrive phase-aligned, so '
                 'explanations serialise')
    fig.tight_layout()
    return save(fig, out_dir, 'fig4_capacity.png')


# ----------------------------------------------------------------------
# Figure 5: placement
# ----------------------------------------------------------------------

def fig_placement(r, out_dir):
    methods = ['none', 'shap_ucuz', 'lime_ucuz', 'lime', 'shap']
    paths = {
        'edge': {'none': '07_edge_xai/edge_xai_none.json',
                 'shap_ucuz': '07_edge_xai/edge_xai_shapucuz.json',
                 'lime_ucuz': '07_edge_xai/edge_xai_limeucuz.json',
                 'lime': '07_edge_xai/edge_xai_lime.json',
                 'shap': '07_edge_xai/edge_xai_shap.json'},
        'fog': {'none': '04_fog/fog_xai_none.json',
                'shap_ucuz': '04_fog/fog_xai_shapucuz.json',
                'lime_ucuz': '04_fog/fog_xai_limeucuz.json',
                'lime': '04_fog/fog_xai_lime.json',
                'shap': '04_fog/fog_xai_shap.json'},
        'cloud': {'shap_ucuz': '13_bulut/cloud_xai_shapucuz.json',
                  'lime_ucuz': '13_bulut/cloud_xai_limeucuz.json',
                  'lime': '13_bulut/cloud_xai_lime.json',
                  'shap': '13_bulut/cloud_xai_shap.json'},
    }
    # Cloud baseline lives in a compare_deployments file, different shape.
    cloud_base = None
    cb = load(os.path.join(r, '13_bulut', 'cloud_taban_3tur.json'))
    if cb and 'by_deployment' in cb:
        cloud_base = list(cb['by_deployment'].values())[0]['metrics']['response']['median']['median']

    vals = {}
    for place, mp in paths.items():
        vals[place] = []
        for m in methods:
            if place == 'cloud' and m == 'none':
                vals[place].append(cloud_base)
                continue
            s = load(os.path.join(r, *mp[m].split('/'))) if m in mp else None
            vals[place].append(s['metrics']['response']['median_ms'] if s else None)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.26
    x = np.arange(len(methods))
    pal = {'edge': '#4c72b0', 'fog': '#55a868', 'cloud': '#c44e52'}
    for i, place in enumerate(('edge', 'fog', 'cloud')):
        ys = [v if v is not None else np.nan for v in vals[place]]
        ax.bar(x + (i - 1) * width, ys, width, label=place, color=pal[place])

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=15)
    ax.set_ylabel('median response time (ms)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    ax.legend(title='placement')
    ax.set_title('Placement matters without explanation, and stops mattering with it')
    fig.tight_layout()
    return save(fig, out_dir, 'fig5_placement.png')


# ----------------------------------------------------------------------
# Figure 6: calibration
# ----------------------------------------------------------------------

def fig_calibration(r, out_dir):
    cal = load(os.path.join(r, '08_kalibrasyon', 'calibration_d1.json'))
    if cal is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    for ax, subj in zip(axes, ('S11', 'S14')):
        block = ((cal.get('per_subject') or {}).get(subj) or {}).get('uncalibrated')
        if not block:
            continue
        bins = [b for b in block['reliability_bins'] if b['n']]
        conf = [b['avg_confidence'] for b in bins]
        acc = [b['accuracy'] for b in bins]
        n = np.array([b['n'] for b in bins], dtype=float)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='perfect calibration')
        ax.scatter(conf, acc, s=20 + 180 * n / n.max(), alpha=0.75,
                   color='#1f77b4', edgecolor='white', zorder=3,
                   label='bins (area = #windows)')
        ax.set_xlabel('mean predicted confidence')
        ax.set_ylabel('observed accuracy')
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.set_title(f"{subj} — ECE {block['ece']:.3f}, "
                     f"accuracy {block['accuracy']:.3f}")
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Confidence calibration differs sharply between held-out subjects')
    fig.tight_layout()
    return save(fig, out_dir, 'fig6_calibration.png')


FIGURES = [fig_cost_breakdown, fig_selective, fig_pareto,
           fig_capacity, fig_placement, fig_calibration]


def main():
    cfg = config
    ap = argparse.ArgumentParser(description="Paper figures (English labels)")
    ap.add_argument('--out-dir', default=os.path.join(cfg.RESULTS_DIR, '14_figures'))
    ap.add_argument('--only', type=int, nargs='+',
                    help="figure numbers to build, e.g. --only 1 3")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sel = args.only or list(range(1, len(FIGURES) + 1))
    made = 0
    for i, fn in enumerate(FIGURES, start=1):
        if i not in sel:
            continue
        print(f"[fig {i}] {fn.__name__}")
        try:
            if fn(cfg.RESULTS_DIR, args.out_dir) is None:
                print("  [!] source data missing, skipped")
            else:
                made += 1
        except Exception as e:
            print(f"  [!] failed: {type(e).__name__}: {e}")
    print(f"\n{made} figure(s) written to {args.out_dir}")


if __name__ == "__main__":
    main()
