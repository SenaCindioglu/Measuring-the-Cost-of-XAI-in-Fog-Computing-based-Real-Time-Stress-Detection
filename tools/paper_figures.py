"""
tools/paper_figures.py
=======================
Makale sekilleri, v2 verisinden (16_coloc + 16_pi_perf). paper_numbers.py
ile AYNI dosyalardan okur; elle sayi girilmez. Cikti vektorel PDF olarak
paper/icaiic/ altina yazilir ve build_paper.sh tarafindan uretilir.

fig_cost.pdf: aciklayici basina medyan yanit suresi, iki yerlesim yan yana,
log eksen; 1 s gercek-zamanlilik siniri (ShaTS) ve 5 s hop cizgili.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'paper', 'icaiic')

COLOC = {'none': 'coloc_none', 'shap_cheap': 'coloc_shap_cheap',
         'lime_cheap': 'coloc_lime_cheap', 'lime': 'coloc_lime', 'shap': 'coloc_shap'}
FOG = {'none': 'fog_pi_none_perf', 'shap_cheap': 'fog_pi_shap_perf',
       'lime_cheap': 'fog_pi_lime_perf', 'lime': 'fog_pi_lime_full_perf',
       'shap': 'fog_pi_shap_full_perf'}
CLOUD = {'none': 'fog_cloud_baseline_retry1',
         'shap_cheap': 'fog_cloud_xai_shap_ucuz',
         'lime_cheap': 'fog_cloud_xai_limeucuz',
         'lime': 'fog_cloud_xai_lime', 'shap': 'fog_cloud_xai_shap'}
ORDER = ['none', 'shap_cheap', 'lime_cheap', 'lime', 'shap']
LABEL = {'none': 'no explanation', 'shap_cheap': 'SHAP-cheap',
         'lime_cheap': 'LIME-cheap', 'lime': 'LIME', 'shap': 'SHAP'}

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 8, 'pdf.fonttype': 42})


def med_resp(sub, fn):
    p = os.path.join(R, sub, fn + '.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return d['metrics']['response']['median_ms'], d['metrics']['response']['p95_ms']


def fig_cost():
    rows = []
    for m in ORDER:
        c = med_resp('16_coloc', COLOC[m])
        g = med_resp('16_pi_perf', FOG[m])
        k = med_resp('17_bulut_openloop', CLOUD[m])
        rows.append((LABEL[m], c, g, k))

    has_cloud = any(r[3] for r in rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.02 if has_cloud else 1.85))
    y = range(len(rows))
    h = 0.27 if has_cloud else 0.38
    # y ekseni ters: negatif ofset ustte; cubuklar etiket etrafinda ortali
    for i, (lab, c, g, k) in enumerate(rows):
        bars = [(c, -h if has_cloud else -h / 2, '#8dafd4', 'laptop (co-located)'),
                (g, 0 if has_cloud else h / 2, '#c0504d', 'fog (Raspberry Pi 4)')]
        if has_cloud:
            bars.append((k, h, '#9bbb59', 'cloud (VPS)'))
        for val, off, color, lab_ in bars:
            if not val:
                continue
            ax.barh(i + off, val[0], height=h, color=color,
                    edgecolor='black', linewidth=0.4,
                    label=lab_ if i == 0 else None)
            ax.errorbar(val[1], i + off, xerr=None, fmt='|', color='black',
                        markersize=5)
    ax.axvline(1000, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(5000, color='black', linestyle=':', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlim(3, 7000)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.text(1000, -0.52, '1 s ', fontsize=7, ha='right', va='bottom')
    ax.text(5000, -0.52, '5 s ', fontsize=7, ha='right', va='bottom')
    ax.set_xlabel('median response time per window (ms)')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2,
              frameon=False, columnspacing=1.0, handlelength=1.4)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, 'fig_cost.pdf')
    fig.savefig(path, bbox_inches='tight')
    print('[i]', path)


def fig_load():
    """Cok istemcili yuk: olculen yanit sureleri cubuk, kuyruk modelinin
    tahminleri cubuk ustunde siyah cizgi isareti (Pi, SHAP-cheap her
    pencere, ondemand governor, istemciler eszamanli)."""
    ns, meas_mean, pred_mean, meas_p95, pred_p95 = [], [], [], [], []
    for n, fn in [(1, 'fog_pi_load1'), (2, 'fog_pi_load2'), (4, 'fog_pi_load4')]:
        path = os.path.join(R, '15_pi_ondemand', fn + '.json')
        with open(path) as f:
            d = json.load(f)
        pool = d.get('pooled') or {k: d['metrics'][k] for k in ('response', 'execution')}
        e = pool['execution']['mean_ms']
        ns.append(n)
        meas_mean.append(pool['response']['mean_ms'])
        pred_mean.append((n + 1) / 2 * e)
        meas_p95.append(pool['response']['p95_ms'])
        pred_p95.append(n * e)

    xs = range(len(ns))
    w = 0.32
    fig, ax = plt.subplots(figsize=(3.4, 1.52))
    ax.bar([x - w / 2 for x in xs], meas_mean, w, color='#8dafd4',
           edgecolor='black', linewidth=0.4, label='mean, measured')
    ax.bar([x + w / 2 for x in xs], meas_p95, w, color='#c0504d',
           edgecolor='black', linewidth=0.4, label='p95, measured')
    ax.plot([x - w / 2 for x in xs], pred_mean, linestyle='none', marker='_',
            markersize=12, markeredgewidth=1.5, color='black',
            label='queuing model')
    ax.plot([x + w / 2 for x in xs], pred_p95, linestyle='none', marker='_',
            markersize=12, markeredgewidth=1.5, color='black')
    ax.axhline(5000, color='black', linestyle=':', linewidth=0.8)
    ax.text(2.45, 5060, '5 s hop', fontsize=7, va='bottom', ha='right')
    ax.set_xticks(list(xs), [str(n) for n in ns])
    ax.set_xlabel('concurrent clients')
    ax.set_ylabel('response time (ms)')
    ax.set_ylim(0, 5650)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.97), frameon=False,
              fontsize=7, handlelength=1.3, labelspacing=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, 'fig_load.pdf')
    fig.savefig(path, bbox_inches='tight')
    print('[i]', path)


def fig_governor():
    """Pi'de ondemand ve performance governor karsilastirmasi: ayni
    asamalarin medyan sureleri (fig_cost stilinde yatay cubuk, log eksen)."""
    def grab(sub, files):
        out = {}
        for key, fn in files.items():
            with open(os.path.join(R, sub, fn + '.json')) as f:
                d = json.load(f)
            st = d['server_stages']
            out[key] = {'exec': d['metrics']['execution']['median_ms'],
                        'inf': st['inference']['median_ms'],
                        'xai': st['xai']['median_ms'] if 'xai' in st else None}
        return out

    od = grab('15_pi_ondemand', {'none': 'fog_pi_none',
                                 'lime_cheap': 'fog_pi_lime',
                                 'shap_cheap': 'fog_pi_shap'})
    pf = grab('16_pi_perf', {'none': 'fog_pi_none_perf',
                             'lime_cheap': 'fog_pi_lime_perf',
                             'shap_cheap': 'fog_pi_shap_perf'})

    def busy_inf(t):
        a, b = t['lime_cheap']['inf'], t['shap_cheap']['inf']
        return (a + b) / 2

    rows = [
        ('pipeline, no XAI',        od['none']['exec'],       pf['none']['exec']),
        ('inference (idle run)',    od['none']['inf'],        pf['none']['inf']),
        ('inference (busy runs)',   busy_inf(od),             busy_inf(pf)),
        ('LIME-cheap explanation',  od['lime_cheap']['xai'],  pf['lime_cheap']['xai']),
        ('SHAP-cheap explanation',  od['shap_cheap']['xai'],  pf['shap_cheap']['xai']),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 1.42))
    h = 0.36
    for i, (lab, v_od, v_pf) in enumerate(rows):
        ax.barh(i - h / 2, v_od, height=h, color='#b8b8b8',
                edgecolor='black', linewidth=0.4,
                label='on-demand governor' if i == 0 else None)
        ax.barh(i + h / 2, v_pf, height=h, color='#8dafd4',
                edgecolor='black', linewidth=0.4,
                label='performance governor' if i == 0 else None)
    ax.set_xscale('log')
    ax.set_xlim(4, 2600)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('median time on the Pi (ms, log scale)')
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.2)
    path = os.path.join(OUT, 'fig_governor.pdf')
    fig.savefig(path, bbox_inches='tight')
    print('[i]', path)


if __name__ == '__main__':
    fig_cost()
    fig_load()
