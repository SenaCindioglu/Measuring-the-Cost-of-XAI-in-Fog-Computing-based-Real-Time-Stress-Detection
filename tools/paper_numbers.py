"""
tools/paper_numbers.py
=======================
Makaledeki HER sayiyi sonuc JSON'larindan uretir ve LaTeX makrosu olarak
paper/icaiic/numbers.tex dosyasina yazar. Makale sayi icermez, yalnizca
\\FogShapCRespMed gibi makrolar icerir.

NEDEN: v1 denetiminde ayni olcum alti belgede alti farkli sayi olarak
dolasiyordu (ortalama/medyan karisimi, eski koşudan kalan satirlar, elle
yuvarlama). Tek kaynak bu betik; istatistik politikasi da burada sabit:

    gecikme turu metrikler  -> MEDYAN ve p95 (pencere bazli)
    v2 cok turlu kosu       -> turlarin medyani (tekrar birimi KOSU)
    karisim dagilimlari     -> ORTALAMA, adiyla (secici XAI, kapasite)
    bulut tek yon gecikme   -> raporlanmaz; ag payi (resp-exec-queue)/2

Kullanim:
    python -m tools.paper_numbers                 # v1 (mevcut) veriler
    python -m tools.paper_numbers --v2 results/15_v2
v2 dizininde compare_deployments ciktilari beklenir:
    deployments_tiers_{none,shap_cheap,lime_cheap,lime,shap}.json
    (yerlesim adlari: coloc, fog, cloud)
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')

TIERS = [('coloc', 'Coloc'), ('fog', 'Fog'), ('cloud', 'Cloud')]
METHODS = [('none', 'None'), ('shap_cheap', 'ShapC'), ('lime_cheap', 'LimeC'),
           ('lime', 'Lime'), ('shap', 'Shap')]
# v1 dosya adlari eski Turkce kisaltmalari tasiyor.
V1_FILE = {'none': 'none', 'shap_cheap': 'shapucuz', 'lime_cheap': 'limeucuz',
           'lime': 'lime', 'shap': 'shap'}
V1_DIR = {'coloc': ('07_edge_xai', 'edge_xai_'), 'fog': ('04_fog', 'fog_xai_'),
          'cloud': ('13_bulut', 'cloud_xai_')}


def load(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


def fmt(x, nd):
    return f"{x:,.{nd}f}".replace(',', '{,}')


class Macros:
    def __init__(self):
        self.lines = []
        self.src = {}

    def put(self, name, value, nd=1, src=''):
        if value is None:
            text = r'\textbf{??}'
        elif isinstance(value, str):
            text = value
        else:
            text = fmt(float(value), nd)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{text}\\xspace}}"
                          + (f"  % {src}" if src else ''))


# ---------------------------------------------------------------- tiers --
def tier_cell_v1(tier, method):
    """Tek koşuluk v1 hucresi -> sozluk."""
    if tier == 'cloud' and method == 'none':
        d = load(R, '13_bulut', 'cloud_taban_3tur.json')
        e = next(iter(d['by_deployment'].values()))
        m = e['metrics']
        med = {k: m[k]['median']['median'] for k in m}
        return {'resp': med['response'], 'resp95': m['response']['p95']['median'],
                'exec': med['execution'], 'xai': None,
                'net': (med['response'] - med['execution'] - med['queuing']) / 2,
                'rss': (e.get('server_rss_mb') or {}).get('median'),
                'cpu': (e.get('server_cpu_percent') or {}).get('median'),
                'kbps': e['bandwidth_kbps']['median'], 'n': None,
                'hw': e.get('server_hardware') or {},
                'src': '13_bulut/cloud_taban_3tur.json'}
    d_, pre = V1_DIR[tier]
    fn = f'{pre}{V1_FILE[method]}.json'
    d = load(R, d_, fn)
    m = d['metrics']
    srv = d.get('server') or {}
    res = srv.get('resources') or {}
    st = (d.get('server_stages') or {}).get('xai')
    return {'resp': m['response']['median_ms'], 'resp95': m['response']['p95_ms'],
            'exec': m['execution']['median_ms'],
            'xai': st['median_ms'] if st else None,
            'net': (m['response']['median_ms'] - m['execution']['median_ms']
                    - m['queuing']['median_ms']) / 2,
            'rss': (res.get('process_rss_mb') or {}).get('max'),
            'cpu': (res.get('process_cpu_percent') or {}).get('mean'),
            'kbps': d['bandwidth']['kbps_total'], 'n': d.get('n_measured'),
            'hw': srv.get('hardware') or {}, 'src': f'{d_}/{fn}'}


def tier_cell_v2(v2dir, tier, method):
    fn = f'deployments_tiers_{method}.json'
    d = load(v2dir, fn)
    e = d['by_deployment'][tier]
    m = e['metrics']
    med = {k: m[k]['median']['median'] for k in m if k in
           ('latency', 'queuing', 'execution', 'response')}
    return {'resp': med['response'], 'resp95': m['response']['p95']['median'],
            'exec': med['execution'],
            'xai': (e.get('xai_median_ms') or {}).get('median'),
            'net': (med['response'] - med['execution'] - med['queuing']) / 2,
            'rss': (e.get('server_rss_mb') or {}).get('median'),
            'cpu': (e.get('server_cpu_percent') or {}).get('median'),
            'kbps': e['bandwidth_kbps']['median'],
            'n': int(np.median([x for x in e.get('n_measured', []) if x] or [0])) or None,
            'runs': e['n_runs'], 'spread': m['response']['median']['range_pct'],
            'hw': e.get('server_hardware') or {}, 'src': f'{os.path.basename(v2dir)}/{fn}'}


def hw_label(hw):
    dev = hw.get('device_model')
    if dev:
        return dev.replace('Raspberry Pi', 'Raspberry~Pi')
    proc = hw.get('processor') or hw.get('machine') or '?'
    if 'Intel' in proc:
        proc = 'Intel x86-64'
    n = hw.get('cpu_count')
    return f"{proc}, {n} {'core' if n == 1 else 'cores'}"


def stack_label(hw):
    t = hw.get('torch') or '?'
    box = ', container' if hw.get('in_container') else ''
    return f"Py {hw.get('python')}, torch {t}{box}"


# ---------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2', default=None, help="v2 sonuc dizini (compare ciktilari)")
    ap.add_argument('--out', default=os.path.join(ROOT, 'paper', 'icaiic', 'numbers.tex'))
    args = ap.parse_args()

    M = Macros()
    version = 'v2' if args.v2 else 'v1'
    M.put('DataVersion', version)
    # \ifvone: makale v1 verisiyle derlenirken uyari kutusu basar.
    M.lines.append(r'\newif\ifvone' + (r' \vonetrue' if version == 'v1' else ''))

    # Dongu kipi kontrolu (yerel, SHAP-200): kapali dongunun sensor akisini
    # ne kadar durdurdugu. v2'de yerine 15_v2/loop_check gelir.
    lc = os.path.join(R, '15_v2', 'loop_check')
    try:
        for mode, Mode in [('closed', 'Closed'), ('open', 'Open')]:
            d = load(lc, f'fog_t_{mode}_shap.json')
            M.put(f'Loop{Mode}LagPNF', d['metrics']['send_lag']['p95_ms'], 0,
                  f'15_v2/loop_check/fog_t_{mode}_shap.json')
            M.put(f'Loop{Mode}LagMax', d['metrics']['send_lag']['max_ms'], 0)
            M.put(f'Loop{Mode}N', d['n_measured'], 0)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[!] loop_check eksik: {exc}", file=sys.stderr)

    cells = {}
    for t, T in TIERS:
        for mth, Mth in METHODS:
            try:
                c = (tier_cell_v2(args.v2, t, mth) if args.v2
                     else tier_cell_v1(t, mth))
            except (FileNotFoundError, KeyError) as exc:
                print(f"[!] eksik hucre {t}/{mth}: {exc}", file=sys.stderr)
                c = None
            cells[(t, mth)] = c
            p = f'{T}{Mth}'
            g = (lambda k: c.get(k) if c else None)
            M.put(p + 'RespMed', g('resp'), 1, g('src') or '')
            M.put(p + 'RespPNF', g('resp95'), 1)
            M.put(p + 'ExecMed', g('exec'), 1)
            M.put(p + 'XaiMed', g('xai'), 0)
            M.put(p + 'Net', g('net'), 1)
            M.put(p + 'RSS', g('rss'), 0)
            M.put(p + 'CPU', g('cpu'), 1)
            M.put(p + 'N', g('n'), 0)
        c0 = cells[(t, 'none')]
        if c0:
            M.put(T + 'HW', hw_label(c0['hw']))
            M.put(T + 'Stack', stack_label(c0['hw']))
            M.put(T + 'RAM', (c0['hw'].get('total_ram_mb') or 0) / 1024, 0)
            M.put(T + 'Kbps', c0['kbps'], 0)

    # Turetilmis oranlar (fog ici: yigin karsilastirmasindan bagimsiz)
    def ratio(a, b):
        return a / b if a is not None and b else None

    fn = cells[('fog', 'none')]
    xs = [cells[('fog', m)]['xai'] for m, _ in METHODS[1:] if cells[('fog', m)]]
    if fn and xs:
        M.put('FogXaiOverExecLow', ratio(min(xs), fn['exec']), 0, 'min xai / none exec')
        M.put('FogXaiOverExecHigh', ratio(max(xs), fn['exec']), 0)
        M.put('FogXaiMinMs', min(xs), 0)
        M.put('FogXaiMaxMs', max(xs), 0)
        M.put('FogRespGrowth', ratio(cells[('fog', 'shap')]['resp'], fn['resp']), 0)
        M.put('FogNoneHopPct', 100 * fn['exec'] / 5000.0, 2, 'none exec / 5 s hop')
    for mth, Mth in METHODS:
        a, b = cells[('cloud', mth)], cells[('coloc', mth)]
        if a and b:
            M.put(f'CloudOverColoc{Mth}', ratio(a['resp'], b['resp']), 2)
        a = cells[('fog', mth)]
        if a and b:
            M.put(f'FogOverColoc{Mth}', ratio(a['resp'], b['resp']), 2)
        if a and b and a.get('xai') and b.get('xai'):
            M.put(f'FogOverColocXai{Mth}', ratio(a['xai'], b['xai']), 2)

    # ----------------------------------------------------- secici XAI (fog)
    sel = load(R, '09_secici_xai', 'fog_sel80_tam.json')
    alw = load(R, '04_fog', 'fog_xai_shapucuz.json')
    non = load(R, '04_fog', 'fog_xai_none.json')
    sx = sel['xai']
    M.put('SelThr', ((sel['server'] or {}).get('xai') or {}).get('threshold'), 2,
          '09_secici_xai/fog_sel80_tam.json')
    M.put('SelN', sel['n_measured'], 0)
    M.put('SelMinutes', sel['wall_seconds'] / 60.0, 0)
    M.put('SelNExpl', sx['n_explained'], 0)
    M.put('SelRate', 100 * sx['explain_rate'], 1)
    M.put('SelPerExpl', sx['mean_ms_per_explanation'], 0)
    M.put('SelRespMean', sel['metrics']['response']['mean_ms'], 1)
    M.put('SelRespMedian', sel['metrics']['response']['median_ms'], 1)
    M.put('SelRespPNF', sel['metrics']['response']['p95_ms'], 0)
    M.put('SelExecMean', sel['metrics']['execution']['mean_ms'], 1)
    M.put('SelQueueMax', sel['metrics']['queuing']['p99_ms'], 2)
    M.put('AlwRespMean', alw['metrics']['response']['mean_ms'], 1)
    M.put('AlwExecMean', alw['metrics']['execution']['mean_ms'], 1)
    M.put('NonRespMean', non['metrics']['response']['mean_ms'], 1)
    M.put('NonExecMean', non['metrics']['execution']['mean_ms'], 2)
    M.put('AlwN', alw['n_measured'], 0)
    M.put('SelSpeedup', alw['metrics']['response']['mean_ms']
          / sel['metrics']['response']['mean_ms'], 1)
    hop = 5000.0
    for name, ex in [('Non', non['metrics']['execution']['mean_ms']),
                     ('Sel', sel['metrics']['execution']['mean_ms']),
                     ('Alw', alw['metrics']['execution']['mean_ms'])]:
        M.put(f'Cap{name}', int(hop // ex), 0, 'hop / mean execution')
    M.put('CapGain', (hop // sel['metrics']['execution']['mean_ms'])
          / (hop // alw['metrics']['execution']['mean_ms']), 1)

    proj = load(R, '09_secici_xai', 'selective_xai_sweep_d2.json')
    row = next(r for r in proj['satirlar'] if r['aciklayici'] == 'shap_cheap'
               and r['denek'] == 'S11' and abs(r['esik'] - 0.8) < 1e-9)
    M.put('ProjRespMean', row['response_mean_ms'], 1, 'selective_xai_sweep_d2')
    M.put('ProjErr', 100 * abs(row['response_mean_ms'] - sel['metrics']['response']['mean_ms'])
          / sel['metrics']['response']['mean_ms'], 1)
    M.put('ProjRateErr', 100 * abs(row['aciklama_orani'] - sx['explain_rate'])
          / sx['explain_rate'], 1)

    # ----------------------------------------------------- kalibrasyon
    cal = load(R, '08_kalibrasyon', 'calibration_d1.json')
    for s, S in [('S11', 'Eleven'), ('S14', 'Fourteen')]:
        u = cal['per_subject'][s]['uncalibrated']
        M.put(f'Acc{S}', u['accuracy'], 3, '08_kalibrasyon/calibration_d1.json')
        M.put(f'Conf{S}', u['avg_confidence'], 3)
        M.put(f'ECE{S}', u['ece'], 3)
        M.put(f'CalN{S}', u['n'], 0)
        for th in u['thresholds']:
            tag = {0.6: 'Sixty', 0.7: 'Seventy', 0.8: 'Eighty', 0.9: 'Ninety'}.get(
                round(th['threshold'], 2))
            if tag:
                M.put(f'Rate{S}{tag}', 100 * th['explain_rate'], 1)
                M.put(f'Catch{S}{tag}', 100 * th['error_recall'], 1)
        x = cal['cross_subject'][s]
        M.put(f'ECEts{S}', x['ece_after'], 3)

    # ----------------------------------------------------- kalite (Pareto)
    par = load(R, '11_pareto', 'pareto_c5.json')
    key = {'shap_cheap': 'ShapC', 'lime_cheap': 'LimeC', 'lime': 'Lime', 'shap': 'Shap'}
    for p in par['noktalar']:
        k = key.get(p['nokta'])
        if not k:
            continue
        M.put(f'Faith{k}', p['faithfulness'], 3, '11_pareto/pareto_c5.json')
        M.put(f'FaithSem{k}', p['faithfulness_sem'], 3)
        M.put(f'Mono{k}', p['monotonicity'], 3)
        M.put(f'Sens{k}', p['sensitivity'], 3)
    M.put('QualN', par['config']['n_windows'], 0)
    sens = {p['nokta']: p['sensitivity'] for p in par['noktalar']}
    M.put('SensRatio', np.mean([sens['shap'], sens['shap_cheap']])
          / np.mean([sens['lime'], sens['lime_cheap']]), 1)

    # ----------------------------------------------------- dogruluk, tel
    st = load(R, '01_streaming', 'streaming_summary_native_fixed.json')
    c = st['classification']
    M.put('RtAcc', c['accuracy'], 3, '01_streaming (S11+S14, causal norm)')
    M.put('RtFOne', c['stress_f1'], 3)
    M.put('RtN', c['n_labeled_windows'], 0)
    M.put('ValFOne', st['run_config']['checkpoint_meta']['best_val_f1'], 3)
    M.put('NParams', st['run_config']['architecture']['num_params'], 0)
    sys.path.insert(0, ROOT)
    from serving import protocol as P
    from data.loader import NATIVE_STREAMS
    M.put('PayloadBytes', P.payload_bytes(NATIVE_STREAMS, 500) + P.HEADER.size, 0,
          'one 500 ms data frame')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write("% AUTO-GENERATED by tools/paper_numbers.py -- do not edit by hand.\n")
        f.write(f"% data version: {version}\n")
        f.write("\n".join(M.lines) + "\n")
    print(f"[i] {len(M.lines)} makro -> {args.out} ({version})")


if __name__ == '__main__':
    main()
