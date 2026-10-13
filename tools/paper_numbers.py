"""
tools/paper_numbers.py
=======================
Makaledeki HER olculmus sayiyi sonuc JSON'larindan uretir ve LaTeX makrosu
olarak paper/icaiic/numbers.tex dosyasina yazar. Makale metni sayi icermez,
yalnizca \\FogShapCRespMed gibi makrolar icerir.

Ayrica makaledeki her NITEL iddiayi (orn. "Pi'de SHAP-cheap 1 s sinirini
asar") veriyle denetler. Bir iddia veriyle celisirse betik hata koduyla
biter: metin yeniden yazilmadan makale derlenmemeli.

Istatistik politikasi (tek yerde, burada):
    pencere bazli gecikmeler -> medyan ve p95 (gerekirse p99)
    karisim dagilimlari      -> ortalama, adiyla (secici XAI, yuk, kapasite)
    tekrar birimi            -> kosu; her yapilandirma tek kosu (Limitations)

Veri kaynaklari (v2, acik dongu, ayni PyTorch 2.14.1 CPU):
    results/16_coloc/coloc_{none,shap_cheap,lime_cheap,lime,shap}.json
    results/<PI_DIR>/fog_pi_{none,lime,shap}.json       (lime/shap = *_cheap)
    results/<PI_DIR>/fog_pi_sel_{lime,shap}.json        (tau = 0.80)
    results/<PI_DIR>/fog_pi_load{1,2,4}.json            (shap_cheap, hep)
PI_DIR: 15_pi_ondemand (governor ondemand). Governor 'performance' ile
tekrarlanan kosular 16_pi_perf'e gelince PI_DIR degistirilir; metindeki
DVFS paragrafi o zaman yeniden okunmali (iddia denetimi uyarir).
Modelden bagimsiz analizler (v1'den gecerli):
    08_kalibrasyon, 11_pareto, 01_streaming, 15_v2/loop_check

Kullanim:  python -m tools.paper_numbers
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'paper', 'icaiic', 'numbers.tex')
HOP_MS = 5000.0
PI_PERF = '16_pi_perf'        # governor = performance (tek istemci profili)
PI_OD = '15_pi_ondemand'      # governor = ondemand (secici + cok istemci)
PI_DIR = PI_OD
REALTIME_BOUND_MS = 1000.0          # ShaTS: >1 s / pencere gercek zamanli degil

COLOC = {'none': 'coloc_none', 'shap_cheap': 'coloc_shap_cheap',
         'lime_cheap': 'coloc_lime_cheap', 'lime': 'coloc_lime', 'shap': 'coloc_shap'}
FOG = {'none': 'fog_pi_none_perf', 'lime_cheap': 'fog_pi_lime_perf',
       'shap_cheap': 'fog_pi_shap_perf', 'lime': 'fog_pi_lime_full_perf',
       'shap': 'fog_pi_shap_full_perf'}          # eksik olan atlanir
FOGOD = {'none': 'fog_pi_none', 'lime_cheap': 'fog_pi_lime', 'shap_cheap': 'fog_pi_shap'}
# Bulut: yalnizca v2 protokoluyle (acik dongu, torch 2.14.1+cpu imaji)
# yeniden olculurse girer. Dizin yoksa makale iki yerlesimle derlenir;
# dizin varsa BES dosya da zorunludur ve metindeki bulut iddialari
# denetlenir. v1 bulut kosulari (13_bulut) kapali donguyle alindi, girmez.
CLOUD_DIR = '17_bulut_openloop'
CLOUD = {'none': 'fog_cloud_baseline_retry1', 'shap_cheap': 'fog_cloud_xai_shap_ucuz',
         'lime_cheap': 'fog_cloud_xai_limeucuz', 'lime': 'fog_cloud_xai_lime',
         'shap': 'fog_cloud_xai_shap'}
WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
SEL = {'lime_cheap': 'fog_pi_sel_lime', 'shap_cheap': 'fog_pi_sel_shap'}
LOAD = {1: 'fog_pi_load1', 2: 'fog_pi_load2', 4: 'fog_pi_load4'}
TAG = {'none': 'None', 'shap_cheap': 'ShapC', 'lime_cheap': 'LimeC',
       'lime': 'Lime', 'shap': 'Shap'}


def load(*parts):
    with open(os.path.join(R, *parts)) as f:
        return json.load(f)


def canonical(m):
    return {'lime_ucuz': 'lime_cheap', 'shap_ucuz': 'shap_cheap'}.get(m, m)


def check_kunye(d, name, method, threshold, n_clients=1):
    """Dosyanin gercekten beklenen kosu oldugunu kunyeden dogrular."""
    x = (d.get('server') or {}).get('xai') or {}
    cfg = d.get('config') or {}
    problems = []
    if canonical(x.get('method') or 'none') != method:
        problems.append(f"method {x.get('method')} != {method}")
    if abs(float(x.get('threshold') or 0.0) - threshold) > 1e-9:
        problems.append(f"threshold {x.get('threshold')} != {threshold}")
    if not cfg.get('realtime'):
        problems.append("realtime degil")
    if (d.get('loop') or cfg.get('loop')) != 'open':
        problems.append("acik dongu degil")
    if cfg.get('n_clients', 1) != n_clients:
        problems.append(f"istemci {cfg.get('n_clients')} != {n_clients}")
    if problems:
        raise SystemExit(f"[KUNYE] {name}: " + "; ".join(problems))


def check_kunye_cloud(d, name, method):
    """deploy dalinin istemcisi XAI yontemini/esigini JSON'a yazmiyor;
    yontem dosya adindan gelir, burada yalnizca tutarliligi denetlenir."""
    cfg = d.get('config') or {}
    st = d.get('server_stages') or {}
    problems = []
    if not cfg.get('realtime'):
        problems.append("realtime degil")
    if not cfg.get('open_loop'):
        problems.append("acik dongu degil")
    if cfg.get('n_clients', 1) != 1:
        problems.append(f"istemci {cfg.get('n_clients')} != 1")
    if d.get('n_measured') != 184:
        problems.append(f"n_measured {d.get('n_measured')} != 184")
    if ('xai' in st) != (method != 'none'):
        problems.append(f"xai asamasi {'var' if 'xai' in st else 'yok'}, yontem {method}")
    if problems:
        raise SystemExit(f"[KUNYE] {name}: " + "; ".join(problems))


class Out:
    def __init__(self):
        self.lines = []
        self.claims = []

    def put(self, name, value, nd=1, src=''):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise SystemExit(f"[EKSIK] {name}")
        if isinstance(value, str):
            text = value
        else:
            text = f"{float(value):,.{nd}f}".replace(',', '{,}')
        self.lines.append(f"\\newcommand{{\\{name}}}{{{text}\\xspace}}"
                          + (f"  % {src}" if src else ''))

    def claim(self, text, ok):
        self.claims.append((text, bool(ok)))


def cell(d):
    m = d['metrics']
    st = d.get('server_stages') or {}
    res = ((d.get('server') or {}).get('resources')) or {}
    c = {
        'resp_med': m['response']['median_ms'], 'resp_p95': m['response']['p95_ms'],
        'resp_p99': m['response']['p99_ms'], 'resp_mean': m['response']['mean_ms'],
        'exec_med': m['execution']['median_ms'], 'exec_mean': m['execution']['mean_ms'],
        'age_med': (m.get('age') or {}).get('median_ms'),
        'age_p95': (m.get('age') or {}).get('p95_ms'),
        'lat_med': m['latency']['median_ms'], 'que_med': m['queuing']['median_ms'],
        'jit': m['response']['jitter_mean_abs_delta_ms'],
        'inf_med': st['inference']['median_ms'],
        'xai_med': st['xai']['median_ms'] if 'xai' in st else None,
        'xai_p95': st['xai']['p95_ms'] if 'xai' in st else None,
        'rss': (res.get('process_rss_mb') or {}).get('max'),
        'cpu': (res.get('process_cpu_percent') or {}).get('mean'),
        'n': d['n_measured'], 'kbps': d['bandwidth']['kbps_total'],
        'xai_per_expl': (d.get('xai') or {}).get('mean_ms_per_explanation'),
        'hw': (d.get('server') or {}).get('hardware') or {},
    }
    c['net'] = (c['resp_med'] - c['exec_med'] - c['que_med']) / 2
    c['pipe'] = c['exec_med'] - (c['xai_med'] or 0.0)
    return c


def main():
    o = Out()
    o.lines.append(r'\newif\ifvone')        # v1 uyari kutusu kalkti

    # ------------------------------------------------ tier hucreleri
    cells = {}
    has_cloud = os.path.isdir(os.path.join(R, CLOUD_DIR))
    o.lines.append(r'\newif\ifcloud' + (r' \cloudtrue' if has_cloud else ''))
    tiers = [('coloc', COLOC, 'Coloc', '16_coloc'),
             ('fog', FOG, 'Fog', PI_PERF),
             ('fogod', FOGOD, 'FogOD', PI_OD)]
    if has_cloud:
        tiers.append(('cloud', CLOUD, 'Cloud', CLOUD_DIR))
    for tier, files, T, sub in tiers:
        for m, fn in files.items():
            if not os.path.exists(os.path.join(R, sub, fn + '.json')):
                if m in ('none', 'lime_cheap', 'shap_cheap') or tier == 'cloud':
                    raise SystemExit(f"[EKSIK] {sub}/{fn}.json")
                print(f"[i] henuz yok, atlandi: {sub}/{fn}.json")
                continue
            d = load(sub, fn + '.json')
            if tier == 'cloud':
                check_kunye_cloud(d, fn, m)
            else:
                check_kunye(d, fn, m, 0.0 if m == 'none' else 1.0)
            c = cells[(tier, m)] = cell(d)
            p = T + TAG[m]
            src = f'{sub}/{fn}.json'
            o.put(p + 'RespMed', c['resp_med'], 1, src)
            o.put(p + 'RespPNF', c['resp_p95'], 1)
            o.put(p + 'RespPNN', c['resp_p99'], 1)
            o.put(p + 'RespMean', c['resp_mean'], 1)
            o.put(p + 'ExecMed', c['exec_med'], 1)
            o.put(p + 'ExecMean', c['exec_mean'], 1)
            if c['age_med'] is not None:
                o.put(p + 'AgeMed', c['age_med'], 1)
                o.put(p + 'AgePNF', c['age_p95'], 1)
            o.put(p + 'LatMed', c['lat_med'], 2)
            o.put(p + 'QueMed', c['que_med'], 2)
            o.put(p + 'Jit', c['jit'], 1)
            o.put(p + 'Net', c['net'], 1)
            o.put(p + 'RSS', c['rss'], 0)
            o.put(p + 'CPU', c['cpu'], 1)
            o.put(p + 'N', c['n'], 0)
            o.put(p + 'Kbps', c['kbps'], 1)
            o.put(p + 'Inf', c['inf_med'], 1)
            st = d.get('server_stages') or {}
            for sk, sn in [('resampling', 'Resamp'), ('preprocessing', 'Prep'),
                           ('tensor_prep', 'Tensor'), ('postprocessing', 'Post')]:
                if sk in st:
                    o.put(p + sn, st[sk]['median_ms'], 2)
            if m == 'none' and all(k in st for k in
                                   ('resampling', 'preprocessing', 'tensor_prep',
                                    'inference', 'postprocessing')):
                tot = sum(st[k]['median_ms'] for k in
                          ('resampling', 'preprocessing', 'tensor_prep',
                           'inference', 'postprocessing'))
                o.claim(f'{tier}: asama medyanlari toplami execution %15 icinde',
                        abs(tot - c['exec_med']) / c['exec_med'] < 0.15)
            o.put(p + 'HopPct', 100 * c['exec_med'] / HOP_MS, 1)
            o.put(p + 'Cap', math.floor(HOP_MS / c['exec_mean']), 0)
            if c['xai_med'] is not None:
                o.put(p + 'XaiMed', c['xai_med'], 0)
                o.put(p + 'XaiPNF', c['xai_p95'], 0)
                o.put(p + 'Pipe', c['pipe'], 1)
                o.put(p + 'XaiOverPipe', c['xai_med'] / c['pipe'], 0)

    # Donanim / yigin kunyesi (JSON'dan; SHAP surumu Pi'de JSON'da yok,
    # paper/pi_env.md'den)
    hc = cells[('coloc', 'none')]['hw']
    hf = cells[('fog', 'none')]['hw']
    o.put('ColocCores', hc['cpu_count'], 0)
    o.put('ColocRAM', hc['total_ram_mb'] / 1024, 0)
    o.put('ColocPy', hc['python'])
    o.put('ColocTorch', hc['torch'].split('+')[0])
    o.put('ColocShapVer', hc['shap'])
    o.put('FogModel', hf['device_model'].replace(' Rev 1.1', ''))
    o.put('FogRev', hf['device_model'].split('Rev ')[-1])
    o.put('FogCores', hf['cpu_count'], 0)
    o.put('FogGHz', hf['cpu_mhz_max'] / 1000, 1)
    o.put('FogRAM', round(hf['total_ram_mb'] / 1024), 0)
    o.put('FogPy', hf['python'])
    o.put('FogTorch', hf['torch'].split('+')[0])
    o.put('FogShapVer', '0.53.0', src='paper/pi_env.md')
    o.claim('ayni PyTorch surumu', hc['torch'] == hf['torch'])

    if has_cloud:
        hk = cells[('cloud', 'none')]['hw']
        o.put('CloudCores', hk['cpu_count'], 0)
        o.put('CloudRAM', round(hk['total_ram_mb'] / 1024), 0)
        o.put('CloudPy', hk['python'])
        o.put('CloudTorch', hk['torch'].split('+')[0])
        o.put('CloudShapVer', hk['package_versions']['shap'])
        o.claim('bulut: ayni PyTorch surumu', hk['torch'] == hc['torch'])
        kc = {m: cells[('cloud', m)] for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap')}
        # Yontem dosya adindan geldigi icin ornek sayisi oranlariyla dogrula
        o.claim('bulut: LIME / LIME-cheap aciklama orani ~2 (fark < 0.3)',
                abs(kc['lime']['xai_med'] / kc['lime_cheap']['xai_med'] - 2) < 0.3)
        o.claim('bulut: SHAP / SHAP-cheap aciklama orani ~4 (fark < 0.6)',
                abs(kc['shap']['xai_med'] / kc['shap_cheap']['xai_med'] - 4) < 0.6)
        o.claim('bulut: SHAP-cheap LIME-cheap\'ten hizli (laptop gibi, Pi\'nin tersi)',
                kc['shap_cheap']['xai_med'] < kc['lime_cheap']['xai_med'])
        o.claim('bulut: aciklama execution suresinin >= %90\'i',
                min(c['xai_med'] / c['exec_med'] for c in kc.values()) >= 0.90)
        for m in ('lime_cheap', 'shap_cheap'):
            o.put(f'FogOverCloudXai{TAG[m]}',
                  cells[('fog', m)]['xai_med'] / kc[m]['xai_med'], 1)
            o.put(f'CloudOverColocXai{TAG[m]}',
                  kc[m]['xai_med'] / cells[('coloc', m)]['xai_med'], 1)
            o.claim(f'bulut {m}: WAN gecikmesine ragmen yanit Pi\'den hizli',
                    kc[m]['resp_med'] < cells[('fog', m)]['resp_med'])
            o.claim(f'bulut {m}: aciklama laptoptan yavas, Pi\'den hizli',
                    cells[('coloc', m)]['xai_med'] < kc[m]['xai_med']
                    < cells[('fog', m)]['xai_med'])
        o.claim('bulut: tum XAI yanitlari medyanda 1 s alti (tam SHAP haric)',
                all(kc[m]['resp_med'] < REALTIME_BOUND_MS
                    for m in ('shap_cheap', 'lime_cheap', 'lime')))
        o.claim('bulut: tam SHAP medyani 1 s ustu',
                kc['shap']['resp_med'] > REALTIME_BOUND_MS)

    # Turetilmis
    cn, fn_ = cells[('coloc', 'none')], cells[('fog', 'none')]
    fod = cells[('fogod', 'none')]
    FM = [m for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap') if ('fog', m) in cells]
    o.lines.append(r'\newif\ifpishapfull' + (r' \pishapfulltrue' if 'shap' in FM else ''))
    o.put('FogNPoints', WORD[len(FM)])
    for m in FM:
        r = cells[('fog', m)]['xai_med'] / cells[('coloc', m)]['xai_med']
        o.put(f'FogOverColocXai{TAG[m]}', r, 1)
        o.claim(f'Pi {m} aciklamasi co-located\'dan yavas', r > 1)
    o.put('FogOverColocNoneExec', fn_['exec_med'] / cn['exec_med'], 1)
    o.claim('siralama tersine: hostta SHAP-cheap LIME-cheap\'ten hizli, Pi\'de yavas',
            cells[('coloc', 'shap_cheap')]['xai_med'] < cells[('coloc', 'lime_cheap')]['xai_med']
            and cells[('fog', 'shap_cheap')]['xai_med'] > cells[('fog', 'lime_cheap')]['xai_med'])
    o.claim('hostta SHAP-cheap tum aciklayicilarin en hizlisi',
            cells[('coloc', 'shap_cheap')]['xai_med'] == min(
                cells[('coloc', m)]['xai_med'] for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap')))
    # "Maliyet ornek sayisiyla olcekleniyor" cumlesinin denetimi
    o.claim('coloc: LIME / LIME-cheap aciklama orani ~2 (fark < 0.3)',
            abs(cells[('coloc', 'lime')]['xai_med']
                / cells[('coloc', 'lime_cheap')]['xai_med'] - 2) < 0.3)
    o.claim('coloc: SHAP / SHAP-cheap aciklama orani ~4 (fark < 0.6)',
            abs(cells[('coloc', 'shap')]['xai_med']
                / cells[('coloc', 'shap_cheap')]['xai_med'] - 4) < 0.6)
    xc = [cells[('coloc', m)]['xai_med'] for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap')]
    o.put('ColocXaiMin', min(xc), 0)
    o.put('ColocXaiMax', max(xc), 0)
    rc = [cells[('coloc', m)]['xai_med'] / cells[('coloc', m)]['pipe']
          for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap')]
    share = [cells[(t, m)]['xai_med'] / cells[(t, m)]['exec_med']
             for t, ms in [('coloc', ('shap_cheap', 'lime_cheap', 'lime', 'shap')),
                           ('fog', FM)] for m in ms]
    o.put('XaiShareMin', 100 * min(share), 0)
    o.put('XaiShareMax', 100 * max(share), 0)
    o.claim('aciklama execution suresinin >= %90\'i (tum XAI hucreleri)', min(share) >= 0.90)
    xf = [cells[('fog', m)]['xai_med'] for m in FM]
    o.put('FogXaiMin', min(xf), 0)
    o.put('FogXaiMax', max(xf), 0)
    # Cevap - execution: pencere BAZINDA farkin medyani (medyan farki degil).
    # Ileri yon gecikmesi + kuyruk + cevap yolunu birlikte icerir.
    def resid(sub, fn):
        r = load(sub, fn + '.json')['raw']
        return float(np.median(np.array(r['response_ms']) - np.array(r['execution_ms'])))
    o.put('FogLimeCResidual', resid(PI_PERF, FOG['lime_cheap']), 1)
    o.put('FogNoneResidual', resid(PI_PERF, FOG['none']), 1)
    o.put('FogShapCResidual', resid(PI_PERF, FOG['shap_cheap']), 1)
    # Pi'de LIME'in ortalama kullandigi cekirdek: CPU% / (100 * mesgul oran)
    fl = cells[('fog', 'lime_cheap')]
    o.put('FogLimeCCores', fl['cpu'] / (100 * fl['exec_mean'] / HOP_MS), 1)
    o.put('FogLimeCBusy', 100 * fl['exec_mean'] / HOP_MS, 1)
    # Gonderim gecikmesi (acik dongu sadakati), tek istemcili Pi kosulari
    lags = [load(PI_PERF, FOG[m] + '.json')['metrics']['send_lag'] for m in ['none'] + FM]
    lags += [load(PI_OD, fn + '.json')['metrics']['send_lag'] for fn in SEL.values()]
    o.put('PiLagPNFMax', max(l['p95_ms'] for l in lags), 0)
    o.put('PiLagMax', max(l['max_ms'] for l in lags), 0)
    o.put('ColocRatioMin', min(rc), 0)
    o.put('ColocRatioMax', max(rc), 0)
    rf = [cells[('fog', m)]['xai_med'] / cells[('fog', m)]['pipe'] for m in FM]
    o.put('FogRatioMin', min(rf), 0)
    o.put('FogRatioMax', max(rf), 0)
    # Pi'de frekans olcekleme: ondemand'da ayni asamalar CPU bostayken yavas;
    # performance governor'da fark kayboluyor.
    inf_idle = fod['inf_med']
    inf_busy_od = np.median([cells[('fogod', m)]['inf_med'] for m in ('lime_cheap', 'shap_cheap')])
    inf_perf = fn_['inf_med']
    inf_busy_perf = np.median([cells[('fog', m)]['inf_med'] for m in ('lime_cheap', 'shap_cheap')])
    o.put('FogInfIdle', inf_idle, 1)
    o.put('FogInfBusy', inf_busy_od, 1)
    o.put('FogDvfs', inf_idle / inf_busy_od, 1)
    o.put('FogInfPerf', inf_perf, 1)
    o.put('FogInfBusyPerf', inf_busy_perf, 1)
    o.put('FogGovGain', fod['exec_med'] / fn_['exec_med'], 1)
    o.claim('ondemand: bostaki cikarim mesguldekinden >=1.5x yavas', inf_idle / inf_busy_od >= 1.5)
    o.claim('performance: bostaki ve mesguldeki cikarim %15 icinde',
            abs(inf_perf - inf_busy_perf) / inf_busy_perf < 0.15)
    o.claim("performance governor XAI'siz execution'i >=1.5x hizlandirdi",
            fod['exec_med'] / fn_['exec_med'] >= 1.5)
    for m in ('lime_cheap', 'shap_cheap'):
        g = cells[('fogod', m)]['xai_med'] / cells[('fog', m)]['xai_med']
        o.put(f'FogGovXai{TAG[m]}', 100 * (g - 1), 0)
        o.claim(f"governor {m} aciklama suresini %15'ten az degistirdi", abs(g - 1) < 0.15)
    ci = cn['inf_med']
    o.claim('co-located: XAI\'siz ve LIME kosusunda cikarim farki <%25',
            abs(cells[('coloc', 'lime_cheap')]['inf_med'] - ci) / ci < 0.25)

    # Gercek zamanli sinir (ShaTS, 1 s)
    o.claim('Pi SHAP-cheap response medyani 1 s ustu',
            cells[('fog', 'shap_cheap')]['resp_med'] > REALTIME_BOUND_MS)
    shap_min = min(load(PI_PERF, FOG['shap_cheap'] + '.json')['raw']['response_ms'])
    o.put('FogShapCRespMin', shap_min, 0)
    o.claim('Pi SHAP-cheap: TUM yanitlar 1 s ustu', shap_min > REALTIME_BOUND_MS)
    o.claim('Pi LIME-cheap response p95 1 s alti',
            cells[('fog', 'lime_cheap')]['resp_p95'] < REALTIME_BOUND_MS)
    if 'lime' in FM:
        o.claim('Pi tam LIME medyani 1 s ustu', cells[('fog', 'lime')]['resp_med'] > REALTIME_BOUND_MS)
    if 'shap' in FM:
        o.claim('Pi tam SHAP medyani 1 s ustu', cells[('fog', 'shap')]['resp_med'] > REALTIME_BOUND_MS)
    o.claim('co-located tum aciklayicilarda p95 1 s alti',
            all(cells[('coloc', m)]['resp_p95'] < REALTIME_BOUND_MS
                for m in ('shap_cheap', 'lime_cheap', 'lime', 'shap')))
    # LAN yerlesimleri; bulutta saat senkronu belirsizligi (~50 ms) tek yon
    # gecikmeden buyuk, orada yerine yarim RTT (Net) raporlanir.
    lat = [c['lat_med'] for k, c in cells.items() if k[0] in ('coloc', 'fog')]
    o.put('LatMaxAll', max(lat), 1)
    o.claim('tek yon gecikme medyani tum LAN tek-istemci kosularinda < 5 ms', max(lat) < 5)
    que = [c['que_med'] for k, c in cells.items() if k[0] != 'fogod']
    o.put('QueMaxAll', max(que), 1)
    o.claim('kuyruk medyani tek istemcide <= 0.25 ms', max(que) <= 0.25)
    if has_cloud:
        nets = [c['net'] for k, c in cells.items() if k[0] == 'cloud']
        o.put('CloudNetMin', min(nets), 0)
        o.put('CloudNetMax', max(nets), 0)
    kb = [c['kbps'] for k, c in cells.items() if k[0] != 'fogod']
    o.put('KbpsAll', np.median(kb), 0)
    o.claim('bant genisligi tum kosularda ayni (fark < %1)',
            (max(kb) - min(kb)) / min(kb) < 0.01)
    grow = [cells[(t, m)]['rss'] / cells[(t, 'none')]['rss']
            for t, ms in [('coloc', ('shap_cheap', 'lime_cheap', 'lime', 'shap')),
                          ('fog', FM)] for m in ms]
    o.put('RssGrowMin', min(grow), 1)
    o.put('RssGrowMax', max(grow), 1)
    o.put('FogRssPct', 100 * max(cells[('fog', m)]['rss'] for m in FM)
          / cells[('fog', 'none')]['hw']['total_ram_mb'], 0)
    o.claim('aciklayici yuklu iken bellek >= 1.5x', min(grow) >= 1.5)
    o.claim('XAI\'siz pipeline hop\'un %1\'inden az',
            max(cn['exec_med'], fn_['exec_med']) < 0.01 * HOP_MS)

    # ------------------------------------------------ secici XAI (Pi)
    sel_counts = []
    for m, fn in SEL.items():
        d = load(PI_DIR, fn + '.json')
        check_kunye(d, fn, m, 0.80)
        s, a = cell(d), cells[('fogod', m)]
        x = d['xai']
        p = 'Sel' + TAG[m]
        src = f'{PI_DIR}/{fn}.json'
        sel_counts.append(x['n_explained'])
        o.put(p + 'N', d['n_measured'], 0, src)
        o.put(p + 'NExpl', x['n_explained'], 0)
        o.put(p + 'Rate', 100 * x['explain_rate'], 1)
        o.put(p + 'PerExpl', x['mean_ms_per_explanation'], 0)
        o.put(p + 'ExecMean', s['exec_mean'], 1)
        o.put(p + 'ExecMed', s['exec_med'], 1)
        o.put(p + 'RespMean', s['resp_mean'], 1)
        o.put(p + 'RespMed', s['resp_med'], 1)
        o.put(p + 'RespPNF', s['resp_p95'], 0)
        o.put(p + 'RSS', s['rss'], 0)
        o.put(p + 'CPU', s['cpu'], 1)
        o.put(p + 'Cap', math.floor(HOP_MS / s['exec_mean']), 0)
        o.put(p + 'ExecGain', a['exec_mean'] / s['exec_mean'], 1)
        o.put(p + 'RespGain', a['resp_mean'] / s['resp_mean'], 1)
        o.put(p + 'CapGain', math.floor(HOP_MS / s['exec_mean'])
              / math.floor(HOP_MS / a['exec_mean']), 1)
        dpe = abs(x['mean_ms_per_explanation'] - a['xai_per_expl']) / a['xai_per_expl']
        o.put(p + 'PerExplDiff', 100 * dpe, 1)
        o.claim(f'{m}: aciklama basi maliyet secici ve hep kosularinda %2 icinde', dpe < 0.02)
        # Karisim modeli: E[exec] = E[exec | aciklama yok] + r * E[xai]
        pred = fod['exec_mean'] + x['explain_rate'] * x['mean_ms_per_explanation']
        err = abs(pred - s['exec_mean']) / s['exec_mean']
        o.put(p + 'ProjExec', pred, 1)
        o.put(p + 'ProjErr', 100 * err, 1)
        o.claim(f'{m}: karisim modeli olcumu %3 icinde tahmin ediyor', err < 0.03)
        o.claim(f'{m}: aciklanan pencerede yanit yine 1 s ' +
                ('ustu' if m == 'shap_cheap' else 'alti'),
                (s['resp_p95'] > REALTIME_BOUND_MS) == (m == 'shap_cheap'))
        raw = d['raw']
        ex = [r for r, e in zip(raw['response_ms'], raw['explained']) if e]
        o.put(p + 'ExplRespMin', min(ex), 0)
        o.put(p + 'ExplRespMax', max(ex), 0)
        if m == 'shap_cheap':
            o.claim('secici SHAP-cheap: aciklanan TUM pencereler 1 s ustu',
                    min(ex) > REALTIME_BOUND_MS)
        else:
            o.claim('secici LIME-cheap: aciklanan TUM pencereler 1 s alti',
                    max(ex) < REALTIME_BOUND_MS)
        if m == 'lime_cheap':
            c = d['classification']
            o.put('SelAcc', c['accuracy'], 3, src)
            o.put('SelFOne', c['stress_f1'], 3)
            o.put('SelNLab', c['n_labeled'], 0)
            o.put('SelMinutes', d['wall_seconds'] / 60, 0)
    o.claim('iki aciklayicida aciklanan pencere sayisi ayni', len(set(sel_counts)) == 1)
    diffs, errs, gains, capg = [], [], [], []
    for m in SEL:
        d = load(PI_DIR, SEL[m] + '.json')
        a = cells[('fogod', m)]
        x = d['xai']
        diffs.append(abs(x['mean_ms_per_explanation'] - a['xai_per_expl']) / a['xai_per_expl'])
        pred = fod['exec_mean'] + x['explain_rate'] * x['mean_ms_per_explanation']
        errs.append(abs(pred - d['metrics']['execution']['mean_ms'])
                    / d['metrics']['execution']['mean_ms'])
        gains.append(a['exec_mean'] / d['metrics']['execution']['mean_ms'])
        capg.append(math.floor(HOP_MS / d['metrics']['execution']['mean_ms'])
                    / math.floor(HOP_MS / a['exec_mean']))
    o.put('SelPerExplDiffMax', 100 * max(diffs), 1)
    o.put('SelProjErrMax', 100 * max(errs), 1)
    o.put('SelProjErrRound', round(100 * max(errs)), 0)
    o.put('SelExecGainMin', min(gains), 1)
    o.put('SelExecGainMax', max(gains), 1)
    o.put('SelCapGainMin', min(capg), 1)
    o.put('SelCapGainMax', max(capg), 1)
    o.claim('kapasite kazanci ile maliyet kazanci benzer (fark < %10)',
            abs(np.mean(capg) - np.mean(gains)) / np.mean(gains) < 0.10)
    o.put('AlwMinutes', load(PI_DIR, 'fog_pi_none.json')['wall_seconds'] / 60, 0)

    # ------------------------------------------------ cok istemci (Pi, SHAP-cheap)
    load_err, load_lag = [], []
    for n, fn in LOAD.items():
        d = load(PI_DIR, fn + '.json')
        load_lag.append(d['metrics']['send_lag']['max_ms'])
        check_kunye(d, fn, 'shap_cheap', 1.0, n_clients=n)
        pool = d.get('pooled') or {k: d['metrics'][k] for k in ('response', 'execution')}
        e_mean = pool['execution']['mean_ms']
        r = pool['response']
        name = {1: 'One', 2: 'Two', 4: 'Four'}[n]
        src = f'{PI_DIR}/{fn}.json'
        o.put(f'Load{name}N', d['n_measured'], 0, src)
        o.put(f'Load{name}RespMean', r['mean_ms'], 0)
        o.put(f'Load{name}RespMed', r['median_ms'], 0)
        o.put(f'Load{name}RespPNF', r['p95_ms'], 0)
        o.put(f'Load{name}Util', 100 * n * e_mean / HOP_MS, 0)
        pm = (n + 1) / 2 * e_mean
        o.put(f'Load{name}PredMean', pm, 0)
        o.put(f'Load{name}PredMax', n * e_mean, 0)
        load_err += [abs(pm - r['mean_ms']) / r['mean_ms'],
                     abs(n * e_mean - r['p95_ms']) / r['p95_ms']]
        o.claim(f'{n} istemci: ortalama yanit (N+1)/2*E modeliyle %5 icinde',
                abs(pm - r['mean_ms']) / r['mean_ms'] < 0.05)
        o.claim(f'{n} istemci: p95 yanit N*E modeliyle %5 icinde',
                abs(n * e_mean - r['p95_ms']) / r['p95_ms'] < 0.05)
    o.put('LoadErrMax', 100 * max(load_err), 1)
    o.put('LoadLagMax', max(load_lag) / 1000, 1)
    o.put('LoadExec', load(PI_DIR, 'fog_pi_load4.json')['pooled']['execution']['mean_ms'], 0)

    # ------------------------------------------------ kalibrasyon (model ozelligi)
    cal = load('08_kalibrasyon', 'calibration_d1.json')
    for s, S in [('S11', 'Eleven'), ('S14', 'Fourteen')]:
        u = cal['per_subject'][s]['uncalibrated']
        o.put(f'Acc{S}', u['accuracy'], 3, '08_kalibrasyon/calibration_d1.json')
        o.put(f'Conf{S}', u['avg_confidence'], 3)
        o.put(f'ECE{S}', u['ece'], 3)
        o.put(f'CalN{S}', u['n'], 0)
        for th in u['thresholds']:
            tag = {0.7: 'Seventy', 0.8: 'Eighty', 0.9: 'Ninety'}.get(round(th['threshold'], 2))
            if tag:
                o.put(f'Rate{S}{tag}', 100 * th['explain_rate'], 1)
                o.put(f'Catch{S}{tag}', 100 * th['error_recall'], 1)
                o.put(f'Caught{S}{tag}', th['n_errors_caught'], 0)
                o.put(f'Errors{S}{tag}', th['n_errors_total'], 0)
        o.put(f'ECEts{S}', cal['cross_subject'][s]['ece_after'], 3)
        o.claim(f'{s}: sicaklik olcekleme ECE\'yi kotulestirdi',
                cal['cross_subject'][s]['ece_after'] > u['ece'])
    o.put('ECERatio', cal['per_subject']['S14']['uncalibrated']['ece']
          / cal['per_subject']['S11']['uncalibrated']['ece'], 1)

    # ------------------------------------------------ aciklama kalitesi
    par = load('11_pareto', 'pareto_c5.json')
    key = {'shap_cheap': 'ShapC', 'lime_cheap': 'LimeC', 'lime': 'Lime', 'shap': 'Shap'}
    pts = {p['nokta']: p for p in par['noktalar'] if p['nokta'] in key}
    for k, p in pts.items():
        o.put(f'Faith{key[k]}', p['faithfulness'], 3, '11_pareto/pareto_c5.json')
        o.put(f'FaithSem{key[k]}', p['faithfulness_sem'], 3)
        o.put(f'Mono{key[k]}', p['monotonicity'], 3)
        o.put(f'Sens{key[k]}', p['sensitivity'], 3)
    o.put('QualN', par['config']['n_windows'], 0)
    o.put('QualSensN', par['config']['sensitivity_windows'], 0)
    o.put('QualTrials', par['config']['trials'], 0)
    o.put('SensRatio', np.mean([pts['shap']['sensitivity'], pts['shap_cheap']['sensitivity']])
          / np.mean([pts['lime']['sensitivity'], pts['lime_cheap']['sensitivity']]), 1)
    names = list(pts)
    o.claim('sadakat farklari standart hata cubuklari icinde (cift bazinda)',
            all(abs(pts[a]['faithfulness'] - pts[b]['faithfulness'])
                < pts[a]['faithfulness_sem'] + pts[b]['faithfulness_sem']
                for i, a in enumerate(names) for b in names[i + 1:]))
    o.claim('LIME duyarliligi SHAP\'in yarisindan az',
            max(pts['lime']['sensitivity'], pts['lime_cheap']['sensitivity'])
            < 0.5 * min(pts['shap']['sensitivity'], pts['shap_cheap']['sensitivity']))

    # ------------------------------------------------ dogruluk, tel, dongu kontrolu
    st = load('01_streaming', 'streaming_summary_native_fixed.json')
    c = st['classification']
    o.put('RtAcc', c['accuracy'], 3, '01_streaming (S11+S14, causal norm)')
    o.put('RtFOne', c['stress_f1'], 3)
    o.put('RtN', c['n_labeled_windows'], 0)
    o.put('ValFOne', st['run_config']['checkpoint_meta']['best_val_f1'], 3)
    o.put('NParams', st['run_config']['architecture']['num_params'], 0)
    sys.path.insert(0, ROOT)
    from serving import protocol as P
    from data.loader import NATIVE_STREAMS
    o.put('PayloadBytes', P.payload_bytes(NATIVE_STREAMS, 500) + P.HEADER.size, 0,
          'one 500 ms data frame')
    for mode, Mode in [('closed', 'Closed'), ('open', 'Open')]:
        # v1 kosulari eski_ve_hatali/ altina tasindi; bu dosya kasitli olarak
        # "kapali dongunun kusuru"nun kaniti olarak kullaniliyor.
        lc = next(p for p in (('15_v2', 'loop_check'),
                              ('eski_ve_hatali', '15_v2', 'loop_check'))
                  if os.path.exists(os.path.join(R, *p, f'fog_t_{mode}_shap.json')))
        d = load(*lc, f'fog_t_{mode}_shap.json')
        o.put(f'Loop{Mode}LagPNF', d['metrics']['send_lag']['p95_ms'], 0,
              f'15_v2/loop_check/fog_t_{mode}_shap.json')
        o.put(f'Loop{Mode}N', d['n_measured'], 0)
        o.put(f'Loop{Mode}Packets', d['n_packets'], 0)
        o.put(f'Loop{Mode}Torch', d['server']['hardware']['torch'].split('+')[0])

    # Kosular arasi yayilim (tekrarli olcum, tek iplik, host)
    b = load('02_benchmark', 'benchmark_thread_sweep.json')['by_threads']['1']
    o.put('RepeatRuns', b['n_runs'], 0, '02_benchmark/benchmark_thread_sweep.json')
    o.put('RepeatSpread', b['stages']['total']['median_of_run_medians']['range_pct'], 1)
    o.put('RepeatSpreadPNF', b['stages']['total']['median_of_run_p95']['range_pct'], 1)

    # Saat belirsizligi: kullanilan tum kosularin en buyugu
    unc = []
    for sub, fns in [('16_coloc', COLOC.values()),
                     (PI_PERF, [FOG[m] for m in ['none'] + FM]),
                     (PI_OD, list(FOGOD.values()) + list(SEL.values()) + list(LOAD.values()))]:
        for fn in fns:
            unc.append(load(sub, fn + '.json')['clock_sync']['uncertainty_ms'])
    o.put('ClockUncMax', max(unc), 1)

    # ------------------------------------------------ yaz
    failed = [t for t, ok in o.claims if not ok]
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write("% AUTO-GENERATED by tools/paper_numbers.py -- do not edit by hand.\n")
        f.write("% Claim checks: " + ("ALL PASSED" if not failed else "FAILED") + "\n")
        for t, ok in o.claims:
            f.write(f"%   [{'ok' if ok else 'FAIL'}] {t}\n")
        f.write("\n".join(o.lines) + "\n")
    print(f"[i] {len(o.lines)} makro, {len(o.claims)} iddia -> {OUT}")
    for t, ok in o.claims:
        if not ok:
            print(f"[FAIL] {t}")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
