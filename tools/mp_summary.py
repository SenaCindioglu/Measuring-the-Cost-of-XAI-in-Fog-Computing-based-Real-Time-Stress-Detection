import json, glob, os, re

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
HOP = 5000.0
rows = []
for p in glob.glob(os.path.join(R, '18_pi_multiprocess', 'fog_mp_*.json')):
    d = json.load(open(p, encoding='utf-8-sig'))
    m = re.search(r'fog_mp_p(\d+)(b?)_c(\d+)_', os.path.basename(p))
    P, blas, C = int(m.group(1)), m.group(2) == 'b', int(m.group(3))
    s = d['server']
    po = d.get('pooled') or d['metrics']
    st = d['server_stages']
    res = s.get('resources') or {}
    th = s.get('thermal') or {}
    wpp = s.get('windows_per_process') or {}
    ex = po['execution']['mean_ms']
    rows.append(dict(
        tag=os.path.basename(p)[7:-5], P=P, blas=blas, C=C,
        P_json=s.get('processes'), blas_json=s['hardware'].get('blas_threads_limited'),
        n=d['n_measured'], xai=st['xai']['median_ms'], xai95=st['xai']['p95_ms'],
        ex=ex, que=po['queuing']['mean_ms'], que95=po['queuing']['p95_ms'],
        rmean=po['response']['mean_ms'], r95=po['response']['p95_ms'],
        rmax=po['response']['max_ms'],
        load=C * ex / (P * HOP),            # sunulan yuk / surec kapasitesi
        cap=P * HOP / ex,                    # kac istemciye yetisir
        cpu=(res.get('process_cpu_percent') or {}).get('mean'),
        cpumax=(res.get('process_cpu_percent') or {}).get('max'),
        rss=(res.get('process_rss_mb') or {}).get('max'),
        wpp=sorted(wpp.values()),
        t0=(th.get('start') or {}).get('cpu_temp_c'), t1=(th.get('end') or {}).get('cpu_temp_c'),
        thr=(th.get('end') or {}).get('throttled'),
        lost=(d.get('clock_sync') or {}).get('lost_replies'),
    ))

rows.sort(key=lambda r: (r['blas'], r['P'], r['C']))
print(f"{'kosu':12s} {'P':>2} {'C':>2} {'n':>4} {'xai':>6} {'xai95':>6} {'exec':>6} {'que':>7} "
      f"{'resp':>7} {'r95':>7} {'rmax':>7} {'yuk':>5} {'kap':>5} {'cpu%':>5} {'cpuMx':>5} {'rss':>5}  temp  thr  wpp")
for r in rows:
    print(f"{r['tag']:12s} {r['P']:>2} {r['C']:>2} {r['n']:>4} {r['xai']:6.0f} {r['xai95']:6.0f} {r['ex']:6.0f} "
          f"{r['que']:7.0f} {r['rmean']:7.0f} {r['r95']:7.0f} {r['rmax']:7.0f} {r['load']:5.2f} "
          f"{r['cap']:5.1f} {r['cpu']:5.0f} {r['cpumax']:5.0f} {r['rss']:5.0f}  "
          f"{r['t0']}-{r['t1']} {r['thr']}  {r['wpp']}"
          + ("" if (r['P_json'] == r['P'] and bool(r['blas_json']) == r['blas']) else
             f"  [!] json P={r['P_json']} blas={r['blas_json']}"))

