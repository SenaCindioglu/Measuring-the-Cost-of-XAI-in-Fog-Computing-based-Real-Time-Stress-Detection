"""
compare_deployments.py
=======================
Ayni deneyi birden fazla YERLESIMDE (edge / fog / cloud) kosar ve tek bir
karsilastirma tablosu uretir.

NEDEN AYRI BIR SCRIPT
---------------------
Uc ortami elle kosup sayilari yan yana koymak iki hata uretir:

  1) Adil olmayan kosullar. Farkli denek, farkli sure ya da farkli tick
     kullanilirsa tablo karsilastirilamaz. Burada tum ortamlar ayni
     parametrelerle kosar, degisen tek sey sunucunun yeri.

  2) Isinma kaymasi. Ayni ortamin tekrarlari arka arkaya kosulursa islemci
     isinir ve SON kosulan ortam sistematik olarak yavas gorunur. Bu script
     DONUSUMLU sirada kosar (tur 1: edge, fog / tur 2: edge, fog ...),
     boylece kayma hepsine esit dagilir.

Tek kosunun sonucu guvenilir degildir (daha once %15'e varan oynama olculdu),
bu yuzden her ortam icin medyan ve tekrarlar arasi yayilim birlikte verilir.

KULLANIM
--------
    # edge (yerelde sunucu baslatilir) ile fog (Pi) karsilastirmasi
    python compare_deployments.py --subject S11 --realtime --max-ticks 600 \
        --repeat 3 --deployment edge=local fog=192.168.1.42

    # bulut da eklenirse tek satir yeter
    python compare_deployments.py --subject S11 --realtime --max-ticks 600 \
        --repeat 3 --deployment edge=local fog=192.168.1.42 cloud=203.0.113.7

'local' yazarsan sunucu bu makinede otomatik baslatilir ve kosu bitince
kapatilir. Diger adresler icin sunucunun O MAKINEDE ZATEN CALISIYOR olmasi
gerekir:

    python -m serving.server --host 0.0.0.0 --port 9099
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

from config import config
from data.loader import load_wesad_subject_native
from serving.client import summarize
from run_fog_experiment import _ClientArgs, wait_for_port, run_all, aggregate

DEFAULT_PORT = 9099

# Tabloda gosterilecek metrikler: (anahtar, etiket)
METRICS = [
    ('latency', 'latency'),
    ('queuing', 'queuing delay'),
    ('execution', 'execution'),
    ('response', 'response time'),
]


def parse_deployment(spec):
    """'ad=host[:port]' ya da 'ad=local' -> (ad, host, port, yerel_mi)"""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f"Yerlesim '{spec}' hatali. Bicim: ad=host[:port] ya da ad=local")
    name, target = spec.split('=', 1)
    if target.strip().lower() == 'local':
        return name, '127.0.0.1', DEFAULT_PORT, True
    if ':' in target:
        host, port = target.rsplit(':', 1)
        return name, host, int(port), False
    return name, target, DEFAULT_PORT, False


def spread(values):
    v = np.array(values, dtype=float)
    return {
        'median': float(np.median(v)),
        'min': float(v.min()), 'max': float(v.max()),
        'range_pct': float((v.max() - v.min()) / v.mean() * 100.0) if v.mean() else 0.0,
        'runs': [float(x) for x in v],
    }


def start_local_server(args, host, port):
    cmd = [sys.executable, '-m', 'serving.server', '--host', host, '--port', str(port),
           '--model', args.model, '--norm-mode', args.norm_mode,
           '--threads', str(args.threads)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not wait_for_port(host, port):
        proc.kill()
        raise SystemExit(f"Yerel sunucu baslatilamadi ({host}:{port})")
    return proc


def one_deployment_run(args, cfg, loaded, subjects, host, port):
    ca_list = [_ClientArgs(host=host, port=port, subject=s,
                           tick_ms=args.tick_ms, norm_mode=args.norm_mode,
                           realtime=args.realtime, max_ticks=args.max_ticks)
               for s in subjects]
    results = asyncio.run(run_all(ca_list, cfg, loaded))
    summaries = []
    for (records, bs, br, w, clock, srv), subj in zip(results, subjects):
        s = summarize(records, bs, br, w, args.warmup_windows, cfg, clock, srv)
        s['subject'] = subj
        summaries.append(s)
    return aggregate(summaries)


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', nargs='+', required=True,
                        help="ad=host[:port] ya da ad=local (birden fazla verilebilir)")
    parser.add_argument('--subject', nargs='+', required=True)
    parser.add_argument('--clients', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=3, help="Her ortam icin tur sayisi")
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE)
    parser.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS)
    parser.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS)
    parser.add_argument('--threads', type=int, default=cfg.DEFAULT_THREADS,
                        help="Yalnizca YEREL sunucu icin; uzak sunucunun ipligi orada ayarlanir")
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--max-ticks', type=int, default=0)
    parser.add_argument('--tag', default='deployments')
    args = parser.parse_args()

    deployments = [parse_deployment(d) for d in args.deployment]
    n_clients = args.clients or len(args.subject)
    subjects = [args.subject[i % len(args.subject)] for i in range(n_clients)]

    print("=" * 78)
    print("YERLESIM KARSILASTIRMASI")
    print("=" * 78)
    for name, host, port, local in deployments:
        print(f"  {name:<10} {host}:{port}" + ("   (yerelde baslatilacak)" if local else ""))
    print(f"  denek(ler) : {subjects}   istemci: {n_clients}")
    print(f"  tekrar     : {args.repeat} tur x {len(deployments)} ortam "
          f"= {args.repeat * len(deployments)} kosu   (donusumlu sirada)")
    print(f"  tick {args.tick_ms}ms | realtime {args.realtime} | "
          f"max-ticks {args.max_ticks or 'sinirsiz'} | norm {args.norm_mode}")
    print("=" * 78)

    # Uzak sunucularin acik oldugunu once dogrula: yarim saat kostuktan sonra
    # "baglanti yok" hatasi almak istemiyoruz.
    for name, host, port, local in deployments:
        if not local and not wait_for_port(host, port, timeout=5.0):
            raise SystemExit(
                f"'{name}' ({host}:{port}) erisilemiyor.\n"
                f"O makinede sunucuyu baslat:\n"
                f"    python -m serving.server --host 0.0.0.0 --port {port}")

    loaded = {}
    for s in set(subjects):
        st, lb = load_wesad_subject_native(s, cfg.DATA_PATH)
        if st is None:
            raise SystemExit(f"{s} yuklenemedi")
        loaded[s] = (st, lb)

    runs = {name: [] for name, _, _, _ in deployments}

    for r in range(args.repeat):
        for name, host, port, local in deployments:     # donusumlu
            proc = start_local_server(args, host, port) if local else None
            try:
                t0 = time.perf_counter()
                agg = one_deployment_run(args, cfg, loaded, subjects, host, port)
                agg['wall_s'] = time.perf_counter() - t0
                runs[name].append(agg)
                m = agg['metrics']
                print(f"  tur {r+1}/{args.repeat} | {name:<8} -> "
                      f"yanit {m['response']['median_ms']:7.3f} ms | "
                      f"cikarim {m['execution']['median_ms']:7.3f} ms | "
                      f"ag {m['latency']['median_ms']:6.3f} ms")
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    time.sleep(0.5)      # port serbest kalsin

    # ---------------- ozet ----------------
    report = {'config': {
        'deployments': [{'name': n, 'host': h, 'port': p, 'local': l}
                        for n, h, p, l in deployments],
        'subjects': subjects, 'n_clients': n_clients, 'repeat': args.repeat,
        'tick_ms': args.tick_ms, 'realtime': args.realtime,
        'norm_mode': args.norm_mode, 'max_ticks': args.max_ticks,
        'warmup_windows': args.warmup_windows,
    }, 'by_deployment': {}}

    for name, host, port, local in deployments:
        aggs = runs[name]
        entry = {'host': f'{host}:{port}', 'local': local, 'n_runs': len(aggs),
                 'metrics': {}, 'clock_synced_same_machine': local}
        for key, _ in METRICS:
            entry['metrics'][key] = {
                'median': spread([a['metrics'][key]['median_ms'] for a in aggs]),
                'p95': spread([a['metrics'][key]['p95_ms'] for a in aggs]),
                'jitter': spread([a['metrics'][key]['jitter_mean_abs_delta_ms'] for a in aggs]),
            }
        entry['bandwidth_kbps'] = spread([a['bandwidth']['kbps_total'] for a in aggs])
        srv = next((a.get('server') for a in aggs if a.get('server')), None)
        if srv:
            entry['server_hardware'] = srv.get('hardware')
            cpu = [a['server']['resources']['process_cpu_percent']['mean']
                   for a in aggs if a.get('server', {}).get('resources', {}).get('process_cpu_percent')]
            ram = [a['server']['resources']['process_rss_mb']['max']
                   for a in aggs if a.get('server', {}).get('resources', {}).get('process_rss_mb')]
            if cpu:
                entry['server_cpu_percent'] = spread(cpu)
            if ram:
                entry['server_rss_mb'] = spread(ram)
        if aggs[0].get('classification'):
            entry['classification'] = aggs[0]['classification']
        report['by_deployment'][name] = entry

    print("\n" + "=" * 78)
    print(f"SONUC  ({args.repeat} turun medyani; parantez icinde tekrarlar arasi yayilim)")
    print("=" * 78)
    print(f"{'ortam':<10}" + ''.join(f"{lab:>17}" for _, lab in METRICS) + f"{'kbps':>10}")
    for name, _, _, _ in deployments:
        e = report['by_deployment'][name]
        row = f"{name:<10}"
        for key, _ in METRICS:
            m = e['metrics'][key]['median']
            row += f"{m['median']:>10.3f} (%{m['range_pct']:>3.0f})"
        row += f"{e['bandwidth_kbps']['median']:>10.1f}"
        print(row)

    print(f"\n{'ortam':<10}{'donanim':<34}{'CPU %':>9}{'bellek MB':>12}")
    for name, _, _, _ in deployments:
        e = report['by_deployment'][name]
        hw = e.get('server_hardware') or {}
        dev = (hw.get('device_model') or hw.get('processor') or hw.get('machine') or '?')[:32]
        cpu = e.get('server_cpu_percent', {}).get('median')
        ram = e.get('server_rss_mb', {}).get('median')
        print(f"{name:<10}{dev:<34}"
              f"{(f'{cpu:.1f}' if cpu is not None else '-'):>9}"
              f"{(f'{ram:.0f}' if ram is not None else '-'):>12}")

    # --- yorum ---
    print("\n" + "-" * 78)
    resp = {n: report['by_deployment'][n]['metrics']['response']['median']
            for n, _, _, _ in deployments}
    best = min(resp, key=lambda n: resp[n]['median'])
    worst = max(resp, key=lambda n: resp[n]['median'])
    gap = resp[worst]['median'] - resp[best]['median']
    worst_spread = max(resp[n]['range_pct'] for n in resp)
    print(f"En dusuk yanit suresi: {best} ({resp[best]['median']:.3f} ms)")
    print(f"Ortamlar arasi en buyuk fark: {gap:.3f} ms | "
          f"tekrarlar arasi en buyuk yayilim: %{worst_spread:.1f}")
    if len(resp) > 1 and gap / max(resp[best]['median'], 1e-9) * 100 < worst_spread:
        print(">> Fark olcum gurultusunun altinda; ortamlar ayirt edilemiyor.")

    hop_ms = cfg.HOP_DURATION * 1000
    win_ms = cfg.WINDOW_DURATION * 1000
    print(f"\nOLCEK NOTU: en buyuk ortam farki {gap:.1f} ms. Kullanicinin bekledigi")
    print(f"sure ise {win_ms:.0f} ms pencere + {hop_ms:.0f} ms hop. Yerlesim farki, "
          f"pencere suresinin")
    print(f"binde {gap / win_ms * 1000:.1f}'i kadar. Uctan uca gecikmeyi belirleyen sey "
          f"yerlesim degil,")
    print("pencere uzunlugudur.")

    if any(not l for _, _, _, l in deployments):
        print("\n[!] Uzak ortamlarda saatler farkli sayaclar kullanir. Saat farki NTP")
        print("    mantigiyla olculup duzeltilir ama 'latency' sutununun hata payi")
        print("    gidis-donus suresinin yarisi kadardir. queuing/execution (sunucu")
        print("    saati) ve response/jitter (istemci saati) bundan ETKILENMEZ.")

    print("\n[i] Aginin hic olmadigi dordüncü durum (model giyilebilir cihazin icinde)")
    print("    olculemez ama hesaplanabilir: python -m tools.mcu_feasibility")

    path = os.path.join(cfg.RESULTS_DIR, f'deployments_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=1)
    print(f"\n[i] Kaydedildi: {path}")


if __name__ == "__main__":
    main()
