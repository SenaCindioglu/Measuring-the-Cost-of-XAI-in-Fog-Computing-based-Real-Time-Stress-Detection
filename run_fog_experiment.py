"""
run_fog_experiment.py
======================
Sunucuyu baslatir, N istemciyi es zamanli kosturur, sonuclari tek raporda
toplar ve sunucuyu kapatir.

Tek komutla ucdan uca deney:
    python run_fog_experiment.py --subject S11 --realtime --tag edge_1user
    python run_fog_experiment.py --subject S11 S14 --clients 4 --realtime --tag load_4

--clients N verilirse N istemci ayni anda baglanir; subject listesi dongusel
olarak dagitilir (4 istemci, 2 subject -> S11, S14, S11, S14). Boylece tek bir
worker'in kac es zamanli kullaniciyi tasidigi olculebilir.

Uzaktaki bir sunucuya baglanmak icin --server-host ver; o zaman sunucu burada
baslatilmaz, sadece istemciler kosulur.

    python run_fog_experiment.py --server-host 192.168.1.42 --subject S11 --realtime

UYARI: istemci ve sunucu ayri makinelerdeyse saatler senkron olmadigi icin
tek yonlu gecikme (latency) ve kuyruk gecikmesi guvenilir degildir; o durumda
yalnizca response time (istemci saati) ve execution time (sunucu saati)
raporlanmalidir. Script bunu tespit edip uyarir.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

from config import config
from data.loader import load_wesad_subject_native, NATIVE_STREAMS
from serving import protocol as P
from serving.client import run_client, summarize, print_summary


class _ClientArgs:
    """serving.client.run_client'in bekledigi argparse benzeri tasiyici."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def check_enough_ticks(cfg, tick_ms, max_ticks, warmup_windows):
    """Kosu, isinmadan sonra olculecek pencere birakacak kadar uzun mu?

    Ilk tahmin pencere dolunca gelir (10 s), sonrasi her hop'ta (5 s) bir.
    Cok kisa bir kosuda tum tahminler isinmaya gider, ozet bos kalir ve
    rapor asamasinda cokerdi; bu yuzden basta kontrol edilir.
    """
    if not max_ticks:
        return
    fill = int(round(cfg.WINDOW_DURATION * 1000 / tick_ms))
    per_window = max(1, int(round(cfg.HOP_DURATION * 1000 / tick_ms)))
    n_pred = 0 if max_ticks < fill else 1 + (max_ticks - fill) // per_window
    if n_pred <= warmup_windows:
        need = fill + (warmup_windows + 1) * per_window
        raise SystemExit(
            f"--max-ticks {max_ticks} cok kisa: {n_pred} tahmin uretir, "
            f"bunun {min(n_pred, warmup_windows)} tanesi isinma olarak dislanir "
            f"ve olculecek pencere kalmaz.\n"
            f"En az --max-ticks {need} kullan (ya da --warmup-windows degerini dusur).")


def wait_for_port(host, port, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


async def run_all(client_args_list, cfg, loaded):
    tasks = []
    for ca in client_args_list:
        streams, labels = loaded[ca.subject]
        cid = f"{ca.subject}-{uuid.uuid4().hex[:6]}"
        tasks.append(run_client(ca, cfg, streams, labels, cid))
    return await asyncio.gather(*tasks)


def aggregate(summaries):
    """Birden fazla istemcinin ozetini birlestirir."""
    if len(summaries) == 1:
        return summaries[0]

    agg = {'n_clients': len(summaries),
           'n_packets': sum(s['n_packets'] for s in summaries),
           'n_predictions': sum(s['n_predictions'] for s in summaries),
           'n_measured': sum(s['n_measured'] for s in summaries),
           'n_warmup_excluded': sum(s['n_warmup_excluded'] for s in summaries),
           'wall_seconds': max(s['wall_seconds'] for s in summaries),
           'metrics': {}, 'per_client': summaries}

    for key in ['latency', 'queuing', 'execution', 'response']:
        vals = [s['metrics'][key] for s in summaries if s['metrics'].get(key)]
        if not vals:
            continue
        agg['metrics'][key] = {
            'mean_ms': float(np.mean([v['mean_ms'] for v in vals])),
            'median_ms': float(np.median([v['median_ms'] for v in vals])),
            'p95_ms': float(np.max([v['p95_ms'] for v in vals])),      # en kotu istemci
            'p99_ms': float(np.max([v['p99_ms'] for v in vals])),
            'min_ms': float(np.min([v['min_ms'] for v in vals])),
            'max_ms': float(np.max([v['max_ms'] for v in vals])),
            'std_ms': float(np.mean([v['std_ms'] for v in vals])),
            'jitter_mean_abs_delta_ms': float(np.mean([v['jitter_mean_abs_delta_ms'] for v in vals])),
        }

    bs = sum(s['bandwidth']['bytes_sent'] for s in summaries)
    br = sum(s['bandwidth']['bytes_recv'] for s in summaries)
    w = agg['wall_seconds'] or 1.0
    agg['clock_sync'] = summaries[0].get('clock_sync')
    agg['server'] = next((s['server'] for s in summaries if s.get('server')), None)
    agg['bandwidth'] = {'bytes_sent': bs, 'bytes_recv': br,
                        'kbps_sent': bs * 8 / 1000.0 / w,
                        'kbps_recv': br * 8 / 1000.0 / w,
                        'kbps_total': (bs + br) * 8 / 1000.0 / w}

    stage_names = ['resampling', 'preprocessing', 'tensor_prep', 'inference',
                   'postprocessing', 'xai']
    stages = {}
    for st in stage_names:
        vals = [s['server_stages'][st] for s in summaries
                if s.get('server_stages', {}).get(st)]
        if vals:
            stages[st] = {'mean_ms': float(np.mean([v['mean_ms'] for v in vals])),
                          'median_ms': float(np.median([v['median_ms'] for v in vals])),
                          'p95_ms': float(np.max([v['p95_ms'] for v in vals])),
                          'p99_ms': float(np.max([v['p99_ms'] for v in vals]))}
    if stages:
        agg['server_stages'] = stages

    cls = [s['classification'] for s in summaries if 'classification' in s]
    if cls:
        cm = {k: sum(c['confusion'][k] for c in cls) for k in ['tn', 'fp', 'fn', 'tp']}
        n = sum(c['n_labeled'] for c in cls)
        pr = cm['tp'] / (cm['tp'] + cm['fp']) if cm['tp'] + cm['fp'] else 0.0
        rc = cm['tp'] / (cm['tp'] + cm['fn']) if cm['tp'] + cm['fn'] else 0.0
        agg['classification'] = {
            'n_labeled': n,
            'accuracy': (cm['tp'] + cm['tn']) / n if n else 0.0,
            'stress_precision': pr, 'stress_recall': rc,
            'stress_f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            'confusion': cm,
        }
    return agg


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', nargs='+', required=True,
                        help="Istemcilere dongusel dagitilacak subject listesi")
    parser.add_argument('--clients', type=int, default=0,
                        help="Es zamanli istemci sayisi (varsayilan: subject sayisi)")
    parser.add_argument('--server-host', default=None,
                        help="Verilirse sunucu burada baslatilmaz, bu adrese baglanilir")
    parser.add_argument('--port', type=int, default=9099)
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE)
    parser.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS)
    parser.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS)
    parser.add_argument('--threads', type=int, default=cfg.DEFAULT_THREADS)
    parser.add_argument('--realtime', '--live', dest='realtime', action='store_true',
                        help="Sensor gercek hizinda gonder (kuyruk gecikmesi ancak boyle anlamli)")
    parser.add_argument('--open-loop', action='store_true',
                        help="Yaniti beklemeden tick hizinda gonder; --realtime gibi zamanlar")
    parser.add_argument('--max-ticks', type=int, default=0)
    parser.add_argument('--measured-windows', type=int, default=0,
                        help="Warm-up disinda tam bu sayida tahmin penceresi olc")
    parser.add_argument('--clock-resync-s', type=float, default=60.0,
                        help="Saat farki kac saniyede bir yeniden olculsun "
                             "(0 = yalnizca baslangicta; uzun kosularda latency kayar)")
    parser.add_argument('--tag', default='fog')
    args = parser.parse_args()

    if args.measured_windows < 0:
        parser.error('--measured-windows pozitif olmali')
    if args.measured_windows:
        if args.max_ticks:
            parser.error('--measured-windows ve --max-ticks birlikte kullanilamaz')
        fill_ticks = int(round(cfg.WINDOW_DURATION * 1000 / args.tick_ms))
        ticks_per_prediction = max(1, int(round(cfg.HOP_DURATION * 1000 / args.tick_ms)))
        args.max_ticks = fill_ticks + (args.warmup_windows + args.measured_windows - 1) * ticks_per_prediction

    host = args.server_host or '127.0.0.1'
    local_server = args.server_host is None
    n_clients = args.clients or len(args.subject)
    subjects = [args.subject[i % len(args.subject)] for i in range(n_clients)]

    proc = None
    if local_server:
        cmd = [sys.executable, '-m', 'serving.server', '--host', host,
               '--port', str(args.port), '--model', args.model,
               '--norm-mode', args.norm_mode, '--threads', str(args.threads)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if not wait_for_port(host, args.port):
            proc.kill()
            raise SystemExit("Sunucu baslatilamadi (port acilmadi)")

    print("=" * 70)
    print("FOG DENEYI")
    print("=" * 70)
    print(f"Sunucu       : {host}:{args.port} " +
          ("(bu makinede baslatildi)" if local_server else "(UZAK)"))
    print(f"Istemci      : {n_clients} adet -> {subjects}")
    print(f"Tick         : {args.tick_ms}ms | Realtime: {args.realtime} | "
          f"Norm: {args.norm_mode} | Sunucu ipligi: {args.threads}")
    print(f"Gonderim     : {'open-loop' if args.open_loop else 'stop-and-wait'}")
    if not local_server:
        print("[!] UZAK sunucu: saat farki NTP mantigiyla olculup duzeltilir, ama")
        print("    latency'nin belirsizligi gidis-donus suresinin yarisi kadardir.")
        print("    queuing/execution (sunucu saati) ve response (istemci saati) bundan etkilenmez.")
    print("=" * 70)

    check_enough_ticks(cfg, args.tick_ms, args.max_ticks, args.warmup_windows)

    try:
        loaded = {}
        for s in set(subjects):
            st, lb = load_wesad_subject_native(s, cfg.DATA_PATH)
            if st is None:
                raise SystemExit(f"{s} yuklenemedi")
            loaded[s] = (st, lb)

        if args.measured_windows:
            available_ticks = min(
                len(loaded[s][0][name]) // int(round(rate * args.tick_ms / 1000.0))
                for s in set(subjects)
                for name, rate, _ in NATIVE_STREAMS
                if int(round(rate * args.tick_ms / 1000.0)) > 0)
            if args.max_ticks > available_ticks:
                raise SystemExit(
                    f"{args.measured_windows} measured windows plus warm-up require "
                    f"{args.max_ticks} ticks, but the selected data supports "
                    f"only {available_ticks} ticks.")

        ca_list = [_ClientArgs(host=host, port=args.port, subject=s,
                               tick_ms=args.tick_ms, norm_mode=args.norm_mode,
                               realtime=args.realtime, max_ticks=args.max_ticks,
                               open_loop=args.open_loop,
                               clock_resync_s=args.clock_resync_s)
                   for s in subjects]

        t0 = time.perf_counter()
        results = asyncio.run(run_all(ca_list, cfg, loaded))
        wall = time.perf_counter() - t0

        summaries = []
        for (records, bs, br, w, clock, srv), subj in zip(results, subjects):
            s = summarize(records, bs, br, w, args.warmup_windows, cfg, clock, srv)
            s['subject'] = subj
            summaries.append(s)

        agg = aggregate(summaries)
        agg['config'] = {
            'subjects': subjects, 'n_clients': n_clients, 'tick_ms': args.tick_ms,
            'norm_mode': args.norm_mode, 'realtime': args.realtime,
            'open_loop': args.open_loop,
            'requested_measured_windows': args.measured_windows or None,
            'max_ticks': args.max_ticks, 'warmup_windows': args.warmup_windows,
            'threads': args.threads, 'model': os.path.basename(args.model),
            'server': f'{host}:{args.port}', 'local_server': local_server,
            'clock_synced': local_server, 'total_wall_s': wall,
            'clock_resync_s': args.clock_resync_s,
        }

        print_summary(agg, title=f"{n_clients} istemci, toplam {wall:.1f} s")

        if n_clients > 1:
            print("\n   -- istemci bazinda response time (ms) --")
            for s in summaries:
                r = s['metrics']['response']
                print(f"   {s['subject']:>6} ({s['n_measured']:>4} tahmin): "
                      f"medyan {r['median_ms']:.2f}  p95 {r['p95_ms']:.2f}")

        path = os.path.join(cfg.RESULTS_DIR, f'fog_{args.tag}.json')
        with open(path, 'w') as f:
            json.dump(agg, f, indent=1)
        print(f"\n[i] Kaydedildi: {path}")

    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
