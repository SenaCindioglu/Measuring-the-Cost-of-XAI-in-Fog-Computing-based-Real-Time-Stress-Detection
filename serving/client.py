"""
serving/client.py
==================
Sensor istemcisi: bir WESAD kaydini HAM ornekleme hizlarinda, tick tick
sunucuya gonderir ve donen tahminleri zaman damgalariyla birlikte kaydeder.

Olculen metrikler (makaledeki tanimlarla birebir):
    latency        = t_recv  - t_send    paket istemciden sunucuya ulasana kadar
    queuing delay  = t_start - t_recv    sunucuda islenmeyi bekleme suresi
    execution time = t_end   - t_start   pipeline'in calisma suresi
    response time  = t_reply - t_send    istemcinin gorurdugu toplam sure
    jitter                                response time'in ardisik degisimi
    bandwidth                             gonderilen + alinan bayt / sure

--realtime verilirse tick'ler arasinda gercek zaman beklenir; boylece
sistem gercek sensor hizinda yuklenir ve kuyruk gecikmesi anlam kazanir.
Verilmezse mumkun oldugunca hizli gonderilir (dayanikliik/throughput testi).

Calistirma:
    python -m serving.client --subject S11 --realtime --tag c1
"""

import argparse
import asyncio
import json
import os
import time
import uuid

import numpy as np

from config import config
from data.loader import load_wesad_subject_native, NATIVE_STREAMS, LABEL_RATE
from serving import protocol as P


CLOCK_RESYNC_S = 60.0     # saat farki kac saniyede bir yeniden olculur


def _offset_at(t_ns, samples):
    """Verilen istemci saati anindaki saat farkini tahmin eder.

    Iki olcum arasinda DOGRUSAL ara deger kullanilir: kayma (drift) iki
    kristal arasindaki sabit hiz farkindan gelir, yani zamanla duzgun
    ilerler. Olculen aralik disinda en yakin olcum kullanilir.
    """
    if not samples:
        return 0
    if len(samples) == 1 or t_ns <= samples[0]['t_local_ns']:
        return samples[0]['offset_ns']
    if t_ns >= samples[-1]['t_local_ns']:
        return samples[-1]['offset_ns']
    for a, b in zip(samples, samples[1:]):
        if a['t_local_ns'] <= t_ns <= b['t_local_ns']:
            span = b['t_local_ns'] - a['t_local_ns']
            if span <= 0:
                return a['offset_ns']
            w = (t_ns - a['t_local_ns']) / span
            return a['offset_ns'] + w * (b['offset_ns'] - a['offset_ns'])
    return samples[-1]['offset_ns']


def _clock_report(samples, resync_every_s):
    """Saat olcumlerinden ozet: belirsizlik, toplam kayma, kayma hizi."""
    if not samples:
        return {'offset_ns': 0, 'rtt_ns': 0, 'uncertainty_ms': 0.0, 'n_samples': 0}
    rtts = sorted(s['rtt_ns'] for s in samples)
    rtt_med = rtts[len(rtts) // 2]
    span_s = (samples[-1]['t_local_ns'] - samples[0]['t_local_ns']) / 1e9
    drift_ns = samples[-1]['offset_ns'] - samples[0]['offset_ns']
    return {
        # Geriye uyumluluk: ilk olcum, eski alan adlariyla
        'offset_ns': int(samples[0]['offset_ns']),
        'rtt_ns': int(rtt_med),
        'uncertainty_ms': rtt_med / 2 / 1e6,
        # Periyodik yeniden olcumden gelenler
        'method': 'periodik yeniden olcum + dogrusal ara deger',
        'resync_every_s': resync_every_s,
        'n_samples': len(samples),
        'span_s': span_s,
        'rtt_min_ns': int(rtts[0]),
        'rtt_max_ns': int(rtts[-1]),
        'drift_ns': int(drift_ns),
        'drift_ms': drift_ns / 1e6,
        # Kaymanin hizi: iki kristal arasindaki bagil frekans farki
        'drift_ppm': (drift_ns / (span_s * 1e3)) if span_s > 0 else 0.0,
        'samples': [{'t_rel_s': (s['t_local_ns'] - samples[0]['t_local_ns']) / 1e9,
                     'offset_ns': int(s['offset_ns']), 'rtt_ns': int(s['rtt_ns'])}
                    for s in samples],
    }


async def run_client(args, cfg, streams, raw_labels, client_id):
    per_tick = P.samples_per_tick(NATIVE_STREAMS, args.tick_ms)
    tick_s = args.tick_ms / 1000.0

    n_ticks = min(len(streams[name]) // per_tick[name]
                  for name, _, _ in NATIVE_STREAMS if per_tick[name] > 0)
    if args.max_ticks:
        n_ticks = min(n_ticks, args.max_ticks)

    reader, writer = await asyncio.open_connection(args.host, args.port)

    hello = {'client_id': client_id, 'subject': args.subject,
             'tick_ms': args.tick_ms, 'norm_mode': args.norm_mode}
    writer.write(P.pack_hello(hello))
    await writer.drain()

    # Sunucu saatiyle aramizdaki farki olc. Latency bunsuz hesaplanamaz.
    #
    # Olcum TEK SEFER yapilmaz: iki makinenin kristali birebir ayni hizda
    # tiklemedigi icin fark zamanla kayar. 87 dakikalik bir kosuda bu kayma
    # onlarca milisaniyeye ulasti ve latency medyani NEGATIF okundu, yani
    # metrik gecersizdi. Bu yuzden olcum periyodik olarak tekrarlanir ve her
    # paketin latency'si o ana ait ara deger ile duzeltilir.
    resync_every_s = float(getattr(args, 'clock_resync_s', CLOCK_RESYNC_S) or 0.0)
    clock_offset, clock_rtt = await P.measure_clock_offset(reader, writer)
    clock_samples = [{'t_local_ns': time.perf_counter_ns(),
                      'offset_ns': clock_offset, 'rtt_ns': clock_rtt}]

    tag = f"[{args.subject}]"
    eta = f", tahmini {n_ticks * tick_s / 60:.1f} dk" if args.realtime else ""
    print(f"{tag} baglandi, {n_ticks} tick gonderilecek{eta}", flush=True)

    records = []
    bytes_sent = bytes_recv = 0
    cursor = {name: 0 for name, _, _ in NATIVE_STREAMS}
    elapsed_s = 0.0
    t_wall0 = time.perf_counter()
    PROGRESS_EVERY_S = 10.0
    t_next_progress = t_wall0 + PROGRESS_EVERY_S
    t_next_resync = (t_wall0 + resync_every_s) if resync_every_s > 0 else float('inf')

    for msg_id in range(n_ticks):
        chunks = {}
        for name, _, _ in NATIVE_STREAMS:
            k = per_tick[name]
            c = cursor[name]
            chunks[name] = streams[name][c:c + k]
            cursor[name] = c + k
        elapsed_s += tick_s

        t_send = time.perf_counter_ns()
        packet = P.pack_data(msg_id, t_send, chunks, NATIVE_STREAMS)
        writer.write(packet)
        await writer.drain()
        bytes_sent += len(packet)

        body = await P.read_frame(reader)
        t_reply = time.perf_counter_ns()
        if body is None:
            break
        bytes_recv += len(body) + P.HEADER.size
        rep = P.unpack_reply(body)

        rec = {
            'msg_id': rep['msg_id'],
            'predicted': rep['predicted'],
            # Yalnizca latency saat farki duzeltmesi ister; digerleri ayni
            # saatten. Latency kosu bitince, o ana ait saat farkiyla
            # hesaplanir; ham damgalar bu yuzden saklanir.
            't_send_ns': t_send,
            't_recv_ns': rep['t_recv'],
            'queuing_ms':   (rep['t_start'] - rep['t_recv']) / 1e6,
            'execution_ms': (rep['t_end']   - rep['t_start']) / 1e6,
            'response_ms':  (t_reply - t_send) / 1e6,
        }
        if rep['predicted']:
            # Pencerenin kapsadigi araligin cogunluk etiketi (yalniz raporlama icin;
            # sunucu etiketi hic gormez)
            end_i = int(round(elapsed_s * LABEL_RATE))
            start_i = int(round((elapsed_s - cfg.WINDOW_DURATION) * LABEL_RATE))
            span = raw_labels[max(0, start_i):min(end_i, len(raw_labels))]
            if len(span):
                majority = int(np.bincount(span.astype(int)).argmax())
                if majority in cfg.VALID_LABELS:
                    rec['true_label'] = 1 if majority == 2 else 0
                else:
                    rec['true_label'] = -1     # gecersiz etiketli pencere
            rec['pred_label'] = rep['pred']
            rec['confidence'] = rep['confidence']
            rec['stages'] = rep.get('stages')

        records.append(rec)

        # Olcum zaman damgalarindan sonra basilir; realtime'da bekleme suresi
        # bunu emer, yani olculen metrikleri etkilemez.
        now = time.perf_counter()
        if now >= t_next_progress:
            t_next_progress = now + PROGRESS_EVERY_S
            done = msg_id + 1
            left = (f", kalan ~{(n_ticks - done) * tick_s / 60:.1f} dk"
                    if args.realtime else "")
            print(f"{tag} tick {done}/{n_ticks} (%{100 * done / n_ticks:.0f}), "
                  f"son response {rec['response_ms']:.1f} ms{left}", flush=True)

        # Saat farkini yeniden olc. Burasi guvenli an: son cevap alindi, bir
        # sonraki paket henuz gonderilmedi, yani hatta bekleyen veri yok.
        # Tur sayisi baslangictakinden az (4 yerine 12): amac akisi az
        # kesmek, dogruluk zaten en dusuk RTT'li turdan geliyor.
        if now >= t_next_resync:
            t_next_resync = now + resync_every_s
            off, rtt = await P.measure_clock_offset(reader, writer, rounds=4)
            if rtt > 0:
                clock_samples.append({'t_local_ns': time.perf_counter_ns(),
                                      'offset_ns': off, 'rtt_ns': rtt})

        if args.realtime:
            target = t_wall0 + (msg_id + 1) * tick_s
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    # Kapanmadan once son bir olcum: kayma egrisinin sag ucu da bilinsin.
    if resync_every_s > 0 and len(clock_samples) > 1:
        off, rtt = await P.measure_clock_offset(reader, writer, rounds=4)
        if rtt > 0:
            clock_samples.append({'t_local_ns': time.perf_counter_ns(),
                                  'offset_ns': off, 'rtt_ns': rtt})

    # Latency'ler simdi hesaplanir: her paket icin o ana ait saat farki.
    for rec in records:
        off = _offset_at(rec['t_send_ns'], clock_samples)
        rec['latency_ms'] = ((rec['t_recv_ns'] - off) - rec['t_send_ns']) / 1e6

    writer.write(P.pack_bye())
    await writer.drain()

    # Sunucu kapanmadan once kendi donanim ve kaynak ozetini gonderir
    server_stats = None
    try:
        body = await asyncio.wait_for(P.read_frame(reader), timeout=5.0)
        if body is not None and P.msg_type(body) == P.TYPE_STATS:
            server_stats = P.unpack_stats(body)
    except Exception:
        pass

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    wall_s = time.perf_counter() - t_wall0
    clock = _clock_report(clock_samples, resync_every_s)
    return records, bytes_sent, bytes_recv, wall_s, clock, server_stats


def summarize(records, bytes_sent, bytes_recv, wall_s, warmup_windows, cfg,
              clock=None, server_stats=None):
    """Makaledeki metrik tanimlariyla ozet uretir."""
    preds = [r for r in records if r['predicted']]
    live = preds[warmup_windows:]                     # isinma pencereleri disarida

    def stat(vals):
        if not vals:
            return None
        a = np.array(vals, dtype=float)
        d = np.abs(np.diff(a)) if len(a) > 1 else np.array([0.0])
        return {
            'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
            'min_ms': float(a.min()), 'max_ms': float(a.max()), 'std_ms': float(a.std()),
            'jitter_mean_abs_delta_ms': float(d.mean()),
        }

    out = {
        'clock_sync': clock,
        'server': server_stats,
        'n_packets': len(records),
        'n_predictions': len(preds),
        'n_measured': len(live),
        'n_warmup_excluded': min(warmup_windows, len(preds)),
        'wall_seconds': wall_s,
        'metrics': {
            'latency':   stat([r['latency_ms'] for r in live]),
            'queuing':   stat([r['queuing_ms'] for r in live]),
            'execution': stat([r['execution_ms'] for r in live]),
            'response':  stat([r['response_ms'] for r in live]),
        },
        'bandwidth': {
            'bytes_sent': bytes_sent,
            'bytes_recv': bytes_recv,
            'kbps_sent': bytes_sent * 8 / 1000.0 / wall_s if wall_s else 0.0,
            'kbps_recv': bytes_recv * 8 / 1000.0 / wall_s if wall_s else 0.0,
            'kbps_total': (bytes_sent + bytes_recv) * 8 / 1000.0 / wall_s if wall_s else 0.0,
        },
    }

    # Sunucu ici asama dokumu
    stage_names = ['resampling', 'preprocessing', 'tensor_prep', 'inference',
                   'postprocessing', 'xai']
    stages = {}
    for st in stage_names:
        vals = [r['stages'][st] for r in live if r.get('stages') and st in r['stages']]
        if vals:
            stages[st] = stat(vals)
    if stages:
        out['server_stages'] = stages

    # Siniflandirma (etiketi gecerli pencereler)
    pairs = [(r['true_label'], r['pred_label']) for r in live
             if r.get('true_label', -1) >= 0 and 'pred_label' in r]
    if pairs:
        yt = np.array([a for a, _ in pairs]); yp = np.array([b for _, b in pairs])
        tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out['classification'] = {
            'n_labeled': len(pairs),
            'accuracy': float((yt == yp).mean()),
            'stress_precision': pr, 'stress_recall': rc,
            'stress_f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        }
    return out


def print_summary(s, title=""):
    m = s['metrics']
    print(f"\n[SUMMARY] {title}  ({s['n_measured']} tahmin, "
          f"{s['n_warmup_excluded']} warm-up haric, {s['n_packets']} paket)")
    print(f"   {'metrik':>14} {'ort':>9} {'medyan':>9} {'p95':>9} {'p99':>9} {'jitter':>9}")
    for key, label in [('latency', 'latency'), ('queuing', 'queuing delay'),
                       ('execution', 'execution'), ('response', 'response time')]:
        v = m[key]
        if v:
            print(f"   {label:>14} {v['mean_ms']:>9.3f} {v['median_ms']:>9.3f} "
                  f"{v['p95_ms']:>9.3f} {v['p99_ms']:>9.3f} {v['jitter_mean_abs_delta_ms']:>9.3f}")
    if 'server_stages' in s:
        print("   -- sunucu ici asamalar (ms) --")
        for st, v in s['server_stages'].items():
            print(f"   {st:>14} {v['mean_ms']:>9.3f} {v['median_ms']:>9.3f} "
                  f"{v['p95_ms']:>9.3f} {v['p99_ms']:>9.3f}")
    srv = s.get('server')
    if srv:
        hw = srv.get('hardware', {})
        rs = srv.get('resources', {})
        name = hw.get('device_model') or hw.get('processor') or hw.get('machine', '?')
        print(f"   sunucu        : {name} | {hw.get('cpu_count')} cekirdek, "
              f"{hw.get('torch_threads')} iplik")
        if rs.get('process_cpu_percent'):
            c = rs['process_cpu_percent']
            print(f"   sunucu CPU    : ort %{c['mean']:.1f}  tepe %{c['max']:.1f} "
                  f"(surec; {hw.get('cpu_count')} cekirdekte tavan "
                  f"%{100 * (hw.get('cpu_count') or 1)})")
        if rs.get('process_rss_mb'):
            r = rs['process_rss_mb']
            extra = (f", makinenin %{rs['process_memory_percent']:.1f}'i"
                     if rs.get('process_memory_percent') else "")
            print(f"   sunucu bellek : ort {r['mean']:.0f} MB  tepe {r['max']:.0f} MB{extra}")
    if s.get('clock_sync'):
        c = s['clock_sync']
        print(f"   saat farki    : {c['offset_ns'] / 1e6:.3f} ms "
              f"(belirsizlik +/-{c['uncertainty_ms']:.3f} ms)")
        if c.get('n_samples', 1) > 1:
            print(f"   saat kaymasi  : {c['drift_ms']:+.3f} ms / {c['span_s']:.0f} s "
                  f"({c['drift_ppm']:+.2f} ppm, {c['n_samples']} olcum)")
        # Tutarlilik kontrolu: tek yonlu gecikme, gidis-donusun yarisindan
        # buyuk olamaz ve negatif olamaz. Ikisi de saat duzeltmesinin
        # yeterli olup olmadigini gosterir.
        lat, resp, ex, qu = m['latency'], m['response'], m['execution'], m['queuing']
        if lat and resp and ex and qu:
            net_rt = resp['median_ms'] - ex['median_ms'] - qu['median_ms']
            print(f"   latency kontrol: olculen {lat['median_ms']:.3f} ms, "
                  f"ag gidis-donusun yarisi {net_rt / 2:.3f} ms")
            if lat['median_ms'] < 0:
                print("   [!] latency medyani NEGATIF: saat duzeltmesi yetersiz. "
                      "Bu sutun raporlanmamali, yerine yukaridaki tahmin kullanilmali.")
    b = s['bandwidth']
    print(f"   bant genisligi: gonderilen {b['kbps_sent']:.1f} kbps, "
          f"alinan {b['kbps_recv']:.1f} kbps, toplam {b['kbps_total']:.1f} kbps")
    if 'classification' in s:
        c = s['classification']
        cm = c['confusion']
        print(f"   siniflandirma : acc={c['accuracy']:.4f} stresF1={c['stress_f1']:.4f} "
              f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9099)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS)
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE)
    parser.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS)
    parser.add_argument('--realtime', action='store_true',
                        help="Tick'ler arasinda gercek zaman bekle (sensor hizinda yukle)")
    parser.add_argument('--max-ticks', type=int, default=0, help="Test icin tick sayisini sinirla")
    parser.add_argument('--clock-resync-s', type=float, default=CLOCK_RESYNC_S,
                        help="Saat farki kac saniyede bir yeniden olculsun (0 = kapali)")
    parser.add_argument('--tag', default='client')
    args = parser.parse_args()

    streams, raw_labels = load_wesad_subject_native(args.subject, cfg.DATA_PATH)
    if streams is None:
        raise SystemExit(f"{args.subject} yuklenemedi")

    client_id = f"{args.subject}-{uuid.uuid4().hex[:6]}"
    records, bs, br, wall, clock, srv = asyncio.run(
        run_client(args, cfg, streams, raw_labels, client_id))

    s = summarize(records, bs, br, wall, args.warmup_windows, cfg, clock, srv)
    s['config'] = {'subject': args.subject, 'tick_ms': args.tick_ms,
                   'norm_mode': args.norm_mode, 'realtime': args.realtime,
                   'host': args.host, 'port': args.port, 'client_id': client_id}
    print_summary(s, title=f"{args.subject} @ {args.host}:{args.port}")

    path = os.path.join(cfg.RESULTS_DIR, f'client_{args.tag}.json')
    with open(path, 'w') as f:
        json.dump({'summary': s, 'records': records}, f, indent=1)
    print(f"\n[i] Kaydedildi: {path}")


if __name__ == "__main__":
    main()
