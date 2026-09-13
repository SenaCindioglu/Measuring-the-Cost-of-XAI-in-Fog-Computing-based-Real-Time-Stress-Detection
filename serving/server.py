"""
serving/server.py
==================
Gercek zamanli stres tespiti sunucusu (makaledeki "worker node" karsiligi).

Her istemci icin ayri bir MultiRateBuffer ve normalizasyon durumu tutar.
Pencere hazir oldugunda modeli calistirir ve tahmini istemciye dondurur.

MIMARI: ACIK ISTEK KUYRUGU + TEK WORKER
----------------------------------------
Her baglanti icin bir okuyucu gorevi vardir; gelen paketi aninda okur,
varis zamanini (t_recv) damgalar ve ORTAK KUYRUGA koyar. Tek bir worker
gorevi kuyrugu sirayla bosaltir ve modeli calistirir.

Bu yapi bilerek secildi. Ilk surumde paket okunur okunmaz ayni yerde
isleniyordu; o zaman t_start ile t_recv bitisik oldugu icin queuing delay
her zaman ~0 okuyordu ve bekleme suresi latency'nin icinde gizleniyordu.
Acik kuyrukla:
    latency        = saf ag suresi
    queuing delay  = worker mesgulken kuyrukta gecen sure
ikisi ayri ayri olculebilir. Tek worker varsayimi makaledeki modelle ayni
ve cok kullanicili kapasite olcumunun on kosulu.

Calistirma:
    python -m serving.server --host 127.0.0.1 --port 9099
    python -m serving.server --threads 1 --log-every 200
"""

import argparse
import asyncio
import json
import os
import time

import numpy as np
import torch

from config import config
from data.loader import NATIVE_STREAMS, LABEL_RATE
from data.normalization import ensure_fixed_stats, apply_normalization, RunningNormalizer
from models.temporal_cnn import load_checkpoint
from streaming.buffer import MultiRateBuffer
from serving import protocol as P


class ClientSession:
    """Tek bir istemcinin sunucu tarafindaki durumu."""

    def __init__(self, cfg, hello, fixed_mean, fixed_std, writer=None):
        self.writer = writer
        self.client_id = hello.get('client_id', '?')
        self.subject = hello.get('subject', '?')
        self.tick_ms = hello['tick_ms']
        self.tick_s = self.tick_ms / 1000.0
        self.norm_mode = hello.get('norm_mode', 'running')

        self.per_tick = P.samples_per_tick(NATIVE_STREAMS, self.tick_ms)
        self.buffer = MultiRateBuffer(NATIVE_STREAMS, cfg.WINDOW_DURATION,
                                      cfg.HOP_DURATION, cfg.WINDOW_SIZE)
        self.runner = RunningNormalizer(cfg.NUM_CHANNELS) if self.norm_mode == 'running' else None
        self.fixed_mean, self.fixed_std = fixed_mean, fixed_std
        self.n_windows = 0
        self.n_packets = 0


async def worker(state):
    """
    Tek worker: kuyrugu sirayla bosaltir, modeli calistirir, cevabi yazar.
    Bloklayici oldugu icin ikinci bir istek gelirse kuyrukta bekler --
    queuing delay tam olarak bu beklemeyi olcer.
    """
    cfg, model, queue = state['cfg'], state['model'], state['queue']

    while True:
        sess, body, t_recv = await queue.get()
        try:
            t_start = time.perf_counter_ns()
            msg_id, t_send, chunks = P.unpack_data(body, NATIVE_STREAMS, sess.per_tick)
            ready = sess.buffer.push(chunks, sess.tick_s)

            predicted = False
            pred = conf = None
            stages = None

            if ready:
                stages = {}
                t0 = time.perf_counter()
                raw_window = sess.buffer.get_window()
                stages['resampling'] = (time.perf_counter() - t0) * 1000.0

                if raw_window is not None:
                    t0 = time.perf_counter()
                    if sess.runner is not None:
                        new_part = (raw_window if sess.n_windows == 0
                                    else raw_window[-cfg.HOP_SIZE:])
                        sess.runner.update(new_part)
                        m, sd = sess.runner.mean, sess.runner.std
                    else:
                        m, sd = sess.fixed_mean, sess.fixed_std
                    norm_window = apply_normalization(raw_window, m, sd)
                    stages['preprocessing'] = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    x = torch.FloatTensor(norm_window.transpose(1, 0)).unsqueeze(0)
                    stages['tensor_prep'] = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        logits = model(x)
                    stages['inference'] = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    pred = int(probs.argmax())
                    conf = float(probs.max())
                    stages['postprocessing'] = (time.perf_counter() - t0) * 1000.0

                    predicted = True
                    sess.n_windows += 1

            t_end = time.perf_counter_ns()

            sess.writer.write(P.pack_reply(msg_id, t_recv, t_start, t_end,
                                           predicted=predicted, pred=pred,
                                           confidence=conf, true_label=None,
                                           stages=stages))
            await sess.writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            queue.task_done()


async def handle_client(reader, writer, state):
    cfg, model = state['cfg'], state['model']
    peer = writer.get_extra_info('peername')

    body = await P.read_frame(reader)
    if body is None or P.msg_type(body) != P.TYPE_HELLO:
        writer.close()
        return
    hello = json.loads(body[1:].decode('utf-8'))
    sess = ClientSession(cfg, hello, state['mean'], state['std'], writer=writer)
    state['sessions'] += 1
    print(f"[+] istemci baglandi: {sess.client_id} (subject={sess.subject}, "
          f"tick={sess.tick_ms}ms, norm={sess.norm_mode}) {peer}")

    try:
        while True:
            body = await P.read_frame(reader)
            if body is None:
                break
            t_recv = time.perf_counter_ns()
            mtype = P.msg_type(body)

            if mtype == P.TYPE_BYE:
                break
            if mtype == P.TYPE_PING:
                # Saat olcumu kuyruga GIRMEZ: amaci saf gidis-donus suresini
                # olcmek, kuyruk beklemesi bu olcumu bozardi.
                t1 = P.unpack_ping(body)
                writer.write(P.pack_pong(t1, time.perf_counter_ns()))
                await writer.drain()
                continue
            if mtype != P.TYPE_DATA:
                continue

            sess.n_packets += 1
            await state['queue'].put((sess, body, t_recv))

            if state['log_every'] and sess.n_packets % state['log_every'] == 0:
                print(f"    {sess.client_id}: {sess.n_packets} paket, "
                      f"{sess.n_windows} tahmin")

    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    finally:
        print(f"[-] istemci ayrildi: {sess.client_id} "
              f"({sess.n_packets} paket, {sess.n_windows} tahmin)")
        state['sessions'] -= 1
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main_async(args):
    cfg = config
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    model, arch, meta = load_checkpoint(args.model, torch.device('cpu'), fallback_cfg=cfg)

    mean = std = None
    if args.norm_mode == 'fixed':
        tag = os.path.splitext(os.path.basename(args.model))[0]
        cache = os.path.join(cfg.CHECKPOINT_DIR, f'norm_stats_{tag}.npz')
        mean, std = ensure_fixed_stats(cache, meta.get('train_subjects') or cfg.TRAIN_SUBJECTS, cfg)

    state = {'cfg': cfg, 'model': model, 'mean': mean, 'std': std,
             'sessions': 0, 'log_every': args.log_every,
             'queue': asyncio.Queue()}

    asyncio.create_task(worker(state))

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, state), args.host, args.port)

    print("=" * 66)
    print("STRES TESPITI SUNUCUSU")
    print("=" * 66)
    print(f"Adres        : {args.host}:{args.port}")
    print(f"Model        : {os.path.basename(args.model)} "
          f"(depth={arch['depth']}, {arch['num_params']:,} parametre)")
    print(f"Is parcacigi : {torch.get_num_threads()}")
    print(f"Pencere/hop  : {cfg.WINDOW_SIZE} / {cfg.HOP_SIZE} ornek @ {cfg.TARGET_RATE} Hz")
    print(f"Mimari       : acik istek kuyrugu + tek worker")
    print("=" * 66)
    print("Hazir, istemci bekleniyor. Durdurmak icin Ctrl+C.")

    async with server:
        await server.serve_forever()


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9099)
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--norm-mode', choices=['fixed', 'running'], default=cfg.DEFAULT_NORM_MODE,
                        help="Sunucu tarafi normalizasyon modu")
    parser.add_argument('--threads', type=int, default=cfg.DEFAULT_THREADS)
    parser.add_argument('--log-every', type=int, default=0,
                        help="Kac pakette bir ilerleme yazilsin (0 = yazma)")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")


if __name__ == "__main__":
    main()
