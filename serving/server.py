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
import platform
import time

import numpy as np
import torch

from config import config
from data.loader import NATIVE_STREAMS, LABEL_RATE
from data.normalization import ensure_fixed_stats, apply_normalization, RunningNormalizer
from models.temporal_cnn import load_checkpoint
from streaming.buffer import MultiRateBuffer
from profiling.latency_profiler import _ResourceSampler
from serving import protocol as P

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def hardware_info():
    """Sunucunun uzerinde kostugu donanimi tanimlar.

    Raspberry Pi'de platform.processor() bos doner; model adi
    /proc/device-tree/model ya da /proc/cpuinfo'dan okunur.
    """
    info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or None,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
    }
    for path, key in [('/proc/device-tree/model', 'device_model')]:
        try:
            with open(path) as f:
                info[key] = f.read().strip('\x00').strip()
        except Exception:
            pass
    if not info.get('processor'):
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.lower().startswith('model name'):
                        info['processor'] = line.split(':', 1)[1].strip()
                        break
        except Exception:
            pass
    if _HAS_PSUTIL:
        try:
            info['total_ram_mb'] = round(psutil.virtual_memory().total / (1024 ** 2), 1)
            freq = psutil.cpu_freq()
            if freq:
                info['cpu_mhz_max'] = freq.max or None
        except Exception:
            pass
    return info


def local_addresses():
    """
    Bu makinenin baska cihazlardan erisilebilir IPv4 adresleri.

    Sunucu 0.0.0.0'a baglandiginda kullanicinin istemci tarafina hangi adresi
    yazacagini bilmesi gerekiyor; bunu ayrica ipconfig/hostname kosmadan
    ogrenebilsin diye aciliste basiliyor.
    """
    addrs = []
    if _HAS_PSUTIL:
        try:
            import socket as _s
            for iface, entries in psutil.net_if_addrs().items():
                for e in entries:
                    if e.family == _s.AF_INET and not e.address.startswith('127.'):
                        addrs.append((iface, e.address))
        except Exception:
            pass

    def rank(item):
        # 169.254.x.x baglanti-yerel (APIPA) adresleridir: DHCP alinamadiginda
        # atanir ve baska cihazdan genelde erisilemez. Gercek yerel ag
        # adresleri (192.168 / 10 / 172.16-31) one alinir.
        ip = item[1]
        if ip.startswith('169.254.'):
            return 2
        if ip.startswith(('192.168.', '10.')):
            return 0
        if ip.startswith('172.'):
            second = int(ip.split('.')[1]) if ip.split('.')[1].isdigit() else 0
            return 0 if 16 <= second <= 31 else 1
        return 1

    return sorted(addrs, key=rank)


def resource_summary(sampler):
    """Ornekleyiciden CPU ve bellek ozeti.

    process_cpu_percent cok cekirdekli makinede 100'u asabilir (her cekirdek
    %100 sayilir); system_cpu_percent tum makinenin yuzdesidir. Makaledeki
    rakamlarla karsilastirirken hangisinin kastedildigine dikkat edilmeli.
    """
    import numpy as _np
    out = {}
    if sampler and sampler.cpu:
        c = _np.array(sampler.cpu)
        out['process_cpu_percent'] = {'mean': float(c.mean()), 'max': float(c.max()),
                                      'p95': float(_np.percentile(c, 95)),
                                      'n_samples': int(len(c))}
    if sampler and sampler.rss:
        r = _np.array(sampler.rss)
        out['process_rss_mb'] = {'mean': float(r.mean()), 'max': float(r.max())}
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            out['system_cpu_percent'] = psutil.cpu_percent(interval=None)
            out['system_memory_percent'] = vm.percent
            if sampler and sampler.rss:
                out['process_memory_percent'] = round(
                    max(sampler.rss) / (vm.total / (1024 ** 2)) * 100, 2)
        except Exception:
            pass
    return out


#: --xai-method ile secilebilen aciklayicilar. Iki maliyet noktasi kasitli:
#: secici XAI'nin kazanci aciklama pahalilastikca buyuyor.
XAI_METHODS = ('none', 'lime', 'lime_ucuz', 'shap', 'shap_ucuz')


def build_explainer(method, model, device='cpu'):
    """xai_config.json'daki dondurulmus yapilandirmadan aciklayici kurar.

    Import BILEREK tembel: `xai/` paketi sunucu imajinda bulunmayabilir
    (.dockerignore onu disarida birakiyor, cunku --xai-method none varsayilan
    ve o durumda hic gerekmiyor). Tepede import edilseydi XAI kullanmayan
    bir dagitimda sunucu hic acilmazdi.
    """
    if method in (None, 'none'):
        return None
    from xai.artifacts import load_artifacts
    from xai import lime_ as lime_mod, shap_ as shap_mod

    record = load_artifacts()
    if method == 'lime':
        return lime_mod.from_artifacts(model, record, device=device)
    if method == 'lime_ucuz':
        # CPU'da LIME'in maliyeti ornek sayisiyla dogrusal; dilim sayisi
        # bedava (olculdu). Ucuz nokta bu yuzden num_samples'tan geciyor.
        return lime_mod.from_artifacts(model, record, device=device, num_samples=250)
    if method == 'shap':
        return shap_mod.from_artifacts(model, record, device=device)
    if method == 'shap_ucuz':
        return shap_mod.from_artifacts(model, record, device=device, cheap=True)
    raise ValueError("Bilinmeyen --xai-method: " + str(method))


def warmup_explainer(explainer, cfg):
    """Aciklayiciyi acilista bir kez calistirir ve ilk cagri suresini doner.

    NEDEN GEREKLI: SHAP'in ILK cagrisi bu makinede 12.4 saniye olculdu,
    sonraki cagrilar 0.85 saniye. Fark grafik kurulumu ve onbellek isinmasi.
    Isitilmazsa bu 12 saniye ilk aciklamada worker'i bloklar; tek worker
    oldugu icin butun kuyruk onun arkasinda birikir ve o istemcinin
    queuing delay'i olcum boyunca anlamsizlasir.
    """
    if explainer is None:
        return None
    dummy = np.zeros((cfg.NUM_CHANNELS, cfg.WINDOW_SIZE), dtype=np.float32)
    t0 = time.perf_counter()
    explainer.explain(dummy)
    return (time.perf_counter() - t0) * 1000.0


def should_explain(confidence, threshold):
    """Secici XAI karari: guven esigin ALTINDA ise acikla.

    threshold = 0.0 -> hicbir zaman  (kontrol kosusu: XAI'siz taban)
    threshold = 1.0 -> her pencere   (kontrol kosusu: en pahali durum)
    arasi            -> yalnizca modelin emin olmadigi pencereler

    1.0 ayrica ele alinir: softmax cikti tam 1.0 okuyabildigi icin
    'conf < 1.0' karsilastirmasi nadiren de olsa atlardi.
    """
    if threshold <= 0.0:
        return False
    if threshold >= 1.0:
        return True
    return confidence < threshold


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
        self.n_explained = 0


async def worker(state):
    """
    Tek worker: kuyrugu sirayla bosaltir, modeli calistirir, cevabi yazar.
    Bloklayici oldugu icin ikinci bir istek gelirse kuyrukta bekler --
    queuing delay tam olarak bu beklemeyi olcer.
    """
    cfg, model, queue = state['cfg'], state['model'], state['queue']
    explainer = state.get('explainer')
    xai_threshold = state.get('xai_threshold', 0.0)

    while True:
        sess, body, t_recv = await queue.get()
        try:
            t_start = time.perf_counter_ns()
            msg_id, t_send, chunks = P.unpack_data(body, NATIVE_STREAMS, sess.per_tick)
            ready = sess.buffer.push(chunks, sess.tick_s)

            predicted = False
            explained = False
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

                    # --- Secici XAI ---
                    # Aciklama, tahminle AYNI worker'da ve execution time'in
                    # ICINDE uretilir. Ayri bir iplige atilsaydi maliyeti
                    # gorunmez olurdu; deneyin butun amaci o maliyeti olcmek.
                    if explainer is not None and should_explain(conf, xai_threshold):
                        t0 = time.perf_counter()
                        explainer.explain(
                            np.ascontiguousarray(norm_window.transpose(1, 0),
                                                 dtype=np.float32))
                        stages['xai'] = (time.perf_counter() - t0) * 1000.0
                        explained = True
                        sess.n_explained += 1

                    predicted = True
                    sess.n_windows += 1

            t_end = time.perf_counter_ns()

            sess.writer.write(P.pack_reply(msg_id, t_recv, t_start, t_end,
                                           predicted=predicted, pred=pred,
                                           confidence=conf, true_label=None,
                                           stages=stages, explained=explained))
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
    state['sessions_peak'] = max(state.get('sessions_peak', 0), state['sessions'])
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
                # Kapanmadan once kendi kaynak kullanimimizi gonder
                try:
                    # XAI kunyesi de gonderilir. Onceki hali yalnizca
                    # donanim ve kaynak ozetini yolluyordu; hangi
                    # aciklayicinin hangi esikle kosuldugu SADECE dosya
                    # adinda kaliyordu (fog_xai_lime.json gibi). Dosya adi
                    # degistigi anda bilgi kayboluyor ve sonuclar tekrar
                    # uretilemez hale geliyor, o yuzden JSON'un icine yazilir.
                    xai_info = {'method': state.get('xai_method', 'none'),
                                'threshold': state.get('xai_threshold', 0.0),
                                'warmup_ms': state.get('xai_warmup_ms')}
                    explainer = state.get('explainer')
                    if explainer is not None:
                        try:
                            xai_info['config'] = explainer.config()
                        except Exception:
                            pass
                    writer.write(P.pack_stats({
                        'hardware': state['hardware'],
                        'resources': resource_summary(state.get('sampler')),
                        'sessions_peak': state.get('sessions_peak', 1),
                        'xai': xai_info,
                    }))
                    await writer.drain()
                except Exception:
                    pass
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
                extra = (f", {sess.n_explained} aciklama"
                         if state.get('explainer') is not None else "")
                print(f"    {sess.client_id}: {sess.n_packets} paket, "
                      f"{sess.n_windows} tahmin{extra}")

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

    # XAI: aciklayiciyi ISTEMCI BAGLANMADAN kur ve isit. Kaynak ornekleyici
    # de bundan sonra baslatilir, yoksa isinmanin CPU/RAM tepesi kosunun
    # kaynak ozetine karisirdi.
    explainer = build_explainer(args.xai_method, model, device='cpu')
    warmup_ms = None
    if explainer is not None:
        print(f"[i] XAI hazirlaniyor ({args.xai_method}), ilk cagri yavastir...")
        warmup_ms = warmup_explainer(explainer, cfg)
        print(f"[i] XAI hazir. Isinma cagrisi {warmup_ms:.0f} ms surdu "
              f"(olcume DAHIL DEGIL).")

    hw = hardware_info()
    sampler = None
    if _HAS_PSUTIL:
        sampler = _ResourceSampler(psutil.Process(os.getpid()), 0.25)
        sampler.start()

    state = {'cfg': cfg, 'model': model, 'mean': mean, 'std': std,
             'sessions': 0, 'sessions_peak': 0, 'log_every': args.log_every,
             'queue': asyncio.Queue(), 'hardware': hw, 'sampler': sampler,
             'explainer': explainer, 'xai_threshold': args.xai_threshold,
             'xai_method': args.xai_method, 'xai_warmup_ms': warmup_ms}

    asyncio.create_task(worker(state))

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, state), args.host, args.port)

    print("=" * 66)
    print("STRES TESPITI SUNUCUSU")
    print("=" * 66)
    print(f"Adres        : {args.host}:{args.port}")
    print(f"Model        : {os.path.basename(args.model)} "
          f"(depth={arch['depth']}, {arch['num_params']:,} parametre)")
    print(f"Is parcacigi : {torch.get_num_threads()} / {hw.get('cpu_count')} cekirdek")
    print(f"Donanim      : {hw.get('device_model') or hw.get('processor') or hw['machine']}")
    if hw.get('total_ram_mb'):
        print(f"Bellek       : {hw['total_ram_mb']:.0f} MB")
    print(f"Pencere/hop  : {cfg.WINDOW_SIZE} / {cfg.HOP_SIZE} ornek @ {cfg.TARGET_RATE} Hz")
    if explainer is not None:
        thr = args.xai_threshold
        rule = ("her pencere" if thr >= 1.0 else f"guven < {thr:.2f} olan pencereler")
        print(f"XAI          : {args.xai_method} | {rule} | "
              f"isinma {warmup_ms:.0f} ms")
    else:
        print(f"XAI          : kapali (--xai-method none)")
    print(f"Mimari       : acik istek kuyrugu + tek worker")
    print("=" * 66)
    if args.host in ('0.0.0.0', '::'):
        addrs = local_addresses()
        if addrs:
            print("\nIstemci tarafinda kullanilacak adres(ler):")
            for iface, ip in addrs:
                note = "  <- muhtemelen bu" if ip == addrs[0][1] else (
                    "  (baglanti-yerel, muhtemelen erisilemez)"
                    if ip.startswith('169.254.') else "")
                print(f"    {ip:<16}  ({iface}){note}")
            print("\nDiger makinede ornek komut:")
            print("    python compare_deployments.py --subject S11 --realtime \\")
            print(f"        --deployment edge=local fog={addrs[0][1]}:{args.port} --tag edge_vs_fog")
        else:
            print("\n[!] Yerel IP adresi bulunamadi; 'ipconfig' (Windows) ya da")
            print("    'hostname -I' (Linux) ile ogren.")
    print("\nHazir, istemci bekleniyor. Durdurmak icin Ctrl+C.")

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
    parser.add_argument('--xai-method', choices=XAI_METHODS, default='none',
                        help="Aciklayici. Yapilandirma checkpoints/xai_config.json'dan "
                             "okunur; 'none' ise xai/ paketi hic import edilmez.")
    parser.add_argument('--xai-threshold', type=float, default=0.0,
                        help="Guven esigi: bunun ALTINDAKI pencereler aciklanir. "
                             "0.0 = hic (taban kosusu), 1.0 = her pencere (en pahali).")
    args = parser.parse_args()

    # Sessizce yanlis kosu yapmaktansa basta durdur: bu iki kombinasyon
    # neredeyse her zaman yazim hatasidir ve fark edilmezse "XAI'nin
    # maliyeti sifir cikti" gibi bir sonuca yol acar.
    if args.xai_method != 'none' and args.xai_threshold <= 0.0:
        parser.error("--xai-method verildi ama --xai-threshold 0.0; hicbir pencere "
                     "aciklanmaz. Her pencere icin 1.0, secici icin 0.60-0.90 kullan.")
    if args.xai_method == 'none' and args.xai_threshold > 0.0:
        parser.error("--xai-threshold verildi ama --xai-method none; aciklayici yok.")

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")


if __name__ == "__main__":
    main()
