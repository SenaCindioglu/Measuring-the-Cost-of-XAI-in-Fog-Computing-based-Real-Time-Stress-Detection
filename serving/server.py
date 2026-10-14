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
import multiprocessing as mp
import os
import platform
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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
        'numpy': np.__version__,
        # Konteynerde mi: yigin esitligi iddiasinin kaniti bu alan.
        'in_container': os.path.exists('/.dockerenv'),
        'image_tag': os.environ.get('WESAD_IMAGE_TAG'),
    }
    # Surum paket meta verisinden okunur; import edilseydi XAI'siz kosunun
    # bellek tabani shap'in yuku kadar sisecekti.
    from importlib import metadata as _md
    for dist in ('shap', 'scikit-learn'):
        try:
            info[dist] = _md.version(dist)
        except Exception:
            info[dist] = None
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


def resource_summary(sampler, start=0):
    """Ornekleyiciden CPU ve bellek ozeti, YALNIZCA bu oturumun penceresi.

    Ornekleyici sunucu acildigindan beri calisir. Uzak sunucu birden fazla
    kosu boyunca acik kaldiginda (compare_deployments'in 3 turu) tum gecmisi
    ozetlemek, oturumlar arasi bosta gecen sureyi CPU ortalamasina katar ve
    yerel (her turda yeniden baslatilan) sunucuyla karsilastirmayi bozar.
    `start`, oturum baslarken ornek listesinin uzunlugudur.

    process_cpu_percent cok cekirdekli makinede 100'u asabilir (her cekirdek
    %100 sayilir); system_cpu_percent tum makinenin yuzdesidir. Makaledeki
    rakamlarla karsilastirirken hangisinin kastedildigine dikkat edilmeli.
    """
    import numpy as _np
    out = {}
    cpu = sampler.cpu[start:] if sampler else []
    rss = sampler.rss[start:] if sampler else []
    if cpu:
        c = _np.array(cpu)
        out['process_cpu_percent'] = {'mean': float(c.mean()), 'max': float(c.max()),
                                      'p95': float(_np.percentile(c, 95)),
                                      'n_samples': int(len(c))}
    if rss:
        r = _np.array(rss)
        out['process_rss_mb'] = {'mean': float(r.mean()), 'max': float(r.max())}
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            out['system_cpu_percent'] = psutil.cpu_percent(interval=None)
            out['system_memory_percent'] = vm.percent
            if rss:
                out['process_memory_percent'] = round(
                    max(rss) / (vm.total / (1024 ** 2)) * 100, 2)
        except Exception:
            pass
    return out


def _read_first(paths):
    for p in paths:
        try:
            with open(p) as f:
                return f.read().strip()
        except Exception:
            continue
    return None


def thermal_snapshot():
    """Sicaklik, frekans kisma ve governor: kaynak kisitli dugumde (Pi)
    olcumu sessizce bozan uc etken. Oturum basinda ve sonunda kaydedilir.

    get_throttled bit 0-3 su an, 16-19 acilistan beri: alt-gerilim, frekans
    siniri, kisma, yumusak sicaklik siniri. 0x0 disindaki her deger raporda
    belirtilmeli. Bulunamayan alan None kalir (x86 masaustunde normal).
    """
    snap = {}
    t = _read_first(['/sys/class/thermal/thermal_zone0/temp'])
    try:
        snap['cpu_temp_c'] = round(int(t) / 1000.0, 1) if t else None
    except ValueError:
        snap['cpu_temp_c'] = None
    snap['throttled'] = _read_first([
        '/sys/devices/platform/soc/soc:firmware/get_throttled',
        '/sys/devices/platform/soc/soc:firmware/raspberrypi-hwmon/get_throttled'])
    snap['governor'] = _read_first(
        ['/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'])
    f = _read_first(['/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'])
    try:
        snap['cpu_mhz_now'] = int(f) / 1000.0 if f else None
    except ValueError:
        snap['cpu_mhz_now'] = None
    return snap


#: --xai-method ile secilebilen aciklayicilar. Iki maliyet noktasi kasitli:
#: secici XAI'nin kazanci aciklama pahalilastikca buyuyor.
XAI_METHODS = ('none', 'lime', 'lime_cheap', 'shap', 'shap_cheap',
               # Eski Turkce adlar GERIYE UYUMLULUK icin kabul ediliyor.
               # Ikinci bilgisayarda ve bulut VM'inde kosan komutlar,
               # betikler ve docker-compose dosyalari bunlari kullaniyor;
               # tek seferde degistirmek sessiz kosu hatalari uretirdi.
               # canonical_method() ikisini de ayni kimlige indirger.
               'lime_ucuz', 'shap_ucuz')

#: Eski ad -> yeni ad. Raporda ve grafiklerde yalnizca yeni adlar gorunur.
_METHOD_ALIASES = {'lime_ucuz': 'lime_cheap', 'shap_ucuz': 'shap_cheap'}


def canonical_method(method):
    """Eski Turkce adi yeni Ingilizce karsiligina cevirir."""
    return _METHOD_ALIASES.get(method, method)


def build_explainer(method, model, device='cpu'):
    """xai_config.json'daki dondurulmus yapilandirmadan aciklayici kurar.

    Import BILEREK tembel: `xai/` paketi sunucu imajinda bulunmayabilir
    (.dockerignore onu disarida birakiyor, cunku --xai-method none varsayilan
    ve o durumda hic gerekmiyor). Tepede import edilseydi XAI kullanmayan
    bir dagitimda sunucu hic acilmazdi.
    """
    if canonical_method(method) in (None, 'none'):
        return None
    from xai.artifacts import load_artifacts
    from xai import lime_ as lime_mod, shap_ as shap_mod

    # LIME arka plan kumesine ihtiyac duymuyor; onu okumak 3.4 MB'lik npz'yi
    # diskten yuklemek ve SHA-256'sini dogrulamak demek. Kaynak kisitli bir
    # dugumde bosuna maliyet.
    method = canonical_method(method)
    needs_bg = method in ('shap', 'shap_cheap')
    record = load_artifacts(with_background=needs_bg)

    if method == 'lime':
        return lime_mod.from_artifacts(model, record, device=device)
    if method == 'lime_cheap':
        # CPU'da LIME'in maliyeti ornek sayisiyla dogrusal; dilim sayisi
        # bedava (olculdu). Ucuz nokta bu yuzden num_samples'tan geciyor.
        return lime_mod.from_artifacts(model, record, device=device, num_samples=250)
    if method == 'shap':
        return shap_mod.from_artifacts(model, record, device=device)
    if method == 'shap_cheap':
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
        self.loop = hello.get('loop', 'closed')
        self.sample_start = 0          # kaynak ornekleyicide oturumun baslangici
        self.thermal_start = None


def process_packet(sess, body, cfg, model, explainer, xai_threshold):
    """Bir veri paketini isler; ISCI IPLIGINDE calisir (olay dongusunde degil).

    Neden ayri iplik: bu fonksiyon aciklama acikken saniyeler surer. Olay
    dongusunde calistiginda okuyucu gorevler o sure boyunca paket okuyamaz;
    bekleyen paketlerin t_recv damgasi aciklama bitene kadar atilmaz ve
    kuyruk beklemesi 'latency' sutununa kayar (v1 ve Pi yuk kosularinda
    4 istemcide 'ag gecikmesi' medyani 1.2 s okundu). Tek iplikli executor
    tek-worker semantigini korur; oturum durumu yalnizca bu iplikte degisir.
    """
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
            # ICINDE uretilir. Ayri bir worker'a atilsaydi maliyeti
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
    return msg_id, t_start, t_end, predicted, pred, conf, stages, explained


async def worker(state):
    """
    Tek worker: kuyrugu sirayla bosaltir, modeli calistirir, cevabi yazar.
    Hesap tek iplikli executor'da kosar; o surerken olay dongusu yeni
    paketleri okumaya ve damgalamaya devam eder. Ikinci bir istek kuyrukta
    bekler -- queuing delay tam olarak bu beklemeyi olcer.
    """
    cfg, model, queue = state['cfg'], state['model'], state['queue']
    explainer = state.get('explainer')
    xai_threshold = state.get('xai_threshold', 0.0)
    executor = state['executor']
    loop = asyncio.get_running_loop()

    while True:
        sess, body, t_recv = await queue.get()
        try:
            (msg_id, t_start, t_end, predicted, pred, conf, stages,
             explained) = await loop.run_in_executor(
                executor, process_packet, sess, body, cfg, model,
                explainer, xai_threshold)

            sess.writer.write(P.pack_reply(msg_id, t_recv, t_start, t_end,
                                           predicted=predicted, pred=pred,
                                           confidence=conf, true_label=None,
                                           stages=stages, explained=explained))
            await sess.writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# --processes N > 1: ayni dugumde N hesap sureci (IISEC). N = 1 yukaridaki
# tek-worker yolunu AYNEN kullanir; ICAIIC olcumleri o yoldan alindi.
# ---------------------------------------------------------------------------

def prepare_packet(sess, body, cfg):
    """Oturum durumunu degistiren hafif kisim: tampon + normalizasyon.

    Tek iplikli 'prep' executor'unda kosar; gorevler kuyruktan alindigi
    sirayla (arada await olmadan) gonderildigi icin her oturumun paketleri
    gonderim sirasiyla islenir ve RunningNormalizer bozulmaz.
    """
    msg_id, _, chunks = P.unpack_data(body, NATIVE_STREAMS, sess.per_tick)
    ready = sess.buffer.push(chunks, sess.tick_s)
    if not ready:
        return msg_id, None, None
    stages = {}
    t0 = time.perf_counter()
    raw_window = sess.buffer.get_window()
    stages['resampling'] = (time.perf_counter() - t0) * 1000.0
    if raw_window is None:
        return msg_id, None, stages
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
    sess.n_windows += 1
    return msg_id, np.ascontiguousarray(norm_window, dtype=np.float32), stages


async def worker_mp(state):
    """N esdeger tuketiciden biri. Kuyruktan alir, hafif kismi 'prep'
    ipliginde, agir kismi (cikarim + aciklama) surec havuzunda calistirir.
    Ayni anda en fazla N paket islenir; fazlasi kuyrukta bekler ve bu
    bekleme queuing delay olarak olculur (tek-worker tanimiyla ayni)."""
    from serving import mp_worker
    cfg, queue = state['cfg'], state['queue']
    xai_threshold = state.get('xai_threshold', 0.0)
    prep, pool = state['executor'], state['pool']
    loop = asyncio.get_running_loop()

    while True:
        sess, body, t_recv = await queue.get()
        try:
            t_start = time.perf_counter_ns()
            msg_id, norm_window, stages = await loop.run_in_executor(
                prep, prepare_packet, sess, body, cfg)
            predicted = explained = False
            pred = conf = None
            if norm_window is not None:
                pred, conf, st2, explained, pid = await loop.run_in_executor(
                    pool, mp_worker.child_compute, norm_window, xai_threshold)
                stages.update(st2)
                predicted = True
                if explained:
                    sess.n_explained += 1
                state['pid_windows'][pid] = state['pid_windows'].get(pid, 0) + 1
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


class _TreeSampler(threading.Thread):
    """_ResourceSampler gibi, ama ana surec + TUM alt surecler toplanir.
    Aksi halde hesap alt sureclerde oldugu icin CPU ~0 okunurdu.
    cpu_percent her surec nesnesinin kendi onceki cagrisina gore olcer;
    nesneler pid ile onbellekte tutulur."""

    def __init__(self, proc, interval=0.25):
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.cpu, self.rss = [], []
        self._procs = {}
        self._stop_evt = threading.Event()

    def _members(self):
        try:
            kids = self.proc.children(recursive=True)
        except Exception:
            kids = []
        live = {}
        for p in [self.proc] + kids:
            live[p.pid] = self._procs.get(p.pid) or p
            if p.pid not in self._procs:
                try:
                    live[p.pid].cpu_percent(interval=None)
                except Exception:
                    pass
        self._procs = live
        return list(live.values())

    def run(self):
        self._members()
        while not self._stop_evt.wait(self.interval):
            cpu = rss = 0.0
            for p in self._members():
                try:
                    cpu += p.cpu_percent(interval=None)
                    rss += p.memory_info().rss / (1024 ** 2)
                except Exception:
                    pass
            self.cpu.append(cpu)
            self.rss.append(rss)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)


async def handle_client(reader, writer, state):
    cfg, model = state['cfg'], state['model']
    peer = writer.get_extra_info('peername')

    try:
        body = await P.read_frame(reader)
    except asyncio.IncompleteReadError:
        # Health checks and port probes may connect without sending a frame.
        writer.close()
        await writer.wait_closed()
        return
    if body is None or P.msg_type(body) != P.TYPE_HELLO:
        writer.close()
        return
    hello = json.loads(body[1:].decode('utf-8'))
    sess = ClientSession(cfg, hello, state['mean'], state['std'], writer=writer)
    sampler = state.get('sampler')
    sess.sample_start = len(sampler.cpu) if sampler else 0
    sess.thermal_start = thermal_snapshot()
    sess.pid_start = dict(state.get('pid_windows', {}))
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
                    xai_info = {'method': canonical_method(state.get('xai_method', 'none')),
                                'method_as_given': state.get('xai_method', 'none'),
                                'threshold': state.get('xai_threshold', 0.0),
                                'warmup_ms': state.get('xai_warmup_ms')}
                    explainer = state.get('explainer')
                    if explainer is not None:
                        try:
                            xai_info['config'] = explainer.config()
                        except Exception:
                            pass
                    elif state.get('xai_config') is not None:
                        xai_info['config'] = state['xai_config']
                    writer.write(P.pack_stats({
                        'hardware': state['hardware'],
                        'resources': resource_summary(state.get('sampler'),
                                                      start=sess.sample_start),
                        'resources_scope': 'session',
                        # v2.1: hesap executor'da; latency/queuing ayrimi
                        # cok istemcide de gecerli. Eski kosularda alan yok.
                        # v2.3: BLAS siniri bayraga baglandi (--limit-blas,
                        # varsayilan kapali); fiili durum
                        # hardware.blas_threads_limited alaninda.
                        # v2.4: --processes eklendi; processes == 1 yolu
                        # v2.3 ile ayni kod.
                        'server_build': 'v2.4-multiprocess',
                        'processes': state.get('processes', 1),
                        # Oturum suresince her alt surecin isledigi pencere
                        # (tum istemciler toplam): yuk dagiliminin kaniti.
                        'windows_per_process': {
                            str(k): v - sess.pid_start.get(k, 0)
                            for k, v in state.get('pid_windows', {}).items()},
                        'thermal': {'start': sess.thermal_start,
                                    'end': thermal_snapshot()},
                        'client_loop': sess.loop,
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
                         if state.get('xai_on') else "")
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
    blas_limited = False
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    # BLAS havuzu (numpy/scikit-learn) VARSAYILAN olarak serbesttir:
    # ICAIIC veri setinin tamami (edge/Pi/cloud) serbest BLAS ile olculdu
    # ve makale bunu acikca beyan ediyor ("default multithreading on both
    # placements"). Siniri acmak yeni kosulari eski tablolarla
    # KARSILASTIRILAMAZ yapar: host'ta lime_cheap 287 -> ~478 ms (~1.7x).
    # Cok-worker deneyleri (IISEC, worker basina cekirdek butcesi) icin
    # --limit-blas ile acilir; hangi kosunun hangi modda oldugu
    # hardware.blas_threads_limited alanina yazilir.
    if getattr(args, 'limit_blas', False) and args.threads > 0:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=args.threads)
            blas_limited = True
        except ImportError:
            pass

    model, arch, meta = load_checkpoint(args.model, torch.device('cpu'), fallback_cfg=cfg)

    mean = std = None
    if args.norm_mode == 'fixed':
        tag = os.path.splitext(os.path.basename(args.model))[0]
        cache = os.path.join(cfg.CHECKPOINT_DIR, f'norm_stats_{tag}.npz')
        mean, std = ensure_fixed_stats(cache, meta.get('train_subjects') or cfg.TRAIN_SUBJECTS, cfg)

    # XAI: aciklayiciyi ISTEMCI BAGLANMADAN kur ve isit. Kaynak ornekleyici
    # de bundan sonra baslatilir, yoksa isinmanin CPU/RAM tepesi kosunun
    # kaynak ozetine karisirdi.
    n_proc = max(1, args.processes)
    explainer = pool = xai_config = None
    warmup_ms = None
    if n_proc == 1:
        explainer = build_explainer(args.xai_method, model, device='cpu')
        if explainer is not None:
            print(f"[i] XAI hazirlaniyor ({args.xai_method}), ilk cagri yavastir...")
            warmup_ms = warmup_explainer(explainer, cfg)
            print(f"[i] XAI hazir. Isinma cagrisi {warmup_ms:.0f} ms surdu "
                  f"(olcume DAHIL DEGIL).")
    else:
        from serving import mp_worker
        ctx = mp.get_context('spawn')
        barrier = ctx.Barrier(n_proc)
        pool = ProcessPoolExecutor(
            max_workers=n_proc, mp_context=ctx, initializer=mp_worker.child_init,
            initargs=(args.model, args.xai_method, args.threads,
                      getattr(args, 'limit_blas', False), barrier))
        print(f"[i] {n_proc} hesap sureci aciliyor ve isitiliyor "
              f"({args.xai_method}); bu bir-iki dakika surebilir...")
        loop = asyncio.get_running_loop()
        ready = await asyncio.gather(*[loop.run_in_executor(pool, mp_worker.child_ready)
                                       for _ in range(n_proc)])
        pids = sorted({r['pid'] for r in ready})
        if len(pids) != n_proc:
            raise SystemExit(f"[!] {n_proc} surec bekleniyordu, {len(pids)} acildi")
        warmups = [r['warmup_ms'] for r in ready if r['warmup_ms'] is not None]
        warmup_ms = max(warmups) if warmups else None
        xai_config = ready[0]['xai_config']
        blas_limited = all(r['blas_limited'] for r in ready) if ready else blas_limited
        print(f"[i] Surecler hazir: pid {pids}"
              + (f", en uzun isinma {warmup_ms:.0f} ms (olcume DAHIL DEGIL)"
                 if warmup_ms is not None else ""))

    hw = hardware_info()
    hw['blas_threads_limited'] = blas_limited
    hw['processes'] = n_proc
    sampler = None
    if _HAS_PSUTIL:
        sampler_cls = _ResourceSampler if n_proc == 1 else _TreeSampler
        sampler = sampler_cls(psutil.Process(os.getpid()), 0.25)
        sampler.start()

    state = {'cfg': cfg, 'model': model, 'mean': mean, 'std': std,
             'explainer': explainer, 'xai_threshold': args.xai_threshold,
             'xai_method': args.xai_method, 'xai_warmup_ms': warmup_ms,
             'xai_on': canonical_method(args.xai_method) != 'none',
             'xai_config': xai_config, 'processes': n_proc, 'pool': pool,
             'pid_windows': {},
             'sessions': 0, 'sessions_peak': 0, 'log_every': args.log_every,
             'queue': asyncio.Queue(), 'hardware': hw, 'sampler': sampler,
             'executor': ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix='worker')}

    if n_proc == 1:
        asyncio.create_task(worker(state))
    else:
        for _ in range(n_proc):
            asyncio.create_task(worker_mp(state))

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
    if state['xai_on']:
        thr = args.xai_threshold
        rule = ("her pencere" if thr >= 1.0 else f"guven < {thr:.2f} olan pencereler")
        print(f"XAI          : {args.xai_method} | {rule} | "
              f"isinma {warmup_ms:.0f} ms")
    else:
        print(f"XAI          : kapali (--xai-method none)")
    print(f"Mimari       : acik istek kuyrugu + "
          + ("tek worker" if n_proc == 1 else
             f"{n_proc} hesap sureci (BLAS {'sinirli' if blas_limited else 'serbest'})"))
    th = thermal_snapshot()
    if any(v is not None for v in th.values()):
        print(f"Termal       : {th.get('cpu_temp_c')} C | governor {th.get('governor')} "
              f"| throttled {th.get('throttled')}")
        if th.get('governor') not in (None, 'performance'):
            print("[!] governor 'performance' degil; frekans olcekleme "
                  "olcumu oynatir.")
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
    parser.add_argument('--limit-blas', action='store_true',
                        default=os.environ.get('WESAD_LIMIT_BLAS', '') == '1',
                        help="BLAS havuzunu da --threads degerine sinirla. "
                             "VARSAYILAN KAPALI: ICAIIC olcumleri serbest BLAS "
                             "ile alindi; acmak yeni kosulari eskilerle "
                             "karsilastirilamaz yapar (~1.7x yavas). "
                             "IISEC cok-worker icin. Ortam: WESAD_LIMIT_BLAS=1")
    parser.add_argument('--processes', type=int,
                        default=int(os.environ.get('WESAD_PROCESSES', '1')),
                        help="Hesap sureci sayisi. 1 (varsayilan) = ICAIIC'teki "
                             "tek-worker yolu. N > 1: cikarim + aciklama N alt "
                             "surece dagitilir; her surec modeli ve aciklayiciyi "
                             "ayri yukler (bellek ~N kat). Ortam: WESAD_PROCESSES")
    parser.add_argument('--log-every', type=int, default=0,
                        help="Kac pakette bir ilerleme yazilsin (0 = yazma)")
    # Varsayilanlar ortam degiskeninden okunur: Docker'da bayrak gecirmek
    # icin komutu degistirmek gerekiyor, ortam degiskeni ise
    # docker-compose'dan verilebiliyor (WESAD_XAI_METHOD=shap_ucuz
    # docker compose up). Komut satirinda acikca verilen deger yine kazanir.
    parser.add_argument('--xai-method', choices=XAI_METHODS,
                        default=os.environ.get('WESAD_XAI_METHOD', 'none'),
                        help="Aciklayici. Yapilandirma checkpoints/xai_config.json'dan "
                             "okunur; 'none' ise xai/ paketi hic import edilmez. "
                             "Ortam degiskeni: WESAD_XAI_METHOD")
    parser.add_argument('--xai-threshold', type=float,
                        default=float(os.environ.get('WESAD_XAI_THRESHOLD', '0')),
                        help="Guven esigi: bunun ALTINDAKI pencereler aciklanir. "
                             "0.0 = hic (taban kosusu), 1.0 = her pencere (en pahali). "
                             "Ortam degiskeni: WESAD_XAI_THRESHOLD")
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
