"""
serving/protocol.py
====================
Istemci ile sunucu arasindaki tel formati (wire format) ve zaman damgasi
sozlesmesi.

TASARIM KARARLARI
-----------------
1) Cerceve: 4 bayt big-endian uzunluk + govde. TCP akis protokolu oldugu icin
   mesaj sinirlarini biz koymak zorundayiz.

2) Veri paketi ikili (binary), cevap paketi JSON. Veri paketi her tick'te
   gider ve buyuktur (500 ms'lik ham veri ~11.5 KB), JSON'a cevirmek hem
   bant genisligini hem de serilestirme suresini olcumu bozacak kadar
   artirirdi. Cevap paketi kucuktur, okunabilirlik JSON lehine.

3) Zaman damgalari time.perf_counter_ns() ile alinir.

   Ilk denemede time.time_ns() kullanilmisti; Windows'ta cozunurlugu
   OLCULDU ve 15.6 ms cikti (GetSystemTimeAsFileTime). Loopback uzerindeki
   gecikme ~0.1 ms oldugu icin latency medyani tam sifir okuyordu, yani
   metrik tamamen anlamsizdi. perf_counter ise QueryPerformanceCounter'a
   dayanir ve 100 ns cozunurluk verir.

   perf_counter'in referans noktasi tanimsizdir, bu yuzden iki surecin
   degerleri dogrudan karsilastirilamaz. Cozum: baglantinin basinda NTP
   mantigiyla bir SAAT FARKI (offset) olculur -- asagidaki PING/PONG.
   Boylece hem ayni makinede hem de uzak makinede tek yonlu gecikme
   olculebilir; olcumun belirsizligi gidis-donus suresinin yarisidir ve
   rapora yazilir.

   Offset gerektirmeyen metrikler (iki damga da AYNI saatten):
       queuing delay  = t_start - t_recv    (sunucu saati)
       execution time = t_end   - t_start   (sunucu saati)
       response time  = t_reply - t_send    (istemci saati)
   Offset gerektiren tek metrik: latency = (t_recv - offset) - t_send.

OLCULEN ZAMAN DAMGALARI (makaledeki metrik tanimlariyla ayni)
--------------------------------------------------------------
    t_send   istemci paketi gonderdi
    t_recv   sunucu paketi tam olarak aldi        latency        = t_recv - t_send
    t_start  sunucu islemeye basladi              queuing delay  = t_start - t_recv
    t_end    sunucu islemeyi bitirdi              execution time = t_end - t_start
    t_reply  istemci cevabi aldi                  response time  = t_reply - t_send
                                                  jitter         = response time'in degisimi
"""

import json
import struct

import numpy as np

HEADER = struct.Struct('>I')          # cerceve uzunlugu
DATA_HEADER = struct.Struct('<IQ')    # msg_id (uint32), t_send_ns (uint64)

TYPE_HELLO = 1
TYPE_DATA = 2
TYPE_REPLY = 3
TYPE_BYE = 4
TYPE_PING = 5
TYPE_PONG = 6
TYPE_STATS = 7

PING = struct.Struct('<Q')      # t1 (istemci saati)
PONG = struct.Struct('<QQ')     # t1 yankisi + t2 (sunucu saati)

DTYPE = np.float32                    # tel uzerinde float32; float64 gereksiz yere iki kat yer kaplar


def samples_per_tick(streams_spec, tick_ms):
    """Her akis icin bir tick'te kac ornek gidecegini hesaplar."""
    return {name: int(round(rate * tick_ms / 1000.0)) for name, rate, _ in streams_spec}


def payload_floats(streams_spec, tick_ms):
    """Bir veri paketindeki toplam float sayisi."""
    per_tick = samples_per_tick(streams_spec, tick_ms)
    return sum(per_tick[name] * ch for name, _, ch in streams_spec)


def payload_bytes(streams_spec, tick_ms):
    """Bir veri paketinin govde boyutu (bayt): tip + baslik + veri."""
    return 1 + DATA_HEADER.size + payload_floats(streams_spec, tick_ms) * DTYPE().itemsize


# ----------------------------------------------------------------------
# Cerceveleme
# ----------------------------------------------------------------------

def frame(body: bytes) -> bytes:
    return HEADER.pack(len(body)) + body


async def read_frame(reader):
    """Tam bir cerceve okur. Baglanti kapandiysa None doner."""
    head = await reader.readexactly(HEADER.size) if not reader.at_eof() else b''
    if len(head) < HEADER.size:
        return None
    (n,) = HEADER.unpack(head)
    return await reader.readexactly(n)


# ----------------------------------------------------------------------
# HELLO / BYE (kontrol mesajlari, JSON)
# ----------------------------------------------------------------------

def pack_hello(cfg_dict) -> bytes:
    return frame(bytes([TYPE_HELLO]) + json.dumps(cfg_dict).encode('utf-8'))


def pack_bye() -> bytes:
    return frame(bytes([TYPE_BYE]))


def pack_stats(stats_dict) -> bytes:
    """Sunucu, baglanti kapanirken kendi donanim ve kaynak ozetini gonderir.

    Kaynak kullanimi sunucu tarafinda olculmek zorunda: istemci karsi
    makinenin CPU'sunu goremez. Makaledeki 'resource usage' metriginin
    karsiligi budur.
    """
    return frame(bytes([TYPE_STATS]) + json.dumps(stats_dict).encode('utf-8'))


def unpack_stats(body):
    return json.loads(body[1:].decode('utf-8'))


# ----------------------------------------------------------------------
# Saat senkronizasyonu (NTP mantigi)
# ----------------------------------------------------------------------

def pack_ping(t1_ns) -> bytes:
    return frame(bytes([TYPE_PING]) + PING.pack(t1_ns))


def pack_pong(t1_ns, t2_ns) -> bytes:
    return frame(bytes([TYPE_PONG]) + PONG.pack(t1_ns, t2_ns))


def unpack_ping(body):
    (t1,) = PING.unpack_from(body, 1)
    return t1


def unpack_pong(body):
    return PONG.unpack_from(body, 1)


async def measure_clock_offset(reader, writer, rounds=12):
    """
    Istemci saati ile sunucu saati arasindaki farki olcer.

    Her turda: t1 (istemci) -> t2 (sunucu) -> t3 (istemci)
        offset = ((t2 - t1) + (t2 - t3)) / 2
        rtt    = t3 - t1
    Gidis-donus suresi EN KUCUK olan tur secilir; en az sira bekleyen
    olcum en dogru olanidir (standart NTP pratigi).

    Returns: (offset_ns, rtt_ns) -- belirsizlik +/- rtt/2
    """
    import time as _t
    best = None
    for _ in range(rounds):
        t1 = _t.perf_counter_ns()
        writer.write(pack_ping(t1))
        await writer.drain()
        body = await read_frame(reader)
        t3 = _t.perf_counter_ns()
        if body is None or msg_type(body) != TYPE_PONG:
            continue
        _, t2 = unpack_pong(body)
        rtt = t3 - t1
        offset = ((t2 - t1) + (t2 - t3)) // 2
        if best is None or rtt < best[1]:
            best = (offset, rtt)
    return best if best is not None else (0, 0)


# ----------------------------------------------------------------------
# DATA (ikili)
# ----------------------------------------------------------------------

def pack_data(msg_id, t_send_ns, chunks, streams_spec) -> bytes:
    """
    chunks: dict[ad] -> (n_i, C_i) dizi. Akislar streams_spec sirasiyla,
    duzlestirilmis halde arka arkaya yazilir; boylece uzunluk bilgisi
    tel uzerinde tasinmak zorunda kalmaz (iki taraf da tick'i biliyor).
    """
    flat = [np.ascontiguousarray(chunks[name], dtype=DTYPE).ravel()
            for name, _, _ in streams_spec]
    blob = np.concatenate(flat).tobytes() if flat else b''
    body = bytes([TYPE_DATA]) + DATA_HEADER.pack(msg_id, t_send_ns) + blob
    return frame(body)


def unpack_data(body, streams_spec, per_tick):
    """
    Returns: (msg_id, t_send_ns, chunks)
    """
    msg_id, t_send_ns = DATA_HEADER.unpack_from(body, 1)
    arr = np.frombuffer(body, dtype=DTYPE, offset=1 + DATA_HEADER.size)
    chunks, off = {}, 0
    for name, _, ch in streams_spec:
        n = per_tick[name]
        size = n * ch
        chunks[name] = arr[off:off + size].reshape(n, ch).astype(np.float64)
        off += size
    return msg_id, t_send_ns, chunks


# ----------------------------------------------------------------------
# REPLY (JSON)
# ----------------------------------------------------------------------

def pack_reply(msg_id, t_recv_ns, t_start_ns, t_end_ns,
               predicted=False, pred=None, confidence=None,
               true_label=None, stages=None) -> bytes:
    payload = {
        'msg_id': msg_id,
        't_recv': t_recv_ns,
        't_start': t_start_ns,
        't_end': t_end_ns,
        'predicted': predicted,
    }
    if predicted:
        payload.update({'pred': pred, 'confidence': confidence,
                        'true_label': true_label, 'stages': stages})
    return frame(bytes([TYPE_REPLY]) + json.dumps(payload).encode('utf-8'))


def unpack_reply(body):
    return json.loads(body[1:].decode('utf-8'))


def msg_type(body):
    return body[0] if body else None
