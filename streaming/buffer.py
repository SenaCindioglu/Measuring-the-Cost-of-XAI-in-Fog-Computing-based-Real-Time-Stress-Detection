"""
streaming/buffer.py
====================
Gercek zamanli sensor akisini taklit eden rolling buffer.

Sensorden veri her seferinde kucuk "tick" parcalari halinde gelir (orn.
500ms'lik bloklar). Buffer bu parcalari biriktirir; ilk WINDOW_SIZE ornek
toplandiginda bir pencere "hazir" olur, sonrasinda her HOP_SIZE ornekte bir
yeni pencere hazir olur (%50 overlap, 10s pencere + 5s hop varsayilaniyla).
"""

import numpy as np


class RollingBuffer:
    def __init__(self, num_channels, window_size, hop_size):
        self.num_channels = num_channels
        self.window_size = window_size
        self.hop_size = hop_size

        self._buffer = np.zeros((0, num_channels), dtype=np.float64)
        self._total_pushed = 0
        self._since_last_window = 0
        self._first_window_emitted = False

    def push(self, chunk):
        """
        chunk: (n_samples, num_channels) -- yeni gelen ham veri parcasi.
        Returns: bool -- bu push sonrasi yeni bir pencere hazir mi?
        """
        self._buffer = np.concatenate([self._buffer, chunk], axis=0)
        # Bellek sismesin diye sadece son window_size ornegi tut
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

        self._total_pushed += len(chunk)
        self._since_last_window += len(chunk)

        if not self._first_window_emitted:
            if self._total_pushed >= self.window_size:
                self._first_window_emitted = True
                self._since_last_window = 0
                return True
            return False

        if self._since_last_window >= self.hop_size:
            self._since_last_window = 0
            return True
        return False

    def get_window(self):
        """Buffer henuz tam dolmadiysa None doner."""
        if len(self._buffer) < self.window_size:
            return None
        return self._buffer.copy()  # (window_size, num_channels)


class MultiRateBuffer:
    """
    HAM ornekleme hizlarinda gelen coklu sensor akisini biriktirir ve pencere
    hazir oldugunda ortak hedef hiza (64 Hz) indirerek tek bir (T, C) dizi uretir.

    RollingBuffer'dan farki: orada veri buffer'a girmeden once zaten 64 Hz'e
    indirilmis oluyordu, yani yeniden ornekleme maliyeti olcum disinda kaliyordu.
    Burada her akis kendi hizinda birikir ve indirgeme HER PENCEREDE, olculebilir
    bir is olarak yapilir. Gercek bir sistemde bu is sunucuda yapilir.

    Kenar etkisi notu: pencere basina yeniden ornekleme, tum sinyali bir kerede
    indirmekle birebir ayni sonucu vermez (pencere kenarlarinda interpolasyon
    komsulari farkli). Fark ic bolgede ihmal edilebilir duzeydedir ve zaten
    gercek zamanli bir sistemin yapabileceginin tam karsiligidir: elinde
    yalnizca o ana kadar gelmis veri vardir.
    """

    def __init__(self, streams_spec, window_duration, hop_duration, window_size):
        """
        streams_spec : [(ad, orijinal_hiz_Hz, kanal_sayisi), ...] -- kanal sirasi
                       cikti dizisindeki sirayi belirler
        window_size  : hedef pencere uzunlugu (orn. 640 = 10s @ 64Hz)
        """
        self.spec = list(streams_spec)
        self.window_duration = float(window_duration)
        self.hop_duration = float(hop_duration)
        self.window_size = int(window_size)

        # Her akis icin pencereyi doldurmaya yetecek kadar ornek tutulur
        self._need = {name: int(round(rate * self.window_duration))
                      for name, rate, _ in self.spec}
        self._buf = {name: np.zeros((0, ch), dtype=np.float64)
                     for name, _, ch in self.spec}

        self.elapsed_s = 0.0
        self._since_last = 0.0
        self._first_emitted = False

    def push(self, chunks, tick_s):
        """
        chunks : dict[ad] -> (n_i, C_i) o tick'te gelen ham veri
        tick_s : bu tick'in kac saniyelik veri tasidigi

        Returns: bool -- bu push sonrasi yeni bir pencere hazir mi?
        """
        for name, _, _ in self.spec:
            c = chunks.get(name)
            if c is None or len(c) == 0:
                continue
            b = np.concatenate([self._buf[name], c], axis=0)
            need = self._need[name]
            if len(b) > need:
                b = b[-need:]          # bellek sismesin: sadece bir pencerelik tut
            self._buf[name] = b

        self.elapsed_s += tick_s
        self._since_last += tick_s

        if not self._first_emitted:
            if self.elapsed_s >= self.window_duration and self._full():
                self._first_emitted = True
                self._since_last = 0.0
                return True
            return False

        if self._since_last >= self.hop_duration:
            self._since_last = 0.0
            return True
        return False

    def _full(self):
        return all(len(self._buf[n]) >= self._need[n] for n, _, _ in self.spec)

    def get_window(self):
        """
        Tum akislari hedef uzunluga indirip (window_size, toplam_kanal) dondurur.
        Bu cagri YENIDEN ORNEKLEME MALIYETINI icerir -- olcum bunu kapsamali.
        """
        if not self._full():
            return None

        out = []
        for name, _, _ in self.spec:
            block = self._buf[name]                 # (n_i, C_i) ham hizda
            out.append(_resample_block(block, self.window_size))
        return np.concatenate(out, axis=1)          # (window_size, C_toplam)


def _resample_block(block, target_len):
    """(n, c) bloku dogrusal interpolasyonla (target_len, c) haline getirir."""
    n, c = block.shape
    if n == target_len:
        return block
    src = np.linspace(0.0, n - 1.0, n)
    dst = np.linspace(0.0, n - 1.0, target_len)
    out = np.empty((target_len, c), dtype=np.float64)
    for i in range(c):
        out[:, i] = np.interp(dst, src, block[:, i])
    return out
