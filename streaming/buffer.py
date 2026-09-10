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
