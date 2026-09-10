"""
data/normalization.py
======================
SABIT (pre-computed) normalizasyon.

Offline pipeline'daki subject-bazli normalizasyon (her subject kendi
mean/std'siyle normalize edilir) gercek zamanli sistemde KULLANILAMAZ,
cunku henuz gelmemis verinin istatistigini bilemezsin (future leakage).

Bunun yerine: SADECE train subjectlerinin ham sinyalinden hesaplanmis
sabit bir per-channel mean/std kaydedilir ve deployment'ta (streaming
dahil) hep ayni sabit degerler kullanilir.
"""

import numpy as np

from data.loader import load_wesad_subject


def compute_fixed_stats(train_subjects, config):
    """
    Verilen train subjectlerinin TUM ham sinyalini (pencereleme yapmadan)
    birlestirip her kanal icin mean/std hesaplar.

    Returns
    -------
    mean : np.ndarray, shape (num_channels,)
    std  : np.ndarray, shape (num_channels,)
    """
    all_signals = []
    for subj in train_subjects:
        X, _ = load_wesad_subject(subj, config.DATA_PATH, config.TARGET_RATE)
        if X is not None:
            all_signals.append(X)

    if not all_signals:
        raise RuntimeError("Hicbir train subject yuklenemedi, istatistik hesaplanamadi.")

    full = np.concatenate(all_signals, axis=0)  # (T_total, C)
    mean = full.mean(axis=0)
    std = full.std(axis=0)
    std[std == 0] = 1e-8  # sifira bolme korumasi
    return mean, std


def save_stats(mean, std, path):
    np.savez(path, mean=mean, std=std)
    print(f"[i] Normalizasyon istatistikleri kaydedildi: {path}")


def load_stats(path):
    data = np.load(path)
    return data['mean'], data['std']


def apply_normalization(window, mean, std):
    """
    window: (T, C) veya (C, T) -- son eksen kanal sayisina esit olan
    otomatik algilanir ve buna gore broadcast edilir.
    """
    if window.shape[-1] == mean.shape[0]:
        return (window - mean) / std
    # (C, T) formatindaysa mean/std'yi (C, 1) sekline getir
    return (window - mean[:, None]) / std[:, None]


def ensure_fixed_stats(path, train_subjects, config):
    """
    Sabit istatistik dosyasi varsa yukler, yoksa train subjectlerinden hesaplayip
    kaydeder. Streaming emulasyonu ilk calistirildiginda .npz henuz yoktur;
    bu fonksiyon o bosluğu kapatir (hesaplama bir kere yapilir, sonra cache'lenir).
    """
    import os
    if os.path.exists(path):
        print(f"[i] Sabit normalizasyon istatistikleri yuklendi: {path}")
        return load_stats(path)

    print(f"[i] {path} bulunamadi. {len(train_subjects)} train subject'ten hesaplaniyor "
          f"(bir kerelik, sonraki calistirmalarda cache'ten okunur)...")
    mean, std = compute_fixed_stats(train_subjects, config)
    save_stats(mean, std, path)
    return mean, std


class RunningNormalizer:
    """
    NEDENSEL (causal) online z-score.

    Offline egitimde her subject kendi StandardScaler'i ile normalize edilmisti.
    Gercek zamanli sistemde bunun gelecege sizmayan karsiligi budur: sadece
    O ANA KADAR gorulmus ornekler uzerinden Welford ile calisan mean/std.
    Gelecekteki veri asla kullanilmaz.

    Sabit (fixed) moda gore avantaji: modelin egitildigi girdi dagilimina daha
    yakin kalir. Dezavantaji: ilk pencerelerde istatistik henuz oturmamistir
    (warm-up penceresi bunu zaten disarida birakiyor).
    """

    def __init__(self, num_channels, eps=1e-8):
        self.n = 0
        self._mean = np.zeros(num_channels, dtype=np.float64)
        self._m2 = np.zeros(num_channels, dtype=np.float64)
        self.eps = eps

    def update(self, chunk):
        """chunk: (n_samples, num_channels) -- buffer'a giren her tick ile cagrilir."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        n_b = len(chunk)
        if n_b == 0:
            return
        mean_b = chunk.mean(axis=0)
        m2_b = ((chunk - mean_b) ** 2).sum(axis=0)

        delta = mean_b - self._mean
        n_tot = self.n + n_b
        self._mean = self._mean + delta * (n_b / n_tot)
        self._m2 = self._m2 + m2_b + (delta ** 2) * (self.n * n_b / n_tot)
        self.n = n_tot

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        if self.n < 2:
            return np.ones_like(self._mean)
        s = np.sqrt(self._m2 / self.n)
        s[s < self.eps] = self.eps
        return s
