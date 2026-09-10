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
