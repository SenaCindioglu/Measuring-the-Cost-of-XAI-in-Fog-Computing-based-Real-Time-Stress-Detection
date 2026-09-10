"""
data/windowing.py
==================
Offline egitim icin pencereleme ve sinif dengeleme yardimcilari.
(Streaming tarafi bunlari KULLANMAZ -- orada streaming/buffer.py, pencereyi
gercek zamanli akistan olusturur.)
"""

import numpy as np


def create_windows(X, y, window_size, stride):
    """Kayan pencere + cogunluk oylamasi ile etiket."""
    windows, labels = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        w = X[i:i + window_size]
        lbl = y[i:i + window_size]
        majority = np.bincount(lbl.astype(int)).argmax()
        windows.append(w)
        labels.append(majority)
    return np.array(windows), np.array(labels)


def balance_by_undersampling(X, y, seed=42):
    """SADECE train verisine uygulanir; val/test asla undersample edilmez."""
    rng = np.random.RandomState(seed)
    stress_idx = np.where(y == 1)[0]
    non_stress_idx = np.where(y == 0)[0]

    if len(stress_idx) == 0 or len(non_stress_idx) == 0:
        return X, y

    if len(non_stress_idx) > len(stress_idx):
        chosen = rng.choice(non_stress_idx, size=len(stress_idx), replace=False)
    else:
        chosen = non_stress_idx

    all_idx = np.concatenate([stress_idx, chosen])
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def labels_to_binary(y_raw):
    """WESAD ham etiketlerini binary'e cevirir: 2=stress -> 1, digerleri -> 0."""
    return (y_raw == 2).astype(int)


def filter_valid_labels(X, y_raw, valid_labels):
    mask = np.isin(y_raw, valid_labels)
    return X[mask], y_raw[mask]
