"""
data/loader.py
==============
WESAD .pkl dosyalarindan ham chest+wrist sinyallerini (14 kanal) okur ve
target_rate'e resample eder. Pencereleme YAPMAZ -- ham, surekli zaman
serisi dondurur. Pencereleme offline egitimde data/windowing.py'de,
streaming'de streaming/buffer.py'de yapilir.
"""

import os
import pickle
import numpy as np


def resample_signal(signal, original_rate, target_rate):
    """Dogrusal interpolasyon ile yeniden ornekleme."""
    if original_rate == target_rate:
        return signal
    orig_len = len(signal)
    target_len = int(orig_len * target_rate / original_rate)
    orig_idx = np.linspace(0, orig_len - 1, orig_len)
    target_idx = np.linspace(0, orig_len - 1, target_len)
    if signal.ndim == 1:
        return np.interp(target_idx, orig_idx, signal)
    result = np.zeros((target_len, signal.shape[1]))
    for i in range(signal.shape[1]):
        result[:, i] = np.interp(target_idx, orig_idx, signal[:, i])
    return result


def load_wesad_subject(subject_id, data_path, target_rate):
    """
    Tek bir subject icin chest(8) + wrist(6) = 14 kanal ham sinyali ve
    etiketleri target_rate'e resample edilmis halde dondurur.

    KANAL SIRASI KRITIKTIR -- egitim scriptindeki (wesad_save_best_model.py)
    sira ile BIREBIR ayni olmak zorundadir, aksi halde model karisik kanal
    aliyor ve dogruluk cokuyor:
        0-2  chest ACC (x, y, z)   700 Hz
        3    chest ECG             700 Hz
        4    chest EMG             700 Hz
        5    chest EDA             700 Hz
        6    chest Temp            700 Hz
        7    chest Resp            700 Hz
        8-10 wrist ACC (x, y, z)    32 Hz
        11   wrist BVP              64 Hz
        12   wrist EDA               4 Hz
        13   wrist TEMP              4 Hz

    Returns
    -------
    X : np.ndarray, shape (T, 14)
    y : np.ndarray, shape (T,)   -- orijinal WESAD etiketleri (henuz binary degil)
    (None, None) : dosya yoksa / hata varsa
    """
    subject_path = os.path.join(data_path, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(subject_path):
        print(f"   x {subject_id}: Dosya bulunamadi")
        return None, None

    try:
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        chest = data['signal']['chest']
        wrist = data['signal']['wrist']

        acc_c  = chest['ACC']
        ecg    = chest['ECG'].reshape(-1, 1)
        emg    = chest['EMG'].reshape(-1, 1)
        eda_c  = chest['EDA'].reshape(-1, 1)
        temp_c = chest['Temp'].reshape(-1, 1)
        resp   = chest['Resp'].reshape(-1, 1)

        acc_w  = wrist['ACC']
        bvp    = wrist['BVP'].reshape(-1, 1)
        eda_w  = wrist['EDA'].reshape(-1, 1)
        temp_w = wrist['TEMP'].reshape(-1, 1)

        raw_labels = data['label'].astype(int)
        label_700hz_len = len(raw_labels)
        target_label_len = int(label_700hz_len * target_rate / 700)
        label_indices = np.round(
            np.linspace(0, label_700hz_len - 1, target_label_len)
        ).astype(int)
        labels_resampled = raw_labels[label_indices]

        acc_c_r  = resample_signal(acc_c,  700, target_rate)
        ecg_r    = resample_signal(ecg,    700, target_rate)
        emg_r    = resample_signal(emg,    700, target_rate)
        eda_c_r  = resample_signal(eda_c,  700, target_rate)
        temp_c_r = resample_signal(temp_c, 700, target_rate)
        resp_r   = resample_signal(resp,   700, target_rate)
        acc_w_r  = resample_signal(acc_w,  32,  target_rate)
        bvp_r    = resample_signal(bvp,    64,  target_rate)
        eda_w_r  = resample_signal(eda_w,  4,   target_rate)
        temp_w_r = resample_signal(temp_w, 4,   target_rate)

        # SIRA: chest (8) once, wrist (6) sonra -- egitimdekiyle birebir
        all_signals = [acc_c_r, ecg_r, emg_r, eda_c_r, temp_c_r, resp_r,
                       acc_w_r, bvp_r, eda_w_r, temp_w_r]
        min_len = min(min(len(s) for s in all_signals), len(labels_resampled))

        X = np.concatenate([s[:min_len] for s in all_signals], axis=1)  # (T, 14)
        y = labels_resampled[:min_len]
        return X, y

    except Exception as e:
        print(f"   x {subject_id}: Hata - {e}")
        import traceback; traceback.print_exc()
        return None, None
