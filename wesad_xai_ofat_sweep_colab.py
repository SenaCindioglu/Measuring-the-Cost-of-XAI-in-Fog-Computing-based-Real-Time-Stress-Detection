"""
WESAD Chest+Wrist - ASAMA 3: XAI (LIME + SHAP) OFAT Hiperparametre Sweep
=========================================================================
Bu script, mimari OFAT sweep ile AYNI veri yukleme / pencereleme /
normalizasyon kodunu kullanir; TemporalCNN mimarisi de birebir aynidir.
Ayri olan kisim: model EGITILMIYOR, dondurulmus en iyi checkpoint
(wesad_best_model_f32-128_k9.pt) yuklenir ve uzerinde LIME ile SHAP icin
SIRALI (greedy / One-Factor-At-a-Time) bir hiperparametre taramasi yapilir.

Mantik (hem LIME hem SHAP icin ayni):
    Asama 1: latency'yi en cok etkileyen parametreyi 3 degerle tara -> kazanan
    Asama 2: ikinci parametreyi (1. kazanan sabit) 3 degerle tara -> kazanan
    Asama 3: sadece kaliteyi etkileyen ucuncu parametreyi tara -> kazanan
    -> 3+3+3 = 9 deney / yontem, sonunda "onerilen XAI yapilandirmasi".

LIME sirasi: num_samples -> num_segments -> kernel_width
SHAP sirasi: background_size -> nsamples -> background_method

Kalite metrikleri (iki yontem icin de AYNI tanim):
    faithfulness  = insertion_auc - deletion_auc     (yuksek iyi)
    monotonicity  = insertion egrisinde azalmayan adim orani
    sensitivity   = girdiye kucuk gurultu eklenince atifin degisimi (dusuk iyi)

Her asamanin sonunda uc yapilandirmanin gecikme + kalite tablosu basilir ve
kazanani KULLANICI secer. Otomatik secim tek basina yaniltici: olculen
degerlerde faithfulness farklari yuzde birkac iken gecikme katlarca degisiyor
(ornek: SHAP nsamples 200 -> 400 kaliteye 0.0004 katti, gecikmeyi iki katina
cikardi). Bos girilirse "dengeli oneri" uygulanir: kalitede en fazla
QUALITY_TOLERANCE_PCT kayba izin verip icinde en hizli olani secmek.

Colab'da calistirma (ELLE SECIM ICIN boyle cagir; `!python` ile stdin
olmadigi icin sorular atlanir ve dengeli oneri otomatik secilir):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install shap --quiet
    import wesad_xai_ofat_sweep_colab as w
    w.main()

Hic soru sorulmasin istiyorsan Config.INTERACTIVE = False.

Model dosyasi /content altina dogrudan yuklendiyse ayrica bir sey yapmaya
gerek yok; resolve_paths() orayi ilk sirada arar.
"""

import glob
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

try:
    import psutil
except ImportError:
    psutil = None   # bellek olcumu sessizce atlanir

try:
    import shap
except ImportError:
    raise ImportError("SHAP kurulu degil. Once `!pip install shap` calistirin.")


# ===============================================================
# CONFIGURATION
# ===============================================================

class Config:
    # Bu degerler BASLANGIC tahminidir; resolve_paths() gercek konumu bulur.
    DATA_PATH = '/content/drive/MyDrive/WESAD/'
    MODEL_PATH = '/content/wesad_best_model_f32-128_k9.pt'
    MODEL_NAME = 'wesad_best_model_f32-128_k9.pt'

    # ---- Model mimarisi (dondurulmus en iyi model ile BIREBIR ayni) ----
    CONV_FILTERS = [32, 128]
    KERNEL_SIZE = 9
    CONV_DROPOUT = 0.2
    FC_DROPOUT = 0.5
    NUM_CHANNELS = 14
    NUM_CLASSES = 2
    TARGET_CLASS = 1          # 1 = stres; sadece bu sinif icin acikla

    # ---- Pencere ayarlari (offline pipeline ile ayni) ----
    TARGET_RATE = 64
    WINDOW_DURATION = 10.0
    OVERLAP = 0.5
    WINDOW_SIZE = int(WINDOW_DURATION * TARGET_RATE)     # 640
    STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))            # 320
    VALID_LABELS = [1, 2, 3, 4]
    BINARY = True

    EVAL_SUBJECTS = ['S11', 'S14']                  # tutulmus denekler
    BACKGROUND_SUBJECTS = ['S2', 'S3', 'S4', 'S5']  # SHAP arka plani (AYRI)

    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ileri gecis chunk boyutu: LIME'in 1000 ornekli partisini tek seferde
    # GPU'ya vermek bellegi zorlayabilir, parcali gonderilir. model.eval()
    # oldugu icin BatchNorm kosu istatistiklerini kullanir; parcalama sonucu
    # DEGISTIRMEZ.
    BATCH_CHUNK = 256

    N_EVAL_WINDOWS = 40          # faithfulness/monotonicity/latency icin
    N_SENSITIVITY_WINDOWS = 10   # sensitivity pahali -> daha az pencere
    N_SENSITIVITY_TRIALS = 3

    # ---- LIME ----
    LIME_DEFAULT_NUM_SAMPLES = 500
    LIME_DEFAULT_NUM_SEGMENTS = 10     # zaman dilimi sayisi (x14 kanal)
    LIME_DEFAULT_KERNEL_WIDTH = 0.25

    LIME_SAMPLES_SWEEP = [250, 500, 1000]
    LIME_SEGMENTS_SWEEP = [5, 10, 20]
    LIME_KERNEL_WIDTH_SWEEP = ['x0.5', 'default', 'x2']

    # ---- SHAP ----
    SHAP_DEFAULT_BACKGROUND_SIZE = 50
    SHAP_DEFAULT_NSAMPLES = 200
    SHAP_DEFAULT_BG_METHOD = 'random'

    SHAP_BACKGROUND_SWEEP = [10, 50, 100]
    SHAP_NSAMPLES_SWEEP = [50, 200, 400]
    SHAP_BG_METHOD_SWEEP = ['random', 'kmeans']

    RESULTS_PATH = None
    LIME_SWEEP_CSV = None
    SHAP_SWEEP_CSV = None

    # ---- Asama sonu secimi ----
    # Her asamada uc yapilandirmanin gecikme + kalite tablosu basilir ve
    # kazanani kullanici secer. Bos girilirse dengeli oneri uygulanir:
    # kalitede en fazla QUALITY_TOLERANCE_PCT kayba izin verip icinde en
    # hizli olani almak. INTERACTIVE=False yapilirsa hic sorulmaz.
    INTERACTIVE = True
    QUALITY_TOLERANCE_PCT = 2.0


config = Config()
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


# ===============================================================
# YOL COZUMLEME
# ===============================================================
# Colab'da uc ayri durum var ve hangisinde oldugunu bilmek gerekmesin:
#   - model dogrudan session'a yuklendi   -> /content/<ad>.pt
#   - model Drive'da, ic ice klasorde     -> MyDrive/**/<ad>.pt
#   - veri MyDrive/WESAD/WESAD/S2/S2.pkl gibi ic ice duruyor
# Asagisi her birini kendi bulur ve ne buldugunu yazar.

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_paths(cfg, verbose=True):
    # ---- Model ----
    found = _first_existing([
        cfg.MODEL_PATH,
        os.path.join('/content', cfg.MODEL_NAME),
        os.path.join(cfg.DATA_PATH, cfg.MODEL_NAME),
    ])
    if found is None:
        hits = (glob.glob('/content/' + cfg.MODEL_NAME)
                + glob.glob('/content/drive/MyDrive/**/' + cfg.MODEL_NAME, recursive=True)
                + glob.glob('/content/**/' + cfg.MODEL_NAME, recursive=True))
        if not hits:
            raise FileNotFoundError(
                cfg.MODEL_NAME + " bulunamadi.\n"
                "  - Colab'a dogrudan yuklediysen /content altinda olmali\n"
                "  - Drive'daysa mount edildiginden emin ol"
            )
        found = hits[0]
    cfg.MODEL_PATH = found

    # ---- Veri: S2/S2.pkl iceren KOK klasoru bul (ic ice WESAD/WESAD durumu) ----
    probe = os.path.join(cfg.DATA_PATH, 'S2', 'S2.pkl')
    if not os.path.exists(probe):
        hits = glob.glob('/content/drive/MyDrive/**/S2/S2.pkl', recursive=True)
        if not hits:
            raise FileNotFoundError(
                "WESAD S2/S2.pkl bulunamadi. Drive mount edildi mi, klasor adi dogru mu?"
            )
        cfg.DATA_PATH = os.path.dirname(os.path.dirname(hits[0])) + os.sep

    # ---- Sonuc klasoru: once Drive, yazilamazsa yerel session ----
    for candidate in [os.path.join(cfg.DATA_PATH, 'results_xai_sweep'),
                      '/content/results_xai_sweep']:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe_file = os.path.join(candidate, '.write_test')
            with open(probe_file, 'w') as f:
                f.write('ok')
            os.remove(probe_file)
            cfg.RESULTS_PATH = candidate
            break
        except Exception:
            continue
    if cfg.RESULTS_PATH is None:
        raise RuntimeError("Yazilabilir bir sonuc klasoru bulunamadi.")

    cfg.LIME_SWEEP_CSV = os.path.join(cfg.RESULTS_PATH, 'xai_sweep_lime.csv')
    cfg.SHAP_SWEEP_CSV = os.path.join(cfg.RESULTS_PATH, 'xai_sweep_shap.csv')

    if verbose:
        print("[OK] MODEL_PATH   = " + cfg.MODEL_PATH)
        print("[OK] DATA_PATH    = " + cfg.DATA_PATH)
        print("[OK] RESULTS_PATH = " + cfg.RESULTS_PATH)
        if not cfg.RESULTS_PATH.startswith('/content/drive'):
            print("     UYARI: sonuclar Drive'a DEGIL session'a yaziliyor; "
                  "kosu bitince indir, yoksa kaybolur.")
    return cfg


# ===============================================================
# VERI YUKLEME (mimari-sweep script'i ile BIREBIR ayni)
# ===============================================================

def resample_signal(signal, original_rate, target_rate):
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
    subject_path = os.path.join(data_path, subject_id, subject_id + ".pkl")
    if not os.path.exists(subject_path):
        print("   x " + subject_id + ": Dosya bulunamadi (" + subject_path + ")")
        return None, None
    try:
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        chest = data['signal']['chest']
        c_acc, c_ecg = chest['ACC'], chest['ECG'].reshape(-1, 1)
        c_emg, c_eda = chest['EMG'].reshape(-1, 1), chest['EDA'].reshape(-1, 1)
        c_temp, c_resp = chest['Temp'].reshape(-1, 1), chest['Resp'].reshape(-1, 1)

        wrist = data['signal']['wrist']
        w_acc, w_bvp = wrist['ACC'], wrist['BVP'].reshape(-1, 1)
        w_eda, w_temp = wrist['EDA'].reshape(-1, 1), wrist['TEMP'].reshape(-1, 1)

        raw_labels = data['label'].astype(int)
        label_700hz_len = len(raw_labels)
        target_label_len = int(label_700hz_len * target_rate / 700)
        label_indices = np.round(
            np.linspace(0, label_700hz_len - 1, target_label_len)).astype(int)
        labels_resampled = raw_labels[label_indices]

        c_acc_r = resample_signal(c_acc, 700, target_rate)
        c_ecg_r = resample_signal(c_ecg, 700, target_rate)
        c_emg_r = resample_signal(c_emg, 700, target_rate)
        c_eda_r = resample_signal(c_eda, 700, target_rate)
        c_temp_r = resample_signal(c_temp, 700, target_rate)
        c_resp_r = resample_signal(c_resp, 700, target_rate)
        w_acc_r = resample_signal(w_acc, 32, target_rate)
        w_bvp_r = resample_signal(w_bvp, 64, target_rate)
        w_eda_r = resample_signal(w_eda, 4, target_rate)
        w_temp_r = resample_signal(w_temp, 4, target_rate)

        # SIRA KRITIK: once chest (8), sonra wrist (6) -- egitimdekiyle birebir
        all_signals = [c_acc_r, c_ecg_r, c_emg_r, c_eda_r, c_temp_r, c_resp_r,
                       w_acc_r, w_bvp_r, w_eda_r, w_temp_r]
        min_len = min([len(s) for s in all_signals] + [len(labels_resampled)])
        X = np.concatenate([s[:min_len] for s in all_signals], axis=1)
        labels = labels_resampled[:min_len]
        return X, labels
    except Exception as e:
        print("   x " + subject_id + ": Hata - " + str(e))
        return None, None


def create_windows(X, y, window_size, stride):
    windows, labels = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        windows.append(X[i:i + window_size])
        labels.append(np.bincount(y[i:i + window_size].astype(int)).argmax())
    return np.array(windows), np.array(labels)


def normalize_subject(X):
    from sklearn.preprocessing import StandardScaler
    N, C, T = X.shape
    X_flat = X.transpose(0, 2, 1).reshape(-1, C)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_flat)
    return X_norm.reshape(N, T, C).transpose(0, 2, 1), scaler


def prepare_subject_data(subject_id, data_path, cfg):
    X_raw, y_raw = load_wesad_subject(subject_id, data_path, cfg.TARGET_RATE)
    if X_raw is None:
        return None, None
    mask = np.isin(y_raw, cfg.VALID_LABELS)
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    if cfg.BINARY:
        y_raw = (y_raw == 2).astype(int)
    X_win, y_win = create_windows(X_raw, y_raw, cfg.WINDOW_SIZE, cfg.STRIDE)
    if len(X_win) == 0:
        return None, None
    X_win = X_win.transpose(0, 2, 1)
    X_win, _ = normalize_subject(X_win)
    return X_win.astype(np.float32), y_win


def load_all_subject_data(subject_ids, cfg):
    print("\nVeri yukleniyor (" + str(len(subject_ids)) + " subject)...")
    out_X, out_y = [], []
    for subj in subject_ids:
        X, y = prepare_subject_data(subj, cfg.DATA_PATH, cfg)
        if X is not None:
            out_X.append(X)
            out_y.append(y)
            print("   OK " + subj + ": " + str(len(y)) + " pencere | shape=" + str(X.shape))
        else:
            print("   X  " + subj + ": Yuklenemedi, atlaniyor")
    if not out_X:
        raise RuntimeError("Hicbir subject yuklenemedi. DATA_PATH'i kontrol et.")
    return np.concatenate(out_X, axis=0), np.concatenate(out_y, axis=0)


# ===============================================================
# TEMPORALCNN (offline pipeline ile birebir ayni mimari)
# ===============================================================

class TemporalCNN(nn.Module):
    def __init__(self, num_channels=14, num_classes=2,
                 filters=None, kernel_size=9,
                 conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        if filters is None:
            filters = [32, 128]
        layers = []
        in_ch = num_channels
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(conv_dropout),
            ]
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(filters[-1], 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        return self.fc2(x)


def load_best_model(cfg):
    # weights_only=False SART: checkpoint icinde 'meta', 'filters' gibi tensor
    # olmayan alanlar var. torch>=2.6 varsayilani True ve orada patlar.
    ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Checkpoint kendi mimarisini tasiyorsa ONDAN oku. Sessiz uyusmazlik
    # olmasin diye: load_state_dict zaten patlardi ama once uyarmak daha iyi.
    filters, kernel = list(cfg.CONV_FILTERS), cfg.KERNEL_SIZE
    if isinstance(ckpt, dict) and 'filters' in ckpt:
        ck_filters = list(ckpt['filters'])
        ck_kernel = ckpt.get('kernel_size', cfg.KERNEL_SIZE)
        if ck_filters != filters or ck_kernel != kernel:
            print("   [UYARI] checkpoint mimarisi config'ten farkli: "
                  "filters=" + str(ck_filters) + ", kernel=" + str(ck_kernel)
                  + ". Checkpoint kazanir.")
        filters, kernel = ck_filters, ck_kernel
        cfg.CONV_FILTERS, cfg.KERNEL_SIZE = filters, kernel

    model = TemporalCNN(
        num_channels=cfg.NUM_CHANNELS, num_classes=cfg.NUM_CLASSES,
        filters=filters, kernel_size=kernel,
        conv_dropout=cfg.CONV_DROPOUT, fc_dropout=cfg.FC_DROPOUT,
    ).to(cfg.DEVICE)

    model.load_state_dict(state_dict)   # strict=True: sessiz uyusmazlik olmaz
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("\nModel yuklendi: filters=" + str(filters) + ", kernel=" + str(kernel)
          + ", parametre=" + format(n_params, ','))
    return model


@torch.no_grad()
def predict_prob_batch(model, windows, target_class, device, chunk=256):
    """windows: (N, C, T) numpy -> (N,) hedef sinif olasiliklari.

    Parcali gonderilir: LIME'in 1000 ornekli partisi tek seferde GPU
    bellegini zorlayabiliyor. model.eval() oldugu icin BatchNorm kosu
    istatistiklerini kullanir, yani parcalama sonucu DEGISTIRMEZ.
    """
    out = []
    for i in range(0, len(windows), chunk):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + chunk])).float().to(device)
        probs = torch.softmax(model(x), dim=1)[:, target_class]
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ===============================================================
# OZEL LIME: zaman-dilimi x kanal bolge tabanli perturbasyon + agirlikli Ridge
# ===============================================================

def lime_explain(model, window, num_segments, num_samples, kernel_width,
                 target_class, device, seed=0, chunk=256):
    """
    window: (C, T) numpy.
    num_segments: zaman ekseni kac dilime bolunecek (bolge = num_segments x C).
    Kapatilan bolge o kanalin KENDI ortalamasiyla degistirilir (sifirlamaktan
    fizyolojik olarak daha savunulabilir).
    Donen: attribution (C,T), region_importance (num_regions,)
    """
    C, T = window.shape
    bounds = np.linspace(0, T, num_segments + 1).astype(int)
    regions = [(c, bounds[s], bounds[s + 1])
               for s in range(num_segments) for c in range(C)]
    num_regions = len(regions)

    channel_baseline = window.mean(axis=1)                           # (C,)
    baseline_full = np.repeat(channel_baseline[:, None], T, axis=1)  # (C,T)

    rng = np.random.RandomState(seed)
    masks = rng.randint(0, 2, size=(num_samples, num_regions))
    masks[0, :] = 1   # orijinal pencere de orneklere dahil

    # Bolge kimliklerini (C,T) izgarasina bir kez yaz, sonra tum
    # perturbasyonlari tek vektorel islemle uret. Onceki hali her ornek icin
    # Python dongusuyle bolge bolge kopyaliyordu; olculen latency'nin buyuk
    # kismi modelin degil o dongunun maliyetiydi.
    region_grid = np.empty((C, T), dtype=np.int32)
    for idx, (c, t0, t1) in enumerate(regions):
        region_grid[c, t0:t1] = idx

    keep = masks[:, region_grid].astype(bool)                        # (N, C, T)
    perturbed_batch = np.where(keep, window[None, :, :],
                               baseline_full[None, :, :]).astype(np.float32)

    probs = predict_prob_batch(model, perturbed_batch, target_class, device, chunk)

    # LIME agirliklandirmasi: kapali bolge oranina gore uzaklik + ussel kernel
    distances = 1.0 - masks.mean(axis=1)
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))

    reg = Ridge(alpha=1.0)
    reg.fit(masks, probs, sample_weight=weights)
    region_importance = reg.coef_

    attribution = np.zeros((C, T), dtype=np.float32)
    for idx, (c, t0, t1) in enumerate(regions):
        attribution[c, t0:t1] = region_importance[idx]
    return attribution, region_importance


# ===============================================================
# SHAP: GradientExplainer
# ===============================================================

def build_background(pool_windows, size, method, seed=0):
    if method == 'random':
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pool_windows),
                         size=min(size, len(pool_windows)), replace=False)
        return pool_windows[idx].copy()
    if method == 'kmeans':
        # shap.kmeans'in tablosal ozetlemesinin zaman-serisi karsiligi:
        # pencereleri duzlestir, KMeans merkezlerini "sentetik" arka plan yap.
        flat = pool_windows.reshape(len(pool_windows), -1)
        k = min(size, len(pool_windows))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(flat)
        centers = km.cluster_centers_.reshape(k, *pool_windows.shape[1:])
        return centers.astype(np.float32)
    raise ValueError("Bilinmeyen background methodu: " + str(method))


def make_shap_explainer(model, background_windows, device):
    background_tensor = torch.from_numpy(background_windows).float().to(device)
    explainer = shap.GradientExplainer(model, background_tensor)
    return explainer, 'GradientExplainer'


def shap_explain(explainer, window, nsamples, target_class, device, num_classes=2,
                 seed=0):
    """Atif haritasini (C, T) olarak dondurur.

    SHAP surumden surume FARKLI SEKIL donduruyor ve onceki hali bunu yanlis
    cozuyordu: cok siniflI ciktida (1, C, T, n_classes) gelen dizi
    `.squeeze(-1)` ile duzelmiyor (son eksen 2, squeeze hicbir sey yapmiyor),
    sonuc (C, T, 2) kaliyor ve flatten edilince pencereyle boyutu tutmuyordu.
    Asagisi dort olasi sekli de acikca ele alir ve tutmazsa sessizce
    devam etmek yerine hata verir.
    """
    x = torch.from_numpy(window).float().unsqueeze(0).to(device)

    # SEED SABITLENMEK ZORUNDA. GradientExplainer her cagrida arka plan ile
    # girdi arasinda rastgele `nsamples` nokta ornekler; seed sabit degilse
    # AYNI pencere iki kez aciklandiginda farkli atif cikar. O zaman
    # sensitivity metrigi (girdiye gurultu ekleyip atifin degisimini olcer)
    # yontemin kendi rastgeleligini de sayar ve LIME ile karsilastirilamaz
    # hale gelir -- LIME tarafinda maske ornekleme seed'i sabit tutuluyor.
    # Kendi kararsizligi ayri bir metrik olarak olculuyor (self_instability).
    np.random.seed(seed)
    torch.manual_seed(seed)
    sv = explainer.shap_values(x, nsamples=nsamples)

    if isinstance(sv, list):        # eski surum: sinif basina bir dizi
        a = sv[target_class]
    else:
        a = sv
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)

    if a.ndim == 4:                             # (1, C, T, n_classes)
        a = a[0, ..., target_class]
    elif a.ndim == 3:
        if a.shape[0] == 1:                     # (1, C, T)
            a = a[0]
        elif a.shape[-1] == num_classes:        # (C, T, n_classes)
            a = a[..., target_class]
    if a.shape != window.shape:
        raise ValueError(
            "SHAP atif sekli beklenmedik: " + str(a.shape)
            + ", pencere " + str(window.shape) + ". shap surumunu kontrol edin."
        )
    return np.asarray(a, dtype=np.float32)


# ===============================================================
# KALITE METRIKLERI (LIME ve SHAP icin ORTAK tanim)
# ===============================================================

def _auc(curve):
    """Egrinin [0,1] araligina normalize edilmis alani.

    np.trapz NumPy 2.0'da np.trapezoid oldu; ikisi de desteklenir.
    n noktali egri n-1 aralik kapsar, bolen bu yuzden len(curve)-1.
    (Onceki hali n'e boluyordu; iki yontemde de ayni oldugu icin
    karsilastirma bozulmuyordu ama mutlak deger hafif kayikti.)
    """
    trap = getattr(np, 'trapezoid', None) or np.trapz
    n = len(curve)
    return float(trap(curve)) / max(n - 1, 1)


def compute_faithfulness_monotonicity(model, window, attribution, target_class,
                                      device, num_steps=20, chunk=256):
    """
    deletion : en onemliden baslayarak bolgeleri kanal-ortalamasiyla kapat
               (dusuk AUC = sadik)
    insertion: baseline'dan baslayarak en onemliden ekle
               (yuksek AUC = sadik)
    faithfulness = insertion_auc - deletion_auc
    monotonicity = insertion egrisinde azalmayan adim orani

    Onceki hali her kontrol noktasi icin TEK ORNEKLIK ileri gecis yapiyordu
    (pencere basina 42 cagri). Hepsi tek partide gonderiliyor artik; sonuc
    ayni, sure GPU'da kat kat kisa.
    """
    C, T = window.shape
    baseline = np.repeat(window.mean(axis=1)[:, None], T, axis=1).astype(np.float32)

    flat_w = window.reshape(-1)
    flat_b = baseline.reshape(-1)
    n_feat = flat_w.size

    order = np.argsort(-attribution.reshape(-1))     # en onemli -> en onemsiz
    rank = np.empty(n_feat, dtype=np.int64)
    rank[order] = np.arange(n_feat)

    step = max(1, n_feat // num_steps)
    ks = list(range(0, n_feat + 1, step))
    if ks[-1] != n_feat:
        ks.append(n_feat)
    ks_arr = np.asarray(ks)

    sel = rank[None, :] < ks_arr[:, None]            # (n_ck, n_feat) bool
    del_batch = np.where(sel, flat_b[None, :], flat_w[None, :])
    ins_batch = np.where(sel, flat_w[None, :], flat_b[None, :])

    both = np.concatenate([del_batch, ins_batch]).reshape(-1, C, T).astype(np.float32)
    probs = predict_prob_batch(model, both, target_class, device, chunk)

    n_ck = len(ks)
    del_curve, ins_curve = probs[:n_ck], probs[n_ck:]

    deletion_auc = _auc(del_curve)
    insertion_auc = _auc(ins_curve)
    monotonicity = float(np.mean(np.diff(ins_curve) >= -1e-6))
    return insertion_auc - deletion_auc, monotonicity, deletion_auc, insertion_auc


def compute_sensitivity(explain_fn, window, attribution, noise_scale=0.05,
                        n_trials=3, seed=0):
    """
    Girdiye kucuk Gauss gurultusu eklenince atifin ne kadar degistigi
    (normalize L2 fark). Dusuk = daha kararli aciklama.

    explain_fn ayni seed'i kullandigi icin LIME'in maske ornekleme gurultusu
    sabit kalir; olculen fark yalnizca GIRDI gurultusundan gelir. Istenen bu.
    """
    rng = np.random.RandomState(seed)
    noise_std = noise_scale * window.std()
    flat_attr = attribution.reshape(-1)
    attr_norm = np.linalg.norm(flat_attr) + 1e-8

    diffs = []
    for _ in range(n_trials):
        noisy = (window + rng.normal(0, noise_std, size=window.shape)).astype(np.float32)
        noisy_attr = explain_fn(noisy)
        diffs.append(np.linalg.norm(noisy_attr.reshape(-1) - flat_attr))
    return float(np.mean(diffs) / attr_norm)


# ===============================================================
# BIR KONFIGURASYONU DEGERLENDIRME
# ===============================================================

def compute_self_instability(explain_fn, window, attribution):
    """Ayni pencere yeniden aciklandiginda atif ne kadar degisiyor?

    Girdi hic degismedigi icin deterministik bir yontemde sonuc 0'dir.
    Sifirdan buyuk cikmasi, yontemin kendi ic ornekleme rastgeleliginin
    aciklamayi oynattigi anlamina gelir. sensitivity bu tabanin uzerine
    biner; ikisi ayri raporlanmazsa "kararsiz yontem" ile "girdiye duyarli
    yontem" birbirine karisir.
    """
    attr2 = explain_fn(window)
    flat = attribution.reshape(-1)
    return float(np.linalg.norm(attr2.reshape(-1) - flat)
                 / (np.linalg.norm(flat) + 1e-8))


def evaluate_config(cfg_name, explain_fn, quality_explain_fn, eval_windows,
                    model, target_class, device,
                    n_sensitivity_windows, n_sensitivity_trials,
                    chunk=256, verbose=True):
    """
    explain_fn(window) -> attribution          : latency olcumu (tum pencereler)
    quality_explain_fn(window) -> attribution  : sensitivity icin tekrar cagrilir
    """
    is_cuda = (device.type == 'cuda')
    proc = psutil.Process(os.getpid()) if psutil else None

    latencies, rss_samples = [], []
    faith_list, mono_list, del_list, ins_list = [], [], [], []
    attributions_cache = []

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # ISINMA: ilk cagri her zaman yavastir (tembel yuklemeler, cache, BLAS
    # ilk kurulum). Projedeki diger tum olcumlerde isinma pencereleri hariç
    # tutuluyor; burada tutulmuyordu ve ilk pencere ortalamayi yukari
    # cekiyordu. Sonucu kullanilmadigi icin tek cagri yeterli.
    if len(eval_windows):
        explain_fn(eval_windows[0])

    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter()

    for window in eval_windows:
        # GPU'da senkronizasyon SART: kernel'ler asenkron kuyruga giriyor,
        # sync olmadan olculen sure gercek hesap suresi degil.
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        attribution = explain_fn(window)
        if is_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        faith, mono, del_auc, ins_auc = compute_faithfulness_monotonicity(
            model, window, attribution, target_class, device, chunk=chunk)
        faith_list.append(faith)
        mono_list.append(mono)
        del_list.append(del_auc)
        ins_list.append(ins_auc)
        attributions_cache.append(attribution)

        if proc is not None:
            rss_samples.append(proc.memory_info().rss / (1024 ** 2))

    wall_t1 = time.perf_counter()
    cpu_t1 = time.process_time()

    # NOT: CPU% process_time'dan tureiyor, GPU'da yapilan isi ICERMEZ.
    cpu_percent = 100.0 * (cpu_t1 - cpu_t0) / max(wall_t1 - wall_t0, 1e-9)

    # Sensitivity: pahali oldugu icin sadece ilk n_sensitivity_windows pencerede
    sens_list, self_list = [], []
    for i in range(min(n_sensitivity_windows, len(eval_windows))):
        sens_list.append(compute_sensitivity(
            quality_explain_fn, eval_windows[i], attributions_cache[i],
            n_trials=n_sensitivity_trials))
        self_list.append(compute_self_instability(
            quality_explain_fn, eval_windows[i], attributions_cache[i]))

    lat = np.asarray(latencies)
    result = {
        'config_name': cfg_name,
        'n_windows': len(eval_windows),
        'latency_mean_ms': float(lat.mean()),
        'latency_median_ms': float(np.median(lat)),
        'latency_p95_ms': float(np.percentile(lat, 95)),
        'latency_p99_ms': float(np.percentile(lat, 99)),
        'cpu_percent': float(cpu_percent),
        # Onceki hali resource.getrusage ile SUREC OMRU BOYUNCA tepe RSS
        # okuyordu; o sayi monoton artan oldugu icin ilk konfigurasyondan
        # sonra hepsi ayni degeri yaziyordu ve karsilastirilamazdi.
        # Burada her konfigurasyon KENDI penceresinde olculur.
        'rss_mb_mean': float(np.mean(rss_samples)) if rss_samples else None,
        'rss_mb_max': float(np.max(rss_samples)) if rss_samples else None,
        'gpu_peak_mb': (float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                        if is_cuda else None),
        'faithfulness_mean': float(np.mean(faith_list)),
        'monotonicity_mean': float(np.mean(mono_list)),
        'deletion_auc_mean': float(np.mean(del_list)),
        'insertion_auc_mean': float(np.mean(ins_list)),
        'sensitivity_mean': float(np.mean(sens_list)) if sens_list else None,
        # Yontemin kendi kararsizligi: ayni girdi, degismis hicbir sey yok.
        # Deterministik bir yontemde 0 cikar. sensitivity bunun uzerine
        # binen girdi duyarliligidir; ikisi birlikte okunmali.
        'self_instability_mean': float(np.mean(self_list)) if self_list else None,
    }

    if verbose:
        sens = result['sensitivity_mean']
        sens_txt = ("%.4f" % sens) if sens is not None else "-"
        self_i = result['self_instability_mean']
        self_txt = ("%.4f" % self_i) if self_i is not None else "-"
        print("    -> latency(mean/p95)=%.1f/%.1f ms | faithfulness=%.4f "
              "| monotonicity=%.4f | sensitivity=%s | kendi-kararsizlik=%s"
              % (result['latency_mean_ms'], result['latency_p95_ms'],
                 result['faithfulness_mean'], result['monotonicity_mean'],
                 sens_txt, self_txt))
    return result


def pick_best(df, metric='faithfulness_mean'):
    """En yuksek faithfulness kazanan satir (yalnizca referans olarak)."""
    return df.loc[df[metric].idxmax()]


def balanced_pick(df, tol_pct):
    """Kalite kaybi tolerans icinde kalan EN HIZLI yapilandirma.

    Gerekce: taramada faithfulness farklari cok kucuk cikarken (yuzde birkac)
    gecikme katlarca degisiyor. Ornek olculen degerler: SHAP nsamples 200 ->
    400 faithfulness'a 0.0004 katti, gecikmeyi iki katina cikardi. Boyle bir
    "kazanan" gercek zamanli sistem icin yanlis secim. Bu yuzden once kalitede
    tolerans kadar kayba izin verilir, sonra icinde en hizli olan secilir.
    """
    best_f = df['faithfulness_mean'].max()
    threshold = best_f - abs(best_f) * (tol_pct / 100.0)
    cand = df[df['faithfulness_mean'] >= threshold]
    return cand.loc[cand['latency_mean_ms'].idxmin()]


def _short_label(config_name):
    """'lime_stage1_samples250' -> 'samples250'"""
    parts = config_name.split('_', 2)
    return parts[2] if len(parts) > 2 else config_name


def choose_stage_winner(df, cfg, stage_title):
    """Asamanin sonuclarini tablo olarak basar ve kullaniciya sectirir.

    Otomatik secim (en yuksek faithfulness) tek basina yanlis yonlendiriyor,
    cunku gecikmeyi hic hesaba katmiyor. Karar insanda kalsin diye tablo
    basilip input bekleniyor; bos girilirse dengeli oneri uygulanir.

    stdin yoksa (Colab'da `!python script.py` ile calistirildiginda oyle olur)
    input() EOF verir; o durumda dengeli oneri otomatik secilir ve bu acikca
    yazilir. Elle secim istiyorsan defter hucresinde
        import wesad_xai_ofat_sweep_colab as w; w.main()
    seklinde cagir, `!python` ile degil.
    """
    df = df.reset_index(drop=True)
    best_f_row = pick_best(df)
    bal_row = balanced_pick(df, cfg.QUALITY_TOLERANCE_PCT)
    best_f = float(best_f_row['faithfulness_mean'])
    slowest = float(df['latency_mean_ms'].max())

    print("\n  " + "-" * 92)
    print("  SECIM: " + stage_title)
    print("  %-3s %-22s %18s %13s %10s %10s %8s %8s"
          % ("#", "yapilandirma", "gecikme ort/p95", "faithfulness",
             "monoton.", "sensitiv.", "kalite", "hiz"))
    for i, row in df.iterrows():
        f = float(row['faithfulness_mean'])
        d_pct = 100.0 * (f - best_f) / abs(best_f) if best_f else 0.0
        quality_txt = "en iyi" if row.equals(best_f_row) else ("%+.1f%%" % d_pct)
        speed = slowest / max(float(row['latency_mean_ms']), 1e-9)
        sens = row.get('sensitivity_mean')
        print("  %-3d %-22s %8.1f/%-9.1f %13.4f %10.4f %10s %8s %7.1fx"
              % (i + 1, _short_label(row['config_name']),
                 row['latency_mean_ms'], row['latency_p95_ms'],
                 f, row['monotonicity_mean'],
                 ("%.4f" % sens) if sens is not None else "-",
                 quality_txt, speed))

    bal_i = int(df.index[df['config_name'] == bal_row['config_name']][0]) + 1
    best_i = int(df.index[df['config_name'] == best_f_row['config_name']][0]) + 1
    print("  Dengeli oneri : #%d  (%s)  -- kalite kaybi %%%.0f toleransi icinde en hizli"
          % (bal_i, _short_label(bal_row['config_name']), cfg.QUALITY_TOLERANCE_PCT))
    print("  En yuksek kalite: #%d (%s)"
          % (best_i, _short_label(best_f_row['config_name'])))
    print("  " + "-" * 92)

    if not cfg.INTERACTIVE:
        print("  [otomatik] INTERACTIVE=False -> dengeli oneri secildi: #%d" % bal_i)
        return bal_row

    while True:
        try:
            raw = input("  Secimin [1-%d, bos = dengeli oneri #%d]: "
                        % (len(df), bal_i)).strip()
        except (EOFError, OSError):
            print("\n  [stdin yok -> dengeli oneri secildi: #%d]" % bal_i)
            return bal_row
        if raw == "":
            print("  -> #%d secildi (dengeli oneri)" % bal_i)
            return bal_row
        if raw.isdigit() and 1 <= int(raw) <= len(df):
            sel = df.loc[int(raw) - 1]
            print("  -> #%s secildi (%s)" % (raw, _short_label(sel['config_name'])))
            return sel
        print("  Gecersiz giris. 1-%d arasi bir sayi yaz ya da bos birak." % len(df))


# ===============================================================
# LIME OFAT SWEEP: num_samples -> num_segments -> kernel_width
# ===============================================================

def run_lime_ofat_sweep(model, eval_windows, cfg):
    print("\n" + "=" * 70)
    print("LIME OFAT SWEEP")
    print("=" * 70)

    all_results = []
    kw_map = {
        'x0.5': cfg.LIME_DEFAULT_KERNEL_WIDTH * 0.5,
        'default': cfg.LIME_DEFAULT_KERNEL_WIDTH,
        'x2': cfg.LIME_DEFAULT_KERNEL_WIDTH * 2.0,
    }

    def make_explain_fn(num_segments, num_samples, kernel_width, seed=0):
        def fn(window):
            attribution, _ = lime_explain(
                model, window, num_segments=num_segments, num_samples=num_samples,
                kernel_width=kernel_width, target_class=cfg.TARGET_CLASS,
                device=cfg.DEVICE, seed=seed, chunk=cfg.BATCH_CHUNK)
            return attribution
        return fn

    def run(cfg_name, fn, extra):
        print("  [" + cfg_name + "]")
        res = evaluate_config(cfg_name, fn, fn, eval_windows, model,
                              cfg.TARGET_CLASS, cfg.DEVICE,
                              cfg.N_SENSITIVITY_WINDOWS, cfg.N_SENSITIVITY_TRIALS,
                              chunk=cfg.BATCH_CHUNK)
        res.update(extra)
        all_results.append(res)

    # ---- Sira 1: num_samples ----
    print("\n>>> SIRA 1: num_samples (segments=%d, kernel_width=%s sabit)"
          % (cfg.LIME_DEFAULT_NUM_SEGMENTS, cfg.LIME_DEFAULT_KERNEL_WIDTH))
    for ns in cfg.LIME_SAMPLES_SWEEP:
        run("lime_stage1_samples" + str(ns),
            make_explain_fn(cfg.LIME_DEFAULT_NUM_SEGMENTS, ns,
                            cfg.LIME_DEFAULT_KERNEL_WIDTH),
            {'num_samples': ns, 'num_segments': cfg.LIME_DEFAULT_NUM_SEGMENTS,
             'kernel_width': cfg.LIME_DEFAULT_KERNEL_WIDTH})

    df1 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('lime_stage1')])
    b1 = choose_stage_winner(df1, cfg, "LIME sira 1 -- num_samples")
    best_samples = int(b1['num_samples'])
    print("  >> Sira 1: num_samples=%d (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_samples, b1['faithfulness_mean'], b1['latency_mean_ms']))

    # ---- Sira 2: num_segments ----
    print("\n>>> SIRA 2: num_segments (num_samples=%d, kernel_width=%s sabit)"
          % (best_samples, cfg.LIME_DEFAULT_KERNEL_WIDTH))
    for seg in cfg.LIME_SEGMENTS_SWEEP:
        run("lime_stage2_segments" + str(seg),
            make_explain_fn(seg, best_samples, cfg.LIME_DEFAULT_KERNEL_WIDTH),
            {'num_samples': best_samples, 'num_segments': seg,
             'kernel_width': cfg.LIME_DEFAULT_KERNEL_WIDTH})

    df2 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('lime_stage2')])
    b2 = choose_stage_winner(df2, cfg, "LIME sira 2 -- num_segments")
    best_segments = int(b2['num_segments'])
    print("  >> Sira 2: num_segments=%d (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_segments, b2['faithfulness_mean'], b2['latency_mean_ms']))

    # ---- Sira 3: kernel_width ----
    print("\n>>> SIRA 3: kernel_width (num_samples=%d, num_segments=%d sabit)"
          % (best_samples, best_segments))
    for kw_label in cfg.LIME_KERNEL_WIDTH_SWEEP:
        kw_val = kw_map[kw_label]
        run("lime_stage3_kw_" + kw_label,
            make_explain_fn(best_segments, best_samples, kw_val),
            {'num_samples': best_samples, 'num_segments': best_segments,
             'kernel_width': kw_val, 'kernel_width_label': kw_label})

    df3 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('lime_stage3')])
    b3 = choose_stage_winner(df3, cfg, "LIME sira 3 -- kernel_width")
    best_kw = float(b3['kernel_width'])
    print("  >> Sira 3: kernel_width=%.4f (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_kw, b3['faithfulness_mean'], b3['latency_mean_ms']))

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(cfg.LIME_SWEEP_CSV, index=False)

    print("\n" + "-" * 70)
    print("LIME ONERILEN: num_samples=%d, num_segments=%d, kernel_width=%.4f"
          % (best_samples, best_segments, best_kw))
    print("Tum LIME sonuclari: " + cfg.LIME_SWEEP_CSV)
    print("-" * 70)

    return df_all, {'num_samples': best_samples, 'num_segments': best_segments,
                    'kernel_width': best_kw}


# ===============================================================
# SHAP OFAT SWEEP: background_size -> nsamples -> background_method
# ===============================================================

def run_shap_ofat_sweep(model, eval_windows, background_pool, cfg):
    print("\n" + "=" * 70)
    print("SHAP OFAT SWEEP")
    print("=" * 70)

    all_results = []

    def make_explain_fn(explainer, nsamples):
        def fn(window):
            return shap_explain(explainer, window, nsamples=nsamples,
                                target_class=cfg.TARGET_CLASS, device=cfg.DEVICE,
                                num_classes=cfg.NUM_CLASSES)
        return fn

    def run(cfg_name, fn, extra):
        print("  [" + cfg_name + "]")
        res = evaluate_config(cfg_name, fn, fn, eval_windows, model,
                              cfg.TARGET_CLASS, cfg.DEVICE,
                              cfg.N_SENSITIVITY_WINDOWS, cfg.N_SENSITIVITY_TRIALS,
                              chunk=cfg.BATCH_CHUNK)
        res.update(extra)
        all_results.append(res)

    # ---- Sira 1: background_size ----
    print("\n>>> SIRA 1: background_size (nsamples=%d, method=%s sabit)"
          % (cfg.SHAP_DEFAULT_NSAMPLES, cfg.SHAP_DEFAULT_BG_METHOD))
    for bg_size in cfg.SHAP_BACKGROUND_SWEEP:
        bg = build_background(background_pool, bg_size,
                              cfg.SHAP_DEFAULT_BG_METHOD, seed=cfg.SEED)
        explainer, method_used = make_shap_explainer(model, bg, cfg.DEVICE)
        run("shap_stage1_bg" + str(bg_size),
            make_explain_fn(explainer, cfg.SHAP_DEFAULT_NSAMPLES),
            {'background_size': bg_size, 'nsamples': cfg.SHAP_DEFAULT_NSAMPLES,
             'bg_method': cfg.SHAP_DEFAULT_BG_METHOD, 'explainer_used': method_used})

    df1 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('shap_stage1')])
    b1 = choose_stage_winner(df1, cfg, "SHAP sira 1 -- background_size")
    best_bg = int(b1['background_size'])
    print("  >> Sira 1: background_size=%d (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_bg, b1['faithfulness_mean'], b1['latency_mean_ms']))

    # ---- Sira 2: nsamples ----
    print("\n>>> SIRA 2: nsamples (background_size=%d, method=%s sabit)"
          % (best_bg, cfg.SHAP_DEFAULT_BG_METHOD))
    bg_fixed = build_background(background_pool, best_bg,
                                cfg.SHAP_DEFAULT_BG_METHOD, seed=cfg.SEED)
    explainer_fixed, method_used = make_shap_explainer(model, bg_fixed, cfg.DEVICE)
    for ns in cfg.SHAP_NSAMPLES_SWEEP:
        run("shap_stage2_nsamples" + str(ns),
            make_explain_fn(explainer_fixed, ns),
            {'background_size': best_bg, 'nsamples': ns,
             'bg_method': cfg.SHAP_DEFAULT_BG_METHOD, 'explainer_used': method_used})

    df2 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('shap_stage2')])
    b2 = choose_stage_winner(df2, cfg, "SHAP sira 2 -- nsamples")
    best_ns = int(b2['nsamples'])
    print("  >> Sira 2: nsamples=%d (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_ns, b2['faithfulness_mean'], b2['latency_mean_ms']))

    # ---- Sira 3: background_method ----
    print("\n>>> SIRA 3: background_method (background_size=%d, nsamples=%d sabit)"
          % (best_bg, best_ns))
    for method in cfg.SHAP_BG_METHOD_SWEEP:
        bg = build_background(background_pool, best_bg, method, seed=cfg.SEED)
        explainer, method_used = make_shap_explainer(model, bg, cfg.DEVICE)
        run("shap_stage3_method_" + method,
            make_explain_fn(explainer, best_ns),
            {'background_size': best_bg, 'nsamples': best_ns,
             'bg_method': method, 'explainer_used': method_used})

    df3 = pd.DataFrame([r for r in all_results
                        if r['config_name'].startswith('shap_stage3')])
    b3 = choose_stage_winner(df3, cfg, "SHAP sira 3 -- background_method")
    best_method = str(b3['bg_method'])
    print("  >> Sira 3: background_method=%s (faithfulness=%.4f, gecikme %.1f ms)"
          % (best_method, b3['faithfulness_mean'], b3['latency_mean_ms']))

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(cfg.SHAP_SWEEP_CSV, index=False)

    print("\n" + "-" * 70)
    print("SHAP ONERILEN: background_size=%d, nsamples=%d, background_method=%s"
          % (best_bg, best_ns, best_method))
    print("Tum SHAP sonuclari: " + cfg.SHAP_SWEEP_CSV)
    print("-" * 70)

    return df_all, {'background_size': best_bg, 'nsamples': best_ns,
                    'background_method': best_method}


# ===============================================================
# KARSILASTIRMA TABLOSU
# ===============================================================

COMPARE_COLS = ['latency_mean_ms', 'latency_p95_ms', 'cpu_percent',
                'rss_mb_max', 'gpu_peak_mb', 'faithfulness_mean',
                'monotonicity_mean', 'sensitivity_mean', 'self_instability_mean']


def print_comparison(df_lime, df_shap, lime_reco, shap_reco, cfg):
    print("\n" + "=" * 70)
    print("LIME vs SHAP - ONERILEN YAPILANDIRMA KARSILASTIRMASI")
    print("=" * 70)

    # Yalniz stage3 satirlarinda ara: stage1/stage2'de ayni parametre
    # uclusunun tekrari olabilir ve .iloc[0] yanlis kosuyu secebilirdi.
    lime_s3 = df_lime[df_lime['config_name'].str.startswith('lime_stage3')]
    lime_row = lime_s3[
        (lime_s3['num_samples'] == lime_reco['num_samples'])
        & (lime_s3['num_segments'] == lime_reco['num_segments'])
        & (np.isclose(lime_s3['kernel_width'], lime_reco['kernel_width']))
    ].iloc[0]

    shap_s3 = df_shap[df_shap['config_name'].str.startswith('shap_stage3')]
    shap_row = shap_s3[
        (shap_s3['background_size'] == shap_reco['background_size'])
        & (shap_s3['nsamples'] == shap_reco['nsamples'])
        & (shap_s3['bg_method'] == shap_reco['background_method'])
    ].iloc[0]

    comp = pd.DataFrame([
        dict([('yontem', 'LIME')] + [(k, lime_row[k]) for k in COMPARE_COLS]),
        dict([('yontem', 'SHAP')] + [(k, shap_row[k]) for k in COMPARE_COLS]),
    ])
    print(comp.to_string(index=False))

    comp_csv = os.path.join(cfg.RESULTS_PATH, 'xai_lime_vs_shap_recommended.csv')
    comp.to_csv(comp_csv, index=False)
    print("\nKarsilastirma tablosu kaydedildi: " + comp_csv)


# ===============================================================
# FINAL DOGRULAMA (opsiyonel): en iyi konfigurasyon, TAM veri kumesi
# ===============================================================

def run_final_validation(model, lime_reco, shap_reco, cfg,
                         full_eval_windows, background_pool):
    """
    Sweep asamalari N_EVAL_WINDOWS kadar altkume kullanir (hiz icin).
    Makaledeki NIHAI sayilar icin bunu S11+S14'un TAM pencere kumesiyle
    (~1175) cagirin. Uzun surer.
    """
    print("\n" + "=" * 70)
    print("FINAL DOGRULAMA - onerilen yapilandirmalar, TAM veri kumesi")
    print("=" * 70)

    def lime_fn(window):
        attribution, _ = lime_explain(
            model, window, num_segments=lime_reco['num_segments'],
            num_samples=lime_reco['num_samples'],
            kernel_width=lime_reco['kernel_width'],
            target_class=cfg.TARGET_CLASS, device=cfg.DEVICE,
            seed=cfg.SEED, chunk=cfg.BATCH_CHUNK)
        return attribution

    bg = build_background(background_pool, shap_reco['background_size'],
                          shap_reco['background_method'], seed=cfg.SEED)
    explainer, _ = make_shap_explainer(model, bg, cfg.DEVICE)

    def shap_fn(window):
        return shap_explain(explainer, window, nsamples=shap_reco['nsamples'],
                            target_class=cfg.TARGET_CLASS, device=cfg.DEVICE,
                            num_classes=cfg.NUM_CLASSES)

    n_sens = min(20, len(full_eval_windows))
    res_lime = evaluate_config('lime_final', lime_fn, lime_fn, full_eval_windows,
                               model, cfg.TARGET_CLASS, cfg.DEVICE,
                               n_sens, cfg.N_SENSITIVITY_TRIALS, chunk=cfg.BATCH_CHUNK)
    res_shap = evaluate_config('shap_final', shap_fn, shap_fn, full_eval_windows,
                               model, cfg.TARGET_CLASS, cfg.DEVICE,
                               n_sens, cfg.N_SENSITIVITY_TRIALS, chunk=cfg.BATCH_CHUNK)

    df_final = pd.DataFrame([res_lime, res_shap])
    final_csv = os.path.join(cfg.RESULTS_PATH, 'xai_final_validation.csv')
    df_final.to_csv(final_csv, index=False)
    print(df_final.to_string(index=False))
    print("\nFinal dogrulama kaydedildi: " + final_csv)
    return df_final


# ===============================================================
# MAIN
# ===============================================================

def main(run_final=False):
    print("=" * 70)
    print("WESAD - ASAMA 3: XAI (LIME + SHAP) OFAT HIPERPARAMETRE SWEEP")
    print("=" * 70)
    resolve_paths(config)
    print("Device: " + str(config.DEVICE))
    print("Eval subjectler: " + str(config.EVAL_SUBJECTS)
          + " (N_EVAL_WINDOWS=" + str(config.N_EVAL_WINDOWS) + ")")
    print("Background subjectler (SHAP): " + str(config.BACKGROUND_SUBJECTS))
    print("=" * 70)

    model = load_best_model(config)

    X_eval_all, y_eval_all = load_all_subject_data(config.EVAL_SUBJECTS, config)
    rng = np.random.RandomState(config.SEED)

    # Sinif dengeli kucuk altkume (sweep hizi icin)
    stress_idx = np.where(y_eval_all == 1)[0]
    nonstress_idx = np.where(y_eval_all == 0)[0]
    n_half = config.N_EVAL_WINDOWS // 2
    chosen = np.concatenate([
        rng.choice(stress_idx, size=min(n_half, len(stress_idx)), replace=False),
        rng.choice(nonstress_idx, size=min(n_half, len(nonstress_idx)), replace=False),
    ])
    eval_windows = X_eval_all[chosen]
    print("\nSweep icin secilen pencere sayisi: " + str(len(eval_windows))
          + " (toplam havuz: " + str(len(X_eval_all)) + ")")

    X_bg_pool, _ = load_all_subject_data(config.BACKGROUND_SUBJECTS, config)
    print("SHAP arka plan havuzu: " + str(len(X_bg_pool)) + " pencere")

    df_lime, lime_reco = run_lime_ofat_sweep(model, eval_windows, config)
    df_shap, shap_reco = run_shap_ofat_sweep(model, eval_windows, X_bg_pool, config)
    print_comparison(df_lime, df_shap, lime_reco, shap_reco, config)

    print("\n" + "=" * 70)
    print("SWEEP TAMAMLANDI.")
    print("LIME onerilen: " + str(lime_reco))
    print("SHAP onerilen: " + str(shap_reco))
    print("Bu degerler Faz D'deki secici XAI mekanizmasina sabit girdi olur.")
    print("=" * 70)

    if run_final:
        run_final_validation(model, lime_reco, shap_reco, config,
                             full_eval_windows=X_eval_all, background_pool=X_bg_pool)
    return df_lime, df_shap, lime_reco, shap_reco


if __name__ == "__main__":
    main(run_final=False)
