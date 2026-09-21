"""
xai/artifacts.py
=================
Secilen XAI yapilandirmalarini ve SHAP'in arka plan kumesini KALICI dosyaya
yazar, sonra geri okur.

NEDEN BOYLE BIR DOSYA GEREKIYOR
--------------------------------
LIME ve SHAP egitilen seyler degil, algoritma. Kaydedilecek bir agirlik yok;
model (`wesad_best_model_f32-128_k9.pt`) dondurulmus halde duruyor ve XAI onu
hic degistirmiyor. Ama iki sey kaydedilmek zorunda:

1) SECILEN PARAMETRELER. Tarama sonunda elle secilen degerler (asama sonu
   tablolarina bakilarak, yalnizca faithfulness'a gore degil gecikme de
   hesaba katilarak). Bunlar sunucuda, olcumde ve makalede birebir ayni
   olmali; koda gomulu sabit olarak dagilirsa bir gun biri degisir ve
   sayilar sessizce tekrar uretilemez hale gelir.

2) SHAP'IN ARKA PLAN KUMESI. SHAP "bu deger su kadar katki yapti" derken bir
   referansa gore konusur: "hicbir sey bilmesem ne beklerdim?". O referans
   arka plan kumesi ve SHAP'in cikti degeri dogrudan ona bagli. Uc sebeple
   dosyaya yazilir:
     - Ayni sayilar ciksin: taramada kullanilan arka plan ile sunucudaki
       birebir ayni olmali.
     - Sunucu her acilista KMeans kosmasin: 2355 x 8960 boyutlu veride
       kumeleme saniyeler suruyor, dosyadan okumak aninda.
     - Sizinti belgelenebilsin: arka plan TEST deneklerinden (S11, S14)
       degil, egitim tarafi deneklerinden gelir. Dosya bunu kendi icinde
       tasir, "hangi deneklerden geldi" sorusu dosyaya bakilarak cevaplanir.

LIME tarafinda kaydedilecek hicbir sey yok: bolge maskeleri her pencerede
sabit seed'den uretilir, uydurulan yerel dogru (Ridge) o pencere icin
yasar ve hemen atilir.

DOGRULANABILIRLIK
-----------------
Kaydedilen JSON, model dosyasinin ve arka plan dizisinin SHA-256 ozetini de
tasir. `load_artifacts(verify=True)` bunlari yeniden hesaplar; dosyalardan
biri degismisse hata verir. Boylece "makaledeki sayilar hangi dosyalarla
uretildi" sorusu kesin olarak cevaplanir.

Kullanim:
    python -m xai.artifacts --save        # uret ve yaz
    python -m xai.artifacts --show        # kayitli olani ozetle

    from xai.artifacts import load_artifacts
    art = load_artifacts()
    art['lime']                # {'num_samples': 500, 'num_segments': 5, ...}
    art['shap']                # {'background_size': 100, 'nsamples': 50, ...}
    art['shap_background']     # (100, 14, 640) float32 numpy dizisi
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from config import config


# ----------------------------------------------------------------------
# SECILEN YAPILANDIRMALAR
# ----------------------------------------------------------------------
# Kaynak: wesad_xai_ofat_sweep_colab.py OFAT taramasi, 21 Eylul 2026, GPU.
# Secim ELLE yapildi. Otomatik secim yalnizca faithfulness'a bakiyor ve
# gecikmeyi hic hesaba katmiyor; olculen degerlerde bu yanlis yonlendiriyor
# (ornek: SHAP nsamples 200 -> 400 kaliteye 0.0004 katti, gecikmeyi iki
# katina cikardi). Asagidaki iki yapilandirma, en iyi kaliteden %2-7 geride
# olup 4-8 kat hizli olanlar.

LIME_CONFIG = {
    'num_samples': 500,       # perturbasyon sayisi
    'num_segments': 5,        # zaman dilimi sayisi (bolge = 5 x 14 kanal = 70)
    'kernel_width': 0.25,     # tarama varsayilani
}

SHAP_CONFIG = {
    'background_size': 100,
    'nsamples': 50,
    'background_method': 'kmeans',
}

# Olculen degerler (GPU, 40 pencere, tek kosu). Yerel CPU olcumu ayrica
# yapilacak; burada duruyor ki dosyanin hangi sayilarla secildigi kaybolmasin.
SELECTION_EVIDENCE = {
    'device': 'cuda (Colab)',
    'n_eval_windows': 40,
    'lime': {'latency_mean_ms': 55.5, 'faithfulness': 0.2370},
    'shap': {'latency_mean_ms': 50.5, 'faithfulness': 0.2287},
    'note': ('Secim elle yapildi: kalite-gecikme tablosuna bakilarak. '
             'En iyi kaliteye gore LIME -%7 / 4.3x hizli, SHAP -%2.3 / 7.7x hizli.'),
}

# Arka plan havuzu: TEST deneklerinden AYRI olmak zorunda.
BACKGROUND_SUBJECTS = ['S2', 'S3', 'S4', 'S5']
EVAL_SUBJECTS = ['S11', 'S14']          # yalnizca kayit icin; burada kullanilmaz
SEED = 42

CONFIG_FILENAME = 'xai_config.json'
BACKGROUND_FILENAME = 'shap_background_k100_kmeans.npz'


# ----------------------------------------------------------------------
# Yardimcilar
# ----------------------------------------------------------------------

def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _sha256_array(arr):
    """Dizinin icerigi degisirse ozet degisir (sekil ve dtype dahil)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'],
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


class _SweepCfg:
    """Tarama script'indeki veri hazirlama fonksiyonlarinin bekledigi alanlar.

    Degerler config.py'den alinir; ikisi ayrismasin diye elle yazilmaz.
    """
    def __init__(self, cfg):
        self.TARGET_RATE = cfg.TARGET_RATE
        self.WINDOW_SIZE = cfg.WINDOW_SIZE
        self.STRIDE = cfg.HOP_SIZE
        self.VALID_LABELS = list(cfg.VALID_LABELS)
        self.BINARY = cfg.BINARY


# ----------------------------------------------------------------------
# Uretim
# ----------------------------------------------------------------------

def build_shap_background(cfg=config, subjects=None, size=None, method=None,
                          seed=SEED, verbose=True):
    """Arka plan kumesini sifirdan uretir.

    Veri hazirlama ve kumeleme, TARAMADAKI FONKSIYONLARIN KENDISI ile yapilir
    (wesad_xai_ofat_sweep_colab). Kod kopyalanirsa iki taraf zamanla ayrisir
    ve arka plan sessizce farklilasir; o an SHAP'in cikti degerleri de degisir.
    """
    # Tarama script'i shap'i artik tembel ice aktardigi icin bu import,
    # shap kurulu olmayan makinede de calisir.
    from wesad_xai_ofat_sweep_colab import prepare_subject_data, build_background

    subjects = list(subjects or BACKGROUND_SUBJECTS)
    size = size or SHAP_CONFIG['background_size']
    method = method or SHAP_CONFIG['background_method']

    overlap = set(subjects) & set(EVAL_SUBJECTS)
    if overlap:
        raise SystemExit(
            "Arka plan, degerlendirme denekleriyle kesisiyor: " + str(sorted(overlap))
            + ". Bu sizintidir; SHAP'in referansi test deneginin verisini gormemeli.")

    sweep_cfg = _SweepCfg(cfg)
    pool = []
    for subj in subjects:
        X, _ = prepare_subject_data(subj, cfg.DATA_PATH, sweep_cfg)
        if X is None:
            raise SystemExit(subj + " yuklenemedi; DATA_PATH dogru mu?")
        pool.append(X)
        if verbose:
            print("   OK %-4s %5d pencere" % (subj, len(X)))
    pool = np.concatenate(pool, axis=0)
    if verbose:
        print("   havuz: %d pencere, sekil %s" % (len(pool), pool.shape))
        print("   %s ile %d merkeze indiriliyor..." % (method, size))

    background = build_background(pool, size, method, seed=seed)
    return background.astype(np.float32), {'pool_size': int(len(pool)),
                                           'subjects': subjects,
                                           'method': method, 'seed': seed}


def save_artifacts(cfg=config, out_dir=None, verbose=True):
    """Arka plani uretir, iki dosyayi yazar ve ozet dondurur."""
    out_dir = out_dir or cfg.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("SHAP arka plani uretiliyor (%s)..." % ', '.join(BACKGROUND_SUBJECTS))
    background, prov = build_shap_background(cfg, verbose=verbose)

    bg_path = os.path.join(out_dir, BACKGROUND_FILENAME)
    # Metadata dosyanin ICINE de yazilir: npz tek basina elden ele
    # gectiginde nereden geldigi kaybolmasin.
    np.savez_compressed(
        bg_path,
        background=background,
        subjects=np.array(prov['subjects']),
        method=np.array(prov['method']),
        seed=np.array(prov['seed']),
        pool_size=np.array(prov['pool_size']),
        channel_names=np.array(cfg.CHANNEL_NAMES),
    )

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': _git_commit(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'model': {
            'path': os.path.basename(cfg.MODEL_PATH),
            'sha256': _sha256_file(cfg.MODEL_PATH),
        },
        'window': {
            'target_rate_hz': cfg.TARGET_RATE,
            'window_size': cfg.WINDOW_SIZE,
            'hop_size': cfg.HOP_SIZE,
            'num_channels': cfg.NUM_CHANNELS,
            'channel_names': list(cfg.CHANNEL_NAMES),
        },
        'lime': dict(LIME_CONFIG),
        'shap': dict(SHAP_CONFIG),
        'shap_background': {
            'file': BACKGROUND_FILENAME,
            'shape': list(background.shape),
            'dtype': str(background.dtype),
            'size_mb': round(background.nbytes / (1024 ** 2), 2),
            'sha256': _sha256_array(background),
            'subjects': prov['subjects'],
            'pool_size': prov['pool_size'],
            'method': prov['method'],
            'seed': prov['seed'],
        },
        'eval_subjects': list(EVAL_SUBJECTS),
        'selection': SELECTION_EVIDENCE,
        'target_class': 1,      # 1 = stres; aciklama bu sinif icin uretilir
    }

    cfg_path = os.path.join(out_dir, CONFIG_FILENAME)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1, ensure_ascii=False)

    if verbose:
        print("\n[i] Yazildi:")
        print("    " + cfg_path)
        print("    " + bg_path + "  (%s, %.2f MB)"
              % (list(background.shape), background.nbytes / (1024 ** 2)))
    return record


# ----------------------------------------------------------------------
# Okuma
# ----------------------------------------------------------------------

def load_artifacts(cfg=config, out_dir=None, verify=True, with_background=True):
    """Kayitli yapilandirmayi ve arka plani okur.

    verify=True ise model dosyasinin ve arka plan dizisinin SHA-256 ozetleri
    yeniden hesaplanip kayitla karsilastirilir. Uyusmazsa hata verilir:
    sessizce farkli bir arka planla olcum yapmak, sonuclari tekrar
    uretilemez hale getirir.
    """
    out_dir = out_dir or cfg.CHECKPOINT_DIR
    cfg_path = os.path.join(out_dir, CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        raise SystemExit(
            cfg_path + " yok. Once uret:  python -m xai.artifacts --save")

    with open(cfg_path, encoding='utf-8') as f:
        record = json.load(f)

    if verify:
        actual = _sha256_file(cfg.MODEL_PATH)
        if actual != record['model']['sha256']:
            raise SystemExit(
                "Model dosyasi kayittan FARKLI (sha256 uyusmuyor).\n"
                "  kayitli: " + record['model']['sha256'][:16] + "...\n"
                "  simdiki: " + actual[:16] + "...\n"
                "Arka plan bu modele gore uretildi; yeniden uretmen gerekir.")

    if with_background:
        bg_path = os.path.join(out_dir, record['shap_background']['file'])
        if not os.path.exists(bg_path):
            raise SystemExit(bg_path + " yok. python -m xai.artifacts --save")
        with np.load(bg_path, allow_pickle=False) as z:
            background = z['background'].astype(np.float32)
        if verify:
            actual = _sha256_array(background)
            if actual != record['shap_background']['sha256']:
                raise SystemExit(
                    "Arka plan dosyasi kayittan FARKLI (sha256 uyusmuyor).")
        record['shap_background_array'] = background

    return record


def show(cfg=config, out_dir=None):
    rec = load_artifacts(cfg, out_dir, verify=True, with_background=True)
    bg = rec['shap_background']
    print("XAI yapilandirmasi  (" + rec['created_utc'] + ")")
    print("  model        : %s  sha %s..." % (rec['model']['path'],
                                              rec['model']['sha256'][:12]))
    print("  git           : %s" % (rec.get('git_commit') or '-')[:12])
    print("  LIME         : " + ", ".join("%s=%s" % kv for kv in rec['lime'].items()))
    print("  SHAP         : " + ", ".join("%s=%s" % kv for kv in rec['shap'].items()))
    print("  arka plan    : %s %s, %.2f MB, %s, denekler %s (havuz %d)"
          % (bg['shape'], bg['dtype'], bg['size_mb'], bg['method'],
             bg['subjects'], bg['pool_size']))
    print("  arka plan sha: %s..." % bg['sha256'][:12])
    print("  test denekleri: %s  (arka planla kesismiyor)" % rec['eval_subjects'])
    arr = rec['shap_background_array']
    print("  dogrulandi   : dizi okundu %s, ortalama %.4f, std %.4f"
          % (arr.shape, float(arr.mean()), float(arr.std())))


def main():
    ap = argparse.ArgumentParser(description="XAI yapilandirmasi ve SHAP arka plani")
    ap.add_argument('--save', action='store_true', help="uret ve dosyaya yaz")
    ap.add_argument('--show', action='store_true', help="kayitli olani ozetle")
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    if args.save:
        save_artifacts(out_dir=args.out_dir)
        print()
        show(out_dir=args.out_dir)
    elif args.show:
        show(out_dir=args.out_dir)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
