"""
config.py
=========
Tum proje icin tek merkezi konfigurasyon. Diger moduller buradan import eder.

MIMARI: Bu dosya, streaming emulasyonunda kullanilan SECILI OFFLINE MODEL ile
birebir ayni tutuldu:
    wesad_best_model_f32-128_k9.pt
    -> depth=2, filters=[32, 128], kernel_size=9, seed=0
    -> val F1 = 0.9386 (val subjects: S11, S14), 49.890 parametre
"""

import os
import torch

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _env_path(var, default):
    """Ortam degiskeni varsa onu, yoksa varsayilani kullan."""
    return os.environ.get(var, default)


class Config:
    # --- Veri yollari ---
    # WESAD kok klasoru: icinde S2/, S3/, ... alt klasorleri ve S2/S2.pkl dosyalari olmali.
    # WESAD_DATA_PATH ortam degiskeniyle ezilebilir.
    DATA_PATH = _env_path('WESAD_DATA_PATH', r'd:/Users/senac/Downloads/WESAD/WESAD/')
    CHECKPOINT_DIR = _env_path('WESAD_CHECKPOINT_DIR', os.path.join(_PROJECT_ROOT, 'checkpoints'))
    RESULTS_DIR = _env_path('WESAD_RESULTS_DIR', os.path.join(_PROJECT_ROOT, 'results'))

    # Streaming emulasyonunda kullanilacak varsayilan offline model checkpoint'i.
    MODEL_PATH = os.path.join(_PROJECT_ROOT, 'wesad_best_model_f32-128_k9.pt')

    ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                    'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

    # --- Secili modelin egitim/validasyon ayrimi (checkpoint meta'sindan) ---
    # Sabit normalizasyon istatistikleri SADECE TRAIN_SUBJECTS'ten hesaplanir.
    # VAL_SUBJECTS model tarafindan hic egitilmedi -> replay icin dogru secim bunlar.
    TRAIN_SUBJECTS = ['S2', 'S16', 'S7', 'S10', 'S4', 'S3', 'S17',
                      'S6', 'S9', 'S13', 'S15', 'S5', 'S8']
    VAL_SUBJECTS = ['S11', 'S14']

    # --- Sinyal parametreleri (offline pipeline ile AYNI) ---
    TARGET_RATE = 64          # Hz
    WINDOW_DURATION = 10.0    # saniye
    WINDOW_SIZE = 640         # 10s * 64Hz
    HOP_DURATION = 5.0        # saniye (%50 overlap -> 5s'de bir yeni pencere)
    HOP_SIZE = 320            # 5s * 64Hz

    NUM_CHANNELS = 14         # chest(8) + wrist(6)

    # Kanal sirasi egitim scriptindeki (wesad_save_best_model.py) sira ile
    # BIREBIR ayni olmak zorunda -- data/loader.py bunu uretir.
    CHANNEL_NAMES = ['chest_ACC_x', 'chest_ACC_y', 'chest_ACC_z',
                     'chest_ECG', 'chest_EMG', 'chest_EDA', 'chest_Temp', 'chest_Resp',
                     'wrist_ACC_x', 'wrist_ACC_y', 'wrist_ACC_z',
                     'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
    VALID_LABELS = [1, 2, 3, 4]
    BINARY = True

    # --- Model mimarisi (SECILI MODEL: depth2 / f32-128 / k9) ---
    # NOT: run_streaming_emulation.py mimariyi checkpoint'in kendi meta alanindan
    # okur; bu degerler sadece yeniden egitim ve fallback icin gecerlidir.
    CONV_FILTERS = [32, 128]  # depth=2 (baseline [32, 64, 128] degil)
    KERNEL_SIZE = 9
    CONV_DROPOUT = 0.2
    FC_DROPOUT = 0.5

    # --- Egitim parametreleri (sadece training/train_final_model.py kullanir) ---
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 150
    PATIENCE = 20
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.5
    FOCAL_GAMMA = 2.0

    # --- Streaming/replay parametreleri ---
    DEFAULT_TICK_MS = 500        # buffer'a her seferinde kac ms'lik veri eklenecek
    DEFAULT_WARMUP_WINDOWS = 15  # olcume dahil edilmeyecek ilk pencere sayisi

    # PyTorch CPU is parcacigi sayisi.
    #
    # benchmark_runs.py ile GERCEK pipeline uzerinde, 4 tur donusumlu kosuyla
    # (S11, nedensel normalizasyon) olculen degerler:
    #     iplik   p50      p95      tekrarlar arasi yayilim   CPU
    #       1    3.397ms  4.980ms          %2.9               %91
    #       2    3.933ms  5.774ms         %14.7               %93
    #       4    3.684ms  5.508ms         %10.7              %103
    #
    # Medyanlar arasindaki fark (%15.8) tekrarlar arasi yayilimla ayni
    # mertebede, yani hiz farki ayirt edilemiyor. Varsayilanin 1 olmasinin
    # gerekcesi hiz degil KARARLILIK: tek iplikte olcum bes kat daha az
    # oynuyor, ayrica giyilebilir/gomulu dagitim senaryosuna en yakin olan bu.
    #
    # UYARI: izole mikro-benchmark (ayni tensoru tekrar tekrar model'e vermek)
    # yaniltici sonuc verir; iplik sayisi kararini her zaman tam pipeline
    # uzerinde, donusumlu tekrarlarla ver.
    DEFAULT_THREADS = 1

    # Normalizasyon modu:
    #   'fixed'   -> train subjectlerinden hesaplanmis sabit mean/std (deployment gercegi)
    #   'running' -> o ana kadar gorulen veriden nedensel (causal) mean/std;
    #                offline egitimdeki subject-bazli z-score'un gelecege sizmayan karsiligi
    #
    # Model, egitimde subject-bazli StandardScaler ile normalize edilmis veri
    # gordu (wesad_save_best_model.py: normalize_subject). Onun gelecege
    # sizmayan gercek zamanli karsiligi 'running' oldugu icin varsayilan odur.
    # 'fixed' daha kati bir deployment senaryosudur; ikisini de olcup raporla.
    DEFAULT_NORM_MODE = 'running'

    # Egitimdeki subject-bazli train/val ayrimi (referans)
    VAL_FRACTION = 0.15

    # Secili checkpoint'i ureten egitim scriptinin seed'i.
    # NOT: OFAT sweep'inde "seed 0" varyanti raporlanmisti (LOSO F1=0.9321),
    # ama teslim edilen wesad_best_model_f32-128_k9.pt dosyasini ureten
    # wesad_save_best_model.py SEED=42 kullaniyor. Bu checkpoint'i yeniden
    # uretmek istiyorsan deger 42 kalmali.
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
