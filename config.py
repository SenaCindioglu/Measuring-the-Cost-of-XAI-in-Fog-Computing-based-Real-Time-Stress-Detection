"""
config.py
=========
Tum proje icin tek merkezi konfigurasyon. Diger moduller buradan import eder.
Offline egitim pipeline'indaki (WESAD wrist+chest CNN) sinyal/mimari
parametreleriyle BIREBIR AYNI tutuldu ki sonuclar karsilastirilabilir olsun.
"""

import os
import torch


class Config:
    # --- Veri yollari (kendi ortamina gore duzenle) ---
    DATA_PATH = '/content/drive/MyDrive/WESAD/'
    CHECKPOINT_DIR = '/content/drive/MyDrive/WESAD/checkpoints/'
    RESULTS_DIR = '/content/drive/MyDrive/WESAD/streaming_results/'

    ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                    'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

    # --- Sinyal parametreleri (offline pipeline ile AYNI) ---
    TARGET_RATE = 64          # Hz
    WINDOW_DURATION = 10.0    # saniye
    WINDOW_SIZE = 640         # 10s * 64Hz
    HOP_DURATION = 5.0        # saniye (%50 overlap -> 5s'de bir yeni pencere)
    HOP_SIZE = 320            # 5s * 64Hz

    NUM_CHANNELS = 14         # wrist(6) + chest(8)
    VALID_LABELS = [1, 2, 3, 4]
    BINARY = True

    # --- Model mimarisi (offline pipeline ile AYNI) ---
    CONV_FILTERS = [32, 64, 128]
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

    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
