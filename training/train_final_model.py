"""
training/train_final_model.py
==============================
Streaming emulasyonunda kullanilacak model checkpoint'ini ve sabit
normalizasyon istatistiklerini uretir.

Bu script GERCEK ZAMANLI DENEYIN BIR PARCASI DEGILDIR -- sadece bir kere
calistirilip ciktisi (model_<tag>.pth + norm_stats_<tag>.npz) streaming
tarafindan kullanilir. Mevcut offline LOSO sonuclarinla karismasin diye
ayri, tek seferlik bir egitim scripti olarak tutuldu.

Kullanim (proje kok klasorunden calistir):
    python -m training.train_final_model --leave-out S17 --tag loso_s17
    python -m training.train_final_model --all-subjects --tag deployment_final
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from config import config
from data.loader import load_wesad_subject
from data.windowing import (create_windows, balance_by_undersampling,
                             labels_to_binary, filter_valid_labels)
from data.normalization import compute_fixed_stats, save_stats, apply_normalization
from models.temporal_cnn import TemporalCNN


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        return (alpha_t * ((1 - p_t) ** self.gamma) * ce_loss).mean()


class WESADDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_windows(subject_id, mean, std, cfg):
    """Bir subject'i yukle, filtrele, SABIT istatistiklerle normalize et, pencerele."""
    X_raw, y_raw = load_wesad_subject(subject_id, cfg.DATA_PATH, cfg.TARGET_RATE)
    if X_raw is None:
        return None, None

    X_raw, y_raw = filter_valid_labels(X_raw, y_raw, cfg.VALID_LABELS)
    y_bin = labels_to_binary(y_raw) if cfg.BINARY else y_raw

    X_norm = apply_normalization(X_raw, mean, std)  # (T, C), sabit norm

    X_win, y_win = create_windows(X_norm, y_bin, cfg.WINDOW_SIZE, cfg.HOP_SIZE)
    if len(X_win) == 0:
        return None, None
    X_win = X_win.transpose(0, 2, 1)  # (N, C, T)
    return X_win, y_win


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / len(loader), f1_score(labels, preds, average='weighted', zero_division=0)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--leave-out', type=str,
                        help="Test icin disarida birakilacak subject (orn. S17)")
    group.add_argument('--all-subjects', action='store_true',
                        help="Tum subjectlerle deployment modeli egit")
    parser.add_argument('--tag', type=str, required=True,
                         help="Checkpoint/istatistik dosyalarina eklenecek isim")
    args = parser.parse_args()

    cfg = config
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if args.leave_out:
        pool = [s for s in cfg.ALL_SUBJECTS if s != args.leave_out]
        print(f"LOSO modu: test icin {args.leave_out} disarida birakildi.")
    else:
        pool = list(cfg.ALL_SUBJECTS)
        print("Deployment modu: tum subjectler kullaniliyor.")

    val_subj = pool[-1]
    train_subjs = pool[:-1]
    print(f"Train subjects: {train_subjs}")
    print(f"Val subject:    {val_subj}")

    # 1) SABIT normalizasyon istatistiklerini SADECE train subjectlerinden hesapla
    print("\n[i] Sabit normalizasyon istatistikleri hesaplaniyor (sadece train subjectleri)...")
    mean, std = compute_fixed_stats(train_subjs, cfg)
    stats_path = os.path.join(cfg.CHECKPOINT_DIR, f'norm_stats_{args.tag}.npz')
    save_stats(mean, std, stats_path)

    # 2) Pencereleri olustur (sabit norm ile)
    X_train_list, y_train_list = [], []
    for s in train_subjs:
        X_w, y_w = prepare_windows(s, mean, std, cfg)
        if X_w is None:
            continue
        X_bal, y_bal = balance_by_undersampling(X_w, y_w, seed=cfg.SEED)
        X_train_list.append(X_bal)
        y_train_list.append(y_bal)
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    X_val, y_val = prepare_windows(val_subj, mean, std, cfg)

    print(f"\nTrain: {len(y_train)} pencere (dengelenmis) | Val: {len(y_val)} pencere")

    train_loader = DataLoader(WESADDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WESADDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 3) Model egit
    model = TemporalCNN(num_channels=cfg.NUM_CHANNELS, num_classes=2,
                         filters=cfg.CONV_FILTERS, kernel_size=cfg.KERNEL_SIZE,
                         conv_dropout=cfg.CONV_DROPOUT, fc_dropout=cfg.FC_DROPOUT).to(cfg.DEVICE)
    criterion = FocalLoss(cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA) if cfg.USE_FOCAL_LOSS else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f'model_{args.tag}.pth')
    best_val_f1, patience_ctr = 0, 0

    print("\n[i] Egitim basliyor...")
    for epoch in range(cfg.NUM_EPOCHS):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d} | Train F1={tr_f1:.3f} | Val F1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"  [!] Early stopping @ epoch {epoch+1}")
                break

    print(f"\n[OK] En iyi model kaydedildi: {ckpt_path} (Val F1={best_val_f1:.4f})")
    print(f"[OK] Sabit norm istatistikleri: {stats_path}")
    print("\nBu iki dosyayi streaming emulasyonunda kullan:")
    print(f"  python run_streaming_emulation.py --model {ckpt_path} "
          f"--norm-stats {stats_path} --subject <TEST_SUBJECT>")


if __name__ == "__main__":
    main()
