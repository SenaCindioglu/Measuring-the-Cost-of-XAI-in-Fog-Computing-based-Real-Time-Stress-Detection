"""
models/temporal_cnn.py
=======================
Offline pipeline ile birebir ayni mimari (Conv1d bloklari -> GAP -> FC).
Blok sayisi `filters` listesinin uzunlugu kadardir; secili model depth=2
oldugu icin varsayilan [32, 128]'dir.

Bu dosya kasitli olarak diger modullerden bagimsiz (sadece torch'a bagli)
tutuldu ki checkpoint yukleme her yerden sorunsuz calissin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    def __init__(self, num_channels=14, num_classes=2,
                 filters=None, kernel_size=9,
                 conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        if filters is None:
            filters = [32, 128]  # secili model: depth=2

        layers = []
        in_ch = num_channels
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
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


def load_checkpoint(path, device='cpu', fallback_cfg=None):
    """
    Checkpoint'ten TemporalCNN yukler.

    Iki formati da destekler:
      1) Zengin format (secili model):  {'model_state_dict': ..., 'filters': [...],
         'kernel_size': 9, 'num_channels': 14, 'conv_dropout': ..., 'fc_dropout': ...,
         'meta': {...}}  -> mimari DOGRUDAN checkpoint'ten okunur, config'e bakilmaz.
      2) Duz state_dict (eski training/train_final_model.py ciktisi)
         -> mimari fallback_cfg'den (config.py) alinir.

    Returns
    -------
    model : eval moduna alinmis, device'a tasinmis TemporalCNN
    arch  : mimari parametrelerinin dict'i
    meta  : checkpoint icindeki meta bilgisi (yoksa {})
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        arch = {
            'num_channels': ckpt.get('num_channels', 14),
            'filters': list(ckpt.get('filters', [32, 128])),
            'kernel_size': ckpt.get('kernel_size', 9),
            'conv_dropout': ckpt.get('conv_dropout', 0.2),
            'fc_dropout': ckpt.get('fc_dropout', 0.5),
        }
        meta = ckpt.get('meta', {}) or {}
    else:
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        if fallback_cfg is None:
            raise ValueError(
                "Duz state_dict checkpoint'i icin mimari bilgisi gerekli "
                "(fallback_cfg=config gecir)."
            )
        arch = {
            'num_channels': fallback_cfg.NUM_CHANNELS,
            'filters': list(fallback_cfg.CONV_FILTERS),
            'kernel_size': fallback_cfg.KERNEL_SIZE,
            'conv_dropout': fallback_cfg.CONV_DROPOUT,
            'fc_dropout': fallback_cfg.FC_DROPOUT,
        }
        meta = {}

    model = TemporalCNN(num_classes=2, **arch)
    model.load_state_dict(state_dict)   # strict=True: sessiz mimari uyusmazligi olmaz
    model.to(device)
    model.eval()

    arch['depth'] = len(arch['filters'])
    arch['num_params'] = sum(p.numel() for p in model.parameters())
    return model, arch, meta
