"""
models/temporal_cnn.py
=======================
Offline pipeline ile birebir ayni mimari (Conv1d x3 -> GAP -> FC).
Bu dosya kasitli olarak diger modullerden bagimsiz (sadece torch'a bagli)
tutuldu ki checkpoint yukleme her yerden sorunsuz calissin.
"""

import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    def __init__(self, num_channels=14, num_classes=2,
                 filters=None, kernel_size=9,
                 conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128]

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
