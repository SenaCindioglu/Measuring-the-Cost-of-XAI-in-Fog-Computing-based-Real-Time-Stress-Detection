"""
xai/lime_.py
=============
Bolge tabanli LIME: ham sinyal penceresi uzerinde calisir.

NEDEN HAZIR `lime` PAKETI KULLANILMIYOR. Paketin tabular/image/text
aciklayicilari var; bizim girdimiz (14, 640) = 8 960 degerlik bir zaman
serisi. Her ornegi ayri degisken saymak hem cok pahali hem anlamsiz (tek bir
ornegin katkisi yorumlanamaz). Goruntudeki "super piksel"in buradaki
karsiligi ZAMAN DILIMI x KANAL bolgeleridir: 20 dilim x 14 kanal = 280 bolge.
Bir bolge "kapatildiginda" o kanalin o dilimi kanalin KENDI ortalamasiyla
degistirilir; sifirlamak fizyolojik olarak anlamsiz olurdu (normalize edilmis
sinyalde sifir, o kanalin ortalama seviyesi degil, kaydin ortalamasidir).

Hesap taramadaki lime_explain() ile birebir ayni: ayni bolge izgarasi, ayni
maske ornekleme (sabit seed), ayni ussel kernel agirligi, ayni Ridge.
"""

import numpy as np
from sklearn.linear_model import Ridge

from xai.base import Explainer, predict_prob_batch


class LimeExplainer(Explainer):
    name = 'lime'

    def __init__(self, model, num_samples=500, num_segments=20, kernel_width=0.5,
                 num_channels=14, target_class=1, device='cpu', seed=0, chunk=256):
        super().__init__(model, num_channels, target_class, device)
        self.num_samples = int(num_samples)
        self.num_segments = int(num_segments)
        self.kernel_width = float(kernel_width)
        self.seed = int(seed)
        self.chunk = int(chunk)
        self._grid_cache = {}

    def _regions(self, C, T):
        """Bolge sinirlari ve (C, T) izgarasi. Pencere boyu sabit oldugu icin
        bir kez hesaplanip saklanir; her cagrida yeniden kurmak olculen
        gecikmeye modelin isi olmayan bir maliyet eklerdi."""
        key = (C, T, self.num_segments)
        if key in self._grid_cache:
            return self._grid_cache[key]
        bounds = np.linspace(0, T, self.num_segments + 1).astype(int)
        regions = [(c, bounds[s], bounds[s + 1])
                   for s in range(self.num_segments) for c in range(C)]
        grid = np.empty((C, T), dtype=np.int32)
        for idx, (c, t0, t1) in enumerate(regions):
            grid[c, t0:t1] = idx
        self._grid_cache[key] = (regions, grid)
        return regions, grid

    def _explain(self, window):
        C, T = window.shape
        regions, region_grid = self._regions(C, T)
        num_regions = len(regions)

        # Bolge kapatildiginda konulacak deger: o kanalin kendi ortalamasi
        channel_baseline = window.mean(axis=1)
        baseline_full = np.repeat(channel_baseline[:, None], T, axis=1)

        rng = np.random.RandomState(self.seed)
        masks = rng.randint(0, 2, size=(self.num_samples, num_regions))
        masks[0, :] = 1                      # orijinal pencere de orneklerde

        # Tum perturbasyonlar tek vektorel islemle: (N, C, T)
        keep = masks[:, region_grid].astype(bool)
        perturbed = np.where(keep, window[None, :, :],
                             baseline_full[None, :, :]).astype(np.float32)

        probs = predict_prob_batch(self.model, perturbed, self.target_class,
                                   self.device, self.chunk)

        # LIME agirliklandirmasi: kapali bolge oranina gore uzaklik + ussel kernel
        distances = 1.0 - masks.mean(axis=1)
        weights = np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))

        reg = Ridge(alpha=1.0)
        reg.fit(masks, probs, sample_weight=weights)

        attribution = np.zeros((C, T), dtype=np.float32)
        for idx, (c, t0, t1) in enumerate(regions):
            attribution[c, t0:t1] = reg.coef_[idx]
        return attribution

    def config(self):
        return {'method': self.name, 'num_samples': self.num_samples,
                'num_segments': self.num_segments,
                'kernel_width': self.kernel_width, 'seed': self.seed}


def from_artifacts(model, record, device='cpu', **over):
    """Kayitli yapilandirmadan (xai_config.json) aciklayici kurar."""
    p = dict(record['lime'])
    p.update(over)
    return LimeExplainer(model, num_channels=record['window']['num_channels'],
                         target_class=record.get('target_class', 1),
                         device=device, **p)
