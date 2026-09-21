"""
xai/shap_.py
=============
SHAP GradientExplainer, kayitli arka plan kumesiyle.

ARKA PLAN NEDEN DOSYADAN OKUNUYOR. SHAP "bu deger su kadar katki yapti"
derken bir referansa gore konusur. O referans arka plan kumesi ve ciktinin
degeri dogrudan ona baglidir. Sunucuda her acilista yeniden kumeleme yapmak
(a) saniyeler alir, (b) taramada kullanilan kumeyle bit birebir ayni sonucu
vermez. Bu yuzden kume bir kez uretilip kaydedilir:
    python -m xai.artifacts --save
Arka plan TEST deneklerinden (S11, S14) degil, S2-S5'ten gelir ve sunucunun
gordugu girdi uzayinda (nedensel normalizasyon) uretilir.

SEED SABITLENIYOR. GradientExplainer her cagrida arka plan ile girdi arasinda
rastgele `nsamples` nokta ornekler. Seed sabit degilse ayni pencere iki kez
aciklandiginda farkli atif cikar; o zaman "girdiye duyarlilik" metrigi
yontemin kendi rastgeleligini de sayar. Taramada bu olculdu: seed
sabitlenince SHAP'in sensitivity degeri 0.139'dan 0.075'e dustu, yani farkin
neredeyse yarisi kendi gurultusuymus.
"""

import numpy as np
import torch

from xai.base import Explainer


def _require_shap():
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap kurulu degil. Kurulum: pip install shap\n"
            "(Sunucu tarafinda XAI kullanilmayacaksa gerekmez.)")
    return shap


class ShapExplainer(Explainer):
    name = 'shap'

    def __init__(self, model, background, nsamples=200, num_channels=14,
                 target_class=1, device='cpu', seed=0):
        super().__init__(model, num_channels, target_class, device)
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        bg = np.asarray(background, dtype=np.float32)
        if bg.ndim != 3 or bg.shape[1] != num_channels:
            raise ValueError("Arka plan (N, C, T) olmali, gelen: " + str(bg.shape))
        self.background = bg
        self._explainer = None            # tembel: shap importu ancak burada

    @property
    def explainer(self):
        if self._explainer is None:
            shap = _require_shap()
            bg = torch.from_numpy(self.background).float().to(self.device)
            self._explainer = shap.GradientExplainer(self.model, bg)
        return self._explainer

    def _explain(self, window):
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        sv = self.explainer.shap_values(x, nsamples=self.nsamples)

        # shap surumleri farkli sekil donduruyor; dordunu de acikca ele al,
        # tutmazsa sessizce devam etmek yerine hata ver.
        a = sv[self.target_class] if isinstance(sv, list) else sv
        if hasattr(a, 'detach'):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.ndim == 4:                               # (1, C, T, n_classes)
            a = a[0, ..., self.target_class]
        elif a.ndim == 3:
            if a.shape[0] == 1:                       # (1, C, T)
                a = a[0]
            elif a.shape[-1] != window.shape[1]:      # (C, T, n_classes)
                a = a[..., self.target_class]
        return np.asarray(a, dtype=np.float32)

    def config(self):
        return {'method': self.name, 'nsamples': self.nsamples,
                'background_size': int(len(self.background)), 'seed': self.seed}


def from_artifacts(model, record, device='cpu', cheap=False, **over):
    """Kayitli yapilandirmadan aciklayici kurar.

    cheap=True, xai_config.json'daki ikinci (ucuz) calisma noktasini kullanir:
    nsamples=50, kaliteden %1.4 verip 8.6 kat hizli. Faz D iki maliyet
    noktasinda kosuldugu icin ikisi de gerekiyor.
    """
    key = 'shap_cheap' if cheap else 'shap'
    p = dict(record[key])
    p.pop('background_size', None)        # gercek boyut dizinin kendisinden
    p.pop('background_method', None)      # yalnizca kayit amacli
    p.update(over)

    bg = record.get('shap_background_array')
    if bg is None:
        raise SystemExit(
            "Arka plan dizisi kayitta yok. load_artifacts(with_background=True) "
            "ile oku ya da once: python -m xai.artifacts --save")
    return ShapExplainer(model, bg, num_channels=record['window']['num_channels'],
                         target_class=record.get('target_class', 1),
                         device=device, **p)
