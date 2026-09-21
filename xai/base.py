"""
xai/base.py
============
Iki aciklayicinin (LIME, SHAP) paylastigi altyapi.

TASARIM: aciklayicilar SUNUCUNUN ICINDE, tek pencere uzerinde calisir.
Taramadaki (wesad_xai_ofat_sweep_colab.py) hesabin birebir aynisini yapmak
zorundalar, yoksa makalede raporlanan kalite sayilari ile sunucuda uretilen
aciklamalar farkli seylere ait olur.

PENCERE YONU. Iki farkli yon dolasiyor ve karistirmak sessiz hataya yol acar:
    (T, C) = (640, 14)  -- sunucunun buffer'i ve normalizasyon bu yonde
    (C, T) = (14, 640)  -- modelin bekledigi yon
Asagidaki as_model_layout() kanal sayisini bilerek hangi yonde oldugunu
anlar ve gerekirse cevirir. 14 ile 640 birbirine esit olmadigi icin bu
ayrim guvenli.
"""

import numpy as np
import torch


def as_model_layout(window, num_channels):
    """Pencereyi modelin bekledigi (C, T) yonune getirir."""
    w = np.asarray(window)
    if w.ndim != 2:
        raise ValueError("Pencere iki boyutlu olmali, gelen: " + str(w.shape))
    if w.shape[0] == num_channels:
        return np.ascontiguousarray(w, dtype=np.float32)
    if w.shape[1] == num_channels:
        return np.ascontiguousarray(w.T, dtype=np.float32)
    raise ValueError(
        "Pencerenin hicbir ekseni kanal sayisina (%d) esit degil: %s"
        % (num_channels, w.shape))


@torch.no_grad()
def predict_prob_batch(model, windows, target_class, device='cpu', chunk=256):
    """windows: (N, C, T) -> (N,) hedef sinif olasiliklari.

    Parcali gonderilir: LIME tek pencere icin yuzlerce perturbasyon uretiyor.
    model.eval() oldugu icin BatchNorm kosu istatistiklerini kullanir, yani
    parcalama sonucu DEGISTIRMEZ.
    """
    out = []
    for i in range(0, len(windows), chunk):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + chunk])).float().to(device)
        probs = torch.softmax(model(x), dim=1)[:, target_class]
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class Explainer:
    """Ortak arayuz: explain(window) -> atif haritasi (C, T).

    Alt siniflar _explain() yazar; yon cevrimi ve dogrulama burada yapilir.
    """

    name = 'base'

    def __init__(self, model, num_channels=14, target_class=1, device='cpu'):
        self.model = model
        self.num_channels = num_channels
        self.target_class = target_class
        self.device = device

    def explain(self, window):
        w = as_model_layout(window, self.num_channels)
        attribution = self._explain(w)
        if attribution.shape != w.shape:
            raise ValueError(
                "%s: atif sekli pencereyle uyusmuyor (%s != %s)"
                % (self.name, attribution.shape, w.shape))
        return attribution

    def _explain(self, window):
        raise NotImplementedError

    def config(self):
        """Raporda ve JSON'da yazilacak yapilandirma ozeti."""
        return {'method': self.name}
