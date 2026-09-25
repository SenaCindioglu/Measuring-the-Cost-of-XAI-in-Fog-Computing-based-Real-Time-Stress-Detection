"""
xai/quality.py
===============
FAZ C3: aciklama kalitesi olculeri. LIME ve SHAP icin AYNI tanim, cunku
iki yontem ancak ayni olcutle karsilastirilabilir.

UC OLCU
-------
faithfulness  Aciklama gercekten modelin kullandigi yeri mi gosteriyor?
              Iki egri uzerinden:
                deletion  -- en onemli bolgelerden baslayarak kapat;
                             olasilik ne kadar hizli duserse aciklama o
                             kadar sadik (DUSUK AUC iyi)
                insertion -- taban cizgiden baslayip en onemliyi ekle;
                             olasilik ne kadar hizli yukselirse o kadar
                             sadik (YUKSEK AUC iyi)
              faithfulness = insertion_auc - deletion_auc   (yuksek iyi)

monotonicity  insertion egrisinde azalmayan adimlarin orani. Onem sirasi
              dogruysa her ekleme olasiligi artirmali.

sensitivity   Girdiye kucuk gurultu eklenince atif ne kadar degisiyor
              (normalize L2). DUSUK = kararli aciklama.

TABAN CIZGI SECIMI
------------------
Kapatilan bolge sifirlanmaz, o KANALIN KENDI ORTALAMASIYLA degistirilir.
Fizyolojik sinyalde sifir anlamli bir deger degil (sicaklik 0 C, EDA 0 uS
gibi) ve modele egitimde hic gormedigi bir girdi verir; kanal ortalamasi
"bu kanal hakkinda bilgi yok" durumunun cok daha savunulabilir karsiligi.
Ayni taban cizgi aciklayicilarin kendi perturbasyonunda da kullaniliyor,
yani olcu ile yontem tutarli.

PARTILI HESAP
-------------
Kontrol noktalarinin hepsi TEK ileri gecis partisinde gonderilir. Tek
ornekli cagrilarla (pencere basina 42 cagri) ayni sonucu verir ama
CPU'da kat kat yavastir.
"""

import numpy as np

from xai.base import as_model_layout, predict_prob_batch


def _auc(curve):
    """Egrinin [0,1] araligina normalize edilmis alani.

    np.trapz NumPy 2.0'da np.trapezoid oldu; ikisi de desteklenir.
    n noktali egri n-1 aralik kapsar, bolen bu yuzden len(curve)-1.
    """
    trap = getattr(np, 'trapezoid', None) or np.trapz
    n = len(curve)
    return float(trap(curve)) / max(n - 1, 1)


def channel_baseline(window):
    """(C, T) pencere icin kanal bazli taban cizgi, yine (C, T)."""
    return np.repeat(window.mean(axis=1)[:, None], window.shape[1], axis=1)


def faithfulness(model, window, attribution, target_class=1, device='cpu',
                 num_steps=20, chunk=256):
    """
    Returns: (faithfulness, monotonicity, deletion_auc, insertion_auc)
    """
    w = as_model_layout(window, window.shape[0])
    a = np.asarray(attribution, dtype=np.float64)
    if a.shape != w.shape:
        raise ValueError("atif sekli pencereyle uyusmuyor: %s != %s"
                         % (a.shape, w.shape))

    C, T = w.shape
    base = channel_baseline(w).astype(np.float32)
    flat_w = np.asarray(w, dtype=np.float32).reshape(-1)
    flat_b = base.reshape(-1)
    n_feat = flat_w.size

    order = np.argsort(-a.reshape(-1))        # en onemli -> en onemsiz
    rank = np.empty(n_feat, dtype=np.int64)
    rank[order] = np.arange(n_feat)

    step = max(1, n_feat // num_steps)
    ks = list(range(0, n_feat + 1, step))
    if ks[-1] != n_feat:
        ks.append(n_feat)
    sel = rank[None, :] < np.asarray(ks)[:, None]      # (n_ck, n_feat)

    del_batch = np.where(sel, flat_b[None, :], flat_w[None, :])
    ins_batch = np.where(sel, flat_w[None, :], flat_b[None, :])
    both = np.concatenate([del_batch, ins_batch]).reshape(-1, C, T).astype(np.float32)

    probs = predict_prob_batch(model, both, target_class, device, chunk)
    n_ck = len(ks)
    del_curve, ins_curve = probs[:n_ck], probs[n_ck:]

    d_auc, i_auc = _auc(del_curve), _auc(ins_curve)
    mono = float(np.mean(np.diff(ins_curve) >= -1e-6))
    return i_auc - d_auc, mono, d_auc, i_auc


def sensitivity(explainer, window, attribution, noise_scale=0.05,
                n_trials=3, seed=0):
    """Girdi gurultusune karsi atifin kararliligi (dusuk = kararli).

    Aciklayicinin KENDI rastgeleligi sabit tutulur (aciklayicilar seed
    tasiyor), yoksa olculen fark yontemin ic gurultusunu de sayardi.
    Taramada bu olculdu: SHAP'in seed'i sabitlenince sensitivity 0.139'dan
    0.075'e dustu, yani farkin neredeyse yarisi kendi gurultusuymus.
    """
    w = as_model_layout(window, window.shape[0])
    a = np.asarray(attribution, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(a) + 1e-8

    rng = np.random.RandomState(seed)
    std = noise_scale * float(np.std(w))
    diffs = []
    for _ in range(n_trials):
        noisy = (w + rng.normal(0, std, size=w.shape)).astype(np.float32)
        diffs.append(np.linalg.norm(
            np.asarray(explainer.explain(noisy), dtype=np.float64).reshape(-1) - a))
    return float(np.mean(diffs) / norm)


def evaluate(explainer, windows, model=None, target_class=1, device='cpu',
             n_sensitivity=10, n_trials=3, chunk=256, verbose=False):
    """Bir aciklayiciyi pencere kumesi uzerinde olcer.

    Returns: dict -- faithfulness/monotonicity/sensitivity ortalamalari,
    standart hatalari ve pencere sayilari.
    """
    model = model if model is not None else explainer.model
    faith, mono, dels, inss, attrs = [], [], [], [], []

    for w in windows:
        a = explainer.explain(w)
        f, m, d, i = faithfulness(model, w, a, target_class, device, chunk=chunk)
        faith.append(f); mono.append(m); dels.append(d); inss.append(i)
        attrs.append(a)

    sens = []
    for i in range(min(n_sensitivity, len(windows))):
        sens.append(sensitivity(explainer, windows[i], attrs[i], n_trials=n_trials))

    def stat(v):
        v = np.asarray(v, dtype=np.float64)
        if len(v) == 0:
            return {'mean': None, 'sem': None, 'n': 0}
        # Standart hata: tarama ilk halinde YOKTU ve kazananlar 0.003
        # mertebesindeki farklarla seciliyordu. Farkin gurultu olup
        # olmadigi ancak bununla soylenebilir.
        return {'mean': float(v.mean()),
                'sem': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                'n': int(len(v))}

    out = {
        'config': explainer.config(),
        'faithfulness': stat(faith),
        'monotonicity': stat(mono),
        'sensitivity': stat(sens),
        'deletion_auc': stat(dels),
        'insertion_auc': stat(inss),
    }
    if verbose:
        f, s = out['faithfulness'], out['sensitivity']
        print(f"    faithfulness {f['mean']:.4f} +/- {f['sem']:.4f}  "
              f"monotonicity {out['monotonicity']['mean']:.4f}  "
              f"sensitivity {s['mean']:.4f}")
    return out
