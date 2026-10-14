"""
serving/mp_worker.py
=====================
`--processes N > 1` icin alt surec tarafi. Her alt surec modeli ve
aciklayiciyi KENDISI yukler ve isitir; ana surec yalnizca normalize
edilmis pencereyi gonderir, tahmin ve asama surelerini geri alir.

Ayri modulde olmasinin nedeni: alt surecler 'spawn' ile baslatilir (Pi'de
torch ile 'fork' kilitlenebilir, Windows'ta zaten tek secenek). spawn,
gorev fonksiyonlarini modul yoluyla bulur; __main__ icinde tanimli
fonksiyonlar `python -m serving.server` altinda guvenilir sekilde
bulunamazdi.
"""

import os
import time

import numpy as np

_CHILD = {}


def child_init(model_path, xai_method, threads, limit_blas, barrier):
    """Havuz baslatici: her alt surecte BIR KEZ calisir."""
    import torch
    if threads > 0:
        torch.set_num_threads(threads)

    from config import config as cfg
    from models.temporal_cnn import load_checkpoint
    from serving.server import build_explainer, warmup_explainer

    model, _, _ = load_checkpoint(model_path, torch.device('cpu'), fallback_cfg=cfg)
    explainer = build_explainer(xai_method, model, device='cpu')

    # threadpool_limits yalnizca O AN yuklu kutuphaneleri sinirlar; aciklayici
    # scikit-learn/SHAP'i yukledikten sonra cagrilmali.
    blas_limited = False
    if limit_blas and threads > 0:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=threads)
            blas_limited = True
        except ImportError:
            pass
    warmup_ms = warmup_explainer(explainer, cfg)
    _CHILD.update(model=model, explainer=explainer, warmup_ms=warmup_ms,
                  barrier=barrier, blas_limited=blas_limited)


def child_ready():
    """Aciliste N kez gonderilir. Bariyer, N gorevin N FARKLI surece
    dagilmasini zorlar: boylece tum surecler istemci baglanmadan once
    acilmis ve isinmis olur (ilk aciklamanin 10+ s'lik isinmasi
    olcume karismaz)."""
    _CHILD['barrier'].wait(timeout=900)
    ex = _CHILD['explainer']
    cfg = None
    if ex is not None:
        try:
            cfg = ex.config()
        except Exception:
            pass
    return {'pid': os.getpid(), 'warmup_ms': _CHILD['warmup_ms'],
            'blas_limited': _CHILD['blas_limited'], 'xai_config': cfg}


def child_compute(norm_window, xai_threshold):
    """Cikarim + (gerekirse) aciklama. Asama adlari tek surecli yolla ayni."""
    import torch
    from serving.server import should_explain

    model, explainer = _CHILD['model'], _CHILD['explainer']
    stages = {}

    t0 = time.perf_counter()
    x = torch.FloatTensor(norm_window.transpose(1, 0)).unsqueeze(0)
    stages['tensor_prep'] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    stages['inference'] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred = int(probs.argmax())
    conf = float(probs.max())
    stages['postprocessing'] = (time.perf_counter() - t0) * 1000.0

    explained = False
    if explainer is not None and should_explain(conf, xai_threshold):
        t0 = time.perf_counter()
        explainer.explain(np.ascontiguousarray(norm_window.transpose(1, 0),
                                               dtype=np.float32))
        stages['xai'] = (time.perf_counter() - t0) * 1000.0
        explained = True

    return pred, conf, stages, explained, os.getpid()
