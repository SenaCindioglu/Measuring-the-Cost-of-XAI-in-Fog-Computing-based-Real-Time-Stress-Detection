"""
tools/calibration.py
=====================
FAZ D1: modelin guven degeri (confidence) ne kadar guvenilir?

NEDEN GEREKLI
-------------
Secici XAI'nin tamami tek bir varsayima dayaniyor: "model emin degilse
acikla". Bu varsayimin isleyebilmesi icin `confidence` degerinin GERCEK
dogruluk olasiligini temsil etmesi gerekir. Sinir aglari genelde fazla
ozguvenlidir; %90 dediginde %70 hakli olabilir.

Bu bozuksa iki sekilde patlar:
  - Model her seye 0.95 diyorsa, esigi 0.80 koydugunda HICBIR pencere
    aciklanmaz; mekanizma hic calismaz.
  - Model yanildigi pencerelerde de yuksek guven veriyorsa, aciklama tam
    da gerektigi yerde tetiklenmez.

O yuzden esik taramasindan (D2) ONCE bu olculur.

SUNUCUNUN GIRDI UZAYINDA OLCULUR
--------------------------------
Kalibrasyon, modele giren verinin dagilimina baglidir. Sunucu ham akisi
64 Hz'e indirip NEDENSEL (running) z-score uyguluyor. Kalibrasyonu baska
bir uzayda (orn. denek bazli sizintili normalizasyon) olcmek, dagitima
tasinmayan bir sayi uretir. Bu yuzden asagidaki dongu sunucunun worker
dongusunu birebir taklit eder: MultiRateBuffer -> RunningNormalizer ->
model. Karsilastirilabilirligin kanitı: ayni denekte uretilen tahminler
fog kosularindakiyle ayni olmali.

URETILEN OLCULER
----------------
  ECE   Expected Calibration Error: guven kutulara bolunur, her kutuda
        |ortalama guven - gercek dogruluk| farki, kutu agirligiyla.
  MCE   ayni farkin en buyugu (en kotu kutu).
  Brier ortalama kare hata; kalibrasyon ve ayirt etme gucunu birlikte olcer.
  reliability  kutu kutu dokum, diyagram icin.
  esik dagilimi  her aday esigin ALTINDA kalan pencere orani -- D2'nin
        dogrudan girdisi. Bu sutun bos cikiyorsa secici XAI zaten olanaksiz.

SICAKLIK OLCEKLEME (temperature scaling)
----------------------------------------
Tek parametreli duzeltme: logit'ler T'ye bolunup yeniden softmax alinir.
T > 1 guveni yumusatir (fazla ozguven duzelir), T < 1 keskinlestirir.
T, BIR denekte ogrenilir ve DIGER denekte uygulanir (capraz denek); ayni
denekte ogrenip ayni denekte olcmek kalibrasyonu yapay olarak iyi
gosterirdi.

Kullanim:
    python -m tools.calibration
    python -m tools.calibration --subjects S11 S14 --bins 15 --tag d1
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config import config
from data.loader import load_wesad_subject_native, NATIVE_STREAMS, LABEL_RATE
from data.normalization import apply_normalization, RunningNormalizer
from models.temporal_cnn import load_checkpoint
from streaming.buffer import MultiRateBuffer


# ======================================================================
# 1) Sunucunun girdi uzayinda logit toplama
# ======================================================================

def collect_logits(subject, cfg, model, tick_ms=None, warmup_windows=None,
                   verbose=True):
    """Bir denegi replay eder, her pencere icin logit ve etiket dondurur.

    Sunucunun worker dongusuyle AYNI adimlar: ham akis -> MultiRateBuffer
    (64 Hz'e indirme) -> nedensel z-score -> model. Tek fark, burada
    softmax degil LOGIT saklanir; sicaklik olceklemesi logit uzerinde
    calisir ve guvenden geri hesaplamak gereksiz hata katardi.
    """
    tick_ms = tick_ms or cfg.DEFAULT_TICK_MS
    warmup_windows = (cfg.DEFAULT_WARMUP_WINDOWS if warmup_windows is None
                      else warmup_windows)

    streams, raw_labels = load_wesad_subject_native(subject, cfg.DATA_PATH)
    if streams is None:
        return None

    buf = MultiRateBuffer(NATIVE_STREAMS, cfg.WINDOW_DURATION,
                          cfg.HOP_DURATION, cfg.WINDOW_SIZE)
    runner = RunningNormalizer(cfg.NUM_CHANNELS)

    tick_s = tick_ms / 1000.0
    per_tick = {name: int(round(rate * tick_s)) for name, rate, _ in NATIVE_STREAMS}
    cursor = {name: 0 for name, _, _ in NATIVE_STREAMS}
    n_ticks = min(len(streams[name]) // per_tick[name]
                  for name, _, _ in NATIVE_STREAMS if per_tick[name] > 0)

    logits_all, labels_all = [], []
    n_windows = n_invalid = 0

    for _ in range(n_ticks):
        chunks = {}
        for name, _, _ in NATIVE_STREAMS:
            k = per_tick[name]
            c = cursor[name]
            chunks[name] = streams[name][c:c + k]
            cursor[name] = c + k

        if not buf.push(chunks, tick_s):
            continue

        raw_window = buf.get_window()
        if raw_window is None:
            continue

        # Yalnizca yeni gelen kismi istatistige kat (cift saymamak icin) --
        # sunucudaki davranisin aynisi.
        new_part = raw_window if n_windows == 0 else raw_window[-cfg.HOP_SIZE:]
        runner.update(new_part)
        norm_window = apply_normalization(raw_window, runner.mean, runner.std)

        with torch.no_grad():
            x = torch.FloatTensor(norm_window.transpose(1, 0)).unsqueeze(0)
            logits = model(x).numpy()[0]

        n_windows += 1

        # Pencerenin kapsadigi araligin cogunluk etiketi.
        end_i = int(round(buf.elapsed_s * LABEL_RATE))
        start_i = int(round((buf.elapsed_s - cfg.WINDOW_DURATION) * LABEL_RATE))
        span = raw_labels[max(0, start_i):min(end_i, len(raw_labels))]
        if len(span) == 0:
            continue
        majority = int(np.bincount(span.astype(int)).argmax())
        if majority not in cfg.VALID_LABELS:
            # Gecersiz etiketli pencere: normalizasyon istatistigi ZATEN
            # guncellendi (sunucu etiketi gormez, atlayamaz), sadece
            # kalibrasyon hesabina girmiyor.
            n_invalid += 1
            continue
        if n_windows <= warmup_windows:
            continue          # isinmada running istatistik henuz oturmamis

        logits_all.append(logits)
        labels_all.append(1 if majority == 2 else 0)

    if verbose:
        print(f"   {subject}: {n_windows} pencere islendi, "
              f"{len(labels_all)} tanesi kalibrasyona girdi "
              f"({n_invalid} gecersiz etiket, {min(warmup_windows, n_windows)} isinma haric)")

    return {'subject': subject,
            'logits': np.asarray(logits_all, dtype=np.float64),
            'labels': np.asarray(labels_all, dtype=np.int64),
            'n_windows_total': n_windows,
            'n_invalid': n_invalid}


# ======================================================================
# 2) Kalibrasyon olculeri
# ======================================================================

def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)      # tasma korumasi
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confidence_and_pred(probs):
    return probs.max(axis=1), probs.argmax(axis=1)


def expected_calibration_error(conf, correct, n_bins=15):
    """ECE, MCE ve kutu dokumu.

    Esit GENISLIKTE kutular kullanilir (esit sayida degil): literaturdeki
    standart tanim bu ve baska calismalarla karsilastirilabilir olmasi
    icin degistirilmedi. Bos kutular atlanir.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)

    ece = 0.0
    mce = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Son kutu ust sinira dahil, digerleri degil
        m = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= 1.0)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({'lo': lo, 'hi': hi, 'n': 0,
                         'avg_confidence': None, 'accuracy': None, 'gap': None})
            continue
        avg_conf = float(conf[m].mean())
        acc = float(correct[m].mean())
        gap = abs(avg_conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({'lo': float(lo), 'hi': float(hi), 'n': cnt,
                     'avg_confidence': avg_conf, 'accuracy': acc,
                     'gap': float(avg_conf - acc)})
    return float(ece), float(mce), bins


def brier_score(probs, labels):
    """Ikili Brier: pozitif sinif olasiligi ile 0/1 etiket arasinda kare hata."""
    p1 = np.asarray(probs)[:, 1]
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p1 - y) ** 2))


def nll(logits, labels, temperature=1.0):
    p = softmax(logits, temperature)
    idx = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(p[idx, labels], 1e-12, 1.0))))


def fit_temperature(logits, labels, lo=0.05, hi=10.0, rounds=4, points=60):
    """NLL'i en kucuk yapan T'yi kaba-ince izgara taramasiyla bulur.

    Izgara secildi cunku tek parametreli ve tek tepeli bir problem;
    scipy'ye bagimlilik eklemeden deterministik sonuc verir (ayni veride
    ayni T, kosudan kosuya oynamaz).
    """
    best_t, best_nll = 1.0, nll(logits, labels, 1.0)
    for _ in range(rounds):
        grid = np.linspace(lo, hi, points)
        vals = [nll(logits, labels, t) for t in grid]
        i = int(np.argmin(vals))
        if vals[i] < best_nll:
            best_t, best_nll = float(grid[i]), float(vals[i])
        step = grid[1] - grid[0]
        lo, hi = max(0.01, grid[i] - step), grid[i] + step
    return best_t, best_nll


def threshold_table(conf, correct, thresholds):
    """Her aday esik icin: kac pencere aciklanir, hatalarin kaci yakalanir.

    D2'nin dogrudan girdisi. Iki sutun kritik:
      explain_rate  -> maliyet (kac pencerede aciklama uretilecek)
      error_recall  -> fayda (yanlis siniflanan pencerelerin kaci yakalandi)
    Iyi bir esik dusuk explain_rate ile yuksek error_recall verir.
    """
    conf = np.asarray(conf)
    correct = np.asarray(correct, dtype=bool)
    n = len(conf)
    n_err = int((~correct).sum())
    out = []
    for t in thresholds:
        trig = conf < t
        caught = int((trig & ~correct).sum())
        out.append({
            'threshold': float(t),
            'explain_rate': float(trig.mean()),
            'n_explained': int(trig.sum()),
            'error_recall': (caught / n_err) if n_err else None,
            'n_errors_caught': caught,
            'n_errors_total': n_err,
            # Aciklanan pencerelerin kaci gercekten hataliydi: bosa harcanan
            # aciklama oranini bu gosterir.
            'precision_of_trigger': (caught / int(trig.sum())) if trig.sum() else None,
        })
    return out


def evaluate(logits, labels, n_bins=15, temperature=1.0, thresholds=()):
    probs = softmax(logits, temperature)
    conf, pred = confidence_and_pred(probs)
    correct = (pred == np.asarray(labels))
    ece, mce, bins = expected_calibration_error(conf, correct, n_bins)
    return {
        'temperature': float(temperature),
        'n': int(len(labels)),
        'accuracy': float(correct.mean()),
        'avg_confidence': float(conf.mean()),
        # Pozitifse model fazla ozguvenli, negatifse fazla cekingen
        'overconfidence': float(conf.mean() - correct.mean()),
        'ece': ece,
        'mce': mce,
        'brier': brier_score(probs, labels),
        'nll': nll(logits, labels, temperature),
        'reliability_bins': bins,
        'confidence_percentiles': {
            str(p): float(np.percentile(conf, p)) for p in (1, 5, 25, 50, 75, 95, 99)
        },
        'thresholds': threshold_table(conf, correct, thresholds) if thresholds else [],
    }


# ======================================================================
# 3) Rapor
# ======================================================================

def print_report(name, res):
    print(f"\n  [{name}]  n={res['n']}  T={res['temperature']:.3f}")
    print(f"     dogruluk        {res['accuracy']:.4f}")
    print(f"     ortalama guven  {res['avg_confidence']:.4f}"
          f"   (fazla ozguven {res['overconfidence']:+.4f})")
    print(f"     ECE {res['ece']:.4f}   MCE {res['mce']:.4f}   "
          f"Brier {res['brier']:.4f}   NLL {res['nll']:.4f}")
    p = res['confidence_percentiles']
    print(f"     guven dagilimi  p5={p['5']:.3f}  p25={p['25']:.3f}  "
          f"p50={p['50']:.3f}  p75={p['75']:.3f}  p95={p['95']:.3f}")
    if res['thresholds']:
        print(f"     {'esik':>6} {'aciklanan':>11} {'hata yakalama':>15} {'isabet':>9}")
        for t in res['thresholds']:
            er = f"{t['error_recall']:.3f}" if t['error_recall'] is not None else "-"
            pr = f"{t['precision_of_trigger']:.3f}" if t['precision_of_trigger'] is not None else "-"
            print(f"     {t['threshold']:6.2f} {t['explain_rate']:10.1%} "
                  f"{er:>15} {pr:>9}")


def main():
    cfg = config
    ap = argparse.ArgumentParser(description="D1: guven kalibrasyonu")
    ap.add_argument('--subjects', nargs='+', default=list(cfg.VAL_SUBJECTS),
                    help="Kalibrasyon olculecek denekler (varsayilan: tutulmus denekler)")
    ap.add_argument('--model', default=cfg.MODEL_PATH)
    ap.add_argument('--tick-ms', type=int, default=cfg.DEFAULT_TICK_MS)
    ap.add_argument('--warmup-windows', type=int, default=cfg.DEFAULT_WARMUP_WINDOWS)
    ap.add_argument('--bins', type=int, default=15, help="Reliability kutu sayisi")
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.60, 0.70, 0.80, 0.90],
                    help="D2 icin aday esikler")
    ap.add_argument('--threads', type=int, default=cfg.DEFAULT_THREADS)
    ap.add_argument('--tag', default='d1')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    model, arch, meta = load_checkpoint(args.model, torch.device('cpu'), fallback_cfg=cfg)
    print("=" * 70)
    print("D1 - GUVEN KALIBRASYONU")
    print("=" * 70)
    print(f"Model     : {os.path.basename(args.model)} ({arch['num_params']:,} parametre)")
    print(f"Denekler  : {args.subjects}")
    print(f"Girdi uzayi: native yeniden ornekleme + NEDENSEL z-score "
          f"(sunucunun gordugu uzay)")
    print("=" * 70)

    t0 = time.perf_counter()
    data = {}
    for s in args.subjects:
        d = collect_logits(s, cfg, model, args.tick_ms, args.warmup_windows)
        if d is None:
            print(f"   ! {s} yuklenemedi, atlaniyor")
            continue
        if len(d['labels']) == 0:
            print(f"   ! {s}: etiketli pencere yok, atlaniyor")
            continue
        data[s] = d
    if not data:
        raise SystemExit("Hicbir denek yuklenemedi.")
    print(f"\nToplama suresi {time.perf_counter() - t0:.1f} s")

    out = {'config': {'model': os.path.basename(args.model),
                      'subjects': list(data),
                      'tick_ms': args.tick_ms,
                      'warmup_windows': args.warmup_windows,
                      'bins': args.bins,
                      'thresholds': args.thresholds,
                      'input_space': 'native + causal z-score (sunucuyla ayni)'},
           'per_subject': {}, 'cross_subject': {}}

    # --- Denek bazli, duzeltmesiz ---
    print("\n" + "-" * 70)
    print("DUZELTMESIZ (T = 1)")
    print("-" * 70)
    for s, d in data.items():
        res = evaluate(d['logits'], d['labels'], args.bins, 1.0, args.thresholds)
        res['n_invalid_windows'] = d['n_invalid']
        out['per_subject'][s] = {'uncalibrated': res}
        print_report(s, res)

    # --- Capraz denek sicaklik olcekleme ---
    # T bir denekte ogrenilir, DIGERINDE uygulanir. Ayni denekte ogrenip
    # ayni denekte olcmek kalibrasyonu yapay olarak iyi gosterir.
    if len(data) >= 2:
        print("\n" + "-" * 70)
        print("CAPRAZ DENEK SICAKLIK OLCEKLEME")
        print("-" * 70)
        subs = list(data)
        for i, target in enumerate(subs):
            source = subs[(i + 1) % len(subs)]
            T, src_nll = fit_temperature(data[source]['logits'], data[source]['labels'])
            res = evaluate(data[target]['logits'], data[target]['labels'],
                           args.bins, T, args.thresholds)
            before = out['per_subject'][target]['uncalibrated']
            out['per_subject'][target]['calibrated'] = res
            out['cross_subject'][target] = {
                'temperature_from': source, 'temperature': T,
                'ece_before': before['ece'], 'ece_after': res['ece'],
                'ece_reduction': before['ece'] - res['ece'],
            }
            print(f"\n  T = {T:.3f}  ({source}'de ogrenildi, {target}'e uygulandi)")
            print(f"     ECE {before['ece']:.4f} -> {res['ece']:.4f}"
                  f"   ({before['ece'] - res['ece']:+.4f})")
            print_report(f"{target} / T={T:.3f}", res)
    else:
        print("\n[!] Capraz denek sicaklik olceklemesi icin en az iki denek gerekli.")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, f'calibration_{args.tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[i] Kaydedildi: {path}")
    return out


if __name__ == "__main__":
    main()
