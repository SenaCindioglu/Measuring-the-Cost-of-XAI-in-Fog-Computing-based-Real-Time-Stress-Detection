"""
tools/selective_sweep.py
=========================
FAZ D2: secici XAI esik taramasi.

NE YAPAR
--------
D1'in esik tablosunu (hangi esikte kac pencere aciklanir, hatalarin kaci
yakalanir) 04_fog'un olculen maliyet sayilariyla birlestirir ve her
(aciklayici x esik x denek) noktasi icin uctan uca maliyeti verir.

YENI KOSU GEREKMEZ, cunku iki bilesenin ikisi de olculdu:
  - aciklama orani  -> D1 (results/08_kalibrasyon/calibration_d1.json)
  - aciklama bedeli -> D4'un iki uc durumu (results/04_fog/*.json):
      "hic aciklama"  = taban maliyet
      "her pencere"   = aciklamali maliyet

OLCULEN ve TURETILEN AYRIMI
---------------------------
Asagidaki sayilarin hangisi olculdu hangisi hesaplandi, JSON'da
`kind` alaniyla isaretlenir. Bu ayrim kritik: tablo "olculdu" diye
sunulursa yanlis olur, cunku ara esiklerin kendisi kosulmadi.

  olculdu   : esik 0.0 (taban) ve esik 1.0 (her pencere) satirlari
  turetildi : aradaki her esik

TURETME NASIL YAPILIYOR
-----------------------
Ortalama KESIN: iki bilesenli bir karisimin ortalamasi agirlikli
ortalamadir, dagilimin sekline bagli degildir.
    ortalama = (1 - r) * taban + r * aciklamali        (r = aciklama orani)

p95 icin dagilimin sekli gerekir ve elimizde yalnizca ozet var. Ama bu
ozel durumda iki bilesen HIC ORTUSMUYOR: taban ~3.5 ms, aciklamali
395-1422 ms. Yani karisim temiz bicimde ayrisiyor ve p95'in hangi
bileşende oldugu dogrudan r'den cikiyor:

    r <  0.05  ->  p95 TABAN bolgesinde; aciklama p95'i HIC ETKILEMIYOR
    r >= 0.05  ->  p95 aciklamali bolgede

Bu, tasarim acisindan en onemli sonuc: aciklama orani %5'in altindayken
en kotu durum gecikmesi aciklamadan hic etkilenmiyor.

Bolge icindeki kesin deger, bilinen ceyrekliklerden (medyan, p95, p99,
max) dogrusal ara degerle tahmin edilir; bu bir YAKLASIKTIR ve JSON'da
oyle isaretlenir. D4'un orta satiri (gercek esikle kosu) bu tahmini
dogrulamak icin yapilacak.

CPU ve bellek de dogrusal ara degerle tahmin edilir; ikisi de yaklasik.

Kullanim:
    python -m tools.selective_sweep
    python -m tools.selective_sweep --hop-s 5.0 --tag d2
"""

import argparse
import json
import os

from config import config


#: 04_fog dosya adi -> okunabilir aciklayici adi
EXPLAINERS = [
    ('shap_ucuz', 'fog_xai_shapucuz.json'),
    ('lime_ucuz', 'fog_xai_limeucuz.json'),
    ('lime',      'fog_xai_lime.json'),
    ('shap',      'fog_xai_shap.json'),
]
BASELINE_FILE = 'fog_xai_none.json'


def _load(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d.get('summary', d)


def _quantile_from_summary(stat, q):
    """Ozet istatistikten q ceyrekligini dogrusal ara degerle tahmin eder.

    Elimizde min, medyan, p95, p99, max var. Bunlar arasinda dogrusal
    ara deger alinir. YAKLASIKTIR -- gercek dagilim bu noktalar arasinda
    dogrusal olmak zorunda degil.
    """
    pts = [(0.0, stat['min_ms']), (0.50, stat['median_ms']),
           (0.95, stat['p95_ms']), (0.99, stat['p99_ms']), (1.0, stat['max_ms'])]
    pts = sorted(set(pts))
    if q <= pts[0][0]:
        return pts[0][1]
    if q >= pts[-1][0]:
        return pts[-1][1]
    for (q0, v0), (q1, v1) in zip(pts, pts[1:]):
        if q0 <= q <= q1:
            if q1 == q0:
                return v1
            w = (q - q0) / (q1 - q0)
            return v0 + w * (v1 - v0)
    return pts[-1][1]


def mixture_p95(base_stat, expl_stat, r, target_q=0.95):
    """Karisimin p95'i: hangi bolgede oldugunu ve degerini dondurur.

    Iki bilesen ortusmedigi icin (taban ~3.5 ms, aciklamali 400+ ms)
    karisimin ust r orani tamamen aciklamali bilesendir. Dolayisiyla:
      target_q < 1 - r  ->  taban bolgesinde
      aksi halde        ->  aciklamali bolgede
    """
    if r <= 0.0:
        return 'taban', _quantile_from_summary(base_stat, target_q)
    if r >= 1.0:
        return 'aciklamali', _quantile_from_summary(expl_stat, target_q)

    if target_q < (1.0 - r):
        # Tabanin kendi icindeki hangi ceyrekligine denk geliyor
        q_in = target_q / (1.0 - r)
        return 'taban', _quantile_from_summary(base_stat, q_in)
    q_in = (target_q - (1.0 - r)) / r
    return 'aciklamali', _quantile_from_summary(expl_stat, q_in)


def build(cfg, calib_path, fog_dir, hop_s, thresholds=None):
    calib = json.load(open(calib_path, encoding='utf-8'))
    base = _load(os.path.join(fog_dir, BASELINE_FILE))

    base_exec = base['metrics']['execution']
    base_resp = base['metrics']['response']
    base_res = ((base.get('server') or {}).get('resources') or {})
    base_cpu = (base_res.get('process_cpu_percent') or {}).get('mean')
    base_rss = (base_res.get('process_rss_mb') or {}).get('mean')

    out = {
        'kaynak': {
            'kalibrasyon': os.path.basename(calib_path),
            'maliyet': fog_dir,
            'taban': BASELINE_FILE,
            'hop_s': hop_s,
        },
        'aciklama': (
            "Esik 0.0 ve 1.0 satirlari OLCULDU; aradaki esikler D1 aciklama "
            "orani ile 04_fog maliyetinden TURETILDI. Ortalama kesin, p95 "
            "ve kaynak kullanimi yaklasik. D4'un orta satiri dogrulayacak."
        ),
        'taban': {
            'execution_mean_ms': base_exec['mean_ms'],
            'execution_p95_ms': base_exec['p95_ms'],
            'response_mean_ms': base_resp['mean_ms'],
            'response_p95_ms': base_resp['p95_ms'],
            'cpu_percent': base_cpu, 'rss_mb': base_rss,
        },
        'satirlar': [],
    }

    for name, fname in EXPLAINERS:
        path = os.path.join(fog_dir, fname)
        if not os.path.exists(path):
            continue
        always = _load(path)
        a_exec = always['metrics']['execution']
        a_resp = always['metrics']['response']
        a_res = ((always.get('server') or {}).get('resources') or {})
        a_cpu = (a_res.get('process_cpu_percent') or {}).get('mean')
        a_rss = (a_res.get('process_rss_mb') or {}).get('mean')
        xai_ms = (always.get('xai') or {}).get('mean_ms_per_explanation')

        for subject, block in calib['per_subject'].items():
            src = block.get('uncalibrated') or {}
            rows = src.get('thresholds') or []
            # Esik 0.0 (hic aciklama) ve 1.0 (her pencere) olculen uclar
            cand = [{'threshold': 0.0, 'explain_rate': 0.0,
                     'error_recall': 0.0, 'precision_of_trigger': None,
                     'kind': 'olculdu'}]
            for t in rows:
                if thresholds and t['threshold'] not in thresholds:
                    continue
                cand.append(dict(t, kind='turetildi'))
            cand.append({'threshold': 1.0, 'explain_rate': 1.0,
                         'error_recall': 1.0, 'precision_of_trigger': None,
                         'kind': 'olculdu'})

            for t in cand:
                r = t['explain_rate']
                mean_exec = (1 - r) * base_exec['mean_ms'] + r * a_exec['mean_ms']
                mean_resp = (1 - r) * base_resp['mean_ms'] + r * a_resp['mean_ms']
                region, p95_resp = mixture_p95(base_resp, a_resp, r)
                _, p95_exec = mixture_p95(base_exec, a_exec, r)
                out['satirlar'].append({
                    'aciklayici': name,
                    'denek': subject,
                    'esik': t['threshold'],
                    'kind': t['kind'],
                    'aciklama_orani': r,
                    'hata_yakalama': t.get('error_recall'),
                    'tetikleme_isabeti': t.get('precision_of_trigger'),
                    # Kesin: karisimin ortalamasi agirlikli ortalamadir
                    'execution_mean_ms': mean_exec,
                    'response_mean_ms': mean_resp,
                    # Yaklasik: dagilimin sekli ozetten tahmin edildi
                    'response_p95_ms': p95_resp,
                    'execution_p95_ms': p95_exec,
                    'p95_bolgesi': region,
                    'aciklama_dakikada': r * (60.0 / hop_s),
                    'cpu_percent_tahmin': (None if base_cpu is None or a_cpu is None
                                           else (1 - r) * base_cpu + r * a_cpu),
                    'rss_mb_tahmin': (None if base_rss is None or a_rss is None
                                      else (1 - r) * base_rss + r * a_rss),
                    'aciklama_basina_ms': xai_ms,
                    # Her pencereyi aciklamaya gore ne kadar hesap kurtariliyor
                    'tasarruf_orani': (1.0 - r),
                })
    return out


def print_report(out):
    print("=" * 96)
    print("D2 - SECICI XAI ESIK TARAMASI")
    print("=" * 96)
    b = out['taban']
    print(f"Taban (aciklama yok): execution {b['execution_mean_ms']:.1f} ms, "
          f"response {b['response_mean_ms']:.1f} ms, p95 {b['response_p95_ms']:.1f} ms")
    print(f"Hop {out['kaynak']['hop_s']:.0f} s -> dakikada "
          f"{60 / out['kaynak']['hop_s']:.0f} pencere")
    print()

    by = {}
    for r in out['satirlar']:
        by.setdefault((r['aciklayici'], r['denek']), []).append(r)

    for (expl, subj), rows in by.items():
        print(f"--- {expl}  /  {subj}   (aciklama basina "
              f"{rows[0]['aciklama_basina_ms']:.0f} ms) ---")
        print(f"   {'esik':>5} {'acikla%':>8} {'hata yak':>9} {'ort resp':>10} "
              f"{'p95 resp':>10} {'p95 bolge':>11} {'dk/acikl':>9} {'tasarruf':>9}  kaynak")
        for r in sorted(rows, key=lambda x: x['esik']):
            hy = f"{r['hata_yakalama']:.3f}" if r['hata_yakalama'] is not None else "-"
            print(f"   {r['esik']:5.2f} {r['aciklama_orani']:7.1%} {hy:>9} "
                  f"{r['response_mean_ms']:9.1f}  {r['response_p95_ms']:9.1f}  "
                  f"{r['p95_bolgesi']:>11} {r['aciklama_dakikada']:8.1f} "
                  f"{r['tasarruf_orani']:8.1%}  {r['kind']}")
        print()


def main():
    cfg = config
    ap = argparse.ArgumentParser(description="D2: secici XAI esik taramasi")
    ap.add_argument('--calibration',
                    default=os.path.join(cfg.RESULTS_DIR, '08_kalibrasyon',
                                         'calibration_d1.json'))
    ap.add_argument('--fog-dir', default=os.path.join(cfg.RESULTS_DIR, '04_fog'))
    ap.add_argument('--hop-s', type=float, default=cfg.HOP_DURATION)
    ap.add_argument('--tag', default='d2')
    args = ap.parse_args()

    out = build(cfg, args.calibration, args.fog_dir, args.hop_s)
    print_report(out)

    path = os.path.join(cfg.RESULTS_DIR, f'selective_xai_sweep_{args.tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[i] Kaydedildi: {path}")
    return out


if __name__ == "__main__":
    main()
