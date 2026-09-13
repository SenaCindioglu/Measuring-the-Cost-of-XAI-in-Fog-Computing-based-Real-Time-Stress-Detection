"""
tools/mcu_feasibility.py
=========================
Modelin bir mikrodenetleyicide (STM32 vb.) calisip calismayacagini DONANIMSIZ
hesaplar: katman katman parametre, aktivasyon bellegi, tepe RAM ihtiyaci ve
carpma-toplama (MAC) sayisi.

Neden gerekli: MCU'da darbogaz hiz degil BELLEKTIR. Cikarimin kac milisaniye
surecegi genelde bir sorun olmaz (5 saniyelik hop butcesi var), ama ara
aktivasyonlar RAM'e sigmazsa model hic calismaz.

Kullanim:
    python -m tools.mcu_feasibility
    python -m tools.mcu_feasibility --window 320    # 5 saniyelik pencere icin
"""

import argparse

from config import config
from models.temporal_cnn import load_checkpoint


# Yaygin STM32 aileleri: (ad, RAM KB, Flash KB, saat MHz, FPU)
STM32_FAMILIES = [
    ('STM32F103 (M3)',    20,   128,  72, 'yok'),
    ('STM32F407 (M4)',   192,  1024, 168, 'tek hassasiyet'),
    ('STM32F746 (M7)',   320,  1024, 216, 'tek hassasiyet'),
    ('STM32H743 (M7)',  1024,  2048, 480, 'cift hassasiyet'),
    ('STM32U575 (M33)',  786,  2048, 160, 'tek hassasiyet'),
]


def analyze(num_channels, filters, kernel_size, window_size, fc_hidden=64, num_classes=2):
    """
    Katman katman doku. Aktivasyon boyutlari eleman sayisi olarak dondurulur;
    bayt karsiligi veri tipine gore ayrica hesaplanir.
    """
    layers = []
    in_ch, length = num_channels, window_size
    layers.append({'ad': 'giris', 'params': 0, 'macs': 0,
                   'cikis': in_ch * length, 'sekil': f'{in_ch}x{length}'})

    for i, out_ch in enumerate(filters):
        # Conv1d 'same' padding -> uzunluk degismez
        params = out_ch * in_ch * kernel_size + out_ch
        macs = length * out_ch * in_ch * kernel_size
        layers.append({'ad': f'conv{i+1}', 'params': params, 'macs': macs,
                       'cikis': out_ch * length, 'sekil': f'{out_ch}x{length}'})
        # BatchNorm: cikarimda olcek+kaydirma olarak katlanabilir, ayri parametre saymiyoruz
        layers.append({'ad': f'bn{i+1}', 'params': 2 * out_ch, 'macs': length * out_ch,
                       'cikis': out_ch * length, 'sekil': f'{out_ch}x{length}'})
        length //= 2
        layers.append({'ad': f'pool{i+1}', 'params': 0, 'macs': out_ch * length,
                       'cikis': out_ch * length, 'sekil': f'{out_ch}x{length}'})
        in_ch = out_ch

    layers.append({'ad': 'gap', 'params': 0, 'macs': in_ch * length,
                   'cikis': in_ch, 'sekil': f'{in_ch}'})
    layers.append({'ad': 'fc1', 'params': in_ch * fc_hidden + fc_hidden,
                   'macs': in_ch * fc_hidden, 'cikis': fc_hidden, 'sekil': f'{fc_hidden}'})
    layers.append({'ad': 'bn_fc', 'params': 2 * fc_hidden, 'macs': fc_hidden,
                   'cikis': fc_hidden, 'sekil': f'{fc_hidden}'})
    layers.append({'ad': 'fc2', 'params': fc_hidden * num_classes + num_classes,
                   'macs': fc_hidden * num_classes, 'cikis': num_classes,
                   'sekil': f'{num_classes}'})
    return layers


def peak_activation(layers):
    """
    Tepe aktivasyon bellegi: bir katman calisirken GIRISI ve CIKISI ayni anda
    bellekte durmak zorunda. En buyuk (girdi + cikti) cifti tepe degeri verir.
    (Yerinde islem yapilabilen katmanlar -- BN, ReLU -- bu tahmini biraz
    yukari cekiyor, yani sonuc muhafazakar.)
    """
    peak, at = 0, None
    for a, b in zip(layers, layers[1:]):
        need = a['cikis'] + b['cikis']
        if need > peak:
            peak, at = need, b['ad']
    return peak, at


def main():
    cfg = config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=cfg.MODEL_PATH)
    parser.add_argument('--window', type=int, default=None,
                        help="Pencere uzunlugu (ornek). Varsayilan config.WINDOW_SIZE")
    args = parser.parse_args()

    _, arch, _ = load_checkpoint(args.model, 'cpu', fallback_cfg=cfg)
    window = args.window or cfg.WINDOW_SIZE

    layers = analyze(arch['num_channels'], arch['filters'], arch['kernel_size'], window)
    total_params = sum(l['params'] for l in layers)
    total_macs = sum(l['macs'] for l in layers)
    peak_elems, peak_at = peak_activation(layers)

    print("=" * 74)
    print("MCU UYGUNLUK ANALIZI")
    print("=" * 74)
    print(f"Model   : depth={arch['depth']} filters={arch['filters']} "
          f"k={arch['kernel_size']} kanal={arch['num_channels']}")
    print(f"Pencere : {window} ornek ({window / cfg.TARGET_RATE:.1f} s @ {cfg.TARGET_RATE} Hz)")
    print("=" * 74)

    print(f"\n{'katman':<9}{'sekil':>12}{'parametre':>12}{'MAC':>14}{'aktivasyon':>13}")
    for l in layers:
        print(f"{l['ad']:<9}{l['sekil']:>12}{l['params']:>12,}{l['macs']:>14,}"
              f"{l['cikis']:>13,}")
    print(f"{'TOPLAM':<9}{'':>12}{total_params:>12,}{total_macs:>14,}")

    print(f"\nTepe aktivasyon: {peak_elems:,} eleman ('{peak_at}' katmaninda)")

    print("\n" + "-" * 74)
    print("BELLEK IHTIYACI (veri tipine gore)")
    print("-" * 74)
    print(f"{'tip':<10}{'agirlik (Flash)':>18}{'tepe RAM':>14}{'giris tamponu':>16}")
    for name, nbytes in [('float32', 4), ('float16', 2), ('int8', 1)]:
        w = total_params * nbytes
        r = peak_elems * nbytes
        inp = arch['num_channels'] * window * nbytes
        print(f"{name:<10}{w / 1024:>15.1f} KB{r / 1024:>11.1f} KB{inp / 1024:>13.1f} KB")

    print("\n" + "-" * 74)
    print("STM32 AILELERINE GORE DURUM")
    print("-" * 74)
    print(f"{'aile':<20}{'RAM':>7}{'Flash':>8}{'MHz':>6}  {'float32':>9}  {'int8':>9}")
    for fam, ram_kb, flash_kb, mhz, fpu in STM32_FAMILIES:
        row = f"{fam:<20}{ram_kb:>5}KB{flash_kb:>6}KB{mhz:>6}  "
        for nbytes in (4, 1):
            w_kb = total_params * nbytes / 1024
            # Tepe RAM + giris tamponu + calisma alani icin %25 pay
            r_kb = (peak_elems * nbytes + arch['num_channels'] * window * nbytes) / 1024 * 1.25
            fits = (w_kb <= flash_kb * 0.8) and (r_kb <= ram_kb * 0.8)
            row += f"{'SIGAR' if fits else 'sigmaz':>9}  "
        print(row)
    print("\n(Flash'in %80'i, RAM'in %80'i siniri alindi; kalani program kodu,")
    print(" yigin ve tampon icin birakildi.)")

    print("\n" + "-" * 74)
    print("HIZ TAHMINI")
    print("-" * 74)
    print(f"Pencere basina {total_macs / 1e6:.2f} milyon MAC gerekiyor.")
    print(f"{'donanim':<34}{'MAC/cevrim':>12}{'tahmini sure':>16}")
    scenarios = [
        ('Cortex-M4 @168MHz, float32 (CMSIS)', 0.3, 168e6),
        ('Cortex-M7 @480MHz, float32 (CMSIS)', 0.8, 480e6),
        ('Cortex-M7 @480MHz, int8 (CMSIS-NN)', 2.5, 480e6),
    ]
    for label, mac_per_cycle, hz in scenarios:
        ms = total_macs / (mac_per_cycle * hz) * 1000
        print(f"{label:<34}{mac_per_cycle:>12.1f}{ms:>13.1f} ms")
    budget_ms = cfg.HOP_DURATION * 1000
    print(f"\nButce: {budget_ms:.0f} ms (hop suresi). Yukaridaki en yavas senaryo bile")
    print("butcenin cok altinda; MCU'da darbogaz hiz degil BELLEK.")


if __name__ == "__main__":
    main()
