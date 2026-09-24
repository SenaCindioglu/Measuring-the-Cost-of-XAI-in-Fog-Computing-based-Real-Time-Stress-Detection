# Literatür notları: XAI'nin hesaplama maliyeti ve seçici açıklama

Tarama tarihi: 21 Eylül 2026. İki yönde derin arama yapıldı (SHAP/LIME süre
ölçümleri; güven-tetiklemeli seçici açıklama önceki işleri). Üçüncü tarama
(gömülü donanımda fizyolojik sinyal XAI'si) ayrı bölüm olarak eklenecek.

**Doğrulama durumu.** Aşağıdaki kaynaklar bir tarama ajanı tarafından
getirildi ve büyük kısmı ajan tarafından açılıp okundu. **Makaleye girmeden
önce her atıf elle açılıp kontrol edilecek.** Şüpheli olanlar açıkça
işaretli. Kural: doğrulanmamış hiçbir sayı makaleye yazılmaz.

---

## 1. Neden bu tarama yapıldı

İki soru vardı:

1. Bizim ölçtüğümüz şey (pencere başına açıklama maliyeti) literatürde
   ölçülüyor mu, karşılaştırabileceğimiz sayı var mı?
2. Güven eşiğine bağlı seçici açıklama fikri yeni mi?

Cevaplar: (1) neredeyse hiç ölçülmüyor, bu bizim lehimize; (2) **fikir yeni
değil**, katkı cümlesi değişmek zorunda.

---

## 2. Seçici açıklama: fikir yeni değil

### En yakın çalışmalar

| Çalışma | Mekanizma | Bizden farkı |
|---|---|---|
| **Paes, Wei & Calmon (2024), "Selective Explanations"**, arXiv:2405.19562 | Kapsam parametresi α ile bütçeli seçici açıklama; ucuz açıklayıcının belirsizliği pahalıya yükseltmeye karar veriyor | Kapı *açıklayıcının* belirsizliği, bizde *modelin* güveni; onlar her zaman bir açıklama döndürüyor, biz bazen hiç döndürmüyoruz |
| **Daneshvar ve ark. (2026)**, IRICT 2025 / Springer, DOI 10.1007/978-3-032-23035-5_21 | Faithfulness-kapılı kademe; **"güçlü güven-kapılı temel yöntem" ile karşılaştırıyorlar** | Bizim ana katkı dediğimiz şey burada *karşılaştırma tabanı*. En tehlikeli atıf. |
| **Prabhu & Sunitha (2026)**, Discover AI, DOI 10.1007/s44163-026-01317-w | SHAP modülü ancak olasılık > 0.5 olduğunda tetikleniyor, açıkça verim için; ~25 ms açıklama yükü ölçülmüş | Yüksek olasılıkta tetikliyor (bizde düşük güvende), eşik taraması yok, siber güvenlik alanı |
| **Nguyen (2026)**, MSI Journal of AI and Technology | "Uncertainty-triggered selective explainability": konformal küme büyüklüğü tekse açıklama yok, çoksa TreeSHAP | Adı neredeyse birebir aynı. **Dergi düşük itibarlı görünüyor**, tarih Eylül 2026 (eşzamanlı sayılabilir). Yine de atıf verilmeli. |
| **Artelt, Visser & Hammer (2022/2023)**, arXiv:2205.07623 / Neurocomputing 558 | Yalnızca reddedilen (yeterince emin olunmayan) örnekler için açıklama | Gerekçe epistemik, hesap maliyeti değil; süre ölçmüyorlar |
| **Taguta ve ark. (2026)**, arXiv:2609.14036 | Soğuk zincir izlemede SHAP **koşullu** tetikleniyor (yalnızca sıcaklık kırılımı öngörülünce), Raspberry Pi 4'te 24 günlük canlı dağıtım | Bizim kurulumumuza en yakın *sistem* çalışması; +195 ms, +%28 CPU |

Ata çalışmalar (gerekçe benzeşimi): **BranchyNet** (2017, arXiv:1709.01686)
ve **Berestizshevsky & Even** (2018, arXiv:1805.10982) — softmax güvenine
göre hesabı erken kesme. İkincisi bizim kapı kuralının birebir aynısını
çıkarım için kullanıyor (%15-50 MAC azaltma, ~%1 doğruluk kaybı).

Rakip strateji (hakem soracak): **FastSHAP** (ICLR 2022, arXiv:2107.07436)
ve **Covert ve ark. (2024)**, arXiv:2401.15866 — açıklamayı atlamak yerine
ucuzlatmak. Cevabımız: kapılama bunlarla birlikte de çalışır, alternatif
değil. FastSHAP'ın bir kerelik 146-694 dakikalık eğitim maliyeti var.

### Bizde kalan, savunulabilir kısım

Tarama şunları bulamadı:

1. Mekanizmanın **gerçek zamanlı fizyolojik izleme** (pencereli biyosinyal,
   stres) bağlamında uygulanması.
2. **Kalibre edilmiş** max-softmax üzerinden kapı + eşik taraması ve bunun
   kapsam/gecikme/hata-yakalama eğrisi olarak raporlanması.
3. Metrik kümesi: açıklanan pencere % × yakalanan hata % × pencere başına
   hesap × p95 × CPU/RAM. "Belirli açıklama kapsamında hataların ne kadarı
   açıklandı" eğrisi hiçbir kaynakta bu biçimde yok.

### Katkı cümlesinin yeni hali

Yazılmayacak: "Yeni bir güven-tetiklemeli seçici XAI mekanizması öneriyoruz."

Yazılacak: güven-kapılı hesap dağıtımı çıkarım kademelerinde ve açıklama
üretiminde bilinen bir yaklaşımdır [Paes 2024; Daneshvar 2026; Nguyen 2026;
Prabhu & Sunitha 2026; Artelt 2022; BranchyNet 2017]. Katkımız mekanizma
değil, **sürekli gerçek zamanlı fizyolojik izlemede deneysel
karakterizasyonu**: kalibre edilmiş eşik (ECE ile gösterilmiş), açıklama
kapsamı–hata yakalama çalışma eğrisi ve pencere başına 200-650 ms maliyetli
açıklamalarda tam dağıtım maliyet zarfı.

### Hakem sorularına hazırlık

- **"Neden FastSHAP değil de atlama?"** → Kapılama amortize açıklayıcıyla
  birlikte çalışır; FastSHAP'ın eğitim maliyeti var ve atıf yaklaşık.
- **"Bu sadece seçici tahmin değil mi?"** → Biz *her zaman tahmin ediyoruz*,
  yalnızca açıklamayı koşullu üretiyoruz. Tahmin tarafında kapsam kaybı yok.
- **"max-softmax zayıf bir belirsizlik ölçütü."** → ECE öncesi/sonrası
  raporlanacak; ikinci bir kapı sinyali (entropi) ile mekanizmanın sinyalden
  bağımsız olduğu gösterilecek.
- **"Açıklamalar tam kapının açıldığı yerde kararsız."** → En keskin itiraz.
  Post-hoc açıklamalar karar sınırına yakın oynak; kapı da orada tetikleniyor.
  **Faz D'de kararsızlık metriği yalnızca düşük güvenli pencerelerde ayrıca
  ölçülecek.**

---

## 3. Açıklama maliyeti: literatürdeki sayılar

Karşılaştırma için kullanılabilir, doğrulanmış değerler:

| Çalışma | Ölçülen | Sayı | Donanım / girdi |
|---|---|---|---|
| **Franco de la Peña ve ark. (2025), ShaTS**, arXiv:2506.01450 | Pencere başına süre, **açık gerçek zamanlı sınır** | *"pencere başına bir saniyeden fazla süren bir xAI yöntemi gerçek zamanlı dağıtıma uygun değildir"*; KernelSHAP bu sınırı sıkça aşıyor | i7-14700K + RTX 4080; girdi **10 × 69** (SWaT) — yapısal olarak bizim 10 s × 14 kanala en yakın |
| **Basaran & Dressler (2025), XAI-on-RAN**, arXiv:2511.17514 | Canlı O-RAN test yatağında çıkarım başına XAI yükü | Temel 5.3 ms; SHAP (m=16) ≈ **+15 ms** (toplam ~20.4 ms); IG(k=5) +2.8 ms. Kontrol döngüsü bütçesi 10-100 ms | A100; 5 boyutlu girdi |
| **Taguta ve ark. (2026)**, arXiv:2609.14036 | 24 günlük canlı dağıtımda cihaz üstü XAI | Koşullu SHAP **+195 ms**, **+%28 CPU**, toplam tahmin gecikmesi ~555 ms | **Raspberry Pi 4** fog düğümü |
| **Das ve ark. (2021)**, arXiv:2105.09787, ACM TiiS | Tek örnek için açıklama süresi | **LIME 5.91 s, SHAP 695.2 s**, Anchors 739.1 s | i7-6700K; 33 sensör × 30 adım LSTM |
| **Warnecke ve ark. (2020)**, EuroS&P, arXiv:1906.02108 | Açıklama başına süre, CPU ve GPU | CPU: KernelSHAP 9.1 s / 4.5 s; LIME 3.1e-2 / 7.4e-1 s (l=500) | Xeon E5; MLP ~5000 öznitelik, CNN 530k girdi |
| **Chuang ve ark. (2023)**, arXiv:2302.03225 | Örnek başına gecikme | CIFAR-10: SHAP **27 221 ms**, DeepSHAP 323 ms, FastSHAP 0.4 ms, IG 54.6 ms | Donanım belirtilmemiş |
| **Jethani ve ark. (2022), FastSHAP**, ICLR | 1000 görüntüyü açıklama süresi | FastSHAP 0.04 dk; **KernelSHAP 453-1089 dk** | V100 + 8 çekirdek |
| **Tempel ve ark. (2025)**, Applied Intelligence, arXiv:2412.16003 | Doğrulama kümesi toplamı | SHAP **3.25 saat**, Grad-CAM 0.35 saat | V100; iskelet GCN |
| **Uccello & Nadjm-Tehrani (2025)**, arXiv:2509.10206 | Açıklama başına | **TreeSHAP 0.03 s** (83-478 öznitelik) | Belirtilmemiş; XGBoost |
| **Skliarov ve ark. (2025)**, Sci Rep | Görüntü başına | SHAP ~1.25 s, LIME ~2.50 s (⚠️ şekilden okunmuş, yaklaşık) | T4 (Colab); VGG16/ResNet50 |

**Bizim sayılarımız bu tabloya nasıl oturuyor:** LIME 500 örnek / 20 dilim,
tek CPU ipliği, p50 **710 ms**, p95 **1130 ms**. ShaTS'in koyduğu 1 s/pencere
sınırını medyanda geçiyoruz, p95'te aşıyoruz. 250 örneğe inince p50 490 ms
ve p95 büyük olasılıkla sınırın altında kalıyor — bu, ucuz çalışma noktası
için literatüre dayanan bir gerekçe.

---

## 4. Boşluk nerede (makalenin gerekçesi)

1. **Değerlendirme literatürü bu ekseni kapsam dışı bırakıyor.** Nauta ve
   ark. (2023, ACM CSUR, Co-12 çerçevesi) açıkça yazıyor: *"çalışma süresi
   gibi açıklama kalitesini doğrudan etkilemeyen ölçütler hariç tutulmuştur."*
   Quantus, OpenXAI, XAI-Bench ve BEExAI hiç süre raporlamıyor. Süre sütunu
   olan tek araç **Compare-xAI** (Belaid ve ark. 2022). Bu, "açıklama
   maliyeti" için standart bir metrik, standart bir birim ve liderlik
   tablolarında bir yer olmadığı anlamına geliyor. Alıntılanabilir boşluk
   cümlesi.
2. **Ham çok kanallı fizyolojik sinyalde pencere başına açıklama gecikmesi
   yayınlanmamış.** Yapısal olarak en yakın çalışma (ShaTS, 10 × 69 girdi)
   pencere başına milisaniyeleri yalnızca bir şekil içinde bırakıyor.
3. **Açıklama maliyeti hiçbir yerde taşıma katmanı metrikleriyle birlikte
   verilmemiş.** Kuyruk gecikmesi, jitter, bant genişliği ve kaynak
   kullanımıyla aynı bütçede raporlayan çalışma bulunamadı — bizim yedi
   metrik çerçevemiz tam burada duruyor.
4. **Açıklayıcı hiperparametresi süpürüp hem süre hem kalite veren çalışma
   çok az.** Wang ve ark. (KernelSHAP 40/77/varsayılan örnek), Jethani ve
   ark. (model çağrısı bütçesi), Mikriukov ve ark. (2026, bağıl maliyetle
   Pareto, saat cinsinden süre yok). Fizyolojik pencerelerde LIME örnek
   sayısı / SHAP arka plan boyutu süpürüp ms **ve** faithfulness veren
   çalışma yok.
5. **Negatif kanıt, tam bizim alanımızda:** Ghose ve ark. (2025, Frontiers in
   Digital Health) stres tespitinde LIME kullanıyor ve Jetson Orin Nano'da
   çalışıyor, ama hiçbir açıklama süresi ya da çıkarım gecikmesi
   raporlamıyor; yalnızca eğitim süreleri var.

---

## 5. Atıf verilmeden önce doğrulanacaklar

⚠️ Tarama ajanının açıkça "doğrulanmamış" ya da "tutarsız" işaretlediği
kaynaklar. Makaleye girmeden önce elle açılacak:

- Whitten ve ark. (2026), arXiv:2601.18696 — çıkarılan sayılar kendi içinde
  tutarsız (SHAP hem 2.07 ms hem 250 ms).
- Basaran & Dressler'in "+15 ms" değeri bir kaynakta doğrulanamadı, bir
  kaynakta doğrulandı; PDF elle okunacak.
- Quantus, OpenXAI, XAI-Bench için "süre raporlamıyor" iddiası yalnızca arama
  özetinden geliyor; en az biri açılıp bakılacak.
- Lundberg ve ark. (2020) TreeSHAP karmaşıklık iddiası ikincil kaynaktan.
- **US Patent 12,566,983** — "düşük güven bölgeleri için açıklama üretme"
  iddiası var. Özgünlük iddiası için engelleyici olabilir, elle okunmalı.
- Nguyen (2026) — dergi itibarı sorgulanabilir, ama içerik birebir yakın.

---

## 6. Fizyolojik sinyal ve gömülü donanımda XAI maliyeti

Üçüncü tarama (21 Eylül) bu kesişimi aradı. Sonuç: kesişim neredeyse boş.

### WESAD + açıklanabilirlik: hiçbiri süre vermiyor

| Çalışma | XAI yöntemi | Süre / kaynak sayısı |
|---|---|---|
| **Almadhor ve ark. (2025)**, Front. Bioeng. Biotechnol., DOI 10.3389/fbioe.2025.1659002 — **WESAD + DREAMER, 1D-CNN** | Grad-CAM, Integrated Gradients, dikkat görselleştirme | **Hiçbiri.** Çıkarım süresi, XAI süresi, bellek, parametre sayısı, donanım: hiçbiri yok |
| Velichko & Huyut (2026), BSPC 129, WESAD göğüs solunumu, kompakt 1D-CNN | yorumlanabilir otokorelasyon belirteçleri | Hiçbiri |
| Oliver & Dakshit (2025), arXiv:2502.18733, WESAD, transformer | gömme uzayı görselleştirme | Hiçbiri |

**Makalenin boşluk cümlesi için en güçlü kanıt birincisi:** bizim veri seti
çiftimizi (WESAD + DREAMER) ve model ailemizi (1D-CNN) kullanan bir çalışma,
üstelik XAI de uygulayan bir çalışma, hiçbir hesaplama maliyeti
raporlamıyor.

### Gömülü donanımda XAI maliyetini ölçen iki çalışma (ikisi de fizyolojik değil)

| Çalışma | Donanım | Ölçülen |
|---|---|---|
| **Taguta ve ark. (2026)**, arXiv:2609.14036 | Raspberry Pi 4 fog düğümü | Koşullu SHAP **+%28 CPU**, toplam 555 ms, 0.7 Wh/tahmin. Soğuk zincir, 24 günlük canlı dağıtım |
| **AlNusif (2025)**, IJECS 14(7) | RPi 4, Jetson Nano, Coral | RPi 4 + SHAP: temel 88 ms + **XAI 140 ms** = 228 ms, 130+ MB; RPi 4 + LIME: **~92 ms**, ~98 MB; Saliency <50 ms. "Gerçek zamanlı" ölçütü <200 ms |

⚠️ AlNusif için uyarı: dergi düşük itibarlı ve tablolar kendi içinde tutarsız
(aynı yapılandırma bir yerde 88 ms, başka yerde 42 ms temel gecikme; XAI yükü
140 ms ve 95 ms olarak iki kez geçiyor). Tekrar sayısı, kütüphane sürümü,
varyans yok. "Birileri bunu ölçmüş" demek için kullanılabilir, kesin değer
olarak kullanılamaz.

### Fizyolojik + gerçek gömülü donanım + XAI, ama süre ölçmeyenler

- **EvoApneaFormer** (2026), Front. Bioeng. Biotechnol. — Raspberry Pi 4,
  SHAP + Grad-CAM, çıkarım <50 ms, uçtan uca <300 ms, <2 W. Ama SHAP'ın
  maliyeti çıkarımdan **ayrılmamış**.
- **Ghose ve ark. (2025)**, Front. Digit. Health — Jetson Orin Nano, LIME,
  yalnızca eğitim süreleri.

### Sistem metrikleri + XAI maliyetini birlikte veren: yok

Alan üç ayrık gruba bölünmüş:

1. **Sistem metrikleri, XAI yok:** Hizem ve ark. (2025) RPi 4'te 0.200 ms,
   %15.4 CPU, 193 KB; ve bizim referansımız Erzurumluoğlu ve ark. (ICACT
   2025).
2. **XAI maliyeti, ağ metriği yok:** AlNusif, Basaran & Dressler, ShaTS.
3. **Fizyolojik sinyalde XAI, hiç maliyet yok:** yukarıdaki tablolar.

**El Rashidy ve ark. (2025)**, Front. Comput. Sci. 7 özellikle iyi bir örnek:
*fog mimarisinde duygu tanıma, LIME ve Grad-CAM ile*, ve hâlâ çıkarım başına
gecikme ya da XAI maliyeti yok. Yani fog + XAI kesişiminde bile bu ölçüm
atlanıyor.

### Gerçek zamanlı sınır: test edilen mi, iddia edilen mi

- **Test edilen (yalnızca iki):** ShaTS (1 s/pencere, ölçülüp KernelSHAP'ın
  ihlal ettiği gösterilmiş) ve XAI-on-RAN (10 ms bütçe, SHAP aşıyor).
- **Ölçülmeden iddia edilen:** AlNusif (<200 ms), EXGnet, El Rashidy,
  Almadhor — "gerçek zamanlıya uygun" diyor, ölçüm yok.

### Erişilemeyen ama değerli olabilecek kaynaklar

Üniversite erişimi gerektiriyor, sayılar yalnızca arama özetinden:

1. **MDPI Computers 15(3):139 (2026)**, DOI 10.3390/computers15030139 —
   Raspberry Pi 4 + SHAP, 60 ms çıkarım iddiası. En yüksek öncelik.
2. **Comput. Methods Programs Biomed. (2026)**, PMID 41719782 — "lightweight
   stress monitoring for IoMT", SHAP + Raspberry Pi 5.
3. **Biomed. Signal Process. Control (2026)**, S174680942600488X — stres
   tespitinde SHAP + LIME + MC Dropout.
4. **Wiley Adv. Intell. Syst.**, DOI 10.1002/aisy.70487 — açıklama katmanı
   gecikmesi 210-300 ms, medyan 247 ms, p95 291 ms, 500 ms klinik eşiğine
   karşı. Bizim çerçeveye çok yakın bir raporlama biçimi.

### ✅ Referans makale elde edildi (24 Eylül 2026)

Birincil kaynak okundu, künye ve sayılar doğrulandı. Daha önceki
"bulunamadı" notu geçersiz.

**Künye.** Ömür Fatmanur Erzurumluoğlu, Kadir Özlem, Aslı Tuncay Atalay,
Özgür Atalay, Gökhan İnce. *Fog Computing-based Real-Time Emotion
Recognition using Physiological Signals.* The 27th International Conference
on Advanced Communications Technology (ICACT 2025), s. 442–450.
ISBN 979-11-88428-13-7. İstanbul Teknik Üniversitesi.

**Yöntem (bizimkinden farkları).**

| | Onlar | Biz |
|---|---|---|
| Veri | DREAMER (EEG 14 kanal + ECG 2) | WESAD (göğüs 8 + bilek 6) |
| Görev | 4 sınıf duygu | ikili stres |
| Model | Random Forest | 1D-CNN (49.890 parametre) |
| Girdi | 576 öznitelikten seçilmiş **24 öznitelik** | **ham sinyal**, öznitelik çıkarımı yok |
| Pencere | 3 s, 1 s kayma | 10 s, 5 s kayma |
| İstek hızı | saniyede 32 (31 veri akışı + 1 tahmin) | saniyede 2 tick, 5 s'de bir tahmin |
| Doğruluk | %79.70 (24 öznitelik), %84.21 (104) | — farklı görev, kıyaslanamaz |

**Ölçülen yedi metrik ve donanım.** Cloud: DigitalOcean VPS (Frankfurt).
Broker ve Worker: ikisi de Raspberry Pi 4 Model B (Cortex-A72 @1.8 GHz,
8 GB). Client: Asus ROG Strix G15 (Ryzen 7 4800H, 16 GB, Win 11). Ağ:
Wi-Fi 802.11ac, 100/5 Mbps. Ölçüm: 1. katılımcının 199 s'lik "amusement"
kaydı, 197 tahmin isteği.

| Metrik | Edge | Worker (Fog) | Broker (Fog) | Cloud |
|---|---|---|---|---|
| Latency | **1.6 ms** | 4.7 ms | 4.8 ms | 32.2 ms |
| Queuing delay | 13.7 ms (en kötü) | 0.6 ms | 0.5 ms | 0.8 ms |
| Execution time | 5.7 ms | 34.7 ms | 35.3 ms | 8.9 ms |
| **Response time** | **22.5 ms** | 43.3 ms | 43.9 ms | 73.3 ms |
| Jitter | 0.5 ms | 17.4 ms | 16.8 ms | 36.6 ms |
| Bandwidth | **8.5 kbps** | 373.9 kbps | 392.9 kbps | 384.2 kbps |
| CPU | %0.4 | %6.7 | %6.7 | %6.2 |
| Memory | %41.9 (en kötü) | %7.2 | %7.2 | %12.2 |

**Kapasite iddiası:** 43 ms'lik döngü ve saniyede bir tahminle tek worker
**~20 eşzamanlı kullanıcı** taşıyor (s. 449). A4 (çok istemcili yük)
doğrudan bu sayıyla karşılaştırılacak.

**Dikkat: makale "fog üstün" diyor ama tablosu edge'i gösteriyor.** Edge
latency, response time, jitter ve bant genişliğinde açık ara önde
(22.5 ms'ye karşı 43.3 ms). Yazarlar fog'u şu gerekçelerle seçiyor
(s. 448, VI-B4): çok kullanıcılı eşzamanlı kullanım, giyilebilir cihazın
pil ömrü ve merkezî veri analizi gereksinimi — **ölçülen zaman metrikleri
yüzünden değil**. Bu ayrım bizim makalemizde de tekrarlanmalı, çünkü bizim
`05_yerlesim` sonucumuz da benzer: ortak yerleşim response 7.44 ms, fog
9.97 ms.

**Tek gerçek karşılaştırma noktamız bant genişliği.** Onlar 3 s pencereden
24 öznitelik çıkarıp gönderiyor, worker'da 373.9 kbps; biz 10 s pencerenin
ham sinyalini gönderiyoruz, 187 kbps. Öznitelik göndermek ham sinyalden
**daha pahalı** çıkıyor, çünkü onların istemcisi saniyede 32 istek atıyor.
Zaman metrikleri donanım farkı (Pi 4 ↔ masaüstü) yüzünden doğrudan
kıyaslanamaz; bu makalede açıkça yazılmalı.

**Broker'ın gereksizliği onların kendi sayılarıyla doğrulanıyor:** worker
43.3 ms, broker 43.9 ms. Tek worker varken broker yalnızca bir sıçrama
ekliyor. Bizim mimaride broker olmaması bu yüzden eksiklik değil.

---

## 7. Plana etkisi

- **Faz D'ye eklenen iki ölçüm:** (1) ECE ile kalibrasyon kanıtı, (2)
  açıklama kararsızlığının yalnızca düşük güvenli pencerelerde ölçümü.
- **Faz C'ye eklenen bir eşik:** ShaTS'in 1 s/pencere sınırı, yapılandırma
  seçiminde ölçüt olarak kullanılacak ve makalede atıfla verilecek.
- **Makalede yeni bölüm:** ilgili çalışmalar, yukarıdaki iki grupla.
- **Katkı listesi 4. madde yeniden yazılacak** (bkz. bölüm 2).
