# Aktivasyon Fonksiyonlarının Eğitim Dinamiklerine Etkisi

**Ders:** SWE012 – Deep Learning with Python
**Hoca:** Asst. Prof. Yiğit Bekir Kaya
**Üniversite:** İstinye Üniversitesi — Bilgisayar Mühendisliği Bölümü

## Grup Üyeleri

| İsim | Öğrenci No |
|------|-----------|
| Selvinaz Sayın | 220901755 |
| Ege Karaurgan | 229910141 |
| Vedat Efe Gezer | 229910158 |
| Mehmet Emin Akkaya | 2309011036 |
| Bayram Selim Yılmaz | 2309011053 |

## Proje Hakkında

Bu proje, dört temel aktivasyon fonksiyonunun (Sigmoid, Tanh, ReLU, Leaky ReLU) sinir ağı eğitim dinamikleri üzerindeki etkisini kontrollü deneylerle inceler. Aktivasyon fonksiyonları projenin **derinlik** boyutunu oluştururken, derste işlenen optimizasyon, regularizasyon ve initialization metodolojileri **genişlik** boyutu olarak entegre edilmiştir.

## Deneyler

| # | Deney | Açıklama |
|---|-------|----------|
| 1 | Aktivasyon Karşılaştırması | Sigmoid, Tanh, ReLU, Leaky ReLU — tüm diğer parametreler sabit |
| 2 | Aktivasyon × Optimizer | SGD, Momentum, RMSProp, Adam etkileşimi |
| 3 | Aktivasyon × Regularizasyon | L2, L1, Dropout, BatchNorm, Label Smoothing etkileşimi |
| 4 | Aktivasyon × Initialization | Xavier, He, Random başlangıç stratejileri |
| + | Gradyan Akış Analizi | 5 katmanlı ağda vanishing gradient görselleştirme |
| + | Ölü Nöron Analizi | ReLU vs Leaky ReLU dead neuron karşılaştırması |

## Teknolojiler

- **Python 3.10+**
- **PyTorch** — model tanımlama ve eğitim
- **Matplotlib** — görselleştirme
- **NumPy** — sayısal hesaplamalar

## Veri Seti

**Fashion-MNIST:** 60.000 train / 10.000 test, 10 sınıf, 28×28 gri tonlamalı görüntüler

## Kullanım

```bash
# Gerekli kütüphaneler
pip install torch torchvision matplotlib numpy

# Notebook'u çalıştır
jupyter notebook activation_functions_training_dynamics.ipynb
```

## Dosya Yapısı

```
├── README.md                                        # Bu dosya
├── REPORT.md                                        # Detaylı proje raporu
├── activation_functions_training_dynamics.ipynb      # Ana notebook (kod + analiz)
├── responsibilities/                                # Sorumluluk dosyaları
│   ├── 220901755.md                                 # Selvinaz Sayın
│   ├── 229910141.md                                 # Ege Karaurgan
│   ├── 229910158.md                                 # Vedat Efe Gezer
│   ├── 2309011036.md                                # Mehmet Emin Akkaya
│   └── 2309011053.md                                # Bayram Selim Yılmaz
├── allanoucements.txt                               # Ders duyuruları
└── Deep Learning_merged.txt                         # Ders çalışma kılavuzu
```

## Lisans

Bu proje İstinye Üniversitesi SWE012 dersi kapsamında hazırlanmıştır.
