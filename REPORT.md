# REPORT: Aktivasyon Fonksiyonlarının Eğitim Dinamiklerine Etkisi

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

---

## 1. Proje Özeti

Bu projede, aktivasyon fonksiyonlarının (Sigmoid, Tanh, ReLU, Leaky ReLU) sinir ağı eğitim sürecine etkisi kontrollü deneylerle incelenmektedir. Aktivasyon fonksiyonları **derinlik** boyutunu oluştururken, derste işlenen tüm metodolojiler (optimizasyon, regularizasyon, initialization) **genişlik** boyutu olarak dahil edilmiştir.

**Veri Seti:** Fashion-MNIST (60.000 train, 10.000 test, 10 sınıf, 28×28 gri tonlamalı)
**Framework:** PyTorch
**Yaklaşım:** Tüm parametreleri sabit tutup yalnızca bir değişkeni değiştirerek kontrollü deneyler

---

## 2. Uygulanan Metodolojiler

### 2.1 Aktivasyon Fonksiyonları (Ana Konu — Derinlik)

| Fonksiyon | Formül | Avantaj | Dezavantaj |
|-----------|--------|---------|------------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | Olasılık çıktısı [0,1] | Vanishing gradient, çıktı sıfır merkezli değil |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Sıfır merkezli [-1,1] | Vanishing gradient (doygunluk bölgeleri) |
| ReLU | max(0, x) | Hızlı hesaplama, gradient=1 (x>0) | Dead neuron problemi |
| Leaky ReLU | max(0.01x, x) | Dead neuron sorununu hafifletir | Küçük negatif eğim sabiti seçimi |

**Seçim gerekçesi:** Bu dört fonksiyon, deep learning tarihindeki temel gelişim çizgisini temsil eder. Sigmoid/Tanh (klasik) → ReLU (modern standart) → Leaky ReLU (iyileştirme) sıralaması, vanishing gradient sorununun nasıl aşıldığını gösterir.

### 2.2 Optimizasyon Yöntemleri (Genişlik)

| Optimizer | Mekanizma | Hiperparametreler |
|-----------|-----------|-------------------|
| SGD | w -= lr × ∇J | lr=0.01 |
| SGD + Momentum | Hız birikimi: v = βv + ∇J | lr=0.01, β=0.9 |
| RMSProp | Per-parametre adaptive lr (exp. moving avg.) | lr=0.001 |
| Adam | Momentum + RMSProp + bias correction | lr=0.001, β₁=0.9, β₂=0.999 |

**Seçim gerekçesi:** SGD'den Adam'a kadar olan evrim, optimizasyon algoritmalarının gelişim sürecini yansıtır. Adam, endüstri standardı olarak baseline optimizer seçilmiştir.

### 2.3 Regularizasyon Yöntemleri (Genişlik)

| Yöntem | Hiperparametre | Tuning | Bayesian Yorum |
|--------|---------------|--------|---------------|
| L2 Weight Decay | α = 1e-4 | {1e-3, 1e-4, 1e-5} arasından seçildi | Gaussian prior |
| L1 Lasso | λ = 1e-5 | {1e-4, 1e-5, 1e-6} arasından seçildi | Laplace prior |
| Dropout | p = 0.5 (drop) | Standart hidden layer oranı | Ensemble yaklaşımı |
| Batch Normalization | γ, β (learnable) | Otomatik öğrenilir | Internal covariate shift azaltma |
| Label Smoothing | ε = 0.1 | Standart değer (0.05-0.2 aralığı) | Kalibrasyon iyileştirme |

**Neden bu yöntemler?** Her biri farklı bir overfitting mekanizmasını hedefler:
- L2: Ağırlık büyüklüklerini sınırlar
- L1: Gereksiz bağlantıları sıfırlar (feature selection)
- Dropout: Alt ağ ensemble'ı oluşturur
- BatchNorm: Katmanlar arası dağılım kaymasını önler
- Label Smoothing: Aşırı güvenli tahminleri engeller

### 2.4 Initialization Stratejileri (Genişlik)

| Yöntem | Varyans | Uygun Aktivasyon |
|--------|---------|-----------------|
| Xavier (Glorot) | 2/(n_in + n_out) | Sigmoid, Tanh |
| He (Kaiming) | 2/n_in | ReLU, Leaky ReLU |
| Random (σ=0.5) | Kontrolsüz | — (baseline) |

**Seçim gerekçesi:** Xavier ve He, teorik olarak türetilmiş optimal stratejilerdir. Random initialization, yanlış başlatmanın etkisini göstermek için kontrol grubu olarak kullanılmıştır.

---

## 3. Deneysel Tasarım

### Kontrollü Deney Prensibi
Her deneyde **yalnızca bir değişken** değiştirilmiş, diğer tüm parametreler sabit tutulmuştur:

| Deney | Değişken | Sabit Tutulan |
|-------|----------|--------------|
| Deney 1: Aktivasyon Karşılaştırması | Aktivasyon fonksiyonu | Adam, He init, no reg, 15 epoch |
| Deney 2: Optimizer Etkileşimi | Optimizer × Aktivasyon | He init, no reg, 15 epoch |
| Deney 3: Regularizasyon Etkileşimi | Reg. yöntemi × Aktivasyon | Adam, He init, 15 epoch |
| Deney 4: Initialization Etkileşimi | Init yöntemi × Aktivasyon | Adam, no reg, 15 epoch |

### Eş Zamanlı Uygulanan Yöntemler
- **Her deneyde:** SGD minibatch (batch_size=128), CrossEntropyLoss (Softmax + NLL), backpropagation
- **Deney 3'te:** BatchNorm + Activation birlikte (BN → Activation sırası)
- **Deney 3'te:** Dropout + Activation birlikte (Activation → Dropout sırası)

### Hiperparametre Tuning Süreci
- **Learning rate:** Adam için 0.001 (standart), SGD için 0.01 (daha yüksek gerekli)
- **Batch size:** 128 (SGD noise ile hesaplama maliyeti dengesi)
- **Hidden layers:** [256, 128] (2 katman — yeterli kapasite, aşırı derinlik değil)
- **Epoch:** 15 (yakınsama için yeterli, overfitting gözlemi için uygun)
- **Weight decay:** Grid search ile {1e-3, 1e-4, 1e-5} → 1e-4 optimal
- **L1 lambda:** Grid search ile {1e-4, 1e-5, 1e-6} → 1e-5 optimal
- **Dropout rate:** p=0.5 (hidden layer standardı, Hinton et al. önerisi)
- **Label smoothing:** ε=0.1 (standart, Szegedy et al. önerisi)
- **Seed:** 42 (tüm deneylerde aynı → reproducibility)

---

## 4. Performans Karşılaştırması

### 4.1 Deney 1: Aktivasyon Fonksiyonu Karşılaştırması

| Aktivasyon | Yakınsama Hızı | Test Accuracy | Gradient Flow |
|------------|---------------|---------------|--------------|
| Sigmoid | En yavaş | En düşük | Vanishing (giriş katmanlarında ≈0) |
| Tanh | Yavaş | Orta | Vanishing (Sigmoid'den iyi) |
| ReLU | Hızlı | Yüksek | Stabil (gradient=1) |
| Leaky ReLU | Hızlı | Yüksek | En stabil (dead neuron yok) |

### 4.2 Deney 2: Optimizer Etkileşimi

| Kombinasyon | Gözlem |
|-------------|--------|
| Sigmoid + SGD | En kötü: vanishing gradient + sabit lr |
| Sigmoid + Adam | Adam'ın adaptive lr'si Sigmoid'in yavaşlığını kısmen telafi eder |
| ReLU + SGD | Güçlü gradient akışı sayesinde SGD bile yeterli |
| ReLU + Adam | En stabil ve güvenilir performans |

### 4.3 Deney 3: Regularizasyon Etkileşimi

| Kombinasyon | Gözlem |
|-------------|--------|
| Sigmoid + BatchNorm | **En dramatik iyileşme** — BN, doygunluk sorununu büyük ölçüde çözer |
| ReLU + Dropout | Generalization gap azalır, eğitim yavaşlar |
| L1 | Ağırlıklarda seyreklik oluşturur (feature selection) |
| L2 | Tüm ağırlıkları küçültür ama sıfırlamaz |
| Label Smoothing | Aşırı güvenli tahminleri engelleyerek kalibrasyon iyileştirir |

### 4.4 Deney 4: Initialization Etkileşimi

| Kombinasyon | Gözlem |
|-------------|--------|
| Sigmoid + Xavier | Doğru eşleşme — sinyal varyansı korunur |
| ReLU + He | Doğru eşleşme — ReLU'nun yarıya indirdiği varyans telafi edilir |
| Herhangi biri + Random | Kontrolsüz varyans → kararsız eğitim |

---

## 5. Ek Analizler

### 5.1 Gradyan Akış Analizi
5 katmanlı derin ağda, başlangıç durumunda gradyan normlarının katmanlar arası değişimi ölçüldü:
- **Sigmoid/Tanh:** Giriş katmanlarına doğru gradyan normu logaritmik olarak düşer (vanishing)
- **ReLU/Leaky ReLU:** Gradyan normu katmanlar boyunca yaklaşık sabit kalır

### 5.2 Ölü Nöron Analizi
10 epoch eğitim sonrası, sürekli sıfır çıktı veren nöronların oranı:
- **ReLU:** Belirli bir oranda ölü nöron gözlemlenir
- **Leaky ReLU:** Negatif eğim (0.01) sayesinde ölü nöron oranı önemli ölçüde düşer

### 5.3 Bias-Variance Perspektifi
Her aktivasyon fonksiyonu için train-test loss arasındaki generalization gap hesaplanarak overfitting eğilimi değerlendirildi.

---

## 6. Veri Seti Değerlendirmesi

**Fashion-MNIST** seçim gerekçeleri:
- **Yeterli karmaşıklık:** MNIST'ten daha zor (kıyafet sınıflandırma), aktivasyon farkları belirgin
- **Standart benchmark:** Sonuçlar literatür ile karşılaştırılabilir
- **Makul boyut:** 60K train / 10K test — overfitting analizi için uygun
- **i.i.d. uyumu:** Train ve test setleri aynı dağılımdan örneklenmiş
- **10 sınıf:** Multi-class classification → Softmax + CCE doğal seçim

---

## 7. Sonuç ve Pratik Öneriler

1. **Varsayılan konfigürasyon:** ReLU + He init + Adam + BatchNorm — güvenli başlangıç noktası
2. **Dead neuron sorunu varsa:** Leaky ReLU veya ELU tercih edilmeli
3. **Overfitting varsa:** Dropout + L2 kombinasyonu etkili
4. **Sigmoid/Tanh zorunluluğu varsa:** BatchNorm mutlaka eklenmelive Adam optimizer kullanılmalı
5. **Feature selection gerekiyorsa:** L1 regularizasyon tercih edilmeli

---

## 8. Kapsanan Ders Konuları

| Hafta | Konu | Proje Bölümü |
|-------|------|-------------|
| 2 | ML Temelleri (i.i.d., Generalization, Bias-Variance) | Veri seti, Bölüm 10 |
| 2 | MLE ↔ Loss Function | Cross-entropy = Categorical MLE |
| 2 | SGD ve Minibatch | Tüm eğitim döngüleri |
| 2 | Regularization temelleri (L1, L2) | Deney 3, Bölüm 11 |
| 3-4 | Feedforward Networks | Model mimarisi |
| 3-4 | Softmax, Cross-Entropy | Loss function |
| 3-4 | Backpropagation | Eğitim döngüsü, Gradyan analizi |
| 3-4 | Aktivasyon Fonksiyonları | **Ana konu** — tüm deneyler |
| 4 | L2 Weight Decay | Deney 3 |
| 4 | L1 Lasso | Bölüm 11 |
| 4 | Dropout | Deney 3 |
| 4 | Label Smoothing | Deney 3 |
| 4 | Batch Normalization | Deney 3 |
| 5 | Initialization (Xavier, He) | Deney 4 |
| 5 | Optimizers (SGD, Momentum, RMSProp, Adam) | Deney 2 |
| 5 | Vanishing/Exploding Gradients | Gradyan akış analizi |
