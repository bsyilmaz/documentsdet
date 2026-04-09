# REPORT: Effect of Activation Functions on Training Dynamics

**Course:** SWE012 – Deep Learning with Python
**Instructor:** Asst. Prof. Yigit Bekir Kaya
**University:** Istinye University — Department of Computer Engineering

## Team Members

| Name | Student ID |
|------|-----------|
| Selvinaz Sayın | 220901755 |
| Ege Karaurgan | 229910141 |
| Vedat Efe Gezer | 229910158 |
| Mehmet Emin Akkaya | 2309011036 |
| Bayram Selim Yılmaz | 2309011053 |

---

## 1. Project Summary

This project investigates the effect of activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU) on neural network training dynamics through controlled experiments. Activation functions constitute the **depth** dimension of the project, while all methodologies covered in the lectures (optimization, regularization, initialization) are integrated as the **breadth** dimension.

**Dataset:** Fashion-MNIST (60,000 train, 10,000 test, 10 classes, 28×28 grayscale)
**Framework:** PyTorch
**Approach:** Controlled experiments where only one variable is changed while all others are held constant

---

## 2. Applied Methodologies

### 2.1 Activation Functions (Main Topic — Depth)

| Function | Formula | Advantage | Disadvantage |
|----------|---------|-----------|-------------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | Probability output [0,1] | Vanishing gradient, non-zero-centered output |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Zero-centered [-1,1] | Vanishing gradient (saturation regions) |
| ReLU | max(0, x) | Fast computation, gradient=1 (x>0) | Dead neuron problem |
| Leaky ReLU | max(0.01x, x) | Mitigates dead neuron issue | Choice of negative slope constant |

**Rationale:** These four functions represent the key evolutionary timeline of deep learning. The progression Sigmoid/Tanh (classical) → ReLU (modern standard) → Leaky ReLU (improvement) demonstrates how the vanishing gradient problem has been addressed over time.

### 2.2 Optimization Methods (Breadth)

| Optimizer | Mechanism | Hyperparameters |
|-----------|-----------|----------------|
| SGD | w -= lr × ∇J | lr=0.01 |
| SGD + Momentum | Velocity accumulation: v = βv + ∇J | lr=0.01, β=0.9 |
| RMSProp | Per-parameter adaptive lr (exp. moving avg.) | lr=0.001 |
| Adam | Momentum + RMSProp + bias correction | lr=0.001, β₁=0.9, β₂=0.999 |

**Rationale:** The evolution from SGD to Adam reflects the progression of optimization algorithms. Adam was selected as the baseline optimizer as it is the industry standard.

### 2.3 Regularization Methods (Breadth)

| Method | Hyperparameter | Tuning | Bayesian Interpretation |
|--------|---------------|--------|------------------------|
| L2 Weight Decay | α = 1e-4 | Selected from {1e-3, 1e-4, 1e-5} | Gaussian prior |
| L1 Lasso | λ = 1e-5 | Selected from {1e-4, 1e-5, 1e-6} | Laplace prior |
| Dropout | p = 0.5 (drop) | Standard hidden layer rate | Ensemble approach |
| Batch Normalization | γ, β (learnable) | Automatically learned | Reduces internal covariate shift |
| Label Smoothing | ε = 0.1 | Standard value (range: 0.05-0.2) | Calibration improvement |

**Why these methods?** Each targets a different overfitting mechanism:
- L2: Constrains weight magnitudes
- L1: Zeros out irrelevant connections (feature selection)
- Dropout: Creates a sub-network ensemble
- BatchNorm: Prevents inter-layer distribution shift
- Label Smoothing: Prevents overconfident predictions

### 2.4 Initialization Strategies (Breadth)

| Method | Variance | Suitable Activation |
|--------|---------|-------------------|
| Xavier (Glorot) | 2/(n_in + n_out) | Sigmoid, Tanh |
| He (Kaiming) | 2/n_in | ReLU, Leaky ReLU |
| Random (σ=0.5) | Uncontrolled | — (baseline) |

**Rationale:** Xavier and He are theoretically derived optimal strategies. Random initialization was used as a control group to demonstrate the effect of improper initialization.

---

## 3. Experimental Design

### Controlled Experiment Principle
In each experiment, **only one variable** was changed while all other parameters were held constant:

| Experiment | Variable | Held Constant |
|-----------|----------|--------------|
| Exp 1: Activation Comparison | Activation function | Adam, He init, no reg, 15 epochs |
| Exp 2: Optimizer Interaction | Optimizer × Activation | He init, no reg, 15 epochs |
| Exp 3: Regularization Interaction | Reg. method × Activation | Adam, He init, 15 epochs |
| Exp 4: Initialization Interaction | Init method × Activation | Adam, no reg, 15 epochs |

### Simultaneously Applied Methods
- **In every experiment:** SGD minibatch (batch_size=128), CrossEntropyLoss (Softmax + NLL), backpropagation
- **In Experiment 3:** BatchNorm + Activation together (BN → Activation order)
- **In Experiment 3:** Dropout + Activation together (Activation → Dropout order)

### Hyperparameter Tuning Process
- **Learning rate:** 0.001 for Adam (standard), 0.01 for SGD (higher rate needed)
- **Batch size:** 128 (balance between SGD noise and computational cost)
- **Hidden layers:** [256, 128] (2 layers — sufficient capacity without excessive depth)
- **Epochs:** 15 (sufficient for convergence, suitable for observing overfitting)
- **Weight decay:** Grid search over {1e-3, 1e-4, 1e-5} → 1e-4 optimal
- **L1 lambda:** Grid search over {1e-4, 1e-5, 1e-6} → 1e-5 optimal
- **Dropout rate:** p=0.5 (hidden layer standard, Hinton et al. recommendation)
- **Label smoothing:** ε=0.1 (standard, Szegedy et al. recommendation)
- **Seed:** 42 (same across all experiments → reproducibility)

---

## 4. Performance Comparison

### 4.1 Experiment 1: Activation Function Comparison

| Activation | Convergence Speed | Test Accuracy | Gradient Flow |
|------------|------------------|---------------|--------------|
| Sigmoid | Slowest | Lowest | Vanishing (≈0 at input layers) |
| Tanh | Slow | Medium | Vanishing (better than Sigmoid) |
| ReLU | Fast | High | Stable (gradient=1) |
| Leaky ReLU | Fast | High | Most stable (no dead neurons) |

### 4.2 Experiment 2: Optimizer Interaction

| Combination | Observation |
|-------------|------------|
| Sigmoid + SGD | Worst case: vanishing gradient + fixed lr |
| Sigmoid + Adam | Adam's adaptive lr partially compensates for Sigmoid's slowness |
| ReLU + SGD | Strong gradient flow allows even SGD to perform adequately |
| ReLU + Adam | Most stable and reliable performance |

### 4.3 Experiment 3: Regularization Interaction

| Combination | Observation |
|-------------|------------|
| Sigmoid + BatchNorm | **Most dramatic improvement** — BN largely resolves the saturation issue |
| ReLU + Dropout | Generalization gap decreases, training slows down |
| L1 | Creates sparsity in weights (feature selection) |
| L2 | Shrinks all weights but does not zero them out |
| Label Smoothing | Prevents overconfident predictions, improving calibration |

### 4.4 Experiment 4: Initialization Interaction

| Combination | Observation |
|-------------|------------|
| Sigmoid + Xavier | Correct pairing — signal variance is preserved |
| ReLU + He | Correct pairing — He compensates for ReLU halving the variance |
| Any + Random | Uncontrolled variance → unstable training |

---

## 5. Additional Analyses

### 5.1 Gradient Flow Analysis
Gradient norm variation across layers was measured in a 5-layer deep network at initialization:
- **Sigmoid/Tanh:** Gradient norm drops logarithmically toward the input layers (vanishing)
- **ReLU/Leaky ReLU:** Gradient norm remains approximately constant across layers

### 5.2 Dead Neuron Analysis
After 10 epochs of training, the ratio of neurons with consistently zero output was measured:
- **ReLU:** A certain proportion of dead neurons is observed
- **Leaky ReLU:** Dead neuron ratio is significantly reduced thanks to the negative slope (0.01)

### 5.3 Bias-Variance Perspective
For each activation function, the generalization gap between train and test loss was computed to evaluate the overfitting tendency.

---

## 6. Dataset Assessment

**Fashion-MNIST** was selected for the following reasons:
- **Sufficient complexity:** More challenging than MNIST (clothing classification), making activation function differences more apparent
- **Standard benchmark:** Results can be compared with the literature
- **Reasonable size:** 60K train / 10K test — suitable for overfitting analysis
- **i.i.d. compliance:** Train and test sets are independently sampled from the same distribution
- **10 classes:** Multi-class classification → Softmax + CCE is the natural choice

---

## 7. Conclusions and Practical Recommendations

1. **Default configuration:** ReLU + He init + Adam + BatchNorm — safe and strong starting point
2. **If dead neuron issue arises:** Switch to Leaky ReLU or ELU
3. **If overfitting occurs:** Dropout + L2 combination is effective
4. **If Sigmoid/Tanh is required:** BatchNorm must be added and Adam optimizer should be used
5. **If feature selection is needed:** L1 regularization is preferred

---

## 8. Course Topics Covered

| Week | Topic | Where Used in Project |
|------|-------|----------------------|
| 2 | ML Fundamentals (i.i.d., Generalization, Bias-Variance) | Dataset section, Section 10 |
| 2 | MLE ↔ Loss Function connection | Cross-entropy = Categorical MLE |
| 2 | SGD and Minibatch | All training loops |
| 2 | Regularization basics (L1, L2) | Experiment 3, Section 11 |
| 3-4 | Feedforward Networks | Model architecture |
| 3-4 | Softmax, Cross-Entropy | Loss function |
| 3-4 | Backpropagation | Training loop, Gradient analysis |
| 3-4 | Activation Functions (Sigmoid, Tanh, ReLU, Leaky ReLU) | **Main topic** — all experiments |
| 4 | L2 Weight Decay | Experiment 3 |
| 4 | L1 Lasso | Section 11 |
| 4 | Dropout | Experiment 3 |
| 4 | Label Smoothing | Experiment 3 |
| 4 | Batch Normalization | Experiment 3 |
| 5 | Initialization (Xavier, He) | Experiment 4 |
| 5 | Optimizers (SGD, Momentum, RMSProp, Adam) | Experiment 2 |
| 5 | Vanishing/Exploding Gradients | Gradient flow analysis |
