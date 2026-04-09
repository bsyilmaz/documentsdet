# Effect of Activation Functions on Training Dynamics

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

## About

This project investigates the effect of four fundamental activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU) on neural network training dynamics through controlled experiments. Activation functions form the **depth** dimension of the project, while optimization, regularization, and initialization methodologies covered in the lectures are integrated as the **breadth** dimension.

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Activation Comparison | Sigmoid, Tanh, ReLU, Leaky ReLU — all other parameters held constant |
| 2 | Activation × Optimizer | SGD, Momentum, RMSProp, Adam interaction |
| 3 | Activation × Regularization | L2, L1, Dropout, BatchNorm, Label Smoothing interaction |
| 4 | Activation × Initialization | Xavier, He, Random initialization strategies |
| + | Gradient Flow Analysis | Vanishing gradient visualization in a 5-layer network |
| + | Dead Neuron Analysis | ReLU vs Leaky ReLU dead neuron comparison |

## Technologies

- **Python 3.10+**
- **PyTorch** — model definition and training
- **Matplotlib** — visualization
- **NumPy** — numerical computations

## Dataset

**Fashion-MNIST:** 60,000 train / 10,000 test, 10 classes, 28×28 grayscale images

## Usage

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the notebook
jupyter notebook activation_functions_training_dynamics.ipynb
```

## Repository Structure

```
├── README.md                                        # This file
├── REPORT.md                                        # Detailed project report
├── activation_functions_training_dynamics.ipynb      # Main notebook (code + analysis)
├── responsibilities/                                # Individual responsibility files
│   ├── 220901755.md                                 # Selvinaz Sayın
│   ├── 229910141.md                                 # Ege Karaurgan
│   ├── 229910158.md                                 # Vedat Efe Gezer
│   ├── 2309011036.md                                # Mehmet Emin Akkaya
│   └── 2309011053.md                                # Bayram Selim Yılmaz
├── allanoucements.txt                               # Course announcements
└── Deep Learning_merged.txt                         # Course study guide
```

## License

This project was developed as part of the SWE012 course at Istinye University.
