# Stabilized Lottery Tickets for Robust Deep Hedging

Research project investigating sparse neural networks for robust derivative hedging under transaction costs and stochastic volatility.

## Overview

This project combines three methodologies:
- **Deep Hedging** (Buehler et al., 2019): Neural network policies for derivative hedging
- **Lottery Ticket Hypothesis** (Frankle & Carbin, 2019): Sparse subnetwork discovery via magnitude pruning
- **Adversarial Robustness** (Madry et al., 2018): Robustness to input perturbations via adversarial training

## Project Structure

```
.
├── config.yaml                 # Main experimental configuration
├── config_sanity.yaml          # Black-Scholes sanity check configuration
├── requirements.txt            # Python dependencies
│
├── train_baseline.py           # Dense baseline training
├── train_pruning.py            # Lottery ticket experiments
├── train_adversarial.py        # Adversarial training (FGSM -> PGD)
├── evaluate.py                 # Model evaluation
├── generate_all_figures.py     # Figure generation
│
├── heston.py                   # Heston stochastic volatility simulation (IG scheme)
├── preprocessor.py             # Feature engineering (5 exogenous features)
├── deep_hedging.py             # Policy network with temporal loop
├── losses.py                   # OCE/CVaR loss formulation
├── trainer.py                  # Training infrastructure
├── pruning.py                  # Magnitude pruning and late rewinding
├── fgsm.py                     # Fast Gradient Sign Method attack
├── pgd.py                      # Projected Gradient Descent attack
├── adversarial_trainer.py      # FGSM -> PGD curriculum training
├── baselines.py                # Delta hedging baselines
├── metrics.py                  # Evaluation metrics (CVaR, Sharpe, etc.)
├── visualization.py            # Plotting utilities
├── config.py                   # Configuration management
└── logging.py                  # Logging utilities
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, Matplotlib, PyYAML, tqdm

## Usage

### 1. Train dense baseline
```bash
python train_baseline.py --config config.yaml
```

### 2. Run lottery ticket experiments
```bash
python train_pruning.py --config config.yaml
```

### 3. Run adversarial training
```bash
python train_adversarial.py --config config.yaml
```

### 4. Evaluate models
```bash
python evaluate.py --config config.yaml
```

### 5. Generate figures
```bash
python generate_all_figures.py --config config.yaml
```

## Methodology

### Market Model
- Heston stochastic volatility under risk-neutral measure
- Inverse Gaussian discretization scheme for variance process
- European call option hedging with proportional transaction costs (10 bps)

### Network Architecture
- MLP [256, 256, 128] with ReLU activations
- Temporal loop with explicit state variables (delta_prev, cumulative P&L)
- 5 exogenous input features + 3 recurrent state variables

### Lottery Ticket Protocol
- Global unstructured magnitude pruning
- Late rewinding to epoch 30 (stabilized tickets)
- Sparsity levels: {50%, 80%, 90%, 95%}

### Adversarial Training
- Two-phase curriculum: FGSM (150 epochs) followed by PGD (50 epochs)
- Learning rate warmup for PGD phase
- Perturbations on exogenous features only

## Key Results

| Metric | Dense Baseline | Ticket 80% | Robust Ticket 80% |
|--------|----------------|------------|-------------------|
| Parameters | 101,122 | 20,224 | 20,224 |
| Clean CVaR | -3.00 | -3.00 | -3.08 |
| PGD CVaR | -11.81 | -11.88 | -3.08 |
| Training speedup | 1x | 124x | - |

### Main Findings
1. **Stabilized tickets** require late rewinding (epoch 30) for stable retraining
2. **5x compression** (80% sparsity) with <5% CVaR degradation
3. **124x training speedup** from magnitude-based mask + rewound weights
4. **Standard tickets are adversarially vulnerable**; robust tickets close the gap
5. **Robust tickets generalize better** under regime shifts (-257.7% vs -259.9% degradation)

## Configuration

Key parameters in `config.yaml`:

```yaml
heston:
  S0: 100.0
  K: 100.0
  T: 0.0833  # 1 month
  r: 0.02
  v0: 0.0175
  kappa: 1.5768
  theta: 0.0398
  xi: 0.5751
  rho: -0.5711

training:
  n_epochs: 300
  batch_size: 512
  learning_rate: 0.001
  early_stopping_patience: 30

pruning:
  sparsities: [0.5, 0.8, 0.9, 0.95]
  rewind_epoch: 30

adversarial:
  fgsm:
    epsilon_S: 0.02
    epsilon_v: 0.2
  pgd:
    epsilon_S: 0.05
    epsilon_v: 0.5
    steps: 10
```

## References

1. Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271-1291.

2. Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *ICLR*.

3. Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2020). Stabilizing the Lottery Ticket Hypothesis. *arXiv preprint*.

4. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.

5. Li, Y. et al. (2020). Boosting Tickets: Pruning and Adversarial Training are Complementary. *arXiv preprint*.

6. Abi Jaber, E. (2024). Simulation of Square-Root Processes Made Simple. *arXiv preprint*.

## Authors

- Abdoulaye Traore (ENSTA & ENSAE Paris)
- Tingjia Zhang (ENSTA & ENSAE Paris)
- Franck Wilson Kouassi (ENSTA & ENSAE Paris)

Master's students in Statistics, Finance & Actuarial Science, Institut Polytechnique de Paris.

Supervised by Professor Champonnois.

## License

This project was developed for academic purposes as part of a course project.
