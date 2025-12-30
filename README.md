# Adversarially-Robust Sparse Deep Hedging

**Finding Lottery Tickets that Survive Market Stress**

---

## Project Overview

This project investigates the intersection of three cutting-edge machine learning concepts:

1. **Deep Hedging** - Using deep reinforcement learning for portfolio hedging
2. **Lottery Ticket Hypothesis** - Finding sparse subnetworks that train as well as dense networks
3. **Adversarial Robustness** - Testing model resilience under market perturbations

**Research Questions:**
- Do boosting tickets exist in deep hedging networks?
- Are sparse networks robust to market perturbations (price + volatility)?
- Can we combine sparsity + robustness efficiently?

**Key Contribution:**
- First application of boosting tickets to Deep Hedging
- 40-50% time savings for adversarial training
- Demonstration that sparse robust tickets are achievable with proper protocol

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Deep_Robust_LTH_project

# Install dependencies
pip install -r requirements.txt
```

### Generate Data
```bash
# Generate Heston paths (70,000 paths)
python scripts/generate_data.py --config config.yaml
```

### Train Baseline
```bash
# Train dense baseline model
python scripts/train_baseline.py --config config.yaml
```

### Run Full Pipeline

See notebooks in order:
1. `00_data_exploration.ipynb` - Validate Heston simulation
2. `01_baseline_deep_hedging.ipynb` - Train baseline
3. `02_lottery_tickets.ipynb` - Discover boosting tickets
4. `03_adversarial_attacks.ipynb` - Test robustness
5. `04_regime_shifts.ipynb` - Out-of-distribution testing
6. `05_adversarial_training.ipynb` - Train robust tickets
7. `06_interpretation.ipynb` - Feature analysis

---

## Project Structure
```
Deep_Robust_LTH_project/
├── config.yaml                 # Centralized configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── raw/                    # (empty, simulation only)
│   └── processed/              # Generated Heston paths
│
├── src/
│   ├── data/                   # Heston simulation + features
│   ├── models/                 # MLP architecture + losses
│   ├── pruning/                # Magnitude pruning + masking
│   ├── attacks/                # FGSM + PGD attacks
│   ├── evaluation/             # Metrics + baselines
│   └── utils/                  # Config + logging + viz
│
├── notebooks/                  # Jupyter notebooks (experiments)
├── scripts/                    # CLI scripts
├── experiments/                # Saved results
├── figures/                    # Publication-ready figures
├── report/                     # LaTeX report
└── tests/                      # Unit tests
```

---

## Key Design Decisions

### Architecture
- **Type**: MLP (not LSTM) - confirmed by Buehler et al. (2019)
- **Size**: 512-512-256 (~400k parameters)
- **Regularization**: Dropout (0.2, 0.2, 0.1), BatchNorm, Weight Decay (1e-5)

### Features (8 features)
1. log(S_t / K) - Log-moneyness
2. (S_t - S_{t-1})/S_{t-1} - Return
3. √v_t - Volatility
4. v_t - v_{t-1} - Variance change
5. (T - t) / T - Time to maturity
6. δ_{t-1} - Previous position
7. |δ_t - δ_{t-1}| - Trading volume
8. PnL_{t-1} - Cumulative P&L

### Adversarial Perturbations
- **Type**: Combined price + volatility
- **Norms**: L∞ bounded on both dimensions
- **Epsilons**:
  - FGSM: ε_S=0.02, ε_v=0.2
  - PGD: ε_S=0.05, ε_v=0.5
  - Stress: ε_S=0.10, ε_v=1.0

### Method: Boosting Tickets
- **Phase 1**: FGSM training (100 epochs, LR=0.01)
- **Pruning**: One-shot 80% (magnitude-based)
- **Phase 2**: PGD retraining (40-70 epochs, warmup LR 0.01→0.1)

---

## Expected Results

**Baseline vs Boosting Tickets:**
- Convergence speed: 2-3× faster
- Sparsity: 80% weights pruned
- Performance: Comparable to dense network

**Standard vs Robust Tickets:**
- Robustness gap: Decreases with adversarial training
- Time savings: 40-50% vs dense PGD baseline
- Robustness: Matches or exceeds baseline

---

## References

1. Buehler et al. (2019) - Deep Hedging
2. Frankle & Carbin (2019) - Lottery Ticket Hypothesis
3. Li et al. (2020) - Boosting Tickets for Adversarial Training
4. Madry et al. (2018) - Towards Deep Learning Models Resistant to Adversarial Attacks
5. Ilyas et al. (2019) - Adversarial Examples Are Not Bugs, They Are Features

---

## Team

- Abdoulaye TRAORE
- Franck Wilson KOUASSI
- Tingjia ZHANG

**Supervisor**: Prof. Austin .J

**Deadline**: January 16, 2025

---

## License

This project is for academic research purposes.

---

## Acknowledgments

- Papers with Code for reproducibility standards
- PyTorch team for excellent framework
- Anthropic Claude for assistance