# Adversarially-Robust Sparse Deep Hedging

**Finding Lottery Tickets that Survive Market Stress**

Research project combining:
- **Deep Hedging** (Buehler et al., 2019)
- **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019)
- **Adversarial Robustness** (Madry et al., 2017)

## Project Structure

```
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── run_baseline.py          # Baseline Deep Hedging
├── run_lth.py               # Lottery Ticket experiments
├── run_adversarial.py       # Adversarial training
├── run_full_experiment.py   # Complete pipeline
│
└── src/
    ├── data/
    │   ├── heston.py        # Heston model simulation
    │   └── preprocessor.py  # Feature engineering (5 features)
    │
    ├── models/
    │   ├── deep_hedging.py  # Network with temporal loop
    │   ├── losses.py        # OCE CVaR loss (corrected)
    │   └── trainer.py       # Training infrastructure
    │
    ├── pruning/
    │   ├── magnitude.py     # Iterative Magnitude Pruning
    │   ├── masks.py         # Mask utilities
    │   └── rewind.py        # Weight rewinding
    │
    ├── attacks/
    │   ├── fgsm.py          # FGSM attack
    │   ├── pgd.py           # PGD attack
    │   └── adversarial_trainer.py  # Madry protocol
    │
    ├── evaluation/
    │   ├── baselines.py     # Delta hedging baseline
    │   └── metrics.py       # Evaluation metrics
    │
    └── utils/
        └── helpers.py       # Utility functions
```

## Critical Corrections Applied

### 1. OCE "Free Lunch" Fix (losses.py)
**Problem**: P&L included premium `y`, allowing network to cheat by pushing y→∞

**Solution**: 
```python
# BEFORE (wrong)
pnl = y - Z + trading_pnl - costs

# AFTER (correct)
pnl_naked = -Z + trading_pnl - costs  # No y!
loss = cvar_loss(pnl_naked, y, alpha)  # y added separately in OCE
```

### 2. Temporal Architecture Fix (deep_hedging.py)
**Problem**: Global feed-forward, no memory of previous action

**Solution**: Explicit temporal loop with delta_prev as input
```python
for t in range(n_steps):
    input_t = [market_features_t, delta_{t-1}]
    delta_t = network(input_t)
```

### 3. Feature Fix (preprocessor.py)
**Problem**: 8 features including delta_prev/pnl_prev computed before training

**Solution**: Only 5 exogenous features; delta_prev computed in model loop

### 4. LTH Learning Rate Fix (config.yaml)
**Problem**: Same LR throughout

**Solution**: 
- `initial_lr = 1e-4` (small, for pruning)
- `retrain_lr = 1e-2` (100x larger, for retraining)

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run baseline
```bash
python run_baseline.py --config config.yaml
```

### 3. Run LTH experiments
```bash
python run_lth.py --config config.yaml
```

### 4. Run adversarial training
```bash
python run_adversarial.py --config config.yaml
```

### 5. Run full pipeline
```bash
python run_full_experiment.py --config config.yaml
```

## Expected Results

After training:
- `y` (learned premium) ≈ 1.6 (close to Black-Scholes price)
- Sharpe ratio > 0
- CVaR(5%) improvement over Delta Hedging

For Lottery Tickets:
- Sparse networks (up to 90% sparsity) matching dense performance
- "Winning tickets" that retrain successfully

For Adversarial Training:
- Reduced gap between clean and adversarial performance
- PGD-trained models more robust than naturally trained

## References

1. Buehler, H. et al. (2019). "Deep Hedging"
2. Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis"
3. Madry, A. et al. (2017). "Towards Deep Learning Models Resistant to Adversarial Attacks"
4. Li, Y. et al. (2020). "Boosting Adversarial Training with Hypersphere Embedding"

## Team

- Abdoul (Lead)
- Franck Wilson Kouassi
- Tingjia Zhang

Supervised by Professor Champonnois
