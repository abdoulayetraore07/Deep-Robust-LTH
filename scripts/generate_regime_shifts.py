"""
Script to generate regime shift data
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.heston import HestonSimulator


def main():
    parser = argparse.ArgumentParser(description='Generate regime shift data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    n_paths = config['regime_shifts']['n_paths_per_regime']
    T = config['data']['T']
    n_steps = config['data']['n_steps']
    K = config['data']['heston']['K']
    
    output_dir = Path('data/processed')
    
    # High volatility regime
    print("Generating high volatility regime...")
    params_high = config['data']['heston'].copy()
    params_high['theta'] = config['regime_shifts']['high_vol']['theta']
    params_high['xi'] = config['regime_shifts']['high_vol']['xi']
    
    sim_high = HestonSimulator(params_high)
    S_high, v_high = sim_high.simulate(n_paths, T, n_steps, seed=100)
    Z_high = np.maximum(S_high[:, -1] - K, 0)
    
    regime_dir = output_dir / 'regime_high_vol'
    regime_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(regime_dir / 'S.npy', S_high)
    np.save(regime_dir / 'v.npy', v_high)
    np.save(regime_dir / 'Z.npy', Z_high)
    
    print(f"  Saved to {regime_dir}/")
    
    # Extreme regime
    print("Generating extreme regime...")
    params_extreme = config['data']['heston'].copy()
    params_extreme['theta'] = config['regime_shifts']['extreme']['theta']
    params_extreme['xi'] = config['regime_shifts']['extreme']['xi']
    
    sim_extreme = HestonSimulator(params_extreme)
    S_extreme, v_extreme = sim_extreme.simulate(n_paths, T, n_steps, seed=200)
    Z_extreme = np.maximum(S_extreme[:, -1] - K, 0)
    
    regime_dir = output_dir / 'regime_extreme'
    regime_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(regime_dir / 'S.npy', S_extreme)
    np.save(regime_dir / 'v.npy', v_extreme)
    np.save(regime_dir / 'Z.npy', Z_extreme)
    
    print(f"  Saved to {regime_dir}/")
    print("Regime shift data generation complete")


if __name__ == '__main__':
    main()