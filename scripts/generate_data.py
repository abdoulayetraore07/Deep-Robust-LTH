"""
Script to generate Heston simulation data
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.heston import HestonSimulator


def main():
    parser = argparse.ArgumentParser(description='Generate Heston simulation data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Extract parameters
    data_config = config['data']
    heston_params = data_config['heston']
    
    n_train = data_config['n_train']
    n_val = data_config['n_val']
    n_test = data_config['n_test']
    T = data_config['T']
    n_steps = data_config['n_steps']
    dt = data_config['dt']
    
    seed = config['random_seed']
    
    # Create simulator
    print("Initializing Heston simulator...")
    simulator = HestonSimulator(heston_params)
    
    # Generate training data
    print(f"Generating {n_train} training paths...")
    S_train, v_train = simulator.simulate(n_train, T, n_steps, seed=seed)
    Z_train = np.maximum(S_train[:, -1] - heston_params['K'], 0)
    
    # Validate
    print("Validating simulation...")
    validation_results = simulator.validate_moments(v_train, T)
    print(f"  Mean v_T - Theoretical: {validation_results['E_v_T_theoretical']:.6f}, Empirical: {validation_results['E_v_T_empirical']:.6f}")
    print(f"  Var v_T - Theoretical: {validation_results['Var_v_T_theoretical']:.6f}, Empirical: {validation_results['Var_v_T_empirical']:.6f}")
    print(f"  Min variance: {validation_results['min_v']:.6f}")
    print(f"  Positivity check: {validation_results['positivity_pass']}")
    
    if not validation_results['E_v_T_pass']:
        print("WARNING: Mean validation failed")
    if not validation_results['Var_v_T_pass']:
        print("WARNING: Variance validation failed")
    if not validation_results['positivity_pass']:
        print("ERROR: Positivity check failed")
        return
    
    # Generate validation data
    print(f"Generating {n_val} validation paths...")
    S_val, v_val = simulator.simulate(n_val, T, n_steps, seed=seed+1)
    Z_val = np.maximum(S_val[:, -1] - heston_params['K'], 0)
    
    # Generate test data
    print(f"Generating {n_test} test paths...")
    S_test, v_test = simulator.simulate(n_test, T, n_steps, seed=seed+2)
    Z_test = np.maximum(S_test[:, -1] - heston_params['K'], 0)
    
    # Save data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving data...")
    np.save(output_dir / 'S_train.npy', S_train)
    np.save(output_dir / 'v_train.npy', v_train)
    np.save(output_dir / 'Z_train.npy', Z_train)
    
    np.save(output_dir / 'S_val.npy', S_val)
    np.save(output_dir / 'v_val.npy', v_val)
    np.save(output_dir / 'Z_val.npy', Z_val)
    
    np.save(output_dir / 'S_test.npy', S_test)
    np.save(output_dir / 'v_test.npy', v_test)
    np.save(output_dir / 'Z_test.npy', Z_test)
    
    print(f"Data saved to {output_dir}/")
    
    print("Statistics:")
    print(f"  Training: S in [{S_train.min():.2f}, {S_train.max():.2f}], v in [{v_train.min():.6f}, {v_train.max():.6f}]")
    print(f"  Mean payoff: {Z_train.mean():.4f}")
    print("Data generation complete")


if __name__ == '__main__':
    main()