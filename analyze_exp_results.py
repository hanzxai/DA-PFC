import pickle
import pathlib
import numpy as np
import torch
from analysis.analyzer import PFCAnalyzer

def analyze_results(output_dir):
    save_dir = pathlib.Path(output_dir)
    print(f"Analyzing results in: {save_dir}")
    
    try:
        with open(save_dir / 'raw_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    analyzer = PFCAnalyzer(data)
    da_onset = analyzer.da_onset
    print(f"DA Onset: {da_onset} ms")
    
    # Define time windows
    # Baseline: 0 to da_onset
    # Early Post-DA: da_onset to da_onset + 20000 (20s)
    # Late Post-DA: end - 20000 to end
    
    total_time = analyzer.duration
    print(f"Total Time: {total_time} ms")
    
    windows = {
        "Baseline": (0, da_onset),
        "Early Post-DA": (da_onset, da_onset + 20000),
        "Late Post-DA": (total_time - 20000, total_time)
    }
    
    groups = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']
    
    print("\n--- Average Firing Rates (Hz) ---")
    print(f"{'Group':<10} | {'Baseline':<10} | {'Early Post':<10} | {'Late Post':<10} | {'Change (Late vs Base)':<20}")
    print("-" * 70)
    
    results = {}
    
    for grp in groups:
        if grp not in analyzer.groups:
            continue
            
        # Get spike times for the group
        # analyzer.compute_group_rate returns (centers, rate)
        # We can use compute_group_rate but need to average over specific windows
        
        # Let's use raw spikes for accurate mean rate
        # spikes is a list of tensors or arrays? 
        # analyzer.spikes is [batch_idx][neuron_idx] -> spike_times
        
        # Actually, using compute_group_rate is easier as it gives time course
        centers, rate = analyzer.compute_group_rate(batch_idx=1, group_name=grp, time_win=1000.0) # 1s bin for stability
        
        if rate is None:
            continue
            
        # Helper to get mean rate in window
        def get_mean_rate(t_start, t_end):
            mask = (centers >= t_start) & (centers < t_end)
            if np.sum(mask) == 0:
                return 0.0
            return np.mean(rate[mask])
            
        base_rate = get_mean_rate(*windows["Baseline"])
        early_rate = get_mean_rate(*windows["Early Post-DA"])
        late_rate = get_mean_rate(*windows["Late Post-DA"])
        
        change = (late_rate - base_rate) / base_rate * 100 if base_rate > 0 else 0.0
        
        print(f"{grp:<10} | {base_rate:<10.2f} | {early_rate:<10.2f} | {late_rate:<10.2f} | {change:>+6.1f}%")
        
        results[grp] = {
            "base": base_rate,
            "late": late_rate,
            "change": change
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp_results.py <exp_dir>")
        print("  e.g. python analyze_exp_results.py outputs/exp_2026-03-10_10-40-25")
        sys.exit(1)
    analyze_results(sys.argv[1])
