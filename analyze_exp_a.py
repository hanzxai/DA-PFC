#!/usr/bin/env python3
"""Quick analysis of Experiment A results."""
import numpy as np
import json

result_dir = "outputs/exp_a_pulse_2026-03-31_22-01-25"

# Load JSON results
with open(f"{result_dir}/exp_a_results.json") as f:
    results = json.load(f)

# Load alpha traces
data = np.load(f"{result_dir}/alpha_traces.npz")
times = data['times']
a_d1 = data['alpha_d1']  # (n_records, 2) -> [ctrl, exp]
a_d2 = data['alpha_d2']

with open(f"{result_dir}/analysis_detail.txt", 'w') as out:
    out.write("=== Experiment A: DA Pulse Response — Detailed Analysis ===\n\n")
    out.write(f"Alpha traces shape: times={times.shape}, alpha_d1={a_d1.shape}\n")
    out.write(f"Time range: {times[0]:.1f} - {times[-1]:.1f} s, dt={times[1]-times[0]:.4f} s\n\n")

    out.write("=== Key Time Points ===\n")
    for label, t_start, t_end in [
        ("Pre-pulse baseline (15-20s)", 15, 20),
        ("Pulse peak (45-50s)", 45, 50),
        ("Just after pulse (50-55s)", 50, 55),
        ("30s after pulse (78-82s)", 78, 82),
        ("100s after pulse (148-152s)", 148, 152),
        ("200s after pulse (248-252s)", 248, 252),
        ("End (295-300s)", 295, 300),
    ]:
        mask = (times >= t_start) & (times < t_end)
        if np.any(mask):
            out.write(f"\n{label}:\n")
            out.write(f"  alpha_D1: ctrl={a_d1[mask,0].mean():.4f}, exp={a_d1[mask,1].mean():.4f}\n")
            out.write(f"  alpha_D2: ctrl={a_d2[mask,0].mean():.4f}, exp={a_d2[mask,1].mean():.4f}\n")
            out.write(f"  D1-D2 (exp): {(a_d1[mask,1]-a_d2[mask,1]).mean():.4f}\n")

    out.write("\n\n=== D1 Afterglow Analysis ===\n")
    post_mask = times >= 50
    exp_d1_post = a_d1[post_mask, 1]
    exp_d2_post = a_d2[post_mask, 1]
    times_post = times[post_mask]
    afterglow = exp_d1_post > exp_d2_post * 1.1
    if np.any(afterglow):
        ag_start = times_post[afterglow][0]
        ag_end = times_post[afterglow][-1]
        out.write(f"Afterglow window: {ag_start:.1f}s - {ag_end:.1f}s (duration: {ag_end-ag_start:.1f}s)\n")
        idx_start = np.where(afterglow)[0][0]
        out.write(f"At afterglow start ({ag_start:.1f}s): D1={exp_d1_post[idx_start]:.4f}, D2={exp_d2_post[idx_start]:.4f}\n")
        diff = exp_d1_post - exp_d2_post
        max_diff_idx = np.argmax(diff)
        out.write(f"Max D1-D2 diff at t={times_post[max_diff_idx]:.1f}s: D1={exp_d1_post[max_diff_idx]:.4f}, D2={exp_d2_post[max_diff_idx]:.4f}, diff={diff[max_diff_idx]:.4f}\n")

    # Half-decay times
    d2_at_offset = a_d2[np.argmin(np.abs(times-50)), 1]
    d2_baseline = a_d2[(times >= 15) & (times < 20), 1].mean()
    d2_half = (d2_at_offset + d2_baseline) / 2
    d2_post = a_d2[post_mask, 1]
    half_mask = d2_post <= d2_half
    if np.any(half_mask):
        d2_half_time = times_post[half_mask][0]
        out.write(f"\nD2 half-decay time: {d2_half_time - 50:.1f}s after pulse offset\n")
        out.write(f"  (D2 at offset={d2_at_offset:.4f}, baseline={d2_baseline:.4f}, half={d2_half:.4f})\n")

    d1_at_offset = a_d1[np.argmin(np.abs(times-50)), 1]
    d1_baseline = a_d1[(times >= 15) & (times < 20), 1].mean()
    d1_half = (d1_at_offset + d1_baseline) / 2
    d1_post = a_d1[post_mask, 1]
    half_mask_d1 = d1_post <= d1_half
    if np.any(half_mask_d1):
        d1_half_time = times_post[half_mask_d1][0]
        out.write(f"D1 half-decay time: {d1_half_time - 50:.1f}s after pulse offset\n")
        out.write(f"  (D1 at offset={d1_at_offset:.4f}, baseline={d1_baseline:.4f}, half={d1_half:.4f})\n")

    out.write(f"\nD1 peak alpha: {results['alpha_d1_exp_peak']:.4f}\n")
    out.write(f"D2 peak alpha: {results['alpha_d2_exp_peak']:.4f}\n")

print(f"Analysis written to {result_dir}/analysis_detail.txt")
