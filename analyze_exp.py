import pickle
import torch
import numpy as np

EXP = 'outputs/exp_2026-03-09_21-40-54/raw_data.pkl'
print(f"Loading {EXP} ...")
data = pickle.load(open(EXP, 'rb'))

print('=== Data Structure ===')
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, dict):
        print(f'  {k}: dict')
        for kk, vv in v.items():
            print(f'      {kk}: {vv}')
    else:
        print(f'  {k}: {type(v).__name__} = {v}')

spikes = data['spikes']
print(f'\nspikes shape: {spikes.shape}')
print(f'col0(time) range: {spikes[:,0].min().item()} - {spikes[:,0].max().item()}')
print(f'col1(batch) unique: {torch.unique(spikes[:,1]).tolist()}')
print(f'col2(neuron) range: {spikes[:,2].min().item()} - {spikes[:,2].max().item()}')

# Parameters
cfg = data.get('config', {})
gi  = data.get('groups_info', {})
duration_ms = float(cfg.get('duration', 100000))
da_onset_ms = float(cfg.get('da_onset', 500))
N_E = int(gi.get('N_E', 800))
N_I = int(gi.get('N_I', 200))
N = N_E + N_I
print(f'\nduration={duration_ms}ms, da_onset={da_onset_ms}ms, N_E={N_E}, N_I={N_I}')

# Per-batch analysis
for batch_id in sorted(torch.unique(spikes[:,1]).tolist()):
    batch_id = int(batch_id)
    mask = spikes[:,1] == batch_id
    bs = spikes[mask]

    # Pre-DA / Post-DA split
    pre = bs[bs[:,0] < da_onset_ms]
    post = bs[bs[:,0] >= da_onset_ms]

    # E/I split
    pre_E = pre[pre[:,2] < N_E]
    pre_I = pre[pre[:,2] >= N_E]
    post_E = post[post[:,2] < N_E]
    post_I = post[post[:,2] >= N_E]

    pre_s = da_onset_ms / 1000.0
    post_s = (duration_ms - da_onset_ms) / 1000.0
    total_s = duration_ms / 1000.0

    # Total E/I
    all_E = bs[bs[:,2] < N_E]
    all_I = bs[bs[:,2] >= N_E]

    label = "No DA (Control)" if batch_id == 0 else f"DA = {cfg.get('da_level', '?')} nM"
    print(f'\n{"="*60}')
    print(f'  Batch {batch_id}: {label}')
    print(f'{"="*60}')
    print(f'  Total spikes: {bs.shape[0]:,}')
    print(f'  Total E spikes: {all_E.shape[0]:,} -> mean rate = {all_E.shape[0]/(N_E*total_s):.2f} Hz')
    print(f'  Total I spikes: {all_I.shape[0]:,} -> mean rate = {all_I.shape[0]/(N_I*total_s):.2f} Hz')
    print(f'  --- Pre-DA (0 ~ {da_onset_ms:.0f}ms, {pre_s:.1f}s) ---')
    print(f'    E: {pre_E.shape[0]:,} spikes -> {pre_E.shape[0]/(N_E*pre_s):.2f} Hz')
    print(f'    I: {pre_I.shape[0]:,} spikes -> {pre_I.shape[0]/(N_I*pre_s):.2f} Hz')
    print(f'  --- Post-DA ({da_onset_ms:.0f} ~ {duration_ms:.0f}ms, {post_s:.1f}s) ---')
    print(f'    E: {post_E.shape[0]:,} spikes -> {post_E.shape[0]/(N_E*post_s):.2f} Hz')
    print(f'    I: {post_I.shape[0]:,} spikes -> {post_I.shape[0]/(N_I*post_s):.2f} Hz')

    # Time-windowed analysis (every 10s)
    print(f'  --- Time-windowed (10s bins) ---')
    T_steps = int(duration_ms)
    bin_ms = 10000.0
    for t_start in np.arange(0, T_steps, bin_ms):
        t_end = min(t_start + bin_ms, T_steps)
        seg = bs[(bs[:,0] >= t_start) & (bs[:,0] < t_end)]
        seg_E = seg[seg[:,2] < N_E]
        seg_I = seg[seg[:,2] >= N_E]
        dur_s = (t_end - t_start) / 1000.0
        r_E = seg_E.shape[0] / (N_E * dur_s)
        r_I = seg_I.shape[0] / (N_I * dur_s)
        print(f'    [{t_start/1000:.0f}s-{t_end/1000:.0f}s] E={r_E:.2f}Hz  I={r_I:.2f}Hz')

# V traces
v = data['v_traces']
rec = data['record_indices']
print(f'\n=== V traces ===')
print(f'v_traces shape: {v.shape}')
print(f'record_indices: {rec}')
if v.ndim >= 2:
    print(f'V mean per recorded neuron: {v.mean(dim=0)}')
    print(f'V max  per recorded neuron: {v.max(dim=0).values}')

print('\nDone.')
