import pickle
import numpy as np

EXP_DIR = "outputs/exp_2026-03-08_21-28-29/raw_data.pkl"

print("Loading data...")
with open(EXP_DIR, "rb") as f:
    data = pickle.load(f)

cfg = data['config']
print("\n=== Config ===")
for k, v in cfg.items():
    print(f"  {k} = {v}")

N_E       = int(cfg['N_E'])
N_I       = int(cfg['N_I'])
duration  = float(cfg['duration'])   # ms
dt        = float(cfg['dt'])         # ms
da_onset  = float(cfg['da_onset'])   # ms
da_level  = float(cfg['da_level'])
mode      = cfg['mode']
print(f"\nN_E={N_E}, N_I={N_I}, duration={duration}ms, da_onset={da_onset}ms, da_level={da_level}nM, mode={mode}")

# spikes: (N_events, 3) -> [time_step, batch_idx, neuron_idx]
spk = data['spikes'].numpy()   # convert to numpy
print(f"\nSpike events total: {len(spk)}")
print(f"  time range : {spk[:,0].min()} ~ {spk[:,0].max()} (steps)")
print(f"  batch range: {spk[:,1].min()} ~ {spk[:,1].max()}")
print(f"  neuron range: {spk[:,2].min()} ~ {spk[:,2].max()}")

batches = np.unique(spk[:,1])
print(f"\n=== Per-Batch Analysis ===")
for b in batches:
    mask_b = spk[:,1] == b
    spk_b  = spk[mask_b]
    
    # E neurons: 0 ~ N_E-1, I neurons: N_E ~ N_E+N_I-1
    mask_E = spk_b[:,2] < N_E
    mask_I = spk_b[:,2] >= N_E
    
    n_E_spk = mask_E.sum()
    n_I_spk = mask_I.sum()
    
    dur_s = duration / 1000.0
    rate_E = n_E_spk / (N_E * dur_s)
    rate_I = n_I_spk / (N_I * dur_s)
    
    print(f"\n  Batch {b} ({'Control' if b==0 else 'DA-Exp'}):")
    print(f"    E spikes={n_E_spk:,}  mean_rate={rate_E:.2f} Hz")
    print(f"    I spikes={n_I_spk:,}  mean_rate={rate_I:.2f} Hz")
    
    # 分段: pre-DA vs post-DA
    da_step = int(da_onset / dt)
    T_total = int(duration / dt)
    
    pre_mask  = spk_b[:,0] < da_step
    post_mask = spk_b[:,0] >= da_step
    
    pre_E  = (pre_mask  & mask_E).sum()
    post_E = (post_mask & mask_E).sum()
    pre_I  = (pre_mask  & mask_I).sum()
    post_I = (post_mask & mask_I).sum()
    
    pre_dur_s  = da_onset / 1000.0
    post_dur_s = (duration - da_onset) / 1000.0
    
    if pre_dur_s > 0:
        print(f"    Pre-DA  (0~{da_onset:.0f}ms):  E={pre_E/(N_E*pre_dur_s):.2f}Hz  I={pre_I/(N_I*pre_dur_s):.2f}Hz")
    if post_dur_s > 0:
        print(f"    Post-DA ({da_onset:.0f}~{duration:.0f}ms): E={post_E/(N_E*post_dur_s):.2f}Hz  I={post_I/(N_I*post_dur_s):.2f}Hz")

# v_traces
vt = data['v_traces'].numpy()
ri = data['record_indices'].numpy()
print(f"\n=== V_traces ===")
print(f"  shape={vt.shape}  (T x n_recorded)")
print(f"  record_indices={ri}")
print(f"  V range: {vt.min():.2f} ~ {vt.max():.2f} mV")
print(f"  V mean : {vt.mean():.2f} mV")
