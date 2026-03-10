import pickle, torch, numpy as np, sys

EXP = 'outputs/exp_2026-03-10_10-55-43'
print(f'Loading {EXP}/raw_data.pkl ...')
data = pickle.load(open(f'{EXP}/raw_data.pkl','rb'))
print('Loaded.')

# === Structure ===
print('\n=== Data Keys ===')
for k,v in data.items():
    if hasattr(v,'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, dict):
        print(f'  {k}: dict keys={list(v.keys())}')
    else:
        print(f'  {k}: {type(v).__name__}')

# === Config & Groups ===
cfg = data.get('config', {})
gi = data.get('groups_info', {})
print('\n=== Config ===')
for k,v in cfg.items(): print(f'  {k}: {v}')
print('\n=== Groups Info ===')
for k,v in gi.items(): print(f'  {k}: {v}')

# === Masks ===
# NOTE: mask keys are 'd1' and 'd2' (NOT 'd1_mask'/'d2_mask'/'other_mask')
# 'other' group is derived as ~d1 & ~d2
masks = data.get('masks', {})
print('\n=== Masks ===')
print(f'  Available keys: {list(masks.keys())}')
for k,v in masks.items():
    if hasattr(v,'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}, sum={v.sum().item()}, first10={v[:10].tolist()}')
    else:
        print(f'  {k}: {v}')

# === Spikes ===
spikes = data['spikes']
print(f'\n=== Spikes: shape={spikes.shape}, dtype={spikes.dtype} ===')
sp = spikes.numpy() if hasattr(spikes,'numpy') else np.array(spikes)
# Column layout: col0=time_step, col1=batch_id, col2=neuron_id
print(f'col0(time) range: {sp[:,0].min()}-{sp[:,0].max()}')
print(f'col1(batch) unique: {np.unique(sp[:,1]).tolist()}')
print(f'col2(neuron) range: {sp[:,2].min()}-{sp[:,2].max()}')

N_E = cfg.get('N_E', 800)
N_I = cfg.get('N_I', 200)
N = N_E + N_I
duration_ms = float(cfg.get('duration', 100000))
da_onset = float(cfg.get('da_onset', 500.0))

# === Build subgroup masks (correct key: 'd1', 'd2') ===
m_d1 = masks['d1'].numpy()   # shape (N,) bool
m_d2 = masks['d2'].numpy()   # shape (N,) bool
ids_all = np.arange(N)
is_exc = ids_all < N_E
is_inh = ~is_exc

groups = {
    'E-D1':    is_exc & m_d1,
    'E-D2':    is_exc & m_d2,
    'E-Other': is_exc & (~m_d1) & (~m_d2),   # derived: ~d1 & ~d2
    'I-D1':    is_inh & m_d1,
    'I-D2':    is_inh & m_d2,
    'I-Other': is_inh & (~m_d1) & (~m_d2),   # derived: ~d1 & ~d2
    'All-E':   is_exc,
    'All-I':   is_inh,
}
print('\n=== Subgroup sizes ===')
for gname, gmask in groups.items():
    print(f'  {gname}: {gmask.sum()} neurons')

# === Mean Firing Rate Analysis (spike-count method) ===
# NOTE: This gives MEAN FIRING RATE (Hz) — different from FFT dominant frequency.
# FFT dominant frequency = oscillation frequency of the population rate curve (e.g. gamma ~40Hz).
# Mean firing rate = total spikes / (n_neurons * duration_s).
# They measure different things and should NOT be directly compared.
print(f'\n=== Mean Firing Rate Analysis (spike-count) ===')
print(f'  NOTE: These are MEAN FIRING RATES, not FFT oscillation frequencies.')
print(f'  FFT in analysis_report.txt shows dominant oscillation frequency of the rate curve.\n')

times = sp[:, 0]
batches = sp[:, 1]
neurons = sp[:, 2].astype(int)

phases = [
    ('Baseline (0-500ms)',       0,    500),
    ('DA Early (500-1500ms)',    500,  1500),
    ('Late DA (1500-100000ms)', 1500, 100000),
]

for batch_id in [0, 1]:
    b_mask = batches == batch_id
    b_times = times[b_mask]
    b_neurons = neurons[b_mask]

    label = "CONTROL - No DA modulation" if batch_id==0 else f"EXPERIMENTAL - DA = {cfg.get('da_level','?')} nM"
    print(f'\n{"="*75}')
    print(f'  Batch {batch_id} ({label})')
    print(f'{"="*75}')
    print(f'  Total spike events: {b_mask.sum()}')

    for phase_name, t0, t1 in phases:
        p_mask = (b_times >= t0) & (b_times < t1)
        p_neurons = b_neurons[p_mask]
        dur_s = (t1 - t0) / 1000.0

        print(f'\n  {phase_name}:')
        print(f'    ┌──────────┬──────────────┬──────────────┐')
        print(f'    │ Group    │  Count       │  Rate (Hz)   │')
        print(f'    ├──────────┼──────────────┼──────────────┤')

        for gname, gmask in groups.items():
            valid_ids = np.where(gmask)[0]
            n_neurons_g = len(valid_ids)
            if n_neurons_g == 0:
                continue
            count = np.isin(p_neurons, valid_ids).sum()
            rate = count / (n_neurons_g * dur_s)
            print(f'    │ {gname:<8} │  {count:>10}  │  {rate:>10.2f}  │')

        print(f'    └──────────┴──────────────┴──────────────┘')

# === V Traces ===
v = data['v_traces']
rec = data['record_indices']
print(f'\n{"="*75}')
print(f'  V Traces')
print(f'{"="*75}')
print(f'  record_indices: {rec}')
print(f'  v_traces shape: {v.shape}')
v_np = v.numpy() if hasattr(v,'numpy') else np.array(v)

for col_idx in range(min(v_np.shape[-1], 8)):
    vm = v_np[:, col_idx]
    pre_v = vm[:int(da_onset)]
    post_v = vm[int(da_onset):]
    print(f'  Col {col_idx}: overall mean={vm.mean():.2f}mV, '
          f'pre-DA mean={pre_v.mean():.2f}mV, post-DA mean={post_v.mean():.2f}mV, '
          f'min={vm.min():.2f}mV, max={vm.max():.2f}mV')

print('\nDone.')
