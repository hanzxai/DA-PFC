import pickle, numpy as np, sys, time

N_E, N_I = 800, 200
N = N_E + N_I
DT = 1.0
DURATION = 100000.0
DA_ONSET = 5000.0

def analyze(exp, da):
    print(f'\n========== DA={da}nM  ({exp}) ==========')
    t0 = time.time()
    with open(f'/data/workspace/code/git-code/sss/DA-PFC/outputs/{exp}/raw_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'  Loaded in {time.time()-t0:.1f}s')

    spikes = data['spikes'].numpy()
    gi = data['groups_info']
    e_d1_end    = gi['e_d1_end']
    e_d2_end    = gi['e_d2_end']
    e_other_end = gi['e_other_end']

    mask_b0 = spikes[:, 0] == 0
    sp  = spikes[mask_b0]
    nid = sp[:, 1]
    ts  = sp[:, 2].astype(float) * DT

    def fr(n_start, n_end, t_start, t_end):
        m = (nid >= n_start) & (nid < n_end) & (ts >= t_start) & (ts < t_end)
        n_neurons = n_end - n_start
        t_sec = (t_end - t_start) / 1000.0
        return int(m.sum()) / n_neurons / t_sec

    groups = [
        ('E-D1',    0,          e_d1_end,    e_d1_end),
        ('E-D2',    e_d1_end,   e_d2_end,    e_d2_end - e_d1_end),
        ('E-other', e_d2_end,   e_other_end, e_other_end - e_d2_end),
        ('I',       N_E,        N,           N_I),
        ('ALL-E',   0,          N_E,         N_E),
        ('ALL',     0,          N,           N),
    ]

    print(f'  {"Group":<10} {"N":>5} {"Baseline(Hz)":>14} {"DA-period(Hz)":>14} {"Delta":>8}')
    print(f'  {"-"*55}')
    for name, ns, ne, cnt in groups:
        fb = fr(ns, ne, 0,         DA_ONSET)
        fd = fr(ns, ne, DA_ONSET,  DURATION)
        print(f'  {name:<10} {cnt:>5} {fb:>14.2f} {fd:>14.2f} {fd-fb:>+8.2f}')

    print(f'  Total spikes (batch0): {mask_b0.sum():,}')
    print(f'  Unique neurons fired : {len(np.unique(nid))} / {N}')

    # v_traces
    vt = data['v_traces'].numpy()
    ri = data['record_indices'].numpy()
    print(f'  v_traces shape: {vt.shape}  (T x n_recorded)')
    print(f'  record_indices: {ri}')
    print(f'  V range: [{vt.min():.2f}, {vt.max():.2f}] mV')

analyze('exp_2026-03-08_19-57-31', 3.0)
analyze('exp_2026-03-08_20-00-51', 10.0)
