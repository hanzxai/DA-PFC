# DA-PFC Spiking Neural Network Simulation

> Dopamine modulation of Prefrontal Cortex — LIF network with D1/D2 receptor dynamics (Langmuir binding kinetics)

---

## 1. Project Structure

```
DA-PFC/
├── main.py                    # Entry point (CLI)
├── config.py                  # Global parameters (Single Source of Truth)
├── utils.py                   # Experiment folder management, serialization
├── simulation/
│   ├── __init__.py
│   └── runners.py             # 4 simulation runners (static / stepped / D1 / D1+D2)
├── models/
│   ├── __init__.py
│   ├── network.py             # Network structure builder (W matrix, D1/D2 masks)
│   ├── pharmacology.py        # Pharmacological parameter computation (Sigmoid)
│   └── kernels.py             # JIT-compiled simulation kernels (@torch.jit.script)
├── analysis/
│   ├── __init__.py
│   ├── analyzer.py            # PFCAnalyzer: firing rates, FFT, reports
│   └── plotting.py            # Plotting functions (full-scale + zoom-in)
├── analyze_exp.py             # Post-hoc analysis script for saved experiments
├── analyze_exp_results.py     # Batch experiment result analysis
├── analyze_quick.py           # Quick analysis script
└── outputs/                   # Experiment outputs (git-ignored)
```

---

## 2. Network Architecture

### 2.1 Population Composition

| Population | Count | Proportion | Receptor Expression |
|---|---|---|---|
| Excitatory (E) | 800 | 80% | D1R: 25% of E; D2R: 15% of E |
| Inhibitory (I) | 200 | 20% | D1R: 30% of I; D2R: 10% of I |
| **Total** | **1000** | 100% | — |

### 2.2 Receptor Subpopulation Counts

| Subpopulation | Count | Neuron Index Range |
|---|---|---|
| E-D1 neurons | 200 | [0, 199] |
| E-D2 neurons | 120 | [200, 319] |
| E-Other | 480 | [320, 799] |
| I-D1 neurons | 60 | [800, 859] |
| I-D2 neurons | 20 | [860, 879] |
| I-Other | 120 | [880, 999] |

> Neurons are indexed as $[0, \ldots, 799]$ = E, $[800, \ldots, 999]$ = I.  
> `mask_d1`, `mask_d2` are binary vectors of shape $(N,)$ marking receptor-expressing neurons.

### 2.3 Connectivity

| Parameter | Value | Notes |
|---|---|---|
| Connection probability | 0.20 | Random sparse Erdős–Rényi |
| Excitatory weight $w_E$ | 5.0 pA | E→any |
| Inhibitory weight $w_I$ | −25.0 pA | I→any |

Synaptic weight matrix $W \in \mathbb{R}^{N \times N}$, transposed to $W^T$ for efficient batch matmul: `I_syn += spikes @ W_t`.

### 2.4 Simulation Batches

Two parallel batches run simultaneously with **identical initial conditions and shared noise**:

| Batch | DA concentration | Role |
|---|---|---|
| 0 | 0 nM (always) | Control |
| 1 | `da_level` nM (after onset) | Experiment |

> **Design:** Both batches share identical initial membrane potentials (`V_init` generated once with shape `(1, N)` then expanded) and identical background noise at every time step (`I_bg` generated with shape `(1, N)` then expanded). The **only** difference between batches is the DA concentration parameter.

---

## 3. Mathematical Formulation

### 3.1 LIF Neuron Dynamics

$$
C_m \frac{dV}{dt} = -\frac{V - V_{rest}}{R_{eff}} + I_{total}
$$

$$
I_{total} = I_{syn} \cdot scale_{syn} + I_{bg} + I_{mod}
$$

**Euler integration:**

$$
V(t+dt) = V(t) + \frac{1}{C_m} \left[ -\frac{V - V_{rest}}{R_{eff}} + I_{total} \right] \cdot dt
$$

**Spike generation:** if $V > V_{th}$, emit spike and reset $V \leftarrow V_{reset}$.

**Refractory period:** neuron is clamped during $[t_{spike},\ t_{spike} + t_{ref}]$.

**Synaptic current decay (exponential):**

$$
I_{syn}(t + dt) = I_{syn}(t) \cdot e^{-dt / \tau_{syn}} + \sum_j W_{ji} \cdot s_j(t)
$$

where $s_j(t) = 1$ if neuron $j$ fires at time $t$.

---

### 3.2 DA Receptor Activation — Sigmoid Target

The instantaneous sigmoid target activation for receptor $r \in \{D1, D2\}$:

$$
s_r(t) = \frac{1}{1 + e^{-\beta \left( DA(t) - EC50_r \right)}}
$$

Before DA onset or for the control batch: $s_r = 0$.

---

### 3.3 Receptor Kinetics — Langmuir Binding

Receptor occupancy $\alpha_r$ follows a Langmuir binding ODE:

$$
\frac{d\alpha_r}{dt} = k_{on,r} \cdot s_r(t) \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r
$$

**Steady state:**

$$
\alpha_r^{ss} = \frac{k_{on} \cdot s_r}{k_{on} \cdot s_r + k_{off}} = \frac{s_r}{s_r + K_d}, \quad K_d = \frac{k_{off}}{k_{on}}
$$

**Euler discretization:**

$$
\alpha_r(t + dt) = \alpha_r(t) + \left[ k_{on,r} \cdot s_r \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r \right] \cdot dt
$$

> Note: $\alpha_{ss} < s_r$ due to the Langmuir saturation (unlike the tau or k_on/k_off methods where $\alpha_{ss} = s_r$).

---

### 3.4 DA Neuromodulation — Parameter Assembly

**Effective membrane resistance** (per neuron):

$$
R_{eff} = R_{base} \cdot \left(1 + \varepsilon_{D1} \cdot \alpha_{D1} \cdot mask_{D1} - \varepsilon_{D2} \cdot \alpha_{D2} \cdot mask_{D2}\right)
$$

**Modulation current:**

$$
I_{mod} = BIAS_{D1} \cdot \alpha_{D1} \cdot mask_{D1} + BIAS_{D2} \cdot \alpha_{D2} \cdot mask_{D2}
$$

**Synaptic scaling factor:**

$$
scale_{syn} = 1 + \lambda_{D1} \cdot \alpha_{D1} \cdot mask_{D1} - \lambda_{D2} \cdot \alpha_{D2} \cdot mask_{D2}
$$

---

### 3.5 Membrane Time Constant

$$
\tau_m = R_{base} \cdot C_m
$$

| Neuron type | $C_m$ (pF) | $\tau_m$ (ms) |
|---|---|---|
| Excitatory (E) | 250 | 25.0 |
| Inhibitory (I) | 90 | 9.0 |

> Note: The config module defines `TAU_MEM = 20.0 ms` as a reference, but actual $\tau_m$ varies by neuron type via $C_m$.

---

## 4. Parameter Tables

### 4.1 Network Architecture Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `N_E` | $N_E$ | 800 | — | Number of excitatory neurons |
| `N_I` | $N_I$ | 200 | — | Number of inhibitory neurons |
| `N_TOTAL` | $N$ | 1000 | — | Total neurons |
| `CONN_PROB` | $p$ | 0.20 | — | Connection probability |
| `W_EXC` | $w_E$ | 5.0 | pA | Excitatory synaptic weight |
| `W_INH` | $w_I$ | −25.0 | pA | Inhibitory synaptic weight |
| `FRAC_E_D1` | — | 0.25 | — | Fraction of E neurons expressing D1R |
| `FRAC_E_D2` | — | 0.15 | — | Fraction of E neurons expressing D2R |
| `FRAC_I_D1` | — | 0.30 | — | Fraction of I neurons expressing D1R |
| `FRAC_I_D2` | — | 0.10 | — | Fraction of I neurons expressing D2R |

---

### 4.2 LIF Neuron Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `V_REST` | $V_{rest}$ | −70.0 | mV | Resting membrane potential |
| `V_RESET` | $V_{reset}$ | −75.0 | mV | Post-spike reset potential (hyperpolarized) |
| `V_TH` | $V_{th}$ | −50.0 | mV | Spike threshold |
| `R_BASE` | $R_{base}$ | 0.1 | GΩ (= mV/pA = 100 MΩ) | Baseline membrane resistance |
| `C_E` | $C_E$ | 250.0 | pF | Excitatory membrane capacitance |
| `C_I` | $C_I$ | 90.0 | pF | Inhibitory membrane capacitance |
| `TAU_SYN` | $\tau_{syn}$ | 5.0 | ms | Synaptic current decay time constant |
| `T_REF` | $t_{ref}$ | 5.0 | ms | Absolute refractory period |
| `BG_MEAN` | $\mu_{bg}$ | 190.0 | pA | Background current mean |
| `BG_STD` | $\sigma_{bg}$ | 25.0 | pA | Background current std dev |

> **Background current calibration:** $V_{ss} = V_{rest} + R_{base} \times BG\_MEAN = -70 + 0.1 \times 190 = -51$ mV $\approx V_{th}$. This places the steady-state just below threshold, enabling noise-driven firing.

---

### 4.3 Receptor Kinetics Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `EC50_D1` | $EC50_{D1}$ | 4.0 | nM | D1 receptor half-maximal concentration |
| `EC50_D2` | $EC50_{D2}$ | 8.0 | nM | D2 receptor half-maximal concentration |
| `BETA` | $\beta$ | 1.0 | nM⁻¹ | Sigmoid slope coefficient |
| `TAU_ON_D1` | $\tau_{on,D1}$ | 30876.1 | ms | D1 activation rise time constant |
| `TAU_OFF_D1` | $\tau_{off,D1}$ | 164472.5 | ms | D1 activation decay time constant |
| `TAU_ON_D2` | $\tau_{on,D2}$ | 10000.0 | ms | D2 activation rise time constant (~3× faster than D1) |
| `TAU_OFF_D2` | $\tau_{off,D2}$ | 50000.0 | ms | D2 activation decay time constant (~3× faster than D1) |

Derived Langmuir rate constants (used in `run_dynamic_d1_d2_kernel`):

| Derived | Formula | Value | Unit |
|---|---|---|---|
| $k_{on,D1}$ | $1 / (\tau_{on,D1} - 3000)$ | $\approx 3.59 \times 10^{-5}$ | ms⁻¹ |
| $k_{off,D1}$ | $1 / (\tau_{off,D1} + 3000)$ | $\approx 5.97 \times 10^{-6}$ | ms⁻¹ |
| $k_{on,D2}$ | $1 / \tau_{on,D2}$ | $1.00 \times 10^{-4}$ | ms⁻¹ |
| $k_{off,D2}$ | $1 / \tau_{off,D2}$ | $2.00 \times 10^{-5}$ | ms⁻¹ |
| $K_{d,D1}$ | $k_{off,D1} / k_{on,D1}$ | $\approx 0.1664$ | — |
| $K_{d,D2}$ | $k_{off,D2} / k_{on,D2}$ | $0.2$ | — |

> Note: D1 rate constants have a ±3000 ms offset applied to $\tau$ values before computing $k_{on}$ and $k_{off}$.

---

### 4.4 DA Modulation Strength Parameters

| Parameter | Symbol | Value | Unit | Effect at $\alpha=1$ |
|---|---|---|---|---|
| `EPS_D1` | $\varepsilon_{D1}$ | 0.15 | — | $R_{eff} = R_{base} \times 1.15$ (+15% resistance, ↑ excitability) |
| `EPS_D2` | $\varepsilon_{D2}$ | 0.10 | — | $R_{eff} = R_{base} \times 0.90$ (−10% resistance, ↓ excitability) |
| `BIAS_D1` | $BIAS_{D1}$ | +3.0 | pA | Excitatory bias current added to D1 neurons |
| `BIAS_D2` | $BIAS_{D2}$ | −3.0 | pA | Inhibitory bias current added to D2 neurons |
| `LAM_D1` | $\lambda_{D1}$ | 0.3 | — | Synaptic scaling factor for D1 neurons |
| `LAM_D2` | $\lambda_{D2}$ | 0.2 | — | Synaptic scaling factor for D2 neurons |

---

### 4.5 Simulation Control Parameters

| Parameter | Symbol | Default | Unit | Description |
|---|---|---|---|---|
| `DT` | $dt$ | 1.0 | ms | Integration time step |
| `DEFAULT_DURATION` | — | 2000.0 | ms | Default duration in config |
| `--duration` (CLI) | $T$ | 100000.0 | ms (= 100 s) | CLI default simulation duration |
| `--da` (CLI) | $[DA]$ | 3.0 | nM | DA concentration during drug window |
| `DEFAULT_DA_ONSET` | $t_{onset}$ | 500.0 | ms | DA application onset time |
| `RANDOM_SEED` | — | 42 | — | Random seed for reproducibility |

---

## 5. Simulation Flow

```
main.py
  └─ run_simulation_d1_d2_kinetics()             [simulation/runners.py]
       ├─ _init_network()
       │    ├─ Set random seed (42)
       │    └─ create_network_structure()          [models/network.py]
       │         ├─ Build W matrix (sparse random, p=0.20)
       │         └─ Build mask_d1, mask_d2 vectors
       ├─ _build_record_indices()
       └─ run_dynamic_d1_d2_kernel()               [models/kernels.py, @torch.jit.script]
            ├─ Initialize: V_init (1,N) → expand to (2,N)  [identical across batches]
            └─ For each time step t:
                 ├─ Compute DA(t): B0=0, B1=da_level (if t≥onset)
                 ├─ Update alpha_D1 via Langmuir ODE  (k_on≈3.59e-5, k_off≈5.97e-6)
                 ├─ Update alpha_D2 via Langmuir ODE  (k_on=1e-4,    k_off=2e-5)
                 ├─ Assemble: R_eff, I_mod, scale_syn
                 ├─ Synaptic decay: I_syn *= exp(-dt/τ_syn)
                 ├─ Background noise: I_bg (1,N) → expand  [shared across batches]
                 ├─ LIF integration: dV = (leak + I_total) / C_m * dt
                 ├─ Refractory check
                 ├─ Spike detection & reset
                 └─ Synaptic propagation: I_syn += spikes @ W_t
```

---

## 6. Simulation Modes (4 Kernels)

| Kernel | Description | DA Handling | Used By |
|---|---|---|---|
| `run_batch_network` | Static DA | Fixed modulation params | `run_simulation_in_memory` |
| `run_batch_network_stepped` | Stepped DA | Instantaneous switch at onset | `run_simulation_stepped` |
| `run_dynamic_d1_kernel` | D1 dynamic + D2 dynamic | D1 & D2 Langmuir ODE | `run_simulation_d1_kinetics` |
| `run_dynamic_d1_d2_kernel` | D1 + D2 dynamic | Both Langmuir ODE | `run_simulation_d1_d2_kinetics` (**default**) |

---

## 7. D1 vs D2 Receptor Comparison

| Property | D1 Receptor | D2 Receptor |
|---|---|---|
| $EC50$ | 4.0 nM | 8.0 nM |
| $\tau_{on}$ | 30876 ms (~31 s) | 10000 ms (~10 s) |
| $\tau_{off}$ | 164472 ms (~164 s) | 50000 ms (~50 s) |
| $k_{on}$ | ~3.59e-5 ms⁻¹ | 1.0e-4 ms⁻¹ |
| $k_{off}$ | ~5.97e-6 ms⁻¹ | 2.0e-5 ms⁻¹ |
| $K_d$ | ~0.166 | 0.2 |
| Response speed | Slow | ~3× faster than D1 |
| Effect on $R_{eff}$ | +15% (↑ excitability) | −10% (↓ excitability) |
| Bias current | +3.0 pA (excitatory) | −3.0 pA (inhibitory) |
| Synaptic scaling | +0.3 (enhance) | −0.2 (suppress) |
| Net effect | ↑ Excitability | ↓ Excitability |

---

## 8. Analysis & Output

Each experiment saves to `outputs/exp_<timestamp>/`:

| File | Description |
|---|---|
| `config.json` | Experiment parameters |
| `raw_data.pkl` | Full simulation data (spikes, traces, masks) |
| `analysis_report.txt` | FFT frequency report + per-group phase analysis |
| `firing_rates_batch_{0,1}.png` | Population firing rates (6 subgroups) |
| `firing_rates_E_batch_{0,1}.png` | Excitatory subgroup rates |
| `firing_rates_I_batch_{0,1}.png` | Inhibitory subgroup rates |
| `raster_batch_{0,1}.png` | Spike raster plots |
| `*_zoom_batch_{0,1}.png` | Zoom-in plots (DA onset + 300ms) |

---

## 9. Usage

```bash
# Default run (100s, DA=3.0 nM, DA onset at 500ms)
python main.py

# Custom duration and DA concentration
python main.py --duration 200000 --da 5.0

# Specify plot batch (0=Control, 1=Experiment)
python main.py --duration 100000 --da 3.0 --batch 1
```

### Dependencies

- Python 3.8+
- PyTorch (with CUDA recommended)
- NumPy, SciPy, Matplotlib, tqdm
