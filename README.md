# DA-PFC Spiking Neural Network Simulation

> Dopamine modulation of Prefrontal Cortex — LIF network with D1/D2 receptor dynamics

---

## 1. Network Architecture

### 1.1 Population Composition

| Population | Count | Proportion | Receptor Expression |
|---|---|---|---|
| Excitatory (E) | 800 | 80% | D1R: 25% of E; D2R: 15% of E |
| Inhibitory (I) | 200 | 20% | D1R: 30% of I; D2R: 10% of I |
| **Total** | **1000** | 100% | — |

### 1.2 Receptor Subpopulation Counts (approximate)

| Subpopulation | Count | Notes |
|---|---|---|
| E-D1 neurons | ~200 | 25% × 800 |
| E-D2 neurons | ~120 | 15% × 800 |
| I-D1 neurons | ~60 | 30% × 200 |
| I-D2 neurons | ~20 | 10% × 200 |
| E (no receptor) | ~480 | remaining E |
| I (no receptor) | ~120 | remaining I |

> Neurons are indexed as $[0, \ldots, 799]$ = E, $[800, \ldots, 999]$ = I.  
> `mask_d1`, `mask_d2` are binary vectors of shape $(N,)$ marking receptor-expressing neurons.

### 1.3 Connectivity

| Parameter | Value | Notes |
|---|---|---|
| Connection probability | 0.20 | random sparse Erdős–Rényi |
| Excitatory weight $w_E$ | 0.3 | E→any |
| Inhibitory weight $w_I$ | −2.0 | I→any |

Synaptic weight matrix $W \in \mathbb{R}^{N \times N}$, transposed to $W^T$ for efficient batch matmul.

### 1.4 Simulation Batches

Two parallel batches run simultaneously:

| Batch | DA concentration | Role |
|---|---|---|
| 0 | 0 nM (always) | Control |
| 1 | `da_level` nM (after onset) | Experiment |

---

## 2. Mathematical Formulation

### 2.1 LIF Neuron Dynamics

$$
C_m \frac{dV}{dt} = -\frac{V - V_{rest}}{R_{eff}} + I_{total}
$$

$$
I_{total} = I_{syn} \cdot scale_{syn} + I_{bg} + I_{mod}
$$

**Spike generation:** if $V > V_{th}$, emit spike and reset $V \leftarrow V_{reset}$.

**Refractory period:** neuron is clamped during $[t_{spike},\ t_{spike} + t_{ref}]$.

**Synaptic current decay (Euler):**

$$
I_{syn}(t + dt) = I_{syn}(t) \cdot e^{-dt / \tau_{syn}} + \sum_j W_{ji} \cdot s_j(t)
$$

where $s_j(t) = 1$ if neuron $j$ fires at time $t$.

---

### 2.2 DA Receptor Activation — Sigmoid Target

The instantaneous sigmoid target activation for receptor $r \in \{D1, D2\}$:

$$
s_r(t) = \frac{1}{1 + e^{-\beta \left( DA(t) - EC50_r \right)}}
$$

Before DA onset or for the control batch: $s_r = 0$.

---

### 2.3 Receptor Kinetics — Langmuir Binding

Receptor occupancy $\alpha_r$ follows a Langmuir binding ODE:

$$
\frac{d\alpha_r}{dt} = k_{on,r} \cdot s_r(t) \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r
$$

**Steady state:**

$$
\alpha_r^{ss} = \frac{s_r}{s_r + K_d}, \quad K_d = \frac{k_{off}}{k_{on}}
$$

**Euler discretization:**

$$
\alpha_r(t + dt) = \alpha_r(t) + \left[ k_{on,r} \cdot s_r \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r \right] \cdot dt
$$

---

### 2.4 DA Neuromodulation — Parameter Assembly

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

### 2.5 Membrane Time Constant

$$
\tau_m = R_{base} \cdot C_m
$$

| Neuron type | $C_m$ (pF) | $\tau_m$ (ms) |
|---|---|---|
| Excitatory (E) | 250 | 250 (normalized units) |
| Inhibitory (I) | 50 | 50 (normalized units) |

---

## 3. Parameter Tables

### 3.1 Network Architecture Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `N_E` | $N_E$ | 800 | — | Number of excitatory neurons |
| `N_I` | $N_I$ | 200 | — | Number of inhibitory neurons |
| `N_TOTAL` | $N$ | 1000 | — | Total neurons |
| `CONN_PROB` | $p$ | 0.20 | — | Connection probability |
| `W_EXC` | $w_E$ | 0.3 | nA (normalized) | Excitatory synaptic weight |
| `W_INH` | $w_I$ | −2.0 | nA (normalized) | Inhibitory synaptic weight |
| `FRAC_E_D1` | — | 0.25 | — | Fraction of E neurons expressing D1R |
| `FRAC_E_D2` | — | 0.15 | — | Fraction of E neurons expressing D2R |
| `FRAC_I_D1` | — | 0.30 | — | Fraction of I neurons expressing D1R |
| `FRAC_I_D2` | — | 0.10 | — | Fraction of I neurons expressing D2R |

---

### 3.2 LIF Neuron Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `V_REST` | $V_{rest}$ | 0.0 | mV | Resting membrane potential |
| `V_RESET` | $V_{reset}$ | −5.0 | mV | Post-spike reset potential |
| `V_TH` | $V_{th}$ | 20.0 | mV | Spike threshold |
| `R_BASE` | $R_{base}$ | 1.0 | MΩ (normalized) | Baseline membrane resistance |
| `C_E` | $C_E$ | 250.0 | pF | Excitatory membrane capacitance |
| `C_I` | $C_I$ | 50.0 | pF | Inhibitory membrane capacitance |
| `TAU_SYN` | $\tau_{syn}$ | 5.0 | ms | Synaptic current decay time constant |
| `T_REF` | $t_{ref}$ | 5.0 | ms | Absolute refractory period |
| `BG_MEAN` | $\mu_{bg}$ | 25.0 | pA | Background current mean |
| `BG_STD` | $\sigma_{bg}$ | 5.0 | pA | Background current std dev |

---

### 3.3 Receptor Kinetics Parameters

| Parameter | Symbol | Value | Unit | Description |
|---|---|---|---|---|
| `EC50_D1` | $EC50_{D1}$ | 4.0 | nM | D1 receptor half-maximal concentration |
| `EC50_D2` | $EC50_{D2}$ | 8.0 | nM | D2 receptor half-maximal concentration |
| `BETA` | $\beta$ | 1.0 | nM⁻¹ | Sigmoid slope coefficient |
| `TAU_ON_D1` | $\tau_{on,D1}$ | 30876.1 | ms | D1 activation rise time constant |
| `TAU_OFF_D1` | $\tau_{off,D1}$ | 164472.5 | ms | D1 activation decay time constant |
| `TAU_ON_D2` | $\tau_{on,D2}$ | 10000.0 | ms | D2 activation rise time constant (~3× faster than D1) |
| `TAU_OFF_D2` | $\tau_{off,D2}$ | 50000.0 | ms | D2 activation decay time constant (~3× faster than D1) |

Derived rate constants:

| Derived | Formula | Value | Unit |
|---|---|---|---|
| $k_{on,D1}$ | $1 / \tau_{on,D1}$ | $\approx 3.24 \times 10^{-5}$ | ms⁻¹ |
| $k_{off,D1}$ | $1 / \tau_{off,D1}$ | $\approx 6.08 \times 10^{-6}$ | ms⁻¹ |
| $k_{on,D2}$ | $1 / \tau_{on,D2}$ | $1.00 \times 10^{-4}$ | ms⁻¹ |
| $k_{off,D2}$ | $1 / \tau_{off,D2}$ | $2.00 \times 10^{-5}$ | ms⁻¹ |

---

### 3.4 DA Modulation Strength Parameters

| Parameter | Symbol | Value | Unit | Effect at $\alpha=1$ |
|---|---|---|---|---|
| `EPS_D1` | $\varepsilon_{D1}$ | 0.15 | — | $R_{eff} = R_{base} \times 1.15$ (+15% resistance, ↑ excitability) |
| `EPS_D2` | $\varepsilon_{D2}$ | 0.10 | — | $R_{eff} = R_{base} \times 0.90$ (−10% resistance, ↓ excitability) |
| `BIAS_D1` | $BIAS_{D1}$ | +3.0 | pA | Excitatory bias current added to D1 neurons |
| `BIAS_D2` | $BIAS_{D2}$ | −3.0 | pA | Inhibitory bias current added to D2 neurons |
| `LAM_D1` | $\lambda_{D1}$ | 0.3 | — | Synaptic scaling factor for D1 neurons |
| `LAM_D2` | $\lambda_{D2}$ | 0.2 | — | Synaptic scaling factor for D2 neurons |

---

### 3.5 Simulation Control Parameters

| Parameter | Symbol | Default | Unit | Description |
|---|---|---|---|---|
| `DT` | $dt$ | 1.0 | ms | Integration time step |
| `--duration` | $T$ | 100000.0 | ms (= 100 s) | Total simulation duration |
| `--da` | $[DA]$ | 3.0 | nM | DA concentration during drug window |
| `DEFAULT_DA_ONSET` | $t_{onset}$ | 5000.0 | ms | DA application onset time |
| `RANDOM_SEED` | — | 42 | — | Random seed for reproducibility |

---

## 4. Simulation Flow

```
main.py
  └─ run_simulation_d1_d2_kinetics()          [runners.py]
       ├─ Build W matrix (sparse random)
       ├─ Build mask_d1, mask_d2 vectors
       └─ run_dynamic_d1_d2_kernel()           [kernels.py, @torch.jit.script]
            ├─ For each time step t:
            │    ├─ Compute DA(t)
            │    ├─ Compute sigmoid target: s_D1, s_D2
            │    ├─ Update alpha_D1(t) via Langmuir ODE  [k_on=3.24e-5, k_off=6.08e-6 ms⁻¹]
            │    ├─ Update alpha_D2(t) via Langmuir ODE  [k_on=1e-4,    k_off=2e-5   ms⁻¹]
            │    ├─ Compute R_eff, I_mod, scale_syn
            │    ├─ LIF integration
            │    ├─ Spike detection & reset
            │    └─ Synaptic propagation: I_syn += W^T · spikes
            └─ Return spike_records, v_traces
```

---

## 5. D1 vs D2 Receptor Comparison

| Property | D1 Receptor | D2 Receptor |
|---|---|---|
| $EC50$ | 4.0 nM | 8.0 nM |
| $\tau_{on}$ | 30876 ms (~8.6 h) | 10000 ms (~2.8 h) |
| $\tau_{off}$ | 164472 ms (~45.7 h) | 50000 ms (~13.9 h) |
| Response speed | Slow | ~3× faster than D1 |
| Effect on $R_{eff}$ | +15% (↑ excitability) | −10% (↓ excitability) |
| Bias current | +3.0 pA (excitatory) | −3.0 pA (inhibitory) |
| Synaptic scaling | +0.3 (enhance) | −0.2 (suppress) |
| Net effect | ↑ Excitability | ↓ Excitability |

---

## 6. Usage

```bash
# Default run (100s, DA=3.0 nM)
python main.py

# Custom duration and DA concentration
python main.py --duration 200000 --da 5.0

# Specify plot batch (0=Control, 1=Experiment)
python main.py --duration 100000 --da 3.0 --batch 1
```

Output is saved to `outputs/exp_<timestamp>/`.
