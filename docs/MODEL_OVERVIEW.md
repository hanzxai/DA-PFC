# DA-PFC 建模总览（Model Overview）

> **文件作用**：把当前代码（`config.py` / `models/network.py` / `models/kernels.py` 中的 `run_wm_kernel_dual`）里实际跑的 **方程**、**参数**、**时序**整理成一份可参考的建模文档。
>
> **对应实验**：`exp_wm_da_scan.py`（DA Slow-Scan WM Diagnostic）。
> 黄金参数 reference 实验：`outputs/exp_2026-05-12_11-57-40_da_scan_v3_goldparams_DA2to12nM/`。

---

## 1. 网络结构

### 1.1 总体规模

| 群体 | 数量 | 占比 | 备注 |
|------|------|------|------|
| `N_E`  兴奋性 | 824 | 82.4 % | LIF, τ_m = 25 ms |
| `N_I`  抑制性 | 176 | 17.6 % | LIF, τ_m = 9 ms |
| `N_TOTAL` | 1000 | — | — |

### 1.2 子群划分（`models/network.py: create_wm_network`）

兴奋性 824 个被切成（按神经元 index 顺序）：

| 区段 | Index 区间 | 大小 | 受体 | 角色 |
|------|------------|------|------|------|
| **Mem-A**     | `[0, 60)`   | 60  | D1 | WM 池 A（接收 cue） |
| **Mem-B**     | `[60, 120)` | 60  | D1 | WM 池 B（竞争对照） |
| **D1-BG**     | `[120, 173)`| 53  | D1 | E-D1 中余下，做背景 |
| **E-D2**      | `[173, 379)`| 206 | D2 | DA D2 通路 |
| **E-Other**   | `[379, 824)`| 445 | —  | 无 DA 受体 |

抑制性 176 个：

| 区段 | Index 区间 | 大小 | 受体 | 角色 |
|------|------------|------|------|------|
| **I-D1**    | `[824, 876)`  | 52 | D1 | DA D1 抑制 |
| **I-D2**    | `[876, 890)`  | 14 | D2 | DA D2 抑制 |
| **I-Other** | `[890, 940)`  | 50 | —  | 全局背景抑制 |
| **I-WM**    | `[940, 1000)` | 60 | —  | **WTA shared inhibition pool**（特意放在 I-Other 尾部，不被 DA 调制） |

> 关键设计：Mem-A 和 Mem-B 都在 E-D1 段内，**对 D1 调制对称**；I-WM 放在无受体段，使 DA 不会"自掐"WTA 反馈环路。

### 1.3 连接（`create_wm_network`）

1. **基础稀疏连接**（全网通用）：
   - `CONN_PROB = 0.02`，对角线为 0
   - E→任意 权重 `W_EXC = +3.0 pA`
   - I→任意 权重 `W_INH = -20.0 pA`

2. **池内强递归**（被覆盖重画的稠密块）：
   - Mem-A↔Mem-A：`p = 0.5`，`J = 4.0 pA`（`WM_INTRA_PROB`、`WM_INTRA_W`）
   - Mem-B↔Mem-B：同上
   - 跨池 Mem-A↔Mem-B：`p = 0`（硬竞争）

3. **WTA 共享抑制环**（仅在 `WM_USE_WTA = True` 时启用）：
   - Mem-{A,B} → I-WM（兴奋）：`p = 0.20`, `J = +3.0 pA`
   - I-WM → Mem-{A,B}（抑制）：`p = 0.15`, `J = −3.5 pA`

4. **双通道权重拆分**（`run_wm_kernel_dual` 用）：
   - `W_nmda_t`：**只**保留 Mem-A↔Mem-A、Mem-B↔Mem-B 两个块（NMDA，τ=100 ms）
   - `W_ampa_t`：其余所有连接（AMPA / GABA-A，τ=5 ms）

---

## 2. 神经元方程（LIF）

每个神经元 i 都满足：

$$
C_m \frac{dV_i}{dt} = -\frac{V_i - V_{rest}}{R_{eff,i}} + I_{syn,i} + I_{bg,i} + I_{mod,i} - I_{adapt,i}
$$

发放规则：

$$
V_i \ge V_{th} \;\Rightarrow\; \text{spike}, \quad V_i \leftarrow V_{reset}, \quad t_{ref} = 5\,\text{ms}
$$

代码里采用解析的 exponential Euler：

```
V_inf = V_rest + R_eff * I_total
tau_m = R_eff * C_m
V_new = V_inf + (V - V_inf) * exp(-dt / tau_m)
```

### 2.1 参数（来自 `config.py`）

| 符号 | 值 | 含义 |
|------|----|------|
| `V_REST`  | −70 mV | 静息电位 |
| `V_RESET` | −75 mV | 复位电位 |
| `V_TH`    | −50 mV | 阈值 |
| `R_BASE`  | 0.1 GΩ | 基础膜电阻 |
| `C_E`     | 250 pF | E 神经元电容 → τ_m = 25 ms |
| `C_I`     |  90 pF | I 神经元电容 → τ_m =  9 ms |
| `T_REF`   |   5 ms | 不应期 |
| `DT`      |   1 ms | 时间步长 |

### 2.2 背景驱动（fluctuation-driven regime）

$$
I_{bg,i}(t) = \mathcal{N}(\mu_{bg},\; \sigma_{bg}^2) + \Delta I_{mem} \cdot \mathbb{1}[i \in \text{Mem-A} \cup \text{Mem-B}]
$$

- `BG_MEAN = 200 pA` → V_∞ = −70 + 0.1·200 = −50 mV ≈ V_th（临界点驱动）
- `BG_STD  =  25 pA` → σ_V = 2.5 mV（决定 baseline 发放率）
- `WM_MEM_BG_OFFSET`（Scheme-B）：仅施加在 Mem-A∪Mem-B 上的 DC 偏置；当前 v3 实验用 **−18 pA**

---

## 3. 突触动力学（Dual-Channel，Wang-2002）

### 3.1 AMPA 通道（快，无饱和）

发生在 **除池内 E→E 之外的所有连接**：E-BG、I 抑制、E↔I-WM、I-WM→Mem 等。

$$
\tau_{AMPA}\,\frac{dI_{ampa}}{dt} = -I_{ampa} + W_{ampa}^{\top} \cdot \mathbf{spike}(t)
$$

- `TAU_AMPA = 5 ms`
- 线性叠加，无饱和

### 3.2 NMDA 通道（慢，饱和闸门）

仅作用于 **池内 E→E**（Mem-A↔Mem-A、Mem-B↔Mem-B）。每个突触前神经元 j 维护一个门控变量 s_j ∈ [0,1]：

$$
\frac{ds_j}{dt} = -\frac{s_j}{\tau_{NMDA}} + \alpha_{gate}\,(1 - s_j)\,\mathbf{spike}_j(t)
$$

突触后电流：

$$
I_{nmda,i}(t) = \sum_j W_{nmda,ij}\, s_j(t)
$$

- `TAU_NMDA = 100 ms`
- `NMDA_ALPHA_GATE = 0.5`（每个 spike 把 s 推一半到 1）
- **关键非线性**：(1 − s_j) 饱和因子 → 池内递归 EPSC 不会爆发，给出双稳态 f-I 曲线

### 3.3 总突触输入

$$
I_{syn,i}(t) = I_{ampa,i}(t) + I_{nmda,i}(t)
$$

随后被 D1/D2 突触缩放因子 `scale_syn` 调节（见第 4 节）。

---

## 4. DA 受体动力学

### 4.1 Sigmoid 激活靶值

$$
S_{D1}(\text{DA}) = \frac{1}{1 + \exp(-\beta\,(\text{DA} - EC50_{D1}))}, \quad
S_{D2}(\text{DA}) = \frac{1}{1 + \exp(-\beta\,(\text{DA} - EC50_{D2}))}
$$

- `BETA = 1.0`
- `EC50_D1 = 4 nM`（D1 高亲和力，低阈值）
- `EC50_D2 = 8 nM`（D2 低亲和力，高阈值）

### 4.2 Langmuir 受体占用率（一阶动力学）

$$
\frac{d\alpha_{D1}}{dt} = k_{on}^{D1}\, S_{D1}\,(1 - \alpha_{D1}) - k_{off}^{D1}\, \alpha_{D1}
$$

$$
\frac{d\alpha_{D2}}{dt} = k_{on}^{D2}\, S_{D2}\,(1 - \alpha_{D2}) - k_{off}^{D2}\, \alpha_{D2}
$$

| 通道 | k_on 计算 | k_off 计算 | 数值 |
|------|-----------|-----------|------|
| D1 | `1 / (TAU_ON_D1 − 3000)` | `1 / (TAU_OFF_D1 + 3000)` | k_on ≈ 3.6e-5 ms⁻¹, k_off ≈ 6e-6 ms⁻¹ |
| D2 | `1 / TAU_ON_D2`           | `1 / TAU_OFF_D2`         | k_on = 1e-4 ms⁻¹,  k_off = 2e-5 ms⁻¹  |

> D2 比 D1 快 ~3 ×（D2 偶联 Gi，信号链短）。

### 4.3 三种调制路径（作用在 D1/D2 mask 区域）

每个 timestep 用 α(t) 调节当前神经元的：

| 调制 | 公式 | 系数（α=1 时） |
|------|------|---------------|
| **Gain (R 调制)**     | `R_eff = R_base · (1 + EPS_D1·α_D1·m_D1 − EPS_D2·α_D2·m_D2)` | EPS_D1=0.015, EPS_D2=0.01 |
| **Bias (DC 注入)**    | `I_mod += BIAS_D1·α_D1·m_D1 + BIAS_D2·α_D2·m_D2`            | BIAS_D1=+3 pA, BIAS_D2=−3 pA |
| **Synaptic Scaling**  | `scale_syn += LAM_D1·α_D1·m_D1 − LAM_D2·α_D2·m_D2`           | LAM_D1=0.3, LAM_D2=0.2 |

整体上对带 D1 受体的神经元（含 Mem-A/B）：DA↑ → R↑、I_mod↑、突触增益↑（净兴奋）；
带 D2 的神经元（主要是 E-D2、I-D2）：DA↑ → R↓、I_mod↓、突触增益↓（净抑制）。

---

## 5. SFA（Spike-Frequency Adaptation）

仅 Mem-A∪Mem-B 神经元有，模拟 Ca²⁺-依赖的 K⁺ 后超极化电流：

$$
\frac{dw_{adapt}}{dt} = -\frac{w_{adapt}}{\tau_{sfa}}, \qquad w_{adapt} \leftarrow w_{adapt} + b_{sfa}\;\text{on each spike}
$$

$$
I_{adapt,i}(t) = w_{adapt,i}(t) \cdot \mathbb{1}[i \in \text{Mem-A} \cup \text{Mem-B}]
$$

- v3 实验：`SFA_B = 0` → SFA 关闭（让 attractor 平坦），但代码里默认 `WM_SFA_B = 0.8`, `WM_SFA_TAU = 2000 ms`。

---

## 6. 工作记忆任务时序（WM Protocol）

### 6.1 阶段定义（v3 标准实验）

| 阶段 | 区间 (ms) | 持续 | 内容 |
|------|-----------|------|------|
| **Baseline** | `[0, 5000)`        | 5.0 s  | 网络静息（DA=2 nM，无 cue） |
| **Cue-A**    | `[5000, 6500)`     | 1.5 s  | 向 Mem-A 注入 `WM_CUE_AMPLITUDE` (v3=300 pA) DC |
| **Delay**    | `[6500, 76500)`    | 70.0 s | 无外部 cue，**DA 在此期内做梯形扫描**（仅 Batch-1） |
| **Probe**    | `[76500, 77000)`   | 0.5 s  | 探针窗口（v3 用 500 ms） |

总仿真时长：**77 s**。

### 6.2 时间线示意

```
   t (s)   0          5    6.5                                   76.5  77
           │          │    │                                       │   │
   ── Stim ┼──────────┼──■■┼───────────────────────────────────────┼───┼──
   Cue-A   │ Baseline │  CUE  │            Delay (70 s)            │ P │
           │          │      │                                     │ ε │
   ── DA ──┼──────────┼──────┴─╲              ┌────────╲           ┴───┼──
   Batch1  │          │  2 nM   ╲ ramp-up    /          ╲ ramp-dn │ 2  │
           │          │  ↓      ╲ 30 s      / 12 nM      ╲ 30 s   │ nM │
           │          │  ────────╲          ╱  hold  10 s ╲       │    │
           │          │           ╲────────╱                ╲─────┘    │
   ── DA ──┼─── 2 nM constant ───────────────────────────────────────  │
   Batch0  │  (control: 整段恒定 2 nM)                                 │
           │                                                           │
```

### 6.3 关键时间戳（v3 实验，从 `config.json` 读出）

| 事件 | 时刻 (ms) | 含义 |
|------|-----------|------|
| `t = 0`            | — | 仿真开始 |
| `cue_on`           | 5000  | Mem-A 开始注入 +300 pA |
| `cue_off`          | 6500  | Mem-A cue 撤去 |
| `da_pulse_on`      | 6500  | Batch-1 DA 开始上升（从 cue_off 起） |
| `ramp_up_end`      | 36500 | DA 从 2 → 12 nM 上升完成 |
| `ramp_down_begin`  | 46500 | hold 10 s 结束，开始下降 |
| `da_pulse_off`     | 76500 | DA 回到 2 nM |
| `probe_end`        | 77000 | 仿真结束 |

> **重要**：Batch-1 的 DA pulse 是 **紧接 cue 之后**（`pulse_on = cue_off`），不是从 0 开始。两 batch 在 [0, cue_off) 区间完全 bit-exact。

---

## 7. DA 浓度时序（梯形扫描）

由 `models/kernels.py: run_wm_kernel_dual` 内联实现：

```
              ┌──── da_pulse ────┐
              │                  │
       up    ╱                    ╲   down
            ╱                      ╲
da_base ──┴────────────────────────┴────── da_base
          ↑    ↑                ↑    ↑
     pulse_on  ramp_up_end  ramp_dn_beg  pulse_off
```

代码片段（核心 6 行）：

```python
if current_time < pulse_on or current_time >= pulse_off:
    da_exp = da_base
elif current_time < ramp_up_end:                           # 上升段
    frac = (current_time - pulse_on) / ramp_up_ms
    da_exp = da_base + (da_pulse - da_base) * frac
elif current_time >= ramp_down_beg:                        # 下降段
    frac = (pulse_off - current_time) / ramp_down_ms
    da_exp = da_base + (da_pulse - da_base) * frac
else:                                                       # hold
    da_exp = da_pulse
```

### 7.1 v3 黄金参数实验中的 DA(t)

| 参数 | 值 |
|------|-----|
| `da_base`    | 2.0 nM |
| `da_peak`    | 12.0 nM |
| `ramp_up_ms` | 30 000 ms |
| `hold_ms`    | 10 000 ms |
| `ramp_down_ms` | 30 000 ms |

经过 Langmuir 滤波后实际占用率：
- α_D1：约 0.25 → 0.79（DA=12 nM 时）
- α_D2：约 0.01 → 0.80（DA=12 nM 时）

---

## 8. 双 Batch 设计

| Batch | 用途 | DA 协议 |
|-------|------|---------|
| **Batch 0** | Control | 全程恒定 `da_base = 2 nM` |
| **Batch 1** | Experiment | 梯形扫描：`2 → 12 → 2 nM`（在 delay 期内） |

两个 batch **共享同一个 W、同一个网络结构、同一个随机背景流**（kernel 里 `I_bg = randn(1,N).expand(2,-1)`）；唯一区别就是 DA 时序，所以 Δ(B1−B0) 完全是 DA 调制效应。

---

## 9. Cue 注入机制

```python
if cue_a_onset <= current_time < cue_a_offset:
    I_mod = I_mod + cue_a_amplitude * stim_mask_a   # (N,) -> 仅 Mem-A 索引为 1
```

- `cue_a_amplitude` (v3) = **300 pA**
- `stim_mask_a`：(N,) 向量，Mem-A 索引为 1，其余为 0
- Cue 是 **deterministic DC 注入**（不是 Poisson 输入），叠加在 Mem-A 神经元的 `I_total` 上
- v3 没有用 cue-B（`cue_b_amplitude=0`），Mem-B 自始至终只接收 baseline + WTA 反馈

---

## 10. 总电流装配（每 timestep）

```
I_syn   = I_ampa + (s_nmda @ W_nmda_t)            # 突触
I_bg    = N(BG_MEAN, BG_STD) + WM_MEM_BG_OFFSET   # 背景（含 Mem 池偏置）
I_mod   = BIAS_D1*α_D1*m_D1 + BIAS_D2*α_D2*m_D2   # DA bias
        + cue_amp * stim_mask  (在 cue 窗口内)     # Cue
scale_syn = 1 + LAM_D1*α_D1*m_D1 - LAM_D2*α_D2*m_D2
I_adapt = w_adapt * mem_pool_mask                 # SFA
                                                   
I_total = I_syn * scale_syn + I_bg + I_mod - I_adapt
R_eff   = R_base * (1 + EPS_D1*α_D1*m_D1 - EPS_D2*α_D2*m_D2)
```

把 `I_total` 与 `R_eff` 代入第 2 节的 LIF 方程更新 V → 判定 spike → 更新 `I_ampa, s_nmda, w_adapt, t_last_spike`。循环。

---

## 11. 主要参数索引（一表流）

| 类别 | 符号 | 值 | 来源 |
|------|------|-----|------|
| 网络 | N_E / N_I | 824 / 176 | `config.N_E`, `config.N_I` |
| 网络 | CONN_PROB | 0.02 | `config.CONN_PROB` |
| 网络 | W_EXC / W_INH | +3.0 / −20.0 pA | `config.W_*` |
| LIF  | V_th, V_reset, V_rest | −50, −75, −70 mV | `config.V_*` |
| LIF  | C_E, C_I | 250, 90 pF | `config.C_*` |
| LIF  | T_REF | 5 ms | `config.T_REF` |
| 背景 | BG_MEAN, BG_STD | 200, 25 pA | `config.BG_*` |
| 突触 | TAU_AMPA, TAU_NMDA | 5, 100 ms | `config.TAU_*` |
| 突触 | NMDA α_gate | 0.5 | `config.NMDA_ALPHA_GATE` |
| WM 池 | pool_size | 60 | `config.WM_POOL_SIZE` |
| WM 池 | intra_prob, intra_w | 0.5, 4.0 pA | `config.WM_INTRA_*` |
| WTA  | iwm_size | 60 | `config.WM_IWM_SIZE` |
| WTA  | E2I (p, J) | 0.20, +3.0 pA | `config.WM_E2I_*` |
| WTA  | I2E (p, J) | 0.15, −3.5 pA | `config.WM_I2E_*` |
| Cue  | WM_CUE_AMPLITUDE | 350 pA (v3=300) | `config.WM_CUE_AMPLITUDE` |
| Mem 偏置 | WM_MEM_BG_OFFSET | 0 (v3=−18) pA | `config.WM_MEM_BG_OFFSET` |
| SFA  | b_sfa, τ_sfa | 0.8 pA, 2000 ms (v3=0) | `config.WM_SFA_*` |
| DA   | EC50_D1, EC50_D2 | 4, 8 nM | `config.EC50_*` |
| DA   | β | 1.0 | `config.BETA` |
| DA   | EPS_D1, EPS_D2 | 0.015, 0.01 | `config.EPS_*` |
| DA   | BIAS_D1, BIAS_D2 | +3, −3 pA | `config.BIAS_*` |
| DA   | LAM_D1, LAM_D2 | 0.3, 0.2 | `config.LAM_*` |
| DA   | τ_on D1, τ_off D1 | 30876, 164472 ms | `config.TAU_*_D1` |
| DA   | τ_on D2, τ_off D2 | 10000, 50000 ms | `config.TAU_*_D2` |

---

## 12. 实验复现命令（v3 黄金参数）

```bash
python experiments/exp_wm_da_scan.py \
    --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl \
    --da 2.0 \
    --da-peak 12.0 \
    --ramp-up-ms 30000 \
    --hold-ms 10000 \
    --ramp-down-ms 30000 \
    --baseline-ms 5000 \
    --cue-ms 1500 \
    --probe-ms 500 \
    --pool-size 60 \
    --intra-prob 0.5 \
    --intra-w 4.0 \
    --mem-bg-offset -18.0 \
    --cue-amp 300 \
    --sfa-b 0.0 \
    --tag da_scan_v3_goldparams
```

输出目录命名：`outputs/exp_<timestamp>_da_scan_v3_goldparams_DA2to12nM/`。

---

## 13. 与文献对照

| 模块 | 文献 | 复现 |
|------|------|------|
| LIF + AMPA/NMDA dual-channel | Wang 2002 (J Neurosci) | ✅ 双通道 + NMDA 饱和门 |
| 池内 WTA via shared inhibition | Compte-Brunel-Wang 2000 | ✅ I-WM 结构 |
| D1 inverted-U gain | Vijayraghavan 2007 | 🔄 上升支已观察到，下降支待 DA>15 nM 验证 |
| D2 cAMP / Gi 通路 | Seamans-Yang 2004 | ✅ EC50_D2=8 nM、EPS_D2 反向调制 |
| Langmuir receptor occupancy | 文献参数（项目内）| ✅ 一阶动力学 |
| SFA via Ca²⁺-K⁺ | Compte 2000 / Brette-Gerstner 2005 | ✅（v3 关闭，可选） |

---

*文档生成时间：2026-05-12，对应代码 commit 状态：`run_wm_kernel_dual` 的双通道 + WTA + Langmuir + DA trapezoid 完整版本。*
