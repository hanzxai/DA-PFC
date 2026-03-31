# DA-PFC Spiking Neural Network Simulation

> **Dopamine modulation of Prefrontal Cortex** — LIF spiking neural network with D1/D2 receptor dynamics based on Langmuir binding kinetics.

本项目构建了一个前额叶皮层 (PFC) 的脉冲神经网络 (SNN) 模型，用于研究多巴胺 (DA) 通过 D1/D2 受体对 PFC 网络活动的调节作用。模型采用 PyTorch + JIT 编译实现高性能 GPU 加速仿真，支持多种给药模式和自动化分析。

---

## 1. 项目结构

```
DA-PFC/
├── main.py                    # 主入口 (CLI), 支持多种运行模式
├── config.py                  # 全局参数配置 (Single Source of Truth)
├── utils.py                   # 实验文件夹管理、配置保存、数据序列化
├── requirements.txt           # Python 依赖
├── simulation/
│   ├── __init__.py
│   └── runners.py             # 6 种仿真运行器 (静态/分步/D1/D1+D2/两阶段/Checkpoint恢复)
├── models/
│   ├── __init__.py
│   ├── network.py             # 网络结构构建 (连接矩阵 W, D1/D2 受体 Mask)
│   ├── pharmacology.py        # 药理学参数计算 (Sigmoid 激活函数)
│   └── kernels.py             # JIT 编译仿真内核 (@torch.jit.script, 1452 行)
├── analysis/
│   ├── __init__.py
│   ├── analyzer.py            # PFCAnalyzer: 群体发放率、FFT 频谱分析、统计报告
│   └── plotting.py            # 组合绘图函数 (3×2 布局: Full/BeforeDA/AfterDA × Control/Exp)
├── analyze_exp.py             # 后处理分析脚本 (加载已保存实验数据)
├── analyze_exp_results.py     # 批量实验结果分析
├── analyze_quick.py           # 快速分析脚本
├── analyze_current.py         # 电流分析脚本
├── paper/                     # 参考论文
└── outputs/                   # 实验输出 (git-ignored)
    └── exp_<timestamp>/
        ├── config.json            # 实验参数
        ├── raw_data.pkl           # 完整仿真数据 (spikes, traces, masks, final_state)
        ├── analysis_report.txt    # 自动生成的分析报告
        ├── combined_raster.png    # 3×2 Raster 图
        ├── combined_rates_all.png # 3×2 全亚群发放率图
        ├── combined_rates_E.png   # 3×2 兴奋性亚群发放率图
        └── combined_rates_I.png   # 3×2 抑制性亚群发放率图
```

---

## 2. 网络架构

### 2.1 群体组成

| 群体 | 数量 | 比例 | 受体表达 |
|---|---|---|---|
| 兴奋性 (E) | 824 | 82.4% | D1R: 21% of E; D2R: 25% of E |
| 抑制性 (I) | 176 | 17.6% | D1R: 30% of I; D2R: 8% of I |
| **总计** | **1000** | 100% | — |

### 2.2 受体亚群详情

| 亚群 | 数量 | 神经元索引范围 |
|---|---|---|
| E-D1 | 173 (= int(824 × 0.21)) | [0, 172] |
| E-D2 | 206 (= int(824 × 0.25)) | [173, 378] |
| E-Other | 445 | [379, 823] |
| I-D1 | 52 (= int(176 × 0.30)) | [824, 875] |
| I-D2 | 14 (= int(176 × 0.08)) | [876, 889] |
| I-Other | 110 | [890, 999] |

> 神经元索引: $[0, \ldots, 823]$ = E, $[824, \ldots, 999]$ = I.  
> `mask_d1`, `mask_d2` 为 $(N,)$ 布尔向量，标记表达对应受体的神经元。

### 2.3 连接参数

| 参数 | 值 | 说明 |
|---|---|---|
| 连接概率 | 0.02 | 随机稀疏 Erdős–Rényi 连接 |
| 兴奋性权重 $w_E$ | 3.0 pA | E→any |
| 抑制性权重 $w_I$ | −20.0 pA | I→any |

> 等效递归兴奋: $0.02 \times 824 \times 3.0 \times 0.1 \approx 4.9$

突触权重矩阵 $W \in \mathbb{R}^{N \times N}$，转置为 $W^T$ 用于高效批量矩阵乘法: `I_syn += spikes @ W_t`。

### 2.4 仿真 Batch 设计

两个并行 Batch 同时运行，共享**完全相同的初始条件和背景噪声**:

| Batch | DA 浓度 | 角色 |
|---|---|---|
| 0 | 0 nM (始终) | Control (对照组) |
| 1 | `da_level` nM (给药后) | Experiment (实验组) |

> **设计原理:** 两个 Batch 共享相同的初始膜电位 (`V_init` 以 `(1, N)` 生成后 expand) 和每个时间步的背景噪声 (`I_bg` 以 `(1, N)` 生成后 expand)。两组之间的**唯一差异**是 DA 浓度参数，确保观察到的差异完全归因于 DA 调节。

---

## 3. 数学模型

### 3.1 LIF 神经元动力学

**精确积分 (Exact Integration):**

$$
V_{\infty} = V_{rest} + R_{eff} \cdot I_{total}
$$

$$
V(t+dt) = V_{\infty} + (V(t) - V_{\infty}) \cdot e^{-dt / \tau_m}
$$

其中:
$$
\tau_m = R_{eff} \cdot C_m, \quad I_{total} = I_{syn} \cdot scale_{syn} + I_{bg} + I_{mod}
$$

**脉冲产生:** 若 $V > V_{th}$，发放脉冲并复位 $V \leftarrow V_{reset}$。

**不应期:** 神经元在 $[t_{spike},\ t_{spike} + t_{ref}]$ 期间被钳位在 $V_{reset}$。

**突触电流衰减 (指数衰减):**

$$
I_{syn}(t + dt) = I_{syn}(t) \cdot e^{-dt / \tau_{syn}} + \sum_j W_{ji} \cdot s_j(t)
$$

其中 $s_j(t) = 1$ 表示神经元 $j$ 在时刻 $t$ 发放。

---

### 3.2 DA 受体激活 — Sigmoid 目标值

受体 $r \in \{D1, D2\}$ 的瞬时 Sigmoid 目标激活度:

$$
s_r(t) = \frac{1}{1 + e^{-\beta \left( DA(t) - EC50_r \right)}}
$$

给药前或对照组: $s_r = 0$。

---

### 3.3 受体动力学 — Langmuir 结合动力学

受体占有率 $\alpha_r$ 遵循 Langmuir 结合 ODE:

$$
\frac{d\alpha_r}{dt} = k_{on,r} \cdot s_r(t) \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r
$$

**稳态解:**

$$
\alpha_r^{ss} = \frac{k_{on} \cdot s_r}{k_{on} \cdot s_r + k_{off}} = \frac{s_r}{s_r + K_d}, \quad K_d = \frac{k_{off}}{k_{on}}
$$

**Euler 离散化:**

$$
\alpha_r(t + dt) = \alpha_r(t) + \left[ k_{on,r} \cdot s_r \cdot (1 - \alpha_r) - k_{off,r} \cdot \alpha_r \right] \cdot dt
$$

> 注意: 由于 Langmuir 饱和效应，$\alpha_{ss} < s_r$（与 tau 方法或 k_on/k_off 方法中 $\alpha_{ss} = s_r$ 不同）。

---

### 3.4 DA 神经调节 — 参数组装

**有效膜电阻** (逐神经元):

$$
R_{eff} = R_{base} \cdot \left(1 + \varepsilon_{D1} \cdot \alpha_{D1} \cdot mask_{D1} - \varepsilon_{D2} \cdot \alpha_{D2} \cdot mask_{D2}\right)
$$

**调节电流:**

$$
I_{mod} = BIAS_{D1} \cdot \alpha_{D1} \cdot mask_{D1} + BIAS_{D2} \cdot \alpha_{D2} \cdot mask_{D2}
$$

**突触缩放因子:**

$$
scale_{syn} = 1 + \lambda_{D1} \cdot \alpha_{D1} \cdot mask_{D1} - \lambda_{D2} \cdot \alpha_{D2} \cdot mask_{D2}
$$

---

### 3.5 膜时间常数

$$
\tau_m = R_{eff} \cdot C_m
$$

| 神经元类型 | $C_m$ (pF) | 基础 $\tau_m$ (ms) |
|---|---|---|
| 兴奋性 (E) | 250 | 25.0 |
| 抑制性 (I) | 90 | 9.0 |

---

## 4. 参数表

### 4.1 网络架构参数

| 参数 | 符号 | 值 | 单位 | 说明 |
|---|---|---|---|---|
| `N_E` | $N_E$ | 824 | — | 兴奋性神经元数量 (82.4%) |
| `N_I` | $N_I$ | 176 | — | 抑制性神经元数量 (17.6%) |
| `N_TOTAL` | $N$ | 1000 | — | 总神经元数 |
| `CONN_PROB` | $p$ | 0.02 | — | 连接概率 |
| `W_EXC` | $w_E$ | 3.0 | pA | 兴奋性突触权重 |
| `W_INH` | $w_I$ | −20.0 | pA | 抑制性突触权重 |
| `FRAC_E_D1` | — | 0.21 | — | E 神经元中 D1R 表达比例 |
| `FRAC_E_D2` | — | 0.25 | — | E 神经元中 D2R 表达比例 |
| `FRAC_I_D1` | — | 0.30 | — | I 神经元中 D1R 表达比例 |
| `FRAC_I_D2` | — | 0.08 | — | I 神经元中 D2R 表达比例 |

---

### 4.2 LIF 神经元参数

| 参数 | 符号 | 值 | 单位 | 说明 |
|---|---|---|---|---|
| `V_REST` | $V_{rest}$ | −70.0 | mV | 静息膜电位 |
| `V_RESET` | $V_{reset}$ | −75.0 | mV | 脉冲后复位电位 (超极化) |
| `V_TH` | $V_{th}$ | −50.0 | mV | 脉冲阈值 |
| `R_BASE` | $R_{base}$ | 0.1 | GΩ (= mV/pA = 100 MΩ) | 基础膜电阻 |
| `C_E` | $C_E$ | 250.0 | pF | 兴奋性神经元膜电容 |
| `C_I` | $C_I$ | 90.0 | pF | 抑制性神经元膜电容 |
| `TAU_SYN` | $\tau_{syn}$ | 5.0 | ms | 突触电流衰减时间常数 |
| `T_REF` | $t_{ref}$ | 5.0 | ms | 绝对不应期 |
| `BG_MEAN` | $\mu_{bg}$ | 250.0 | pA | 背景电流均值 |
| `BG_STD` | $\sigma_{bg}$ | 50.0 | pA | 背景电流标准差 |

> **背景电流校准:** $V_{ss} = V_{rest} + R_{base} \times BG\_MEAN = -70 + 0.1 \times 250 = -45$ mV，超阈值驱动。噪声幅度: $R_{base} \times BG\_STD = 0.1 \times 50 = 5$ mV。

---

### 4.3 受体动力学参数

| 参数 | 符号 | 值 | 单位 | 说明 |
|---|---|---|---|---|
| `EC50_D1` | $EC50_{D1}$ | 4.0 | nM | D1 受体半效浓度 |
| `EC50_D2` | $EC50_{D2}$ | 8.0 | nM | D2 受体半效浓度 |
| `BETA` | $\beta$ | 1.0 | nM⁻¹ | Sigmoid 斜率系数 |
| `TAU_ON_D1` | $\tau_{on,D1}$ | 30876.1 | ms | D1 激活上升时间常数 |
| `TAU_OFF_D1` | $\tau_{off,D1}$ | 164472.5 | ms | D1 激活衰减时间常数 |
| `TAU_ON_D2` | $\tau_{on,D2}$ | 10000.0 | ms | D2 激活上升时间常数 (~D1 的 1/3) |
| `TAU_OFF_D2` | $\tau_{off,D2}$ | 50000.0 | ms | D2 激活衰减时间常数 (~D1 的 1/3) |

**Langmuir 速率常数 (用于 JIT 内核):**

| 导出量 | 公式 | 值 | 单位 |
|---|---|---|---|
| $k_{on,D1}$ | $1 / (\tau_{on,D1} - 3000)$ | $\approx 3.59 \times 10^{-5}$ | ms⁻¹ |
| $k_{off,D1}$ | $1 / (\tau_{off,D1} + 3000)$ | $\approx 5.97 \times 10^{-6}$ | ms⁻¹ |
| $k_{on,D2}$ | $1 / \tau_{on,D2}$ | $1.00 \times 10^{-4}$ | ms⁻¹ |
| $k_{off,D2}$ | $1 / \tau_{off,D2}$ | $2.00 \times 10^{-5}$ | ms⁻¹ |
| $K_{d,D1}$ | $k_{off,D1} / k_{on,D1}$ | $\approx 0.1664$ | — |
| $K_{d,D2}$ | $k_{off,D2} / k_{on,D2}$ | $0.2$ | — |

> 注意: D1 速率常数对 $\tau$ 值应用了 ±3000 ms 偏移后再计算 $k_{on}$ 和 $k_{off}$。

---

### 4.4 DA 调节强度参数

| 参数 | 符号 | 值 | 单位 | $\alpha=1$ 时的效果 |
|---|---|---|---|---|
| `EPS_D1` | $\varepsilon_{D1}$ | 0.15 | — | $R_{eff} = R_{base} \times 1.15$ (+15% 电阻, ↑ 兴奋性) |
| `EPS_D2` | $\varepsilon_{D2}$ | 0.10 | — | $R_{eff} = R_{base} \times 0.90$ (−10% 电阻, ↓ 兴奋性) |
| `BIAS_D1` | $BIAS_{D1}$ | +3.0 | pA | D1 神经元兴奋性偏置电流 |
| `BIAS_D2` | $BIAS_{D2}$ | −3.0 | pA | D2 神经元抑制性偏置电流 |
| `LAM_D1` | $\lambda_{D1}$ | 0.3 | — | D1 神经元突触缩放因子 |
| `LAM_D2` | $\lambda_{D2}$ | 0.2 | — | D2 神经元突触缩放因子 |

---

### 4.5 仿真控制参数

| 参数 | 符号 | 默认值 | 单位 | 说明 |
|---|---|---|---|---|
| `DT` | $dt$ | 1.0 | ms | 积分时间步长 |
| `DEFAULT_DURATION` | — | 15000.0 | ms (= 15 s) | config 中默认仿真时长 |
| `DEFAULT_DA_ONSET` | $t_{onset}$ | 10000.0 | ms (= 10 s) | 默认给药起始时间 |
| `--duration` (CLI) | $T$ | 100.0 | s (= 100000 ms) | CLI 默认仿真时长 |
| `--da` (CLI) | $[DA]$ | 3.0 | nM | DA 浓度 |
| `RANDOM_SEED` | — | 42 | — | 随机种子 (可复现) |

---

## 5. 仿真模式 (6 种内核)

### 5.1 内核总览

| # | 内核函数 | 说明 | DA 处理方式 | 对应 Runner |
|---|---|---|---|---|
| 1 | `run_batch_network` | 静态 DA | 固定调节参数 | `run_simulation_in_memory` |
| 2 | `run_batch_network_stepped` | 分步给药 | 在 onset 时刻瞬时切换 | `run_simulation_stepped` |
| 3 | `run_dynamic_d1_kernel` | D1 动力学 | D1 Langmuir ODE | `run_simulation_d1_kinetics` |
| 4 | `run_dynamic_d1_d2_kernel` | D1+D2 动力学 | 双 Langmuir ODE | `run_simulation_d1_d2_kinetics` (**默认**) |
| 5 | `run_dynamic_d1_d2_kernel_two_stage` | 两阶段给药 | 三相 DA 调度 | `run_simulation_d1_d2_two_stage` |
| 6 | `run_dynamic_d1_d2_kernel_from_state` | Checkpoint 恢复 | 从保存状态继续 | `run_simulation_from_checkpoint` |

### 5.2 默认模式: D1+D2 受体动力学 (Runner 4)

```
时间轴: |--- Baseline (0 nM) ---|--- DA Phase (target_da nM) ---|
        0                    da_onset                        duration
```

- Batch 0 (Control): 全程 0 nM
- Batch 1 (Experiment): da_onset 后施加 target_da nM
- alpha_D1 和 alpha_D2 均按 Langmuir ODE 缓慢爬升

### 5.3 两阶段给药模式 (Runner 5)

```
时间轴: |--- Baseline ---|--- Resting DA ---|--- DA Challenge ---|
        0             da_onset          phase2_onset           duration
                        (10s)              (30s)               (130s)
```

- **Phase 1** [0, da_onset): 0 nM (基线)
- **Phase 2** [da_onset, phase2_onset): da_level_1 nM (静息态 DA, 如 2.0 nM)
- **Phase 3** [phase2_onset, end): da_level_2 nM (DA 挑战, 如 15.0 nM)

模拟皮层先在静息态 DA 浓度下达到稳态，然后突然施加高浓度 DA 的场景。

### 5.4 Checkpoint 恢复模式 (Runner 6)

从之前仿真保存的 `raw_data.pkl` 中加载 `final_state`（包含 V, I_syn, t_last_spike, alpha_d1, alpha_d2），以此作为初始状态继续仿真，施加新的 DA 浓度。

---

## 6. 仿真流程

```
main.py (CLI 入口)
  ├─ parse_args()                          # 解析命令行参数
  ├─ setup_experiment_folder()             # 创建 outputs/exp_<timestamp>/
  ├─ save_args()                           # 保存 config.json
  │
  ├─ [模式判断] ─────────────────────────────────────────────────
  │   ├─ --resume 指定 → run_simulation_from_checkpoint()
  │   ├─ --da2 指定   → run_simulation_d1_d2_two_stage()
  │   └─ 默认         → run_simulation_d1_d2_kinetics()
  │
  ├─ Runner 内部流程:
  │   ├─ _init_network()
  │   │    ├─ torch.manual_seed(42)
  │   │    └─ create_network_structure()          [models/network.py]
  │   │         ├─ 构建 W 矩阵 (稀疏随机, p=0.02)
  │   │         └─ 构建 mask_d1, mask_d2 向量
  │   ├─ _build_record_indices()
  │   └─ _run_kernel_with_progress()              # 后台线程 + tqdm 进度条
  │        └─ run_dynamic_d1_d2_kernel()          [models/kernels.py, @torch.jit.script]
  │             ├─ 初始化: V_init (1,N) → expand to (2,N)
  │             └─ For each time step t:
  │                  ├─ 计算 DA(t): B0=0, B1=da_level (if t≥onset)
  │                  ├─ 更新 alpha_D1 (Langmuir ODE)
  │                  ├─ 更新 alpha_D2 (Langmuir ODE)
  │                  ├─ 组装: R_eff, I_mod, scale_syn
  │                  ├─ 突触衰减: I_syn *= exp(-dt/τ_syn)
  │                  ├─ 背景噪声: I_bg (1,N) → expand (共享)
  │                  ├─ 精确积分: V_inf + (V - V_inf) * exp(-dt/τ_m)
  │                  ├─ 不应期检查
  │                  ├─ 脉冲检测 & 复位
  │                  └─ 突触传播: I_syn += spikes @ W_t
  │
  ├─ save_raw_data()                       # 保存 raw_data.pkl (含 final_state)
  │
  ├─ 分析与绘图:
  │   ├─ PFCAnalyzer(data)                 # 初始化分析器
  │   ├─ plot_combined_raster()            # 3×2 Raster 图
  │   ├─ plot_combined_rates_all()         # 3×2 全亚群发放率
  │   ├─ plot_combined_rates_E()           # 3×2 兴奋性亚群发放率
  │   ├─ plot_combined_rates_I()           # 3×2 抑制性亚群发放率
  │   └─ analyzer.save_report()            # 生成分析报告
  │
  └─ 计时报告
```

---

## 7. D1 vs D2 受体对比

| 属性 | D1 受体 | D2 受体 |
|---|---|---|
| $EC50$ | 4.0 nM | 8.0 nM |
| $\tau_{on}$ | 30876 ms (~31 s) | 10000 ms (~10 s) |
| $\tau_{off}$ | 164472 ms (~164 s) | 50000 ms (~50 s) |
| $k_{on}$ | ~3.59e-5 ms⁻¹ | 1.0e-4 ms⁻¹ |
| $k_{off}$ | ~5.97e-6 ms⁻¹ | 2.0e-5 ms⁻¹ |
| $K_d$ | ~0.166 | 0.2 |
| 响应速度 | 慢 | ~3× 快于 D1 |
| 对 $R_{eff}$ 的影响 | +15% (↑ 兴奋性) | −10% (↓ 兴奋性) |
| 偏置电流 | +3.0 pA (兴奋性) | −3.0 pA (抑制性) |
| 突触缩放 | +0.3 (增强) | −0.2 (抑制) |
| 净效应 | ↑ 兴奋性 | ↓ 兴奋性 |

---

## 8. 分析与输出

### 8.1 输出文件

每次实验保存到 `outputs/exp_<timestamp>/`:

| 文件 | 说明 |
|---|---|
| `config.json` | 实验参数配置 |
| `raw_data.pkl` | 完整仿真数据 (spikes, v_traces, masks, groups_info, final_state) |
| `analysis_report.txt` | 自动生成的综合分析报告 |
| `combined_raster.png` | 3×2 Raster 图 (Full/BeforeDA/AfterDA × Control/Exp) |
| `combined_rates_all.png` | 3×2 全亚群 (6 组) 发放率曲线 |
| `combined_rates_E.png` | 3×2 兴奋性亚群 (E-D1/E-D2/E-Other) 发放率曲线 |
| `combined_rates_I.png` | 3×2 抑制性亚群 (I-D1/I-D2/I-Other) 发放率曲线 |

### 8.2 绘图布局

所有图表采用统一的 **3×2 布局**:

```
         Control (B0)          Experiment (B1)
Row 0:   Full time-range       Full time-range
Row 1:   Before DA (zoom)      Before DA (zoom)
Row 2:   After DA (zoom)       After DA (zoom)
```

- Y 轴在所有面板间统一，便于公平比较
- Full 行使用 100ms 时间窗计算发放率，Zoom 行使用 5ms 时间窗
- 两阶段模式下自动绘制两条垂直线标记 DA 切换时刻

### 8.3 分析报告内容

`analysis_report.txt` 包含:

1. **Mean Firing Rate 分析表** — 8 个亚群 (E-D1, E-D2, E-Other, All-E, I-D1, I-D2, I-Other, All-I) 的 Baseline/Post-DA/Overall 平均发放率，Control vs Exp 对比
2. **Baseline 一致性检查** — 验证 B0 和 B1 在给药前发放率差异 < 0.5 Hz
3. **DA 效应摘要** — Post-DA 阶段 Exp vs Ctrl 的发放率变化 (Hz 和 %)
4. **FFT 频率分析表** — Welch 功率谱估计的主频对比
5. **逐亚群详细频率报告** — Baseline / DA Phase / Late DA 三阶段频率分析

### 8.4 典型仿真结果示例

以下为两阶段给药模式 (DA: 2.0 nM → 15.0 nM, 130s 仿真) 的典型结果:

| 亚群 | Ctrl Post-DA (Hz) | Exp Post-DA (Hz) | DA 效应 |
|---|---|---|---|
| E-D1 | 18.24 | 20.31 | **+2.07 Hz (+11.4%) ↑** |
| E-D2 | 17.85 | 15.10 | **−2.76 Hz (−15.4%) ↓** |
| E-Other | 18.19 | 18.14 | −0.05 Hz (−0.3%) → |
| I-D1 | 42.85 | 46.84 | **+4.00 Hz (+9.3%) ↑** |
| I-D2 | 43.16 | 37.72 | **−5.44 Hz (−12.6%) ↓** |
| I-Other | 42.71 | 42.63 | −0.08 Hz (−0.2%) → |

> **关键发现:**
> - D1 受体激活 → 兴奋性增强 (E-D1 ↑11.4%, I-D1 ↑9.3%)
> - D2 受体激活 → 兴奋性降低 (E-D2 ↓15.4%, I-D2 ↓12.6%)
> - 非受体神经元 (E-Other, I-Other) 几乎不受影响 (< 0.3%)
> - Baseline 一致性检查通过 (所有亚群 B0 vs B1 差异 < 0.5 Hz)

---

## 9. JIT 内核实现细节

### 9.1 参数一致性校验

`kernels.py` 中的 `@torch.jit.script` 函数无法访问 Python 全局模块，因此所有 LIF 和药理学参数以字面量硬编码在每个内核函数顶部。为防止参数不同步:

- `verify_kernel_params_consistent()` 函数在启动时自动校验 `kernels.py` 硬编码值与 `config.py` 是否一致
- 不一致时立即抛出 `AssertionError`，打印具体差异

### 9.2 alpha 更新函数

`kernels.py` 为 D1 和 D2 各提供 3 种 alpha 更新实现:

| 函数 | 方程形式 | 稳态 | 说明 |
|---|---|---|---|
| `compute_alpha_d{1,2}_step` | $d\alpha/dt = (s - \alpha) / \tau$ | $\alpha_{ss} = s$ | tau 版本 |
| `compute_alpha_d{1,2}_step_kon_koff` | $d\alpha/dt = k_{on}(s-\alpha)^+ - k_{off}(\alpha-s)^+$ | $\alpha_{ss} = s$ | k_on/k_off 版本 |
| `compute_alpha_d{1,2}_step_langmuir` | $d\alpha/dt = k_{on} \cdot s \cdot (1-\alpha) - k_{off} \cdot \alpha$ | $\alpha_{ss} < s$ | **Langmuir 版本 (当前使用)** |

### 9.3 进度条机制

由于 `@torch.jit.script` 内核无法被 Python 直接插桩，采用后台线程 + 渐进式进度估算:

- JIT 编译阶段显示 "compiling"
- 计算阶段按 $1 - 1/(1+kt)$ 曲线推进 (永远 < 100%)
- 内核完成后跳至 100%

---

## 10. 使用方法

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

### 10.2 基本运行

```bash
# 默认运行 (100s, DA=3.0 nM, GPU 0)
python main.py

# 指定 DA 浓度
python main.py --da 4.0

# 指定仿真时长和 GPU
python main.py --duration 50 --da 3.0 --gpu 1

# 无 GPU 时自动回退到 CPU
python main.py --gpu 999
```

### 10.3 两阶段给药

```bash
# 静息态 DA=2.0 nM → 挑战 DA=15.0 nM
python main.py --da 2.0 --da2 15.0

# 自定义第二阶段起始时间 (40s)
python main.py --da 2.0 --da2 15.0 --phase2-onset 40
```

### 10.4 从 Checkpoint 恢复

```bash
# 从之前的实验恢复，施加新的 DA 浓度
python main.py --resume outputs/exp_2026-03-31_17-00-14/raw_data.pkl --da 15.0

# 恢复并指定新的仿真时长
python main.py --resume outputs/exp_xxx/raw_data.pkl --da 10.0 --duration 200
```

### 10.5 后处理分析

```bash
# 分析已保存的实验数据
python analyze_exp.py outputs/exp_2026-03-31_17-12-57
```

### 10.6 CLI 参数一览

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--duration` | float | 100.0 | 仿真总时长 (秒) |
| `--da` | float | 3.0 | 目标给药浓度 (nM) |
| `--da2` | float | None | 两阶段模式: 第二阶段 DA 浓度 (nM) |
| `--phase2-onset` | float | None | 两阶段模式: 第二阶段起始时间 (秒) |
| `--resume` | str | None | Checkpoint 恢复: raw_data.pkl 路径 |
| `--batch` | int | 1 | 绘图 Batch ID (0=Control, 1=Exp) |
| `--gpu` | int | 0 | GPU 卡号 (无效值自动回退 CPU) |

---

## 11. 依赖

| 包 | 最低版本 | 用途 |
|---|---|---|
| PyTorch | ≥ 2.0.0 | GPU 加速 SNN 仿真 + JIT 编译 |
| NumPy | ≥ 1.24.0 | 数组运算、随机种子 |
| SciPy | ≥ 1.10.0 | Welch FFT 频谱分析 |
| Matplotlib | ≥ 3.7.0 | Raster 图、发放率曲线绑图 |
| tqdm | ≥ 4.65.0 | 仿真进度条 |

推荐使用 CUDA GPU 以获得最佳性能。CPU 模式也可运行，但速度较慢。
