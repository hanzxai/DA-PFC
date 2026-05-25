# DA-PFC 建模公式集（Model Equations）

> **范围**：只列建模相关的数学方程；网络结构、参数数值、实验时序请见 [MODEL_OVERVIEW.md](MODEL_OVERVIEW.md)。
>
> **符号约定**：i 为后突触神经元下标，j 为前突触神经元下标；m_D1, m_D2 为 D1/D2 受体掩码（属于则为 1，否则为 0）；𝟙[·] 为指示函数。

---

## 1. LIF 神经元动力学

### 1.1 膜电位演化

$$
C_{m,i}\,\frac{dV_i}{dt} \;=\; -\frac{V_i - V_{rest}}{R_{eff,i}(t)} \;+\; I_{total,i}(t)
$$

### 1.2 发放与复位

$$
V_i(t^-) \ge V_{th} \;\Longrightarrow\;
\begin{cases}
\text{spike}_i(t) = 1 \\
V_i(t^+) \leftarrow V_{reset} \\
t_{ref}\text{-期内 }V_i\text{ 钳位}
\end{cases}
$$

### 1.3 离散时间 Exponential Euler 更新

令 $V_\infty = V_{rest} + R_{eff}\,I_{total}$，$\tau_m = R_{eff}\,C_m$：

$$
V_i(t + \Delta t) \;=\; V_\infty + \big(V_i(t) - V_\infty\big)\,e^{-\Delta t / \tau_m}
$$

---

## 2. 总输入电流装配

$$
\boxed{\;I_{total,i}(t) \;=\; I_{syn,i}(t)\cdot \text{scale}_{syn,i}(t) \;+\; I_{bg,i}(t) \;+\; I_{mod,i}(t) \;-\; I_{adapt,i}(t)\;}
$$

其中：

$$
I_{syn,i}(t) \;=\; I_{ampa,i}(t) \;+\; I_{nmda,i}(t)
$$

$$
I_{mod,i}(t) \;=\; \text{BIAS}_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;+\; \text{BIAS}_{D2}\,\alpha_{D2}(t)\,m_{D2,i} \;+\; I_{cue,i}(t)
$$

---

## 3. 突触动力学（双通道，Wang 2002）

### 3.1 AMPA 通道（线性，无饱和）

后突触电流直接作为状态变量：

$$
\tau_{AMPA}\,\frac{dI_{ampa,i}}{dt} \;=\; -I_{ampa,i} \;+\; \tau_{AMPA}\sum_j W^{ampa}_{ij}\,\sum_k \delta(t - t_j^{(k)})
$$

离散等价：

$$
I_{ampa,i}(t + \Delta t) \;=\; I_{ampa,i}(t)\,e^{-\Delta t / \tau_{AMPA}} \;+\; \sum_j W^{ampa}_{ij}\,\text{spike}_j(t)
$$

### 3.2 NMDA 通道（饱和门控）

每个**前突触神经元** j 维护一个 NMDA 门控变量 $s_j \in [0, 1]$：

$$
\boxed{\;\frac{ds_j}{dt} \;=\; -\frac{s_j}{\tau_{NMDA}} \;+\; \alpha_{gate}\,(1 - s_j)\,\sum_k \delta(t - t_j^{(k)})\;}
$$

离散等价（先衰减后跳变）：

$$
s_j(t + \Delta t) \;=\; s_j(t)\,e^{-\Delta t/\tau_{NMDA}} \;+\; \alpha_{gate}\,(1 - s_j(t))\,\text{spike}_j(t)
$$

后突触 NMDA 电流：

$$
I_{nmda,i}(t) \;=\; \sum_j W^{nmda}_{ij}\, s_j(t)
$$

### 3.3 NMDA 稳态发放-门控关系

在稳态发放率 $r_j$ 下：

$$
s_\infty(r_j) \;=\; \frac{\alpha_{gate}\,\tau_{NMDA}\,r_j}{1 + \alpha_{gate}\,\tau_{NMDA}\,r_j}
$$

> 当 $r \to \infty$ 时 $s_\infty \to 1$，即受体饱和；这是 attractor 双稳态的根源。

---

## 4. 背景电流

$$
I_{bg,i}(t) \;=\; \mu_{bg} \;+\; \sigma_{bg}\,\xi(t) \;+\; \Delta I_{mem}\,\mathbb{1}\!\left[i \in \mathrm{Mem\text{-}A} \cup \mathrm{Mem\text{-}B}\right]
$$

- $\xi(t) \sim \mathcal{N}(0, 1)$ 每 timestep 独立白噪声（**两个 batch 共享同一抽样**）
- $\Delta I_{mem}$：Scheme-B 偏置（仅作用于 Mem 池）

---

## 5. DA 受体动力学

### 5.1 Sigmoid 浓度响应（瞬时靶值）

$$
S_X\!\big(\mathrm{DA}(t)\big) \;=\; \frac{1}{1 + \exp\!\big(-\beta\,(\mathrm{DA}(t) - \mathrm{EC50}_X)\big)}, \qquad X \in \{D1, D2\}
$$

### 5.2 Langmuir 一阶受体占用动力学

$$
\boxed{\;\frac{d\alpha_X}{dt} \;=\; k^X_{on}\,S_X(\mathrm{DA})\,(1 - \alpha_X) \;-\; k^X_{off}\,\alpha_X\;}
$$

稳态：

$$
\alpha_{X,\infty}(\mathrm{DA}) \;=\; \frac{k^X_{on}\,S_X(\mathrm{DA})}{k^X_{on}\,S_X(\mathrm{DA}) + k^X_{off}}
$$

代码使用的速率常数：

$$
k^{D1}_{on} = \tfrac{1}{\tau^{D1}_{on} - 3000\,\mathrm{ms}}, \quad
k^{D1}_{off} = \tfrac{1}{\tau^{D1}_{off} + 3000\,\mathrm{ms}}, \quad
k^{D2}_{on} = \tfrac{1}{\tau^{D2}_{on}}, \quad
k^{D2}_{off} = \tfrac{1}{\tau^{D2}_{off}}
$$

### 5.3 三种 DA 调制路径

#### (a) Gain 调制（膜电阻）

$$
R_{eff,i}(t) \;=\; R_{base}\,\Big(1 \;+\; \varepsilon_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;-\; \varepsilon_{D2}\,\alpha_{D2}(t)\,m_{D2,i}\Big)
$$

#### (b) Bias 注入（DC 电流）

$$
I_{mod,i}^{(\mathrm{DA})}(t) \;=\; \text{BIAS}_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;+\; \text{BIAS}_{D2}\,\alpha_{D2}(t)\,m_{D2,i}
$$

#### (c) Synaptic Scaling（突触增益缩放）

$$
\text{scale}_{syn,i}(t) \;=\; 1 \;+\; \lambda_{D1}\,\alpha_{D1}(t)\,m_{D1,i} \;-\; \lambda_{D2}\,\alpha_{D2}(t)\,m_{D2,i}
$$

---

## 6. Cue 外部刺激

$$
I_{cue,i}(t) \;=\; A_{cue}^{A}\,\mathbb{1}\!\big[t \in [t^{A}_{on}, t^{A}_{off})\big]\,M^{A}_i \;+\; A_{cue}^{B}\,\mathbb{1}\!\big[t \in [t^{B}_{on}, t^{B}_{off})\big]\,M^{B}_i
$$

其中 $M^{A}_i, M^{B}_i \in \{0,1\}$ 分别为 Mem-A、Mem-B 池的成员掩码。

---

## 7. 发放频率适应（SFA）

仅作用于 Mem 池神经元：

$$
\frac{dw_{adapt,i}}{dt} \;=\; -\frac{w_{adapt,i}}{\tau_{sfa}} \;+\; b_{sfa}\sum_k \delta(t - t_i^{(k)})
$$

$$
I_{adapt,i}(t) \;=\; w_{adapt,i}(t)\,\mathbb{1}\!\left[i \in \mathrm{Mem\text{-}A} \cup \mathrm{Mem\text{-}B}\right]
$$

离散等价：先衰减再跳变：

$$
w_{adapt,i}(t + \Delta t) \;=\; w_{adapt,i}(t)\,e^{-\Delta t/\tau_{sfa}} \;+\; b_{sfa}\,\text{spike}_i(t)
$$

---

## 8. DA 浓度时序（梯形协议）

Batch-0 控制：$\mathrm{DA}^{(0)}(t) \equiv \mathrm{DA}_{base}$

Batch-1 实验：

$$
\mathrm{DA}^{(1)}(t) \;=\;
\begin{cases}
\mathrm{DA}_{base} & t < t_{on}\;\text{or}\;t \ge t_{off} \\[4pt]
\mathrm{DA}_{base} + (\mathrm{DA}_{peak} - \mathrm{DA}_{base})\dfrac{t - t_{on}}{T_{up}} & t_{on} \le t < t_{on} + T_{up} \\[10pt]
\mathrm{DA}_{peak} & t_{on} + T_{up} \le t < t_{off} - T_{down} \\[6pt]
\mathrm{DA}_{base} + (\mathrm{DA}_{peak} - \mathrm{DA}_{base})\dfrac{t_{off} - t}{T_{down}} & t_{off} - T_{down} \le t < t_{off}
\end{cases}
$$

---

## 9. 受体掩码与连接权重

### 9.1 受体掩码

$$
m_{D1,i} = \mathbb{1}[i \in \text{D1 segment}], \qquad
m_{D2,i} = \mathbb{1}[i \in \text{D2 segment}]
$$

### 9.2 双通道权重拆分

记 $\mathcal{P}_{intra} = \{(i, j) : i, j \in \mathrm{Mem\text{-}A}\} \cup \{(i, j) : i, j \in \mathrm{Mem\text{-}B}\}$，则：

$$
W^{nmda}_{ij} \;=\;
\begin{cases}
W_{ij} & (i, j) \in \mathcal{P}_{intra} \\
0 & \text{otherwise}
\end{cases}
$$

$$
W^{ampa}_{ij} \;=\; W_{ij} - W^{nmda}_{ij}
$$

---

## 10. 状态变量与更新顺序汇总

每个 timestep $t \to t + \Delta t$ 按以下顺序更新（与 [kernels.py](../models/kernels.py) `run_wm_kernel_dual` 一致）：

```
1. 计算 DA(t) ← 第 8 节梯形函数
2. 算 S_D1, S_D2 ← 第 5.1 节 sigmoid
3. 更新 α_D1, α_D2 ← 第 5.2 节 Langmuir ODE（Euler dt）
4. 算 R_eff, scale_syn, I_mod ← 第 5.3 节
5. 加 cue 注入到 I_mod ← 第 6 节
6. 衰减 I_ampa, s_nmda ← 第 3 节衰减项
7. 算 I_nmda_post = s_nmda · W_nmda
8. 算 I_bg ← 第 4 节
9. 算 I_total ← 第 2 节装配公式
10. 更新 V ← 第 1.3 节 exponential Euler
11. 检测 spikes, V ≥ V_th
12. 重置: V ← V_reset, t_last_spike ← t（spike 处）
13. spike-driven 更新:
      I_ampa += spike · W_ampa
      s_nmda += α_gate · (1 − s_nmda) · spike
      w_adapt += b_sfa · spike
14. 衰减 w_adapt ← 第 7 节
```

---

## 11. 关键的 mean-field 自洽方程（用于 attractor 分析）

设 Mem-A 池稳态发放率 $r_A$，自洽条件：

### 11.1 池内 NMDA 自激电流

$$
I_{rec}(r_A) \;=\; N_{pool}\,p_{intra}\,J_{intra}\,s_\infty(r_A)\,\big(1 + \lambda_{D1}\,\alpha_{D1}\big)
$$

### 11.2 WTA 反馈抑制

$$
I_{wta}(r_A) \;=\; -\,p_{I2E}\,|J_{I2E}|\,N_{IWM}\,r_{IWM}(r_A,\,\alpha_{D1},\,\alpha_{D2})
$$

### 11.3 总驱动 → V_∞

$$
V_{\infty,A}(r_A) \;=\; V_{rest} \;+\; R_{eff}(\alpha_{D1},\alpha_{D2})\,\Big[\, I_{rec}(r_A) + I_{wta}(r_A) + I_{bg} + \Delta I_{mem} + \text{BIAS}_{D1}\,\alpha_{D1}\,\Big]
$$

### 11.4 Siegert 类发放率函数

$$
r_A \;=\; \Phi\!\big(V_{\infty,A}(r_A);\, V_{th}, V_{reset}, \sigma_V, \tau_m, t_{ref}\big)
$$

**双稳态条件**：上式至少存在两个稳定不动点（low-state 与 high-state），中间夹一个不稳定鞍点。

---

## 12. 符号速查表

| 符号 | 含义 | 单位 |
|------|------|------|
| $V_i$ | 神经元 i 膜电位 | mV |
| $V_{rest}, V_{reset}, V_{th}$ | 静息 / 复位 / 阈值电位 | mV |
| $C_{m,i}$ | 膜电容 | pF |
| $R_{base}, R_{eff,i}$ | 基础 / DA 调制后膜电阻 | GΩ |
| $\tau_m = R_{eff}\,C_m$ | 膜时间常数 | ms |
| $t_{ref}$ | 不应期 | ms |
| $W^{ampa}_{ij}, W^{nmda}_{ij}$ | 双通道权重 | pA |
| $I_{ampa,i}, I_{nmda,i}$ | 突触后电流 | pA |
| $s_j$ | NMDA 门控变量 | 无量纲 ∈ [0,1] |
| $\alpha_{gate}$ | NMDA 单 spike 跳变增益 | 无量纲 |
| $\tau_{AMPA}, \tau_{NMDA}$ | 突触时间常数 | ms |
| $\mu_{bg}, \sigma_{bg}$ | 背景电流均值 / 标准差 | pA |
| $\Delta I_{mem}$ | Mem 池专属背景偏置 | pA |
| $\xi(t)$ | 单位高斯白噪声 | 无量纲 |
| $\mathrm{DA}(t)$ | 突触间隙 DA 浓度 | nM |
| $S_X$ | DA sigmoid 靶值 | 无量纲 ∈ [0,1] |
| $\alpha_{D1}, \alpha_{D2}$ | 受体占用率 | 无量纲 ∈ [0,1] |
| $\mathrm{EC50}_X, \beta$ | sigmoid 半效浓度 / 斜率 | nM, nM⁻¹ |
| $k^X_{on}, k^X_{off}$ | 结合 / 解离速率 | ms⁻¹ |
| $\varepsilon_{D1}, \varepsilon_{D2}$ | Gain 调制系数 | 无量纲 |
| $\text{BIAS}_{D1}, \text{BIAS}_{D2}$ | DC bias 调制 | pA |
| $\lambda_{D1}, \lambda_{D2}$ | 突触缩放系数 | 无量纲 |
| $m_{D1,i}, m_{D2,i}$ | 受体掩码 | 0 / 1 |
| $A_{cue}^{A/B}$ | cue 注入幅度 | pA |
| $M^{A}_i, M^{B}_i$ | Mem-A / Mem-B 成员掩码 | 0 / 1 |
| $w_{adapt,i}$ | SFA 适应电流 | pA |
| $b_{sfa}, \tau_{sfa}$ | SFA 单 spike 增量 / 衰减时间 | pA, ms |

---

*文档与 `run_wm_kernel_dual` 实现一一对应；如需查参数数值或代码对应行号，请回到 [MODEL_OVERVIEW.md](MODEL_OVERVIEW.md)。*
