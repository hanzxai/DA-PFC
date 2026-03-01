# config.py
import torch

# 硬件
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络参数
N_E = 800
N_I = 200
N_TOTAL = N_E + N_I

# 仿真参数
DT = 1
DEFAULT_DURATION = 2000.0

# 动力学参数 (半衰期 ms)
TAU_RISE = 30876.0
TAU_DECAY = 164474.0



# ==========================================
# 2. 受体动力学参数 (Receptor Kinetics)
# 数据来源：用户提供的论文截图
# ==========================================

# D1 受体的时间常数 (ms)
# 对应图中: if S > alpha (上升阶段)
TAU_ON_D1 = 30876.1  

# 对应图中: if S <= alpha (衰减阶段)
TAU_OFF_D1 = 164472.5 

# 药理学参数 (用于计算稳态目标值 S_D1)
# 公式: S = 1 / (1 + exp(-beta * (DA - EC50)))
BETA = 1.0
EC50_D1 = 4.0   # nM
EC50_D2 = 8.0   # nM (假设 D2 保持之前的设定，或者你需要修改也可以在此调整)

# D2 的时间常数
# 如果没有特定数据，通常暂时假设与 D1 同级或沿用旧值。
# 这里为了代码运行，我们暂时沿用 D1 的慢速动力学，或者你可以指定数值。
# 暂时设定为与 D1 一致，保证代码不报错：
TAU_ON_D2 = 30876.1
TAU_OFF_D2 = 164472.5

# ==========================================
# 3. 调节强度参数 (Modulation Strength)
# ==========================================
# 这些参数决定了 alpha_D1 = 1.0 时，对膜参数的具体影响幅度
EPS_D1 = 0.3    # Gain 增强比例
EPS_D2 = 0.2    # Gain 减弱比例

BIAS_D1 = 3.0   # 偏置电流 (pA)
BIAS_D2 = -3.0

LAM_D1 = 0.3    # 突触缩放 (Synaptic Scaling)
LAM_D2 = 0.2