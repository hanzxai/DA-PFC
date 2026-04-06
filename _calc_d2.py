import math

DA = 15.0
EC50_D2 = 8.0
BETA = 1.0
k_on_d2 = 1.0 / 10000.0
k_off_d2 = 1.0 / 50000.0

s_d2 = 1.0 / (1.0 + math.exp(-BETA * (DA - EC50_D2)))
alpha_d2_ss = k_on_d2 * s_d2 / (k_on_d2 * s_d2 + k_off_d2)
print(f"s_D2 = {s_d2:.6f}")
print(f"alpha_D2_ss = {alpha_d2_ss:.6f}")

EPS_D2 = 0.10
BIAS_D2 = -20.0
LAM_D2 = 0.2
R_base = 0.1
bg_mean = 200.0
bg_std = 25.0
V_rest = -70.0
V_th = -50.0
V_reset = -75.0

alpha = alpha_d2_ss
R_eff = R_base * (1 - EPS_D2 * alpha)
I_mod = BIAS_D2 * alpha
scale_syn = 1 - LAM_D2 * alpha
I_total = bg_mean + I_mod
V_inf = V_rest + R_eff * I_total
sigma_V = R_eff * bg_std

print(f"\n=== D2 triple modulation (alpha={alpha:.4f}) ===")
print(f"R_eff = {R_eff:.6f} GOhm")
print(f"I_mod = {I_mod:.4f} pA")
print(f"scale_syn = {scale_syn:.4f}")
print(f"I_total = {I_total:.4f} pA")
print(f"V_inf = {V_inf:.4f} mV")
print(f"V_th = {V_th} mV")
print(f"V_inf - V_th = {V_inf - V_th:.4f} mV")
print(f"sigma_V = {sigma_V:.4f} mV")

if V_inf < V_th:
    x_th = (V_th - V_inf) / sigma_V
    print(f"x_th = {x_th:.4f} (sub-threshold)")
    print(f"V below Vth by {V_th - V_inf:.2f} mV = {x_th:.2f} sigma")

# Breakdown of each modulation
print("\n=== Breakdown ===")
V_inf_base = V_rest + R_base * bg_mean
print(f"Baseline V_inf = {V_inf_base:.2f} mV")

# Only BIAS
V_inf_bias_only = V_rest + R_base * (bg_mean + I_mod)
print(f"BIAS only: V_inf = {V_inf_bias_only:.4f} mV (delta = {V_inf_bias_only - V_inf_base:.4f} mV)")

# Only R_eff
V_inf_R_only = V_rest + R_eff * bg_mean
print(f"R_eff only: V_inf = {V_inf_R_only:.4f} mV (delta = {V_inf_R_only - V_inf_base:.4f} mV)")

# Combined
print(f"Combined: V_inf = {V_inf:.4f} mV (delta = {V_inf - V_inf_base:.4f} mV)")

# D1 comparison
print("\n=== D1 comparison ===")
EC50_D1 = 4.0
s_d1 = 1.0 / (1.0 + math.exp(-BETA * (DA - EC50_D1)))
k_on_d1 = 1.0 / (30876.1 - 3000)
k_off_d1 = 1.0 / (164472.5 + 3000)
alpha_d1_ss = k_on_d1 * s_d1 / (k_on_d1 * s_d1 + k_off_d1)
I_mod_d1 = 20.0 * alpha_d1_ss
R_eff_d1 = R_base * (1 + 0.15 * alpha_d1_ss)
V_inf_d1 = V_rest + R_eff_d1 * (bg_mean + I_mod_d1)
print(f"alpha_D1_ss = {alpha_d1_ss:.6f}")
print(f"V_inf_D1 = {V_inf_d1:.4f} mV (above Vth by {V_inf_d1 - V_th:.2f} mV)")
