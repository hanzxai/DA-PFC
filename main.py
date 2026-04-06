# main.py
"""
DA-PFC 仿真主入口 (SNN Simulation)

============================================================
 命令行参数 (CLI Arguments)
============================================================
  --da        : DA 浓度 (nM), 默认 3.0
  --duration  : 仿真总时长 (s), 默认 100
  --gpu       : GPU 卡号, 默认 0 (无效值自动回退 CPU)
  --batch     : 绘图 Batch ID (0=Control, 1=Exp), 默认 1
  --resume    : 从 checkpoint pkl 恢复仿真, 指定 pkl 路径
  --save-ckpt : 仿真结束后保存 checkpoint 到 checkpoints/ (默认不保存)
  --da2       : 两阶段模式: 第二阶段 DA 浓度 (nM)
  --phase2-onset : 两阶段模式: 第二阶段起始时间 (s)

============================================================
 使用方法 (Usage Examples)
============================================================

  1) 基础仿真 — 单一 DA 浓度:
     python main.py --da 2.0 --duration 500
     # DA=2nM, 仿真 500s

  1b) 基础仿真 + 保存 checkpoint (供后续 --resume 使用):
     python main.py --da 2.0 --duration 500 --save-ckpt
     # 完成后额外保存 checkpoint 到 checkpoints/

  2) 从 checkpoint 恢复 — 切换到新 DA 浓度 (需显式指定 --resume):
     python main.py --resume checkpoints/ckpt_DA2nM_500s.pkl --da 15.0
     # 载入 DA=2nM 稳态, 施加 DA=15nM 继续仿真 100s (默认)
     # 可加 --duration 200 指定恢复后的仿真时长
     # 注意: 不传 --resume 则不会走 resume 模式

  3) 两阶段给药 — 同一次仿真内切换 DA:
     python main.py --da 2.0 --da2 15.0
     # 第一阶段 DA=2nM, 第二阶段 DA=15nM
     # 可加 --phase2-onset 30 指定第二阶段起始时间(s)

  4) 指定 GPU / 回退 CPU:
     python main.py --da 3.0 --gpu 1     # 使用 GPU 1
     python main.py --da 3.0 --gpu 999   # 无效号 → 自动 CPU

============================================================
 Checkpoint 机制说明
============================================================
  - 默认不保存 checkpoint, 需要时加 --save-ckpt 显式指定
    文件名格式: ckpt_DA{da}nM_{duration}s.pkl
    内容: final_state tensor + config dict (体积很小, ~24KB)
  - checkpoints/ 目录应纳入 git 管理, 方便 clone 后直接使用
"""
import argparse
import time
import torch
import config
from simulation.runners import run_simulation_d1_d2_kinetics, run_simulation_d1_d2_two_stage, run_simulation_from_checkpoint
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (plot_combined_raster,
                                plot_combined_rates_all,
                                plot_combined_rates_E,
                                plot_combined_rates_I)
from utils import setup_experiment_folder, save_args, save_raw_data, save_checkpoint


def _fmt_elapsed(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m > 0:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def parse_args():
    parser = argparse.ArgumentParser(description="DA-PFC SNN Simulation")
    parser.add_argument("--duration", type=float, default=100.0,
                        help="仿真总时长 (s), 默认 100")
    parser.add_argument("--da", type=float, default=3.0,
                        help="给药浓度 (nM), 默认 3.0")
    parser.add_argument("--resume", type=str, default=None,
                        help="从checkpoint恢复仿真: 指定pkl路径. "
                             "加载之前仿真的最终状态, 施加--da指定的新DA浓度")
    parser.add_argument("--da2", type=float, default=None,
                        help="两阶段模式: 第二阶段DA浓度 (nM). 指定此参数时启用两阶段给药, "
                             "--da 作为静息态DA浓度, --da2 作为挑战DA浓度")
    parser.add_argument("--phase2-onset", type=float, default=None,
                        help="两阶段模式: 第二阶段DA开始时间 (s), 默认=da_onset+20s")
    parser.add_argument("--batch", type=int, default=1,
                        help="绘图 Batch ID (0=Control, 1=Exp), 默认 1")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU卡号, 默认 0. 无GPU或输入无效将回退到CPU")
    parser.add_argument("--save-ckpt", action="store_true", default=False,
                        help="仿真结束后保存 checkpoint 到 checkpoints/ 目录 (默认不保存)")
    return parser.parse_args()


def main():
    args = parse_args()
    t_total_start = time.time()

    # 决定设备
    if torch.cuda.is_available():
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            print(f"⚠️ 指定的 GPU {args.gpu} 无效，将回退到 CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # 1. 准备实验文件夹
    save_dir = setup_experiment_folder()

    # 将秒转换为毫秒
    duration_ms = args.duration * 1000.0
    phase2_onset_ms = args.phase2_onset * 1000.0 if args.phase2_onset is not None else None

    # 判断运行模式
    resume_mode = args.resume is not None
    two_stage = args.da2 is not None and not resume_mode
    # 判断用户是否显式指定了 duration (默认值为 100.0)
    user_specified_duration = (args.duration != 100.0)

    # 2. 实验配置
    if resume_mode:
        exp_config = {
            "duration": duration_ms if user_specified_duration else "auto",
            "checkpoint": args.resume,
            "new_da": args.da,
            "dt": config.DT,
            "device": str(device),
            "note": f"Resume from checkpoint, new DA={args.da}nM",
        }
    elif two_stage:
        exp_config = {
            "duration": duration_ms if user_specified_duration else "auto",
            "da_level_1": args.da,
            "da_level_2": args.da2,
            "phase2_onset": phase2_onset_ms if phase2_onset_ms else "auto",
            "dt": config.DT,
            "device": str(device),
            "note": (f"Two-Stage DA: {args.da}nM → {args.da2}nM "
                     f"(D1 Tau_on={config.TAU_ON_D1}ms, D2 Tau_on={config.TAU_ON_D2}ms)"),
        }
    else:
        exp_config = {
            "duration": duration_ms,
            "target_da": args.da,
            "dt": config.DT,
            "device": str(device),
            "note": f"D1+D2 受体动力学 (D1 Tau_on={config.TAU_ON_D1}ms, D2 Tau_on={config.TAU_ON_D2}ms, DA={args.da}nM, Duration={args.duration}s)",
        }
    save_args(exp_config, save_dir)

    # 3. 运行仿真
    print(f"🚀 Starting Experiment: {exp_config['note']}")
    t_sim_start = time.time()
    if resume_mode:
        # Resume from checkpoint mode
        data = run_simulation_from_checkpoint(
            checkpoint_path=args.resume,
            duration=duration_ms if user_specified_duration else None,
            da_level=args.da,
            device=device,
        )
    elif two_stage:
        # Two-stage DA dosing mode
        data = run_simulation_d1_d2_two_stage(
            duration=duration_ms if user_specified_duration else None,
            da_level_1=args.da,
            da_level_2=args.da2,
            phase2_onset=phase2_onset_ms,
            device=device,
        )
    else:
        # Standard single-stage DA mode
        data = run_simulation_d1_d2_kinetics(
            duration=duration_ms,
            target_da=args.da,
            device=device,
        )
    t_sim_elapsed = time.time() - t_sim_start

    # 4. 保存原始数据
    t_save_start = time.time()
    save_raw_data(data, save_dir)
    # 仅在用户显式指定 --save-ckpt 时保存 checkpoint
    if args.save_ckpt:
        actual_duration_s = data['config']['duration'] / 1000.0
        actual_da = data['config'].get('da_level', args.da)
        save_checkpoint(data, da_level=actual_da, duration_s=actual_duration_s)
    t_save_elapsed = time.time() - t_save_start

    # 5. 分析与绘图
    print("🎨 Generating Plots...")
    t_plot_start = time.time()
    import numpy as np
    analyzer = PFCAnalyzer(data)

    # Pre-compute is no longer needed; combined plots handle y-axis unification internally

    # Generate 4 combined figures (each is a 2×2 grid: full/zoom × Control/Exp)
    print("\n--- Generating Combined Plots (4 figures) ---")
    plot_combined_raster(analyzer, save_dir=save_dir)
    plot_combined_rates_all(analyzer, save_dir=save_dir)
    plot_combined_rates_E(analyzer, save_dir=save_dir)
    plot_combined_rates_I(analyzer, save_dir=save_dir)

    # Frequency analysis reports — save to file
    import os
    analyzer.save_report(os.path.join(save_dir, "analysis_report.txt"))

    t_plot_elapsed = time.time() - t_plot_start

    # 6. 总计时报告
    t_total_elapsed = time.time() - t_total_start
    print("\n" + "=" * 50)
    print("⏱️  计时报告")
    print("=" * 50)
    print(f"  仿真计算:  {_fmt_elapsed(t_sim_elapsed)}")
    print(f"  数据保存:  {_fmt_elapsed(t_save_elapsed)}")
    print(f"  分析绘图:  {_fmt_elapsed(t_plot_elapsed)}")
    print(f"  ────────────────────────")
    print(f"  总计:      {_fmt_elapsed(t_total_elapsed)}")
    print("=" * 50)
    print(f"\n✅ Experiment Complete. Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
