# utils.py
"""工具函数: 实验文件夹管理、配置保存、数据序列化"""
import json
import pickle
from datetime import datetime
from pathlib import Path


def setup_experiment_folder(base_dir: str = "outputs") -> Path:
    """创建带时间戳的实验文件夹, 返回 Path 对象。"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Experiment directory: {exp_dir}")
    return exp_dir


def save_args(args_dict: dict, save_dir: Path):
    """将实验参数配置保存为 JSON。"""
    json_ready = {k: str(v) for k, v in args_dict.items()}
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=4, ensure_ascii=False)


def save_raw_data(data: dict, save_dir: Path, filename: str = "raw_data.pkl"):
    """保存原始仿真数据 (Pickle)。"""
    file_path = save_dir / filename
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Raw data saved to {file_path}")
