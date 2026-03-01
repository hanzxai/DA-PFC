# utils.py
import os
import json
import pickle
import torch
from datetime import datetime
from pathlib import Path

def setup_experiment_folder(base_dir="outputs"):
    """
    创建一个带时间戳的实验文件夹
    返回: pathlib.Path 对象指向新创建的文件夹
    """
    # 获取当前时间，格式：YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 实验文件夹名称
    exp_dir_name = f"exp_{timestamp}"
    exp_dir = Path(base_dir) / exp_dir_name
    
    # 创建文件夹 (如果父文件夹不存在也一并创建)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Experiment Output Directory Created: {exp_dir}")
    return exp_dir
    
def save_args(args_dict, save_dir):
    """保存实验参数配置"""
    # 修改点 1: open 函数里加上 encoding='utf-8'
    with open(save_dir / "config.json", "w", encoding='utf-8') as f:
        
        json_ready = {k: str(v) for k, v in args_dict.items()}
        
        # 修改点 2: dump 函数里加上 ensure_ascii=False
        json.dump(json_ready, f, indent=4, ensure_ascii=False)

def save_raw_data(data, save_dir, filename="raw_data.pkl"):
    """保存原始仿真数据 (Pickle)"""
    file_path = save_dir / filename
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Raw data saved to {file_path}")