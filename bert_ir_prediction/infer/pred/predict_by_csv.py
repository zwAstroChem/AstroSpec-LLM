"""
@author AFelixLiu
@date 2026 3月 10
"""

import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 环境设置
ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src import BERT, BERT4IR, smiles_tokenize

PATH = {
    "vocab": r"data/vocabs/vocab_smiles.pickle",
    "model_low": r"save/models/finetune/task4_smiles_all_charge/BERT4IR_ft_can_all_charge_useRoPE_low_onehot(4x9)_best.pth",
    "model_high": r"save/models/finetune/task4_smiles_all_charge/BERT4IR_ft_can_all_charge_useRoPE_high_onehot(4x9)_best.pth"
}

def load_vocab():
    with open(PATH.get("vocab"), 'rb') as file: return pickle.load(file)

def load_model(vocab_size, out_freq, device):
    ir_bins = 105 if out_freq == "low" else 71
    model_name = PATH.get("model_low") if out_freq == "low" else PATH.get("model_high")
    bert = BERT(vocab_size, hidden=768, n_layers=6, attn_heads=12, dropout=0., use_rope=True)
    model = BERT4IR(bert, ir_bins, normed=True, support_charge=True, charge_vocab=[-1, 0, 1, 2],
                    charge_encoding="onehot", charge_dim=56, onehot_repeat=9, plot=True)
    checkpoint = torch.load(model_name, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['bert4ir_state_dict'])
    return model.to(device).eval()

def predict_specific_charge(data, model, vocab, out_freq, device):
    """
    仅根据输入数据中的 charge 列进行预测
    """
    out_dim = 105 if out_freq == 'low' else 71
    preds = []
    
    with torch.no_grad():
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"预测光谱 ({out_freq})"):
            smi = row["canonical_smiles"]
            # 获取当前行指定的电荷状态
            current_charge = int(row["charge"]) 
            
            tokens = smiles_tokenize(smi)
            token_ids = [vocab["<cls>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens] + [vocab["<sep>"]]
            input_tensor = torch.tensor(token_ids[:384]).unsqueeze(0).to(device)
            
            # 补位 dummy_label
            dummy_label = torch.zeros((1, out_dim)).to(device)
            charge_tensor = torch.tensor(current_charge).unsqueeze(0).to(device)
            
            # 模型推理
            outputs = model(input_tensor, dummy_label, charge_tensor) 
            
            # 记录预测值
            preds.append(outputs[0].squeeze().cpu().tolist())
            
    # 将预测结果作为新列加入原始 DataFrame
    data[f"predicted_spectrum_{out_freq}"] = preds
    return data

if __name__ == '__main__':
    # 创建输出目录
    output_dir = Path("predicted_spectra")
    output_dir.mkdir(parents=True, exist_ok=True) 

    file_path = Path(r"data/pahdb/pahdb_w24146_all_test.csv")
    
    if not file_path.exists():
        print(f"错误: 找不到输入文件 {file_path}")
        sys.exit(1)
        
    # 读取原始数据（包含参考光谱和其他信息）
    original_data = pd.read_csv(file_path)
    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 循环处理低频和高频
    final_df = original_data.copy()
    for freq in ['low', 'high']:
        print(f"\n--- 正在处理 {freq} 频段 ---")
        model = load_model(len(vocab), freq, device)
        
        # 在原有数据基础上追加预测列
        final_df = predict_specific_charge(final_df, model, vocab, freq, device)
        
    # 保存完整结果（含原始参考信息和预测光谱）
    output_file = output_dir / "pahdb_with_predictions.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n处理完成！完整数据已保存至: {output_file}")
    print(f"列说明：'predicted_spectrum_low' (105维) 和 'predicted_spectrum_high' (71维) 已添加。")