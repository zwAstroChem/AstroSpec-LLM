"""
@author AFelixLiu
@date 2025 1月 19
"""

import torch
import torch.nn.functional as F
from torch import nn

from .bert import BERT
from .utils import EMDLoss, normalize, SISLoss


class BERT4IR(nn.Module):

    def __init__(self, bert: BERT, ir_bins, normed=False, support_charge=False, charge_vocab=None, charge_encoding="emb",
                 charge_dim=16, onehot_repeat=1, plot=False):
        """
        :param bert: 底层 BERT 模型
        :param ir_bins: IR 输出维度
        :param normed: 光谱数据是否已经提前归一化
        :param support_charge: 是否启用电荷信息
        :param charge_vocab: 电荷值列表，如 [-1, 0, 1, 2]
        :param charge_encoding: 电荷编码方式：'emb', 'onehot'
        :param charge_dim: 电荷嵌入维度（仅在 charge_encoding == 'emb' 时有效）
        :param onehot_repeat: 电荷 One-Hot 编码重复次数（仅在 charge_encoding == 'onehot' 时有效）
        :param plot: 是否返回中间结果用于绘图
        """

        super().__init__()
        self.bert = bert
        self.normed = normed
        self.support_charge = support_charge
        self.plot = plot

        if not support_charge:  # 不带电荷
            self.head = IRHead(self.bert.hidden, ir_bins)
        elif charge_encoding == "emb":  # 使用 Embedding 编码电荷
            self.head = IRHeadWithChargeEmb(
                hidden=self.bert.hidden,
                ir_bins=ir_bins,
                charge_vocab=charge_vocab,
                charge_dim=charge_dim
            )
        elif charge_encoding == "onehot":  # 使用 One-Hot 编码电荷
            self.head = IRHeadWithChargeOneHot(
                hidden=self.bert.hidden,
                ir_bins=ir_bins,
                charge_vocab=charge_vocab,
                onehot_repeat=onehot_repeat
            )
        else:
            raise SystemExit(f"Unsupported charge encoding: '{charge_encoding}'. Choose from ['emb', 'onehot'].")

        # Setting the loss function
        self.criterion_emd = EMDLoss()
        self.criterion_sis = SISLoss()
        self.criterion_mse = nn.MSELoss(reduction="sum")

    def forward(self, x, y, charges=None):
        bert_output = self.bert(x)

        if self.support_charge:
            assert charges is not None, "charges must be provided when support_charge=True"
            y_hat = self.head(bert_output, charges)
        else:
            y_hat = self.head(bert_output)

        # 使用归一化后的数据进行损失计算
        if not self.normed:
            normed_y_hat, normed_y = normalize(y_hat, y)
        else:
            normed_y_hat, normed_y = y_hat, y
        emd_loss = self.criterion_emd(normed_y_hat, normed_y)
        sis_loss = self.criterion_sis(normed_y_hat, normed_y)
        mse_loss = self.criterion_mse(normed_y_hat, normed_y)

        if self.plot:
            return normed_y_hat, normed_y, emd_loss, sis_loss, mse_loss
        else:
            return emd_loss, sis_loss, mse_loss


class IRHead(nn.Module):
    """Predicting the IR of neutral PAHs without charge information"""

    def __init__(self, hidden, ir_bins):
        """
        :param hidden: BERT model's output size
        :param ir_bins: Number of output bins (IR spectrum dimension)
        """

        super().__init__()
        half = hidden // 2

        self.linear1 = nn.Linear(hidden, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x):
        """
        Args:
            x: BERT输出，shape = [batch_size, seq_len, hidden]
        """

        cls_vec = x[:, 0]  # # 提取 [CLS] 向量，shape = [batch_size, hidden]

        # MLP 处理
        output = self.relu1(self.linear1(cls_vec))
        final = self.linear2(output)

        return torch.abs(final)  # 保证输出非负


class IRHeadWithChargeEmb(nn.Module):
    """Predicting the IR of charged PAHs using Charge Embedding"""

    def __init__(self, hidden, ir_bins, charge_vocab=None, charge_dim=4):
        """
        :param hidden: BERT model's output size
        :param ir_bins: Number of output bins (IR spectrum dimension)
        :param charge_vocab: List of possible charges (e.g., [-2, -1, 0, 1])
        :param charge_dim: Dimension of charge embedding
        """

        super().__init__()

        if charge_vocab is not None:
            self.charge_vocab = charge_vocab
        else:
            self.charge_vocab = [-2, -1, 0, 1]
            print("Use default charge vocab:", self.charge_vocab)

        self.charge_embedding = nn.Embedding(len(self.charge_vocab), charge_dim)  # 定义电荷嵌入层

        # 拼接维度：BERT输出 + 电荷嵌入
        total_dim = hidden + charge_dim
        half = total_dim // 2

        self.linear1 = nn.Linear(total_dim, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x, charges):
        """
        Args:
            x: BERT输出，shape = [batch_size, seq_len, hidden]
            charges: list 表示每个样本的电荷值，例如 [-1, 0, 1]
        """

        # 检查电荷是否在 vocab 中
        for c in charges:
            if c not in self.charge_vocab:
                raise SystemExit(f"Unsupported charge value: {c}. Supported values are: {self.charge_vocab}")

        device = x.device  # 获取设备信息
        charge_indices = torch.tensor([self.charge_vocab.index(c) for c in charges], device=device)  # 将电荷转换为索引

        cls_vec = x[:, 0]  # 提取 [CLS] 向量，shape = [batch_size, hidden]
        charge_emb = self.charge_embedding(charge_indices)  # 获取电荷嵌入，shape = [batch_size, charge_dim]
        combined = torch.cat([cls_vec, charge_emb], dim=-1)  # 拼接，shape = [batch_size, hidden + charge_dim]

        # MLP 处理
        output = self.relu1(self.linear1(combined))
        final = self.linear2(output)

        return torch.abs(final)  # 保证输出非负


class IRHeadWithChargeOneHot(nn.Module):
    """Predicting the IR of charged PAHs using One-Hot encoding for charge"""

    def __init__(self, hidden, ir_bins, charge_vocab=None, onehot_repeat=1):
        """
        :param hidden: BERT model's output size
        :param ir_bins: Number of output bins (IR spectrum dimension)
        :param charge_vocab: List of possible charges (e.g., [-2, -1, 0, 1])
        :param onehot_repeat: The number of times the one-hot encoding is repeated
        """

        super().__init__()

        if charge_vocab is not None:
            self.charge_vocab = charge_vocab
        else:
            self.charge_vocab = [-2, -1, 0, 1]
            print("Use default charge vocab:", self.charge_vocab)

        self.onehot_repeat = onehot_repeat
        self.charge_dim = len(self.charge_vocab)  # One-Hot 的维度等于电荷种类数
        self.onehot_dim = self.charge_dim * onehot_repeat  # 扩展后的 One-Hot 维度

        # 拼接维度：BERT输出 + one-hot电荷
        total_dim = hidden + self.onehot_dim
        half = total_dim // 2

        self.linear1 = nn.Linear(total_dim, half)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(half, ir_bins)

    def forward(self, x, charges):
        """
        Args:
            x: BERT输出，shape = [batch_size, seq_len, hidden]
            charges: list 或 tensor 表示每个样本的电荷值，例如 [-1, 0, 1]
        """

        # 检查电荷是否在 vocab 中
        for c in charges:
            if c not in self.charge_vocab:
                raise SystemExit(f"Unsupported charge value: {c}. Supported values are: {self.charge_vocab}")

        device = x.device  # 获取设备信息
        charge_indices = torch.tensor([self.charge_vocab.index(c) for c in charges], device=device)  # 将电荷转换为索引

        # 构建 One-Hot 编码，shape = [batch_size, charge_dim]
        charge_onehot = F.one_hot(charge_indices, num_classes=self.charge_dim).float()

        # 扩展 one-hot 向量，重复 onehot_repeat 次，shape = [batch_size, charge_dim * repeat]
        charge_onehot_expanded = charge_onehot.repeat(1, self.onehot_repeat)

        cls_vec = x[:, 0]  # 提取 [CLS] 向量，shape = [batch_size, hidden]

        # 拼接，shape = [batch_size, hidden + onehot_dim]
        combined = torch.cat([cls_vec, charge_onehot_expanded], dim=-1)

        # MLP 处理
        output = self.relu1(self.linear1(combined))
        final = self.linear2(output)

        return torch.abs(final)  # 保证输出非负
