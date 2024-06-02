import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.preprocessing import MinMaxScaler

# 这个代码主要内容为训练过程中需要用到的模型相关的函数

# 加载数据集
# 我们选择观测时长为预测前一周，即168小时，预测时长为观测后一天，即24小时，设置步长为1将数据集切片成序列
# __len__返回序列的个数，__getitem__返回相应特征与标签
class DataGenerator(Dataset):
    def __init__(self, dataset, observation_length=120, prediction_length=24, dataset_step=1):
        super().__init__()
        self.prediction_length = prediction_length
        self.labels = torch.from_numpy(dataset.loc[:, "T (degC)"].to_numpy()).type(torch.float32)
        self.features = torch.from_numpy(dataset.loc[:, [k for k in dataset.columns if k not in ["T (degC)"]]].to_numpy()).type(torch.float32)
        self.observation_sequence = [(i, i + observation_length) for i in range(0, len(dataset) - observation_length - prediction_length, dataset_step)]
    
    def __len__(self):
        return len(self.observation_sequence)
    
    def __getitem__(self, idx):
        start, end = self.observation_sequence[idx]
        feature = self.features[start:end]
        label = self.labels[end:end + self.prediction_length].squeeze()
        return feature, label

# 位置编码
# 直接使用Transformer的正弦编码即可
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)

# Transformer架构
# 1. 线性层进行高维映射：(batch_size, observation_length, 4) -> (batch_size, observation_length, 100)
# 2. 位置编码不改变纬度：(batch_size, observation_length, 100) -> (batch_size, observation_length, 100)
# 3. 编码器：(batch_size, observation_length, 100) -> (batch_size, observation_length, 100)
# 4. Flatten：(batch_size, observation_length, 100) -> (batch_size, observation_length * 100)
# 5. 解码器：(batch_size, observation_length * 100) -> (batch_size, prediction_length)
class Transformer(nn.Module):
    def __init__(self, feature_size=100, encoder_layers_number=1, dropout=0.1, prediction_length=24, indicator_length=13, observation_length=120):
        super(Transformer, self).__init__()

        self.input_embedding  = nn.Linear(indicator_length, feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_layers_number, norm=None)
        self.decoder = nn.Linear(feature_size * observation_length, prediction_length)
        self.init_weights()

    def init_weights(self):
        # 采用正态初始化weight，零初始化bias
        init_range = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
            self.src_mask = mask

        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = output.reshape(output.shape[0], -1)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, size):
        # 需要预测24小时的气温序列，mask直接自动设置，只要保证不接收未来信息即可
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# 训练过程
def train(model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, save_path):
    train_losses = []
    val_losses = []
    lrs = []
    pbar = tqdm(range(epochs), desc="Epochs", leave=True, position=0)
    start = time.time()
    for i, epoch in enumerate(pbar):
        total_loss = 0.
        lrs.append(optimizer.param_groups[0]['lr'])
        model.train()
        for batch in train_loader:  
            # 加载数据
            feature = batch[0].to(device)
            label = batch[1].to(device)
            # 前向传播
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            # 计算loss
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # 计算验证集loss
        total_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                feature = batch[0].to(device)
                label = batch[1].to(device)
                output = model(feature)
                loss = criterion(output, label)
                total_loss += loss.item()
            val_losses.append(total_loss / len(val_loader))
        
        if scheduler is not None:
          scheduler.step()

        pbar.set_postfix(train_loss=train_losses[-1], val_loss=val_losses[-1], lr=lrs[-1])
        
        if i % 10 == 0:
            torch.save(model.state_dict(), save_path)
    
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
    
    return model, train_losses, val_losses, lrs

# 评估过程，直接保留前向传播即可，无需进行反向传播
def evaluate(model, criterion, test_loader, device, separate_labels=False, prediction_length=24):
    model.eval()
    losses = [0] * prediction_length
    with torch.no_grad():
        total_loss = 0.
        for batch in test_loader:
            feature = batch[0].to(device)
            label = batch[1].to(device)

            output = model(feature)
            if separate_labels:
                for i in range(output.shape[1]):
                    losses[i] += criterion(output[:, i], label[:, i]).item()
            else:
                total_loss += criterion(output, label).item()
        
        return total_loss / len(test_loader) if not separate_labels else np.array(losses) / len(test_loader)