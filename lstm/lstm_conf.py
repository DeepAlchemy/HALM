#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pandas as pd
import math
import torch.nn.functional as F

DAYS_FOR_TRAIN = 5


# x,query：[batch, seq_len, hidden_dim*2]
def attention_net(x, query, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # scores:[batch, seq_len, seq_len]

    p_attn = F.softmax(scores, dim=-1)
    context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
    return context, p_attn


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x (seq_len, batch, input_size)
        s, b, h = x.shape  # x (seq_len, batch, hidden_size)
        query = self.dropout(x)
        attn_output, attention = attention_net(x, query)
        x = self.fc(attn_output)
        x = x.view(s, b, -1)
        return x


def create_dataset(data, days_for_train=5):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        dataset_x.append(data.iloc[i:(i + days_for_train)])
        dataset_y.append(data.iloc[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


if __name__ == '__main__':
    bitcoin_path = "BCHAIN-MKPRU.csv"
    gold_path = "LBMA-GOLD.csv"
    bitcoin = False
    gold = True

    if bitcoin:
        data = pd.read_csv(bitcoin_path)
        data = data[["Value"]]
    else:
        data = pd.read_csv(gold_path)
        data = data[["USD (PM)"]]

    data = data.astype('float32')
    max_value = np.max(data)
    min_value = np.min(data)
    # data_close = (data - min_value) / (max_value - min_value)
    data_close = data

    model = LSTM_Regression(DAYS_FOR_TRAIN, hidden_size=32, output_size=1, num_layers=2)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    accuracy_list = []

    for date in range(7, 40):
        data_close = data_close[:date]

        dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

        # 划分训练集和测试集
        train_size = len(dataset_x) - 1
        train_x = dataset_x[:train_size]
        train_y = dataset_y[:train_size]
        test_x = dataset_x[train_size:]
        test_y = dataset_y[train_size:]

        # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
        train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
        train_y = train_y.reshape(-1, 1, 1)
        test_x = test_x.reshape(-1, 1, DAYS_FOR_TRAIN)

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        test_x = torch.from_numpy(test_x)

        for i in range(500):
            out = model(train_x)
            loss = loss_function(out, train_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

        # for test
        model = model.eval()
        pred_test = model(test_x)
        mae = abs(pred_test[-1].detach().numpy() - test_y) / test_y
        accuracy_list.append(mae[0][0])
        print(accuracy_list)

    fig = plt.figure(figsize=(12, 4))
    x = range(6, len(accuracy_list) + 6)
    plt.plot(x, accuracy_list, 'r', label='Error Ratio')
    plt.title('Error ratio between true and predicted values on the last day', fontsize=11)
    plt.xticks(x)
    plt.xlabel('Date', color='blue')
    plt.ylabel('Error Ratio', color='blue')
    plt.legend(loc='best')
    plt.show()

# Gold: [0.8970799, 0.8055122, 0.7185276, 0.6353558, 0.55570006, 0.47948197, 0.4067976, 0.33791366, 0.273278, 0.21353458, 0.15953329, 0.112310074, 0.072987504, 0.04252698, 0.021288995, 0.008530334, 0.0022766697, 6.569014e-05, 0.0006846827, 0.0007882817, 0.0007937636, 0.0007949715, 0.00079580775, 0.00079655106, 0.0007972014, 0.00079757307, 0.0007979447, 0.0007983164, 0.00079840934, 0.00079868804, 0.000798781, 0.0007988739, 0.0007989668]
