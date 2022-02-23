import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.data_processor import DataProcessor
from pandas.core.frame import DataFrame


def GM11(x, n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()  # 一次累加
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((len(x) - 1, 1))
    # a为发展系数 b为灰色作用量
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算参数
    result = (x[0] - b / a) * np.exp(-a * (n - 1)) - (x[0] - b / a) * np.exp(-a * (n - 2))
    S1_2 = x.var()  # 原序列方差
    e = list()  # 残差序列
    for index in range(1, x.shape[0] + 1):
        predict = (x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2))
        e.append(x[index - 1] - predict)
    S2_2 = np.array(e).var()  # 残差方差
    C = S2_2 / S1_2  # 后验差比
    if C <= 0.35:
        assess = '后验差比<=0.35，模型精度等级为好'
    elif C <= 0.5:
        assess = '后验差比<=0.5，模型精度等级为合格'
    elif C <= 0.65:
        assess = '后验差比<=0.65，模型精度等级为勉强'
    else:
        assess = '后验差比>0.65，模型精度等级为不合格'
    # 预测数据
    predict = list()
    for index in range(x.shape[0] + 1, x.shape[0] + n + 1):
        predict.append((x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2)))
    predict = np.array(predict)
    return {
        'a': {'value': a, 'desc': '发展系数'},
        'b': {'value': b, 'desc': '灰色作用量'},
        'predicted': {'value': result, 'desc': '第%d个预测值' % n},
        'C': {'value': C, 'desc': assess},
        'predictor': {'value': predict, 'desc': '往后预测%d个的序列' % (n)},
    }


if __name__ == "__main__":
    bitcoin = "../data/BCHAIN-MKPRU.csv"
    gold = "../data/NEW-LBMA-GOLD-PLUS.csv"
    bitcoin_data = DataProcessor(bitcoin).reader()
    gold_data = DataProcessor(gold).reader()

    start_days = 3
    days_number = 100

    b_data = np.array([b[1] for b in bitcoin_data])
    input_b_data = b_data[0:start_days]  # 输入数据
    true_b_data = b_data[start_days:days_number]  # 需要预测的数据
    result = GM11(input_b_data, len(true_b_data))
    predict_b_data = result['predictor']['value']
    predict_b_data = np.round(predict_b_data, 1)
    errors = [abs((true_b_data[i] - predict_b_data[i]) / predict_b_data[i]) for i in range(len(true_b_data))]

    print('真实值:', true_b_data)
    print('预测值:', predict_b_data)
    print('误差值:', errors)

    heads = ["errors", "date"]
    edf = DataFrame(errors)
    edf["1"] = ""
    edf.columns = heads
    edf["date"] = pd.to_datetime([b[0] for b in bitcoin_data][start_days:days_number])


    g_data = np.array([b[1] for b in gold_data])
    input_g_data = g_data[0:start_days]  # 输入数据
    true_g_data = g_data[start_days:days_number]  # 需要预测的数据
    result = GM11(input_g_data, len(true_g_data))
    predict_g_data = result['predictor']['value']
    predict_g_data = np.round(predict_g_data, 1)
    errors = [abs((true_g_data[i] - predict_g_data[i]) / predict_g_data[i]) for i in range(len(true_g_data))]

    print('真实值:', true_g_data)
    print('预测值:', predict_g_data)
    print('误差值:', errors)

    heads = ["errors", "date"]
    gdf = DataFrame(errors)
    gdf["1"] = ""
    gdf.columns = heads
    gdf["date"] = pd.to_datetime([b[0] for b in gold_data][start_days:days_number])
    plt.figure(figsize=(20, 8))
    plt.plot(edf["date"], edf.errors, color="tab:orange", label="Gold")
    plt.plot(gdf["date"], gdf.errors, color="tab:blue", label="BitCoin")
    plt.xlabel("Date")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

    # heads = ["true", "predictor", "date"]
    # rdf = DataFrame(true_b_data)
    # rdf["1"] = ""
    # rdf["2"] = ""
    # rdf.columns = heads
    # rdf["predictor"] = predict_b_data
    # rdf["date"] = pd.to_datetime([b[0] for b in bitcoin_data][3:])
    # plt.figure(figsize=(20, 8))
    # plt.plot(rdf["date"], rdf.true, color="orange", label="Gold True Price")
    # plt.plot(rdf["date"], rdf.predictor, color="gray", label="Gold Predict Price")
    # plt.show()

    # print(result)
