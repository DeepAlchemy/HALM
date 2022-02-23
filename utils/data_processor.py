import csv
import datetime
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt


class DataProcessor(object):

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        with open(self.file_path, "r") as f:
            self.csv_file = csv.reader(f)
            for index, item in enumerate(self.csv_file):
                if index != 0:
                    date_item = item[0].split("/")
                    day = datetime.date(int("20{}".format(date_item[2])), int(date_item[0]),
                                        int(date_item[1])).strftime("%Y-%m-%d")
                    if item[1] != "":
                        self.data.append([day, float(item[1])])
                    else:
                        self.data.append([day, float(0)])

    def reader(self):
        """ 返回list格式的数据对象 """
        return self.data

    @classmethod
    def complete_missing_date(cls, arr_1, arr_2):
        """ 对比比特币数据，缺省值处理，补齐黄金非交易时间为0 """
        new_arr = []
        arr_1_dates = [i[0] for i in arr_1]
        arr_2_dates = [j[0] for j in arr_2]
        for item in arr_1_dates:
            date_item = item.split("-")
            day = "{}/{}/{}".format(int(date_item[1]), int(date_item[2]), int(date_item[0][2:]))
            if item in arr_2_dates:
                for temp in arr_2:
                    if temp[0] == item:
                        new_arr.append([day, temp[1]])
            else:
                new_arr.append([day, 0])
        with open("../resources/data/NEW-LBMA-GOLD.csv", "w", newline="") as gold_file:
            writer = csv.writer(gold_file)
            writer.writerows(new_arr)

    @classmethod
    def complete_missing_date_plus(cls, arr):
        """ 缺省值处理，补齐黄金非交易时间为上一次交易日的收盘价 """
        new_arr = []
        for index, item in enumerate(arr):
            if index == 0:
                date_item = item[0].split("-")
                day = "{}/{}/{}".format(int(date_item[1]), int(date_item[2]), int(date_item[0][2:]))
                new_arr.append([day, item[1]])
            else:
                if item[1] == 0:
                    if index != 0:
                        date_item = item[0].split("-")
                        day = "{}/{}/{}".format(int(date_item[1]), int(date_item[2]), int(date_item[0][2:]))
                        new_arr.append([day, new_arr[index - 1][1]])
                else:
                    date_item = item[0].split("-")
                    day = "{}/{}/{}".format(int(date_item[1]), int(date_item[2]), int(date_item[0][2:]))
                    new_arr.append([day, item[1]])

        with open("../resources/data/NEW-LBMA-GOLD-PLUS.csv", "w", newline="") as gold_file:
            writer = csv.writer(gold_file)
            writer.writerows(new_arr)

    @classmethod
    def normalized(cls, data, min_value, max_value):
        return (data - min_value) / (max_value - min_value)


if __name__ == '__main__':
    bitcoin = "../data/BCHAIN-MKPRU.csv"
    # gold = "../data/LBMA-GOLD.csv"
    # new_gold = "../data/NEW-LBMA-GOLD.csv"
    new_gold_plus = "../data/NEW-LBMA-GOLD-PLUS.csv"
    b_data = DataProcessor(bitcoin).reader()
    g_data = DataProcessor(new_gold_plus).reader()
    # DataProcessor(bitcoin).complete_missing_date(b_data, g_data)
    # DataProcessor(new_gold).complete_missing_date_plus(n_data)

    b_values = [b[1] for b in b_data]
    max_b_value = np.max(b_values)
    min_b_value = np.min(b_values)

    for index in range(len(b_values)):
        b_values[index] = DataProcessor.normalized(b_values[index], min_b_value, max_b_value)

    g_values = [g[1] for g in g_data]
    max_g_value = np.max(g_values)
    min_g_value = np.min(g_values)

    for index in range(len(g_values)):
        g_values[index] = DataProcessor.normalized(g_values[index], min_g_value, max_g_value)

    heads = ["bitcoin", "gold"]
    df = pd.concat([DataFrame(b_data), DataFrame(g_data)], axis=1)
    df.drop(0, axis=1, inplace=True)
    df.columns = heads
    df.index = [item[0] for item in b_data]
    df.bitcoin = b_values
    df.gold = g_values
    new_df = df.copy(deep=True)
    new_df["date"] = [item[0] for item in b_data]
    new_df["date"] = pd.to_datetime(new_df["date"])
    plt.figure(figsize=(20, 8))
    plt.plot(new_df.date, new_df.bitcoin, color="tab:blue")
    plt.plot(new_df.date, new_df.gold, color="tab:orange")

    plt.show()
