import numpy as np
import pandas as pd
import scipy.optimize as sco

from pandas.core.frame import DataFrame
from utils.data_processor import DataProcessor
from utils.common import separate


class MarkowitzPreprocessor(object):

    def __init__(self, data):
        self.data = data
        self.daily_yields = []
        self.daily_yield_calculator()
        self.m = self.mean()
        self.var = self.variance()
        self.std = self.standard_deviation()

    def dataset_length(self):
        return len(self.data)

    def daily_yield_calculator(self):
        """ 计算日收益率 """
        for index, item in enumerate(self.data):
            if index == 0:
                self.daily_yields.append([item[0], 0])
            else:
                closing_price = item[1]
                last_closing_price = self.data[index - 1][1]
                if last_closing_price != 0:
                    daily_yield = (closing_price - last_closing_price) / last_closing_price
                else:
                    daily_yield = 0
                self.daily_yields.append([item[0], daily_yield])
        return self.daily_yields

    def mean(self):
        """ 计算日收益率的均值 """
        return np.mean([dy[1] for dy in self.daily_yields])

    def variance(self):
        """ 计算日收益率的方差 """
        return np.var([dy[1] for dy in self.daily_yields])

    def standard_deviation(self):
        """ 计算日收益率的标准差 """
        return np.std([dy[1] for dy in self.daily_yields])

    @classmethod
    def covariance(cls, arr_1, arr_2):
        """ 日收益率的协方差 """
        return np.cov(np.array(arr_1), np.array(arr_2))


class Markowitz:
    """ 马科维茨模型 """

    @classmethod
    def markowitz_model(cls, bitcoin_data, gold_data):
        """
        马科维茨模型工程化
        返回两个商品各自的权重w_1, w_2
        """
        heads = ["bitcoin", "gold"]
        df = pd.concat([DataFrame(bitcoin_data), DataFrame(gold_data)], axis=1)
        df.drop(0, axis=1, inplace=True)
        df.columns = heads
        df.index = [item[0] for item in b_data]
        print("合并后的数据列表: ")
        print(df)
        separate()

        returns = np.log(df / df.shift(1))  # 计算投资资产的协方差是构建资产组合过程的核心部分
        length = len(df)
        print("投资资产的协方差列表: ")
        print(returns)
        separate()

        print("每日收益平均值: ")
        print(returns.mean())
        separate()

        print("年化收益: ")
        print(returns.mean() * length)
        separate()

        print("相关系数: ")
        print(returns.corr())
        separate()

        print("协方差: ")
        print(returns.cov() * length)
        separate()

        returns.hist(bins=10, figsize=(9, 6))

        """ 给不同资产随机分配初始权重，所有的权重系数均在0-1之间 """
        weights = np.random.random(2)
        weights /= np.sum(weights)

        print("初始化权重: {}".format(weights))

        """ 计算预期组合年化收益、组合方差和组合标准差 """
        print("初始权重收益: {}".format(np.sum(returns.mean() * weights) * length))
        print("初始权重组合方差: {}".format(np.dot(weights.T, np.dot(returns.cov() * length, weights))))
        print("初始权重组合标准差: {}".format(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * length, weights)))))

        """
        用蒙特卡洛模拟产生大量随机组合，给定的一个投资组合如何找到风险和收益平衡的位置
        通过一次蒙特卡洛模拟，产生大量随机的权重向量，并记录随机组合的预期收益和方差
        """
        port_returns = []
        port_variance = []
        for p in range(4000):
            weights = np.random.random(2)
            weights /= np.sum(weights)
            port_returns.append(np.sum(returns.mean() * length * weights))
            port_variance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * length, weights))))
        port_returns = np.array(port_returns)
        port_variance = np.array(port_variance)

        def statistics(weights):
            """
            记录重要的投资组合统计数据（收益，方差和夏普比）,通过对约束最优问题的求解，得到最优解。其中约束是权重总和为1。
            """
            weights = np.array(weights)
            port_returns = np.sum(returns.mean() * weights) * length
            port_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * length, weights)))
            return np.array([port_returns, port_variance, port_returns / port_variance])

        def min_sharpe(weights):
            """
            最优化投资组合的推导是一个约束最优化问题
            最小化夏普指数的负值
            """
            return -statistics(weights)[2]

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 约束是所有参数(权重)的总和为1，使用minimize函数的约定表达即可
        bnds = tuple((0, 1) for x in range(2))  # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
        opts = sco.minimize(min_sharpe, 2 * [1. / 2, ], method='SLSQP', bounds=bnds,
                            constraints=cons)  # 优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)，我们简单的使用平均分布。
        print("权重：{}".format(opts['x'].round(3)))  # 预期收益率、预期波动率、最优夏普指数
        print("最大sharpe指数预期收益: {}".format(statistics(opts['x']).round(3)))
        return opts['x'].round(3)[0], opts['x'].round(3)[1]  # 此处返回两款产品的权重


bitcoin = "../data/BCHAIN-MKPRU.csv"
gold = "../data/NEW-LBMA-GOLD-PLUS.csv"
b_data = DataProcessor(bitcoin).reader()
bm = MarkowitzPreprocessor(b_data)
b_dys = bm.daily_yield_calculator()
b_mean = bm.mean()
b_variance = bm.variance()
b_std = bm.standard_deviation()
print("BitCoin————均值：{}，方差：{}，标准差：{}".format(b_mean, b_variance, b_std))
g_data = DataProcessor(gold).reader()
g_data.insert(0, ["2016-09-11", 1324.6])
gm = MarkowitzPreprocessor(g_data)
g_dys = gm.daily_yield_calculator()
g_mean = gm.mean()
g_variance = gm.variance()
g_std = gm.standard_deviation()
print("Gold————均值：{}，方差：{}，标准差：{}".format(g_mean, g_variance, g_std))
weights = Markowitz.markowitz_model(b_data, g_data)
print("投资权重为: {}".format(weights))
