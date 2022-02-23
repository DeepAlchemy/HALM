import copy
import random

import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

from utils.processor import DataProcessor


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        # print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res) - 1)

        # reflection
        xr = x0 + alpha * (x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho * (x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma * (tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


class Decision:

    def __init__(self):
        self.POFO_next_day = []

    def update_POFO(self, POFO):
        self.POFO_next_day = POFO

    def computer_action_reward(self, x, day, PB_current, PB_future, PG_current, PG_future, P, cost_g, cost_b, POFO):
        # cash
        cash = POFO[0]
        GOLD = POFO[1]
        BTC = POFO[2]

        POFO_next_day = [0.0, 0.0, 0.0]

        total = cash + GOLD * PG_current + BTC * PB_current
        R = random.random()
        # if abs(PB_future / PB_current - 1) > abs(PG_future / PG_current - 1):
        if R >= 0.5:
            x[1] = min(x[1], cash / PB_current)
            x[1] = min(x[1], total * P / PB_current - BTC)
            x[1] = max(x[1], -BTC)
            x[0] = min(x[0], (cash - x[1] * PB_current) / PG_current)
            x[0] = max(x[0], -GOLD)
        else:
            x[0] = min(x[0], cash / PG_current)
            x[0] = max(x[0], -GOLD)
            x[1] = min(x[1], (cash - x[0] * PG_current) / PB_current)
            x[1] = min(x[1], total * P / PB_current - BTC)
            x[1] = max(x[1], -BTC)

        # update POFO
        if R >= 0.5:
            if x[1] >= 0:
                POFO_next_day[0] = cash - x[1] * PB_current
                x[1] = x[1] * (1 - cost_b)
            else:
                POFO_next_day[0] = cash + abs(x[1]) * (1 - cost_b) * PB_current

            if x[0] >= 0:
                POFO_next_day[0] = POFO_next_day[0] - x[0] * PG_current
                x[0] = x[0] * (1 - cost_g)
            else:
                POFO_next_day[0] = POFO_next_day[0] + abs(x[0]) * (1 - cost_g) * PG_current
        else:
            if x[0] >= 0:
                POFO_next_day[0] = cash - x[0] * PG_current
                x[0] = x[0] * (1 - cost_g)
            else:
                POFO_next_day[0] = cash + abs(x[0]) * (1 - cost_g) * PG_current
            if x[1] >= 0:
                POFO_next_day[0] = POFO_next_day[0] - x[1] * PB_current
                x[1] = x[1] * (1 - cost_b)
            else:
                POFO_next_day[0] = POFO_next_day[0] + abs(x[1]) * (1 - cost_b) * PB_current

        if day == 0:
            x[0] = 0
        # cash_new = cash - x[0] * PG_current - x[1] * PB_current
        GOLD_new = GOLD + x[0]
        BTC_new = BTC + x[1]
        # POFO_next_day = [cash_new, GOLD_new, BTC_new]
        POFO_next_day[1] = GOLD_new
        POFO_next_day[2] = BTC_new
        self.update_POFO(POFO_next_day)

        return POFO_next_day[0] + GOLD_new * PG_future + BTC_new * PB_future


if __name__ == "__main__":
    # test
    import math
    import numpy as np

    # PB0 = 32195.46  # 2021.1.3
    # PG0 = 1887.6  # 2021.1.3

    # PB = [PB0, 32000.78, 32035.03, 34046.67, 36860.41, 39486.04, 40670.25, 40240.72]
    # PG = [PG0, 1880.2, 1940.35, 1931.95, 1920.1, 1862.9, 1888.3, 1903.6]

    bitcoin = "../data/BCHAIN-MKPRU.csv"
    new_gold_plus = "../data/NEW-LBMA-GOLD-PLUS.csv"
    b_data = DataProcessor(bitcoin).reader()
    g_data = DataProcessor(new_gold_plus).reader()

    PB = [b[1] for b in b_data]
    PG = [g[1] for g in g_data]

    PB0 = PB[0]
    PG0 = PG[0]

    P = 0.5
    POFO0 = [1000., 0., 0.]

    AST0 = POFO0[0] * 1 + POFO0[1] * PG0 + POFO0[2] * PB0
    cost_g = 0.01
    cost_b = 0.02
    # AST = [AST0, 0, 0, 0, 0, 0, 0, 0]
    decision = Decision()
    decision.POFO_next_day = POFO0
    # x = np.array([0., 0., 0.])
    x1 = np.array([0., 0.])

    returns = []

    for i in range(len(PB) - 1):
        PB0 = PB[i]
        PG0 = PG[i]
        POFO0 = decision.POFO_next_day


        def f(x):
            # return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[2]) + 1))
            return -decision.computer_action_reward(x, 1, PB0, PB[i + 1], PG0, PG[i + 1], P, cost_g, cost_b, POFO0)


        print("Today gold: ", PG0, "Today BTC: ", PB0)
        print("Next day gold: ", PG[i + 1], "Next day BTC: ", PB[i + 1])
        # print()
        ans = nelder_mead(f, x1)
        print("Delta gold: ", ans[0][0], "Delta BTC: ", ans[0][1])
        # print(nelder_mead(f, x1))
        print("Next day total asset: ", -ans[1])
        returns.append(-ans[1])

    dates = [b[0] for b in b_data]
    rdf = DataFrame(returns)
    heads = ["return", "date"]
    rdf["1"] = dates[1:]
    rdf.columns = heads
    rdf["date"] = pd.to_datetime(rdf["date"])
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(rdf.date, rdf["return"], color="b")
    plt.xlabel("Datetime (09/11/2016 - 09/10/2021)")
    plt.ylabel("Portfolio Value (Cash+Bitcoin+Gold) / US Dollars")
    plt.show()
