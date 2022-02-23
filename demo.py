from halm import HALM

""" A demo of HALM """
a = 1000.44  # The closing price of product A on the day
b = 564.11  # The closing price of product B on the day
h_ab = [
    {"2020-11-01": [852.59, 485.44]},
    {"2020-11-02": [891.38, 933.66]},
    {"2020-11-03": [977.14, 996.52]}
]  # Historical closing prices for products A and B
portfolio = HALM(
    price_a=a, price_b=b, historical_prices=h_ab
).halm_decision()  # Portfolio investment strategy for the next trading day
