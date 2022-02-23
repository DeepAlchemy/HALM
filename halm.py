class HALM:
    """ HALM Model """

    def __init__(self, price_a, price_b, historical_prices):
        """
        Initialization method of HALM model
        :param price_a:             Closing price of the day of product A
        :param price_a:             Closing price of the day of product B
        :param historical_prices:   historical closing price of A and B, the data format is [{"date_01":[price_a,price_b]}]
        """
        self.price_a = price_a
        self.price_b = price_b
        self.historical_prices = historical_prices

    def __price_forecast(self):
        """
        Prediction method of price for the next trading day
        :return: n_price_a, n_price_b: The next trading day's price for Product A and Product B
        """
        n_price_a = 0.0
        n_price_b = 0.0
        return n_price_a, n_price_b

    def __risk_assessment(self):
        """
        Risk assessment methods for Product A and Product B
        :return: rf_a, rf_b: Risk factor for product A and product B
        """
        rf_a = 0.0001
        rf_b = 0.0002
        return rf_a, rf_b

    def __portfolio_decision(self):
        """
        Optimal investment portfolio scheme based on Markowitz model
        :return: weight_a, weight_b: Portfolio Weights for Product A and Product B
        """
        weight_a = 0.50
        weight_b = 0.50
        return weight_a, weight_b

    def halm_decision(self):
        portfolio = {"product_a": 0.0, "product_b": 0.0}
        n_price_a, n_price_b = self.__price_forecast()
        rf_a, rf_b = self.__risk_assessment()
        weight_a, weight_b = self.__portfolio_decision()
        print("The next trading day's price: Product A——{} dollar, Product B——{} dollar".format(n_price_a, n_price_b))
        print("Risk factor: Product A——{}, Product B——{}".format(rf_a, rf_b))
        print("Portfolio Weight: Product A——{}, Product B——{}".format(weight_a * 100, weight_b * 100))
        return portfolio
