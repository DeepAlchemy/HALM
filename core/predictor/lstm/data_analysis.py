import matplotlib.pyplot as plt
import pandas as pd


class DataAnalysis:

    @classmethod
    def data_visualize(self, bitcoin_path, gold_path):
        bitcoin_data = pd.read_csv(bitcoin_path)
        gold_data = pd.read_csv(gold_path)

        plt.plot(bitcoin_data['Value'])
        plt.title('Bitcoin daily prices', fontsize=15)
        plt.xlabel('Date', color='blue')
        plt.ylabel('Value', color='blue')
        plt.legend()
        plt.show()

        plt.plot(gold_data['USD (PM)'])
        plt.title('Gold daily prices', fontsize=15)
        plt.xlabel('Date', color='blue')
        plt.ylabel('Price', color='blue')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bitcoin_path = '../../../resources/data/BCHAIN-MKPRU.csv'
    gold_path = '../../../resources/data/LBMA-GOLD.csv'
    DataAnalysis.data_visualize(bitcoin_path, gold_path)