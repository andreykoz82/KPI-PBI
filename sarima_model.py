# %%
import itertools
import statsmodels.api as sm
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
import numpy as np

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")
FONT_COLOR = "(0.8,0.8,0.8)"
BACKGROUND_COLOR = '(0.22, 0.22, 0.22)'
plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR
plt.rcParams['figure.facecolor'] = BACKGROUND_COLOR
plt.rcParams['text.color'] = FONT_COLOR
plt.rcParams['axes.labelcolor'] = FONT_COLOR
plt.rcParams['xtick.color'] = FONT_COLOR
plt.rcParams['ytick.color'] = FONT_COLOR


def load_data(filename="sales_extend.xlsx"):
    sales = pd.read_excel(filename)
    return sales


@dataclass
class SalesTimeSeries:
    data: pd.DataFrame
    item: str
    aggregation: str
    start_date: str
    model: str = None
    predictions: float = None
    forecast: float = None

    def transform(self):
        if self.item == 'All':
            self.data = self.data
        else:
            self.data = self.data[self.data['Номенклатура'] == self.item]
        month_mapping = {'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 'Май': 5, 'Июнь': 6, 'Июль': 7,
                         'Август': 8, 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12, }
        month = self.data['По месяцам'].str.replace(' г.', "").str.split(' ', expand=True)[0].map(month_mapping)
        year = self.data['По месяцам'].str.replace(' г.', "").str.split(' ', expand=True)[1]
        df = month.to_frame().join(year)
        df[0] = df[0].astype('str')
        df[1] = df[1].astype('str')
        df['date'] = df[0] + '-' + df[1]
        df['date'] = pd.to_datetime(df['date'])
        self.data['По месяцам'] = df['date']
        self.data = self.data.set_index('По месяцам')['Количество']
        self.data = self.data[self.data.index >= self.start_date]
        self.data = self.data.groupby(pd.Grouper(freq=self.aggregation)).sum()

    def train(self):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        res_aic = 9999
        arima_param = 0
        arima_param_seas = 0

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    self.model = mod.fit()
                    if self.model.aic < res_aic:
                        res_aic = self.model.aic
                        arima_param = param
                        arima_param_seas = param_seasonal
                except:
                    continue

        mod = sm.tsa.statespace.SARIMAX(self.data,
                                        order=arima_param,
                                        seasonal_order=arima_param_seas,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.model = mod.fit()

    def predict(self):
        self.predictions = self.model.get_prediction(start=self.data.index[0]).predicted_mean

    def make_forecast(self):
        self.forecast = self.model.get_forecast(steps=12).predicted_mean

    def plot_time_series(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.data.index, self.data, marker='o', markersize=3, c='#7FFFD4')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.title(f"{self.item} Time series plot")
        plt.grid(linestyle='--', c='grey')
        plt.tight_layout()
        plt.show()

    def plot_predictions(self):
        plt.figure(figsize=(8, 4))

        plt.plot(self.data.index, self.data, marker='o', markersize=3, c='#7FFFD4')
        plt.plot(self.predictions.index, self.predictions, marker='o', markersize=3, c='red')

        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.title(f"{self.item} Actual vs predicted plot")
        plt.grid(linestyle='--', c='grey')
        plt.tight_layout()
        plt.show()

    def plot_forecast(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.forecast.index, self.forecast, marker='o', markersize=5, c='#7FFFD4', mfc='red')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.title(f"{self.item} Forecast plot", pad=20)
        plt.grid(linestyle='--', c='grey')
        plt.xticks(self.forecast.index, rotation=90)
        for x, y in zip(self.forecast.index, self.forecast):
            label = y
            plt.annotate(f"{round(label):,}", (x, y),
                         xycoords="data",
                         textcoords="offset points",
                         xytext=(0, 10), ha="center")
        plt.tight_layout()
        plt.show()

    def show_metrics(self):
        mape = np.mean(np.abs(self.predictions - self.data) / np.abs(self.data))
        me = np.mean(self.predictions - self.data)
        mae = np.mean(np.abs(self.predictions - self.data))
        mpe = np.mean((self.predictions - self.data) / self.data)
        rmse = np.mean((self.predictions - self.data) ** 2) ** .5
        corr = np.corrcoef(self.predictions, self.data)[0, 1]
        mins = np.amin(np.hstack([self.predictions[:, None],
                                  self.data[:, None]]), axis=1)
        maxs = np.amax(np.hstack([self.predictions[:, None],
                                  self.data[:, None]]), axis=1)
        minmax = 1 - np.mean(mins / maxs)

        metrics = {'mape': mape, 'me': me, 'mae': mae,
                   'mpe': mpe, 'rmse': rmse, 'corr': corr, 'minmax': minmax}
        print(metrics)

    def save_forecast(self):
        self.forecast.to_excel("forecast.xlsx")


# %%
sales = load_data()
data = SalesTimeSeries(data=sales, item='All', aggregation='M', start_date='2016-01-01')
data.transform()
data.train()
data.predict()
data.make_forecast()
data.plot_time_series()
data.plot_predictions()
data.plot_forecast()
data.show_metrics()
data.save_forecast()
# %%
