# %%
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from autots import AutoTS


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
        self.data = pd.DataFrame(self.data).reset_index().rename(columns={'По месяцам': 'date', 'Количество': 'value'})

    def train(self):
        self.model = AutoTS(
            forecast_length=12,
            frequency='infer',
            prediction_interval=0.9,
            ensemble='all',
            no_negatives=True,
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            drop_most_recent=1,
            max_generations=15,
            num_validations=2,
            validation_method="backwards"
        )
        self.model = self.model.fit(
            self.data,
            date_col="date",
            value_col='value',
            id_col=None
        )

    def predict(self):
        self.predictions = self.model.predict()

    def make_forecast(self):
        self.forecast = self.predictions.forecast

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
        print(self.model)

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
# %%

# %%
data.forecast.plot()
plt.show()
