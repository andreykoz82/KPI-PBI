import streamlit as st
import pandas as pd
import itertools
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import mplcyberpunk
import plotly.express as px

plt.style.use("cyberpunk")

st.title('SALES FORECAST (SARIMAX MODEL)')

@st.cache
def load_data(item, filename="sales_extend.xlsx", start_date='2012-01-01',
                                                  end_date='2022-10-31',
                                                  aggregation='M'):
    data = pd.read_excel(filename)
    if item != 'All':
            data = data[data['Номенклатура'] == item]
    if data['По месяцам'].dtype.kind != 'M':
        month_mapping = {'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 'Май': 5, 'Июнь': 6, 'Июль': 7,
                        'Август': 8, 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12, }
        month = data['По месяцам'].str.replace(' г.', "").str.split(' ', expand=True)[0].map(month_mapping)
        year = data['По месяцам'].str.replace(' г.', "").str.split(' ', expand=True)[1]
        df = month.to_frame().join(year)
        df[0] = df[0].astype('str')
        df[1] = df[1].astype('str')
        df['date'] = df[0] + '-' + df[1]
        df['date'] = pd.to_datetime(df['date'])
        data['По месяцам'] = df['date']
    data = data.set_index('По месяцам')['Количество']
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    data = data.groupby(pd.Grouper(freq=aggregation)).sum()
    return data

actual_items = pd.read_excel('actual_items.xlsx')
option = st.selectbox(
    'Выберите номенклатуру для прогноза:',
    actual_items.item.unique())


data_load_state = st.text('Loading data...')
data = load_data(item=option)
data_load_state.text('Loading data...done!')

def train(data):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    res_aic = 9999
    arima_param = 0
    arima_param_seas = 0

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                model = mod.fit()
                if model.aic < res_aic:
                    res_aic = model.aic
                    arima_param = param
                    arima_param_seas = param_seasonal
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(data,
                                    order=arima_param,
                                    seasonal_order=arima_param_seas,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    model = mod.fit()

    return model

def predict(model):
    predictions = model.get_prediction(start=data.index[0]).predicted_mean
    return predictions

def make_forecast(model):
    forecast = model.get_forecast(steps=12).predicted_mean
    forecast = np.exp(np.log(forecast))
    forecast = forecast.fillna(0)   
    return forecast


model = train(data)
forecast = make_forecast(model)

fig = px.line(forecast, x=forecast.index, y="predicted_mean", title="Прогноз продаж", markers=True,
              template="plotly_dark")
fig.update_xaxes(title_text='Дата')
fig.update_yaxes(title_text='Количество, шт.')

st.plotly_chart(fig, use_container_width=True)

if st.button('Рассчитать запасы ГП'):
    st.subheader('Final data')
    def make_data_table(forecast):
        table = pd.DataFrame(forecast).reset_index().rename(columns={'index': 'End_date'})
        table['Begin_date'] = table['End_date'].astype(str).str[:7] + '-' + '01'
        table['Begin_date'] = pd.to_datetime(table['Begin_date'])
        table['bd_number'] = np.busday_count(begindates=table['Begin_date'].values.astype('datetime64[D]'), 
                                                        enddates=table['End_date'].values.astype('datetime64[D]'))
        table['sales_per_day'] = table['predicted_mean'] / table['bd_number']
        table['MAPE'] = np.mean(np.abs((data - predictions) / data)) * 100
        return table

    predictions = predict(model)
    table = make_data_table(forecast)

    time_series = {}
    time_series[option] = table
    stocks = pd.read_excel('finish_goods_stocks.xlsx')


    def calculate_stock_level(item, stocks, forecast):
        date_index = pd.date_range(start=forecast[item]['Begin_date'].iloc[0], 
                    end=forecast[item]['End_date'].iloc[-1], freq='B')
        temp = pd.DataFrame(index=date_index)
        current_date = pd.to_datetime('today').normalize()
        temp = temp[temp.index>=current_date]
        temp['Остаток'] = stocks[stocks['Наименование'] == item]['Количество'].item()
        i = 0
        dead_line = 0
        stock_level = 0
        
        for date, stock in temp.iterrows():
            year = date.year
            month = date.month
            stat = forecast[item][(forecast[item]['End_date'].dt.year == year) & (forecast[item]['End_date'].dt.month == month)]
            sales_per_day = stat['sales_per_day'].item()
            if i == 0:
                temp['Остаток'].iloc[i] == temp['Остаток'][i]
            else:
                temp['Остаток'].iloc[i] = temp['Остаток'].iloc[i - 1] - sales_per_day
                if temp['Остаток'].iloc[i] < 0:
                    temp['Остаток'].iloc[i:] = 0
                    stock_level = temp.astype(bool).sum(axis=0).item()
                    temp['DeadLine'] = 0
                    if i <= 0:
                        temp['DeadLine'].iloc[0] = 'DeadLine'
                    else:
                        temp['DeadLine'].iloc[i - 6] = 'DeadLine'
                    dead_line = temp[temp['DeadLine'] == 'DeadLine'].index
                    dead_line = pd.Series(dead_line.format())[0]
                    break
                if i == temp.shape[0] - 1:
                    stock_level = temp.astype(bool).sum(axis=0).item()
                    break
            i += 1
        item_stock = stocks[stocks['Наименование'] == item]['Количество'].item() 
        mape = forecast[item]['MAPE'][0]
        
        return item_stock, stock_level, mape, temp, dead_line

    final_data = {}

    if option != 'All':
        item_stock, stock_level, mape, temp_data, dead_line = calculate_stock_level(option, stocks, time_series)
        final_data[option] = [actual_items[actual_items['item']==option]['production_line'].item(), 
        stock_level, item_stock, round(mape, 0), temp_data, dead_line]
    else:
        st.write('None')

    final_table = (pd.DataFrame.from_dict(final_data, orient='index')
    .rename(columns={0: 'Линия производства', 
                    1: 'Запасы (дней)', 
                    2: 'Остатки (шт.)', 
                    3: 'MAPE',
                    5: 'DeadLine'})
    .sort_values(by=['Линия производства', 'Запасы (дней)'])
    .drop(4, 1))

    st.write(final_table)

    fig, ax = plt.subplots(figsize=(12, 4))
    nl = '\n'
    idx = (final_data[option][4]['Остаток'] > 0).sum() + 1
    dead_line = final_data[option][4][final_data[option][4]['DeadLine'] == 'DeadLine'].index
    ax.bar(final_data[option][4].iloc[:idx].index, final_data[option][4].iloc[:idx]['Остаток'])
    ax.axvline(x=dead_line, ymin=0, ymax=final_data[option][4].head(20)['Остаток'].max(), color='red', zorder=2)
    ax.set_title(f'Диаграмма изменения запасов ГП для {option}, {nl}DeadLine at {dead_line.strftime("%Y-%m-%d")[0]}')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    myFmt = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(axis='x', rotation=90);
    st.pyplot(fig)
