import base64
import datetime as dt
import pandas as pd
import pandas_datareader as web
import plotly.graph_objects as go
import streamlit as st
pd.options.plotting.backend = "plotly"

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header('Input Options')
against = st.sidebar.selectbox('Select currency', ('EUR', 'USD'))
currency = st.sidebar.selectbox('Select cryptocurrency', ('BTC', 'ETH', 'EGLD'))

# Web scraping
@st.cache
def load_data(currency='BTC', against='EUR'):
    # Get the stock quote
    crypto_currency = currency
    against_currency = against
    data_source ='yahoo'

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime.now()

    ticket = f'{crypto_currency}-{against_currency}'

    df = web.DataReader(ticket, data_source=data_source, start=start, end=end)

    df.reset_index(inplace=True)
    df.set_index(["Date"])

    sma = df['Close'].rolling(window=30).mean()  # Simple moving average (SMA)
    std = df['Close'].rolling(window=30).std()  # Standard deviation
    df['Upper'] = sma + (2 * std)  # Bollinger band
    df['Lower'] = sma - (2 * std)  # Bollinger band

    df['Short'] = df.Close.ewm(span=20, adjust=False).mean()  # Exponential moving average 20 days
    df['Long'] = df.Close.ewm(span=50, adjust=False).mean()  # Exponential moving average 50 days

    df['20MA'] = df['Close'].rolling(window=20).mean()  # 20 moving average (20MA)
    df['50MA'] = df['Close'].rolling(window=50).mean()  # 50 moving average (50MA)

    shortema = df.Close.ewm(span=12, adjust=False).mean()  # Exponential moving average 12 days
    longema = df.Close.ewm(span=26, adjust=False).mean()  # Exponential moving average 26 days
    df['MACD'] = shortema - longema  # MACD
    df['Signal'] = df.MACD.ewm(span=9, adjust=False).mean()  # Exponential moving average 9 days

    df['PClose'] = df['Adj Close'].shift()  # previous Close
    df['High-Low'] = df['High'] - df['Low']  # Condition 1: High - Low
    df['High-PClose'] = abs(df['High'] - df['PClose'])  # Condition 2: High - Previous Close
    df['Low-PClose'] = abs(df['Low'] - df['PClose'])  # Condition 3: Low - Previous Close
    df['TrueRange'] = df[['High-Low', 'High-PClose', 'Low-PClose']].max(axis=1)  # True Range
    df['ATR'] = df.TrueRange.rolling(window=14).mean() / df['Adj Close'] * 100  # Average True Range 14 days %

    return df


df = load_data(currency, against)


st.sidebar.subheader('Price Data of Selected Cryptocurrency')


# Download CSV data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href


st.sidebar.markdown(filedownload(df), unsafe_allow_html=True)


# Graphics
def price_plot(df):
    period = 60
    df = df.loc[df.shape[0] - period:]
    df.reset_index(inplace=True)

    Closed_Price = go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Close',
        line_color='blue',
    )
    MA20 = go.Scatter(
        x=df['Date'],
        y=df['20MA'],
        name='20MA',
        line_color='black',
    )
    MA50 = go.Scatter(
        x=df['Date'],
        y=df['50MA'],
        name='50MA',
        line_color='orange',
    )
    Upper = go.Scatter(
        x=df['Date'],
        y=df['Upper'],
        name='Upper',
        line_color='silver',
    )
    Lower = go.Scatter(
        x=df['Date'],
        y=df['Lower'],
        name='Lower',
        fill='tonexty',  # fill area between
        mode='lines',
        line_color='silver',
    )
    data = [Closed_Price, MA20, MA50, Upper, Lower]
    layout = go.Layout(yaxis=dict())
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        title='Closed Price',
        yaxis_title='Price',
        xaxis_title='Dates',
        legend_title='Indicator',
    )
    return st.write(fig)


def volume_plot(df):
    period = 60
    df = df.loc[df.shape[0] - period:]
    df.reset_index(inplace=True)

    Volume = go.Scatter(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        line_color='blue',
        fill='tonexty',  # fill area between
        mode='lines',
    )
    layout = go.Layout(yaxis=dict())
    fig = go.Figure(data=Volume, layout=layout)
    fig.update_layout(
        title='Volume',
        yaxis_title='Operations',
        xaxis_title='Dates',
        legend_title='Indicator',
    )
    return st.write(fig)


def atr_plot(df):
    period = 60
    df = df.loc[df.shape[0] - period:]
    df.reset_index(inplace=True)

    fig = df['ATR'].plot()
    fig.update_layout(
        title='ATR',
        yaxis_title='ATR %',
        xaxis_title='Dates',
        legend_title='Indicator',
    )
    return st.write(fig)


def macd_plot(df):
    period = 60
    df = df.loc[df.shape[0] - period:]
    df.reset_index(inplace=True)

    MACD = go.Scatter(
        x=df['Date'],
        y=df['MACD'],
        name='MACD',
        line_color='green',
        fill='tonexty',  # fill area between
        mode='lines',
    )
    Signal = go.Scatter(
        x=df['Date'],
        y=df['Signal'],
        name='Signal',
        line_color='purple',
        line_dash='dot',
    )
    data = [MACD, Signal]
    layout = go.Layout(yaxis=dict())
    fig = go.Figure(data=data, layout=layout)
    fig.add_hline(y=0, line_color="green")
    fig.update_layout(
        title='MACD',
        yaxis_title='MACD',
        xaxis_title='Dates',
        legend_title='Indicator',
    )
    return st.write(fig)


st.header(f'Cryptocurrency Dashboard: {currency}-{against}')
st.markdown("""
This app retrieves cryptocurrency prices for the selected cryptocurrency from the **YahooFinance**!
""")
st.subheader('Data')
st.dataframe(df)

st.subheader('Graphics')
if st.button('Price History (20MA, 50MA, Bollinger limits'):
    price_plot(df)
if st.button('Volume'):
    volume_plot(df)
if st.button('ATR'):
    atr_plot(df)
if st.button('MACD'):
    macd_plot(df)
if st.button('All'):
    price_plot(df)
    volume_plot(df)
    atr_plot(df)
    macd_plot(df)
