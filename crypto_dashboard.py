import base64
import datetime as dt
import pandas as pd
import pandas_datareader as web
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
pd.options.plotting.backend = "plotly"

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header('Input Options')
against = st.sidebar.selectbox('Select currency', ('EUR', 'USD'))
currency = st.sidebar.selectbox('Select cryptocurrency', ('BTC', 'ETH', 'EGLD', 'BNB', 'ICX', 'WABI', 'WAVES'))

# Web scraping
@st.cache
def linear_regression(Y, n=1):
    X = [i for i in range(1, n+1)]
    XY = [X[i] * Y[i] for i in range(n)]
    XX = [X[i]**2 for i in range(n)]
    s = (n * sum(XY) - sum(X) * sum(Y)) / (n * sum(XX) - sum(X) ** 2)
    b = (sum(Y) - s * sum(X)) / n

    return b + s * n


def curve(Y, n):
    lrc = [0] * n
    x = 0
    while x <= len(Y) - n:
        lr = linear_regression(Y[x:(x + n)].tolist(), n)
        lrc.append(round(lr, 2))
        x += 1
    return lrc


def load_data(currency='BTC', against='EUR'):
    # Get the stock quote
    crypto_currency = currency
    against_currency = against
    data_source = 'yahoo'

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime.now()

    ticker = f'{crypto_currency}-{against_currency}'

    df = web.DataReader(ticker, data_source=data_source, start=start, end=end)

    df.reset_index(inplace=True)
    df.set_index(["Date"])

    # Exponential moving averages
    df['EMA10'] = df.Close.ewm(span=10, adjust=False).mean()
    df['EMA55'] = df.Close.ewm(span=55, adjust=False).mean()
    df['EMA200'] = df.Close.ewm(span=200, adjust=False).mean()

    # ADX
    df['PClose'] = df['Close'].shift()  # previous Close
    df['High-Low'] = df['High'] - df['Low']  # Condition 1: High - Low
    df['High-PClose'] = abs(df['High'] - df['PClose'])  # Condition 2: High - Previous Close
    df['Low-PClose'] = abs(df['Low'] - df['PClose'])  # Condition 3: Low - Previous Close
    df['TR'] = df[['High-Low', 'High-PClose', 'Low-PClose']].max(axis=1)  # True Range
    del df['High-Low'], df['High-PClose'], df['Low-PClose'], df['PClose']

    df['PHigh'] = df['High'].shift()  # previous high
    df['PLow'] = df['Low'].shift()  # previous low
    df['H-pH'] = df['High'] - df['PHigh']
    df['pL-L'] = df['PLow'] - df['Low']
    df['DMp'] = df.apply(lambda x: 0 if x['H-pH'] <= x['pL-L'] else 0 if x['H-pH'] < 0 else x['H-pH'], axis=1)
    df['DMl'] = df.apply(lambda x: 0 if x['pL-L'] <= x['H-pH'] else 0 if x['pL-L'] < 0 else x['pL-L'], axis=1)

    DIp14 = 100 * df.DMp.rolling(window=14).sum() / df.TR.rolling(window=14).sum()
    DIl14 = 100 * df.DMl.rolling(window=14).sum() / df.TR.rolling(window=14).sum()
    DI14 = abs(DIp14 - DIl14)
    DIS14 = DIp14 + DIl14

    df['DX'] = DI14 / DIS14
    df['ADX'] = 100 * df.DX.rolling(window=14).mean()
    del df['PHigh'], df['PLow'], df['H-pH'], df['pL-L'], df['DMp'], df['DMl']

    # TTM
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['upperBB'] = df['MA20'] + (2 * df.Close.rolling(window=20).std())
    df['lowerBB'] = df['MA20'] - (2 * df.Close.rolling(window=20).std())
    df['upperKC'] = df['MA20'] + (1.5 * df.TR.rolling(window=20).mean())
    df['lowerKC'] = df['MA20'] - (1.5 * df.TR.rolling(window=20).mean())

    df['sqzOn'] = df.apply(lambda x: (x['lowerBB'] > x['lowerKC']) and (x['upperBB'] < x['upperKC']), axis=1)
    df['sqzOff'] = df.apply(lambda x: (x['lowerBB'] < x['lowerKC']) and (x['upperBB'] > x['upperKC']), axis=1)
    df['noSqz'] = df.apply(lambda x: (x['sqzOn'] == False) and (x['sqzOff'] == False), axis=1)

    df['hHigh'] = df.High.rolling(window=20).max()
    df['lLow'] = df.Low.rolling(window=20).min()
    df['MA'] = df[['hHigh', 'lLow']].mean(axis=1)
    df['MA20MA'] = df[['MA', 'MA20']].mean(axis=1)
    df['TTM'] = df['Close'] - df['MA20MA']
    del df['MA20'], df['hHigh'], df['lLow'], df['MA'], df['MA20MA'], df['sqzOn'], df['sqzOff'], df['TR']

    # Linear regression curve
    df.drop(index=df.index[:19], axis=0, inplace=True)
    df.reset_index(inplace=True)
    df['SMI'] = pd.DataFrame(curve(df['TTM'], 19))

    df['ADXp'] = df['ADX'].shift()
    df['SMIp'] = df['TTM'].shift()
    df['EMAr'] = df.apply(lambda x: 'ALC' if x['EMA10'] > x['EMA55'] else 'BAJ', axis=1)
    df['ADXr'] = df.apply(lambda x: 'ASC' if x['ADX'] > x['ADXp'] else 'DES', axis=1)
    df['SMIr'] = df.apply(lambda x: 'ASC' if x['SMI'] > x['SMIp'] else 'DES', axis=1)
    del df['ADXp'], df['SMIp']

    return df


df = load_data(currency, against)
predictions = df.iloc[-1].tolist()

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
    Closed_Price = go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    EMA10 = go.Scatter(x=df['Date'], y=df['EMA10'], name='EMA10', line_color='blue', )
    EMA55 = go.Scatter(x=df['Date'], y=df['EMA55'], name='EMA55', line_color='brown', )
    EMA200 = go.Scatter(x=df['Date'], y=df['EMA200'], name='EMA200', line_color='pink', )
    ADX = go.Scatter(x=df['Date'], y=df['ADX'], name='ADX', line_color='grey', )
    TTM = go.Bar(x=df['Date'], y=df['TTM'], name='TTM', marker=dict(color='#8dc1aa'))
    SMI = go.Scatter(x=df['Date'], y=df['SMI'], name='SMI', line_color='gold', )
    Volume = go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker=dict(color='#00dede'))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.25, 0.25, 0.1])

    fig.add_trace(Closed_Price, row=1, col=1, )
    fig.add_trace(EMA10, row=1, col=1, )
    fig.add_trace(EMA55, row=1, col=1, )
    fig.add_trace(EMA200, row=1, col=1, )
    fig.add_trace(ADX, row=2, col=1)
    fig.add_trace(TTM, row=3, col=1)
    fig.add_trace(SMI, row=3, col=1)
    fig.add_trace(Volume, row=4, col=1)

    fig.add_hline(y=25, line_color="purple", line_dash='dot', row=2, col=1)
    fig.add_hline(y=40, line_color="purple", line_dash='dot', row=2, col=1)
    fig.add_hline(y=60, line_color="purple", line_dash='dot', row=2, col=1)

    fig.update_xaxes(title_text="Dates", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ADX", row=2, col=1)
    fig.update_yaxes(title_text="TTM", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    fig.update_layout(title_text='Price', height=1000)
    fig.layout.xaxis.rangeslider.visible = False

    return st.write(fig)


def volume_plot(df):
    fig = px.histogram(df, x=df['Volume'], y=df['Close'], nbins=150, orientation='h')
    return st.write(fig)


st.header(f'Cryptocurrency Dashboard: {currency}-{against}')
st.markdown("""
This app retrieves cryptocurrency prices for the selected cryptocurrency from the **YahooFinance**!
""")

st.subheader('Data')
st.dataframe(df)

st.subheader('Status')
st.write(f'EMAr: {predictions[-3]} --|-- ADXr: {predictions[-2]} --|-- SMIr: {predictions[-1]}')

st.subheader('Graphics')
price_plot(df)
volume_plot(df)
