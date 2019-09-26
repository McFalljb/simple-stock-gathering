import bs4 as bs
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import requests
import numpy as np
import yfinance as yf
import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
yf.pdr_override()

style.use('ggplot')


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        ticker = str(ticker).replace(".", "-")
        tickers.append(ticker)

    with open('sp500tickers.pickle', "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers


def get_data_from_yahoo(reload_sp500=True):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}/{}.csv'.format(ticker, ticker)):

            os.makedirs('stock_dfs/{}'.format(ticker))
            df = pdr.get_data_yahoo(ticker, period='10y', treads=True)
            df.to_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))

        else:

            dfnd = pdr.get_data_yahoo(ticker, period='1d', treads=True)
            # dfnd.set_index('Date', inplace=True)
            # df = pd.read_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))
            # df.concat(dfnd, ignore_index=True)
            with open('stock_dfs/{}/{}.csv'.format(ticker, ticker), 'a') as f:
                dfnd.to_csv(f, header=False)


get_data_from_yahoo(False)

# rsi out input
n = 14

# used during the rsi calculation for max scalability


def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.append(y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1))


def param():
    with open('sp500tickers.pickle', "rb") as f:
        tickers = pickle.load(f)

    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))

        df['Volume_Diff'] = df['Volume'].diff().fillna(0)
        df['Adj_Close_Diff'] = df['Adj Close'].diff().fillna(0)
        df['Adj_Close_Pct_Change'] = df['Adj Close'].pct_change().fillna(0)
        df['Volume_Pct_Change'] = df['Volume'].pct_change().fillna(0)

        # Calculating RSI
        df['gain'] = df.Adj_Close_Diff.mask(df.Adj_Close_Diff < 0, 0.0)
        df['loss'] = -df.Adj_Close_Diff.mask(df.Adj_Close_Diff > 0, -0.0)
        df.loc[n:, 'avg_gain'] = rma(df.gain[n + 1:].values, n, df.loc[:n, 'gain'].mean())
        df.loc[n:, 'avg_loss'] = rma(df.loss[n + 1:].values, n, df.loc[:n, 'loss'].mean())
        df['rs'] = df.avg_gain / df.avg_loss
        df['rsi_14'] = 100 - (100 / (1 + df.rs))

        # Calculating BB
        df['30 Day MA'] = df['Adj Close'].rolling(window=30).mean()
        df['30 Day STD'] = df['Adj Close'].rolling(window=30).std()
        df['Upper Band'] = df['30 Day MA'] + (df['30 Day STD'] * 2)
        df['Lower Band'] = df['30 Day MA'] - (df['30 Day STD'] * 2)

        df.drop(['Adj_Close_Diff', 'Volume_Diff'], 1, inplace=True)

        df.set_index('Date', inplace=True)
        df.to_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))


param()


def volumediffone():

    with open('sp500tickers.pickle', "rb") as f:
        tickers = pickle.load(f)

    for ticker in tickers:
        df = pd.read_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))
        c = df.index[df['Volume_Pct_Change'] >= 1.2]

        if not os.path.exists('VolumeDiff/{}'.format(ticker)):
            os.makedirs('VolumeDiff/{}'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

        for idx in c:
            d = df.iloc[(idx - 7):(idx + 30)]
            d.to_csv('VolumeDiff/{}/{}{}.csv'.format(ticker, ticker, idx))


volumediffone()


def irondiablo():
    with open('sp500tickers.pickle', "rb") as f:
        tickers = pickle.load(f)

        os.makedirs('IronCondor/Toplist')

    for ticker in tickers:
        if not os.path.exists('IronCondor/Tickers/{}'.format(ticker)):
            os.makedirs('IronCondor/Tickers/{}'.format(ticker))

        else:
            print('Already have {}'.format(ticker))

        df = pd.read_csv('stock_dfs/{}/{}.csv'.format(ticker, ticker))
        pd.options.mode.chained_assignment = None  # default='warn'
        df30 = df.tail(30)

        df30.drop(['avg_gain', 'avg_loss', 'rs', 'rsi_14', 'gain', 'loss',
               'Adj_Close_Pct_Change', 'Volume_Pct_Change',
               'Upper Band', 'Lower Band'], 1, inplace=True)

        df30['irondiablo'] = (df30['30 Day STD'] / df30['30 Day MA']) * 100
        a = df30['irondiabloemean'] = df30['irondiablo'].mean()

        df30.to_csv('IronCondor/Tickers/{}/{}.csv'.format(ticker, ticker))

        if a <= 1.5:
            if not os.path.exists('IronCondor/Toplist/{}'.format(ticker)):
                os.makedirs('IronCondor/Toplist/{}'.format(ticker))

            df30.to_csv('IronCondor/Toplist/{}/{}.csv'.format(ticker, ticker))


irondiablo()