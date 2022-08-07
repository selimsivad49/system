import pandas as pd
import pandas_datareader.data as web

## EMA
# calc_EMA(df_main) は df_main.ewm(span=3).mean() と同じ
def calc_EMA(df_main, df_latest=None, term=3, index=True):
    df = pd.DataFrame(index=df_main.index)
    df['old'] = df_main
    df['latest'] = df_latest
    if df['latest'].sum()==0:
        df['latest'] = df_main

    # 移動平均を計算する
    df['sum_old'] = df['old'].rolling(window = term-1, center = False).sum().shift()
    df['sma'] = (df['latest'] + df['sum_old']) / term
    # 先頭N-1本文NaNになっているため平均で埋める(ewmと合わせるため)
    df.loc[df.index[0],'sum_old'] = 0
    for i in range(1,term-1):
        df.loc[df.index[i],'sum_old'] = df.loc[df.index[i-1],'sum_old']*i + df.loc[df.index[i],'old']

    df.loc[df.index[0],'sma'] = df.loc[df.index[0],'latest']
    for i in range(1,term-1):
        df.loc[df.index[i],'sma'] = (df.loc[df.index[i],'sum_old']*i + df.loc[df.index[i],'latest']) / (i+1)

    # 指数移動平均の計算に使うαを求める    
    # a = 2 / (term + 1)
    # 最初のEMAはSMAを使う
    df['ema'] = (df['sma'].shift() * (term - 1) + df['latest']*2)/(term + 1)
    df['ema'] = df['sma']

    for i in range(1,len(df.index)):
        df.at[df.index[i],'ema'] = (df.loc[df.index[i-1],'ema'] * (term-1) + df.loc[df.index[i],'latest']*2) / (term+1)
        # αを使う場合。↑と同じ
        # df.at[df.index[i],'ema_'] = df.at[df.index[i-1],'ema'] + a*(df.loc[df.index[i],'latest'] - df.at[df.index[i-1],'ema'])

    # print(df)
    return(df['ema'])

## RSI
def calc_RSI_(df,term=14):
    df_diff = df.diff(1)
    # RSI計算のための上昇、下降を算出する
    df_up, df_down = df_diff.copy(), df_diff.copy()
    df_up[df_up < 0] = 0 # 0未満は0を代入
    df_down[df_down > 0] = 0 # 0より大きいは0を代入
    # 14日上昇分の平均値を計算
    df_up_term = df_up.rolling(window = term, center = False).mean()
    # 14日下降分の平均値を計算
    df_down_term = abs(df_down.rolling(window = term, center = False).mean())
    # print(df_up_term)
    # print(df_down_term)
    return((df_up_term / (df_up_term + df_down_term)) * 100)

## RSI
def calc_RSI(df_main,df_latest=None,term=14):

    df = pd.DataFrame(index=df_main.index)
    df['old'] = df_main
    df['latest'] = df_latest
    if df['latest'].sum()==0:
        df['latest'] = df_main

    df['diff_old'] = df['old'].diff(1)
    # RSI計算のための上昇、下降を算出する
    df['diff_old_up'] = df['diff_old'].apply(lambda x : max(x,0))
    df['diff_old_down'] = df['diff_old'].apply(lambda x : min(x,0))
    ## 当日分は比較する値が異なる
    df['diff_latest'] = df['latest'] - df['old'].shift()
    # RSI計算のための上昇、下降を算出する
    df['diff_latest_up'] = df['diff_latest'].apply(lambda x : max(x,0))
    df['diff_latest_down'] = df['diff_latest'].apply(lambda x : min(x,0))

    # 14日上昇分の平均値を計算
    df['diff_up_term'] = (df['diff_old_up'].rolling(window = term-1, center = False).sum().shift() + df['diff_latest_up']) / term
    # 14日下降分の平均値を計算
    df['diff_down_term'] = abs(df['diff_old_down'].rolling(window = term-1, center = False).sum().shift() + df['diff_latest_down']) / term
    df['RSI'] = (df['diff_up_term'] / (df['diff_up_term'] + df['diff_down_term'])) * 100
    # print(df_diff.tail(10))
    return(df['RSI'])

## MACD
def calc_MACD(df_main, df_latest=None, term1=12, term2=26):
    df = pd.DataFrame(index=df_main.index)

    df['ema1'] = calc_EMA(df_main,df_latest=df_latest,term=term1)
    df['ema2'] = calc_EMA(df_main,df_latest=df_latest,term=term2)
    df['macd'] = df['ema1'] - df['ema2']
    df['signal'] = df['macd'].rolling(9).mean()

    return(df['macd'],df['signal'])

if __name__ == '__main__':
    df = web.DataReader('nikkei225', 'fred', '2020-01-01', '2021-01-01')

    df.dropna(how='any',inplace=True)
    ema_terms = [5]
    for term in ema_terms:
        col = 'EMA_' + str(term)
        df[col+'_'] = df['nikkei225'].ewm(span=term, adjust=False).mean()
        df[col] = calc_EMA(df['nikkei225'],df_latest=df['nikkei225'],term=term)
        df['CloseEMAGrad_'+str(term)] = df[col] / df[col].shift()

    df['RSI_14'] = calc_RSI(df['nikkei225'],df_latest=df['nikkei225'],term=14)
    df['RSI_14_'] = calc_RSI(df['nikkei225'])
    df['RSI_14__'] = calc_RSI_(df['nikkei225'],term=14)
    df['MACD'],df['Signal'] = calc_MACD(df['nikkei225'])

    df.to_csv('nikkei225.csv')
    print(df)
