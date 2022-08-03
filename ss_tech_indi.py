import pandas as pd
import pandas_datareader.data as web

## EMA
# calc_EMA(df_main) は df_main.ewm(span=3).mean() と同じ
def calc_EMA(df_main, df_now=None, term=3, index=True):
    df = pd.DataFrame(index=df_main.index)
    df['old'] = df_main
    if df_now == None:
        df['now'] = df_main
    else:
        df['now'] = df_now
    
    # 移動平均を計算する
    df['sum_old'] = df['old'].rolling(window = term-1, center = False).sum().shift()
    df['sma'] = (df['now'] + df['sum_old']) / term
    # 先頭N-1本文NaNになっているため平均で埋める(ewmと合わせるため)
    df.loc[df.index[0],'sum_old'] = 0
    for i in range(1,term-1):
        df.loc[df.index[i],'sum_old'] = df.loc[df.index[i-1],'sum_old']*i + df.loc[df.index[i],'old']

    df.loc[df.index[0],'sma'] = df.loc[df.index[0],'now']
    for i in range(1,term-1):
        df.loc[df.index[i],'sma'] = (df.loc[df.index[i],'sum_old']*i + df.loc[df.index[i],'now']) / (i+1)

    # 指数移動平均の計算に使うαを求める    
    # a = 2 / (term + 1)
    # 最初のEMAはSMAを使う
    df['ema'] = (df['sma'].shift() * (term - 1) + df['now']*2)/(term + 1)
    df['ema'] = df['sma']

    for i in range(1,len(df.index)):
        df.at[df.index[i],'ema'] = (df.loc[df.index[i-1],'ema'] * (term-1) + df.loc[df.index[i],'now']*2) / (term+1)
        # αを使う場合。↑と同じ
        # df.at[df.index[i],'ema_'] = df.at[df.index[i-1],'ema'] + a*(df.loc[df.index[i],'now'] - df.at[df.index[i-1],'ema'])

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
def calc_RSI(df_main,df_now=None,term=14):

    df_diff = pd.DataFrame(index=df_main.index)
    if df_now == None:
        df_diff['old'] = df_main
    else:
        df_diff['now'] = df_now
    df_diff['old'] = df_main

    df_diff['diff_old'] = df_diff['old'].diff(1)
    # RSI計算のための上昇、下降を算出する
    df_diff['diff_old_up'] = df_diff['diff_old'].apply(lambda x : max(x,0))
    df_diff['diff_old_down'] = df_diff['diff_old'].apply(lambda x : min(x,0))
    ## 当日分は比較する値が異なる
    df_diff['diff_now'] = df_diff['now'] - df_diff['old'].shift()
    # RSI計算のための上昇、下降を算出する
    df_diff['diff_now_up'] = df_diff['diff_now'].apply(lambda x : max(x,0))
    df_diff['diff_now_down'] = df_diff['diff_now'].apply(lambda x : min(x,0))

    # 14日上昇分の平均値を計算
    df_diff['diff_up_term'] = (df_diff['diff_old_up'].rolling(window = term-1, center = False).sum().shift() + df_diff['diff_now_up']) / term
    # 14日下降分の平均値を計算
    df_diff['diff_down_term'] = abs(df_diff['diff_old_down'].rolling(window = term-1, center = False).sum().shift() + df_diff['diff_now_down']) / term
    df_diff['RSI'] = (df_diff['diff_up_term'] / (df_diff['diff_up_term'] + df_diff['diff_down_term'])) * 100
    # print(df_diff.tail(10))
    return(df_diff['RSI'])

if __name__ == '__main__':
    df = web.DataReader('^N225', 'yahoo', '2020-01-01', '2021-01-01')

    ema_terms = [5]
    for term in ema_terms:
        col = 'EMA_' + str(term)
        df[col+'_'] = df['Close'].ewm(span=term, adjust=False).mean()
        df[col] = calc_EMA(df['Close'],df_now=df['Close'],term=term)
        df['CloseEMAGrad_'+str(term)] = df[col] / df[col].shift()

    df['RSI_14'] = calc_RSI_(df['Close'],df_now=df['Close'],14)

    print(df)
