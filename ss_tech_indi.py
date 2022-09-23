import pandas as pd
import pandas_datareader.data as web
import numpy as np
## 使い方
## Colab
# import os
# import shutil
# !git clone https://github.com/selimsivad49/system.git
## Colabでテスト実行
# !python3 '/content/system/ss_tech_indi.py'
## 直接ダウンロード
# if not os.path.exists('system/ss_tech_indi.py'):
#     !wget  -O ss_tech_indi.py "https://raw.githubusercontent.com/selimsivad49/system/main/ss_tech_indi.py"
#     !mkdir system
#     !mv ss_tech_indi.py system/
# sys.path.append('system/')
# sys.path.append('/content/system/')

# import ss_tech_indi as ssti

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

def calc_ATR(high, low, close, term=14):

    # ATR  
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(term).mean()

    return(atr)

def calc_SuperTrend(high, low, close, term=14, multiplier=3):

    # ATR  
    # tr1 = pd.DataFrame(high - low)
    # tr2 = pd.DataFrame(abs(high - close.shift(1)))
    # tr3 = pd.DataFrame(abs(low - close.shift(1)))
    # frames = [tr1, tr2, tr3]
    # tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    # atr = tr.ewm(term).mean()
    atr = calc_ATR(high, low, close, term)
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{term}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    # supertrend = supertrend.dropna()[1:]
    supertrend = supertrend.bfill()

    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
            
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    
    return st, upt, dt

def calc_ATR2(high,low,close,term=14):
    h, l, c_prev = high, low, close.shift(1)
    tr = np.max([high - low, (c_prev - h).abs(), (c_prev - l).abs()], axis=0)
    atr = pd.Series(tr).ewm(term).mean().bfill().values

    return(atr)

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

    df['ST'],df['ST_UPT'],df['ST_DT'] = get_supertrend(df.High, df.Low, df.Close, term=14, multiplier=3)
    
    df.to_csv('nikkei225.csv')
    print(df)
