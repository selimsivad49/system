import pandas as pd

## EMAの計算
## 15:15の終値だけで、ewm()で計算すると当日の15:00時点の値が計算できない
# 【Python】移動平均(SMA)と指数移動平均(EMA)を計算する | ミナピピンの研究室
# https://tkstock.site/2019/09/27/indicator-python-sma-ema/

# 指数移動平均線を計算する関数
def calc_ema(df_now,df_old, term, index=True):
    df = pd.DataFrame()
    df['now'] = df_now
    df['old'] = df_old
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

    for i in range(1,len(df_now.index)):
        df.at[df.index[i],'ema'] = (df.loc[df.index[i-1],'ema'] * (term-1) + df.loc[df.index[i],'now']*2) / (term+1)
        # αを使う場合。↑と同じ
        # df.at[df.index[i],'ema_'] = df.at[df.index[i-1],'ema'] + a*(df.loc[df.index[i],'now'] - df.at[df.index[i-1],'ema'])

    # print(df)
    return(df['ema'])

def get_RSI(df,term=14):
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

## 当日の終値が出る前にRSIを予測する
## 前日までは15:15、当日分は15:00の値を使用
def get_RSI_pred(df_now,df_old,term=14):

    df_diff = pd.DataFrame()

    df_diff['now'] = df_now
    df_diff['old'] = df_old

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
  ema_terms = [5]
  for term in ema_terms:
      col = 'CloseEMA_' + str(term)
      # df[col] = df['Close_1515'].ewm(span=3, adjust=False).mean()
      # df[col] = calc_ema(df['Close_1515'],df['Close_1515'],term)
      df[col] = calc_ema(df['Close_End1500'],df['Close_End1515'],term)
      df['CloseEMAGrad_'+str(term)] = df[col] / df[col].shift()


  df['Last15'] = df['Close_End1515'] - df['Open_1500']

  print(df.iloc[:,:5])
  df.iloc[:,-10:]
  
  # RSI
  df['RSI_14'] = get_RSI_pred(df[['Close_End1500']],df[['Close_End1515']],14)

  df.iloc[:,-5:]
