import numpy as np
import scipy.stats as st
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import os
from arch import arch_model

sheet_name = 'DRESSTK_2021_'
cwd = '/home/czj/pycharm_project_tmp_pytorch/VaR/'

file_path = '/home/czj/pycharm_project_tmp_pytorch/VaR/股票数据/'
file_names = os.listdir(file_path)
plt.figure(figsize=(15, 10))
df = pd.DataFrame()
price = pd.DataFrame()

for file_name in file_names:
    # 确认列索引是否在范围内
    price = pd.read_excel(file_path + file_name, sheet_name=sheet_name, header=0)
    new_price = price[['日期_Date', '收盘价_Clpr']]
    new_price.set_index('日期_Date', inplace=True)
    new_price = new_price.dropna()
    new_price.rename(columns={'收盘价_Clpr': '收盘价_Clpr_' + file_name[-10:-4]}, inplace=True)
    if df.empty:
        df = new_price
    else:
        df = df.join(new_price)
    break

index = ['日期_Date', '收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2','成交量_Trdvol',
               '成交金额_Trdsum','日振幅(%)_Dampltd','总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR', '日收益率_Dret', '日资本收益率_Daret',
               '等权平均市场日收益率_Dreteq','流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc', '等权平均市场日资本收益率_Dareteq',
               '总市值加权平均日资本收益_Daretmc']

price_index = ['日期_Date', '收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2','成交量_Trdvol',]

price = price[price_index]
price = price.set_index('日期_Date')
price = price.dropna()

df = price

garch_list = []

for column in df.columns:
    df_column = df[column]
    df_column = np.log(df_column / df_column.shift(1))
    df_column = df_column.dropna() * 100
    model_garch = arch_model(y=df_column, mean='AR', lags=0, vol='GARCH', p=1, o=0, q=1, dist='t')
    result_garch = model_garch.fit()
    garch_list.append(result_garch)

print(garch_list[0].params)
