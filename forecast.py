import numpy as np
import scipy.stats as st
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import os
from arch import arch_model
from model import *
from torch.utils.data import DataLoader as Dataloader
from train_func import train_model_cp

torch.autograd.set_detect_anomaly(True)

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
               '总市值加权平均日资本收益_Daretmc', '日无风险收益率_DRfRet', '市盈率_PE']

criterion_CrossEntropy = nn.CrossEntropyLoss()
criterion_NLLLoss = nn.NLLLoss()
criterion_PoissonNLLLoss = nn.PoissonNLLLoss()
criterion_L1Loss = nn.L1Loss()
criterion_MSELoss = nn.MSELoss()

price = price[index]
price = price.set_index('日期_Date')
train_price = price['2021-01-01':'2023-12-31']
eval_price = price['2024-01-01':'2024-06-15']
dmodel = len(index) - 1
seqlen=12
batch_size = 12
model_trans = Transformer_model_Cp(dmodel, 4, seq_len=seqlen, nlayers=4)
train_data = CustomDataset(train_price, seqlen)
dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

eval_data = CustomDataset(eval_price, seqlen)
dataloader_eval = DataLoader(eval_data, batch_size=1, shuffle=False)

optimizer = optim.Adam(model_trans.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
train_model_cp(model_trans, 100, dataloader_train, criterion_MSELoss, optimizer, scheduler, dataloader_eval)

df = df['2021-01-01':'2021-12-31']

df = np.log(df / df.shift(1)) # 计算日收益率
df = df.dropna()

df['收盘价_Clpr_000001'] = df['收盘价_Clpr_000001'] * 100

returns = df.iloc[:,0]
model_garch = arch_model(y=returns, mean='AR', lags=0, vol='GARCH', p=1, o=0, q=1, dist='t')
result_garch = model_garch.fit()

forecast_len = 120

forecast_garch = result_garch.forecast(horizon=forecast_len, start=len(df) - forecast_len)
predicted_volatility = np.sqrt(forecast_garch.variance.values[-forecast_len:])
window_size = 10
actual_volatility = df['收盘价_Clpr_000001'].rolling(window=window_size).std()

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(df.index[-forecast_len:], actual_volatility[-forecast_len:], label='volatility')
plt.plot(forecast_garch.variance.index[-forecast_len:], predicted_volatility[:,0], label='mean')
plt.legend()
plt.show()
plt.savefig(cwd + 'garch_forcast.png')
