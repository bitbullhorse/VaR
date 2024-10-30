import numpy as np
import scipy.stats as st
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import os
from arch import arch_model
from model import *
from torch.utils.data import DataLoader as Dataloader
from train_func import train_model_cp, train_iTranformer

from iTransformer_model import iTransformer

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
    print(file_name)
    new_price = price[['日期_Date', '收盘价_Clpr']]
    new_price.set_index('日期_Date', inplace=True)
    new_price = new_price.dropna()
    new_price.rename(columns={'收盘价_Clpr': '收盘价_Clpr_' + file_name[-10:-4]}, inplace=True)
    if df.empty:
        df = new_price
    else:
        df = df.join(new_price)
    break

index = ['日期_Date', '收盘价_Clpr', '开盘价_Oppr', '最高价_Hipr', '最低价_Lopr', '复权价1(元)_AdjClpr1', '复权价2(元)_AdjClpr2','成交量_Trdvol',\
               '成交金额_Trdsum','日振幅(%)_Dampltd','总股数日换手率(%)_DFulTurnR', '流通股日换手率(%)_DTrdTurnR', '日收益率_Dret', '日资本收益率_Daret',\
               '等权平均市场日收益率_Dreteq','流通市值加权平均市场日收益率_Drettmv', '总市值加权平均市场日收益率_Dretmc', '等权平均市场日资本收益率_Dareteq',\
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
dmodel = 128
seqlen=12
predlen = 1
batch_size = 12

model = iTransformer(seqlen, predlen, dmodel, 8, 6, 256)
model.to(device)
model = model.to(torch.float64)

dataset_train = Custom_iTransformer_Dataset(train_price,seqlen,predlen)
dataset_eval = Custom_iTransformer_Dataset(eval_price,seqlen,predlen)

train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
eval_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

train_iTranformer(model, 100, train_dataloader, criterion_MSELoss, optimizer, scheduler, eval_dataloader)

