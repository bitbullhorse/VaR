import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
from matplotlib.ticker import ScalarFormatter

I = 100000
n = 8
epsilon = npr.standard_t(df=n, size=I)

sheet_name = 'DRESSTK_2021_'

file_path = './股票数据/'
file_names = os.listdir(file_path)
df = pd.DataFrame()
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

R = np.log(df / df.shift(1)) # 计算日收益率

R_mean = R.mean() * 252
R_vol = R.std() * np.sqrt(252)
dt = 1 / 252

value_port = 1e9
D1 = 1
D2 = 10
X1 = 0.95
X2 = 0.99

W = np.array([0.15, 0.2, 0.5, 0.05, 0.1])  # 假设各资产配置权重

profit_port = 0

for i in range(5):
    P = df.iloc[-1, i]
    P_new = P * np.exp((R_mean[i] - 0.5 * R_vol[i] ** 2) * dt + R_vol[i] * epsilon * np.sqrt(dt))
    profit = (P_new / P - 1) * value_port*W[i]
    profit_port += profit


plt.figure(figsize=(9, 6))
plt.hist(profit_port, bins=50, facecolor='y', edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'Portfolio Value', fontsize=13)
plt.xticks(rotation=45)  # 旋转45度
plt.yticks(fontsize=13)
plt.ylabel(u'频数', fontsize=13)
plt.title(u'直方图', fontsize=13)
plt.grid()
plt.show()

VaR95_1day_MCst = np.abs(np.percentile(a=profit_port, q=(1 - X1) * 100))
VaR99_1day_MCst = np.abs(np.percentile(a=profit_port, q=(1 - X2) * 100))

print(VaR95_1day_MCst, VaR99_1day_MCst)

VaR95_10day_MCst = np.sqrt(D2) * VaR95_1day_MCst
VaR99_10day_MCst = np.sqrt(D2) * VaR99_1day_MCst

print(VaR95_10day_MCst, VaR99_10day_MCst)

P = np.array(df.iloc[-1])

epsilon_normal = npr.standard_normal(I)

P_new = np.zeros(shape=(I, len(R_mean)))

for i in range(len(R_mean)):
    P_new[:,i]=P[i] * np.exp((R_mean[i] - 0.5 * R_vol[i] ** 2) * dt + R_vol[i] * epsilon_normal * np.sqrt(dt))

profit_port_norm = (np.dot(P_new / P - 1, W)) * value_port

plt.figure(figsize=(12, 6))
plt.hist(profit_port_norm, bins=30, facecolor='y', edgecolor='k')
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xticks(fontsize=13)
plt.xlabel('Portfolio', fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('Probability', fontsize=13)
plt.grid()
plt.show()

VaR95_1day_MCnorm = np.abs(np.percentile(a=profit_port_norm, q=(1 - X1) * 100))
VaR99_1day_MCnorm = np.abs(np.percentile(a=profit_port_norm, q=(1 - X2) * 100))

print(VaR95_1day_MCnorm, VaR99_1day_MCnorm)

VaR95_10day_MCnorm = np.sqrt(D2) * VaR95_1day_MCnorm
VaR99_10day_MCnorm = np.sqrt(D2) * VaR99_1day_MCnorm

print(VaR95_10day_MCnorm, VaR99_10day_MCnorm)
