from arch import arch_model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

index_smboard = pd.read_excel('./板块数据/RESSET_DRET_SMBOARD_1_中小板指数.xls', sheet_name='DRET_SMBOARD', header=0)
index_smboard = index_smboard[['日期_Date', '中小板日收益率_等权_Dreteq']]
index_smboard.set_index('日期_Date', inplace=True)
index_smboard = index_smboard['2018-01-01':'2020-12-31']

index_GEM = pd.read_excel('./板块数据/RESSET_DRET_GEM_1.xls', sheet_name='DRET_GEM', header=0)
index_GEM = index_GEM[['日期_Date', '创业板日收益率_流通市值加权_Drettmv']]
index_GEM.set_index('日期_Date', inplace=True)
index_GEM = index_GEM['2018-01-01':'2020-12-31']

index = index_smboard.join(index_GEM)

MS_index = index.iloc[:, 0]

model_arch = arch_model(y=MS_index, mean='Constant', lags=0, vol='ARCH', p=1, o=0, q=0, dist='normal')
result_arch = model_arch.fit()

model_garch = arch_model(y=MS_index, mean='Constant', lags=0, vol='GARCH', p=1, o=0, q=1, dist='t')
result_garch = model_garch.fit()

vol = np.sqrt(result_garch.params[1] / (1 - result_garch.params[2] - result_garch.params[3]))
print(vol)

result_arch.plot()

result_garch.plot()
print(result_garch.summary())

plt.show()
