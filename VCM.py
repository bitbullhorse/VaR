import numpy as np
import scipy.stats as st
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import os


def VaR_VCM(Value, Rp, Vp, X, N):
    """
    定义一个运用方差-协方差法计算VaR的函数
    :param value: 代表价值或市值
    :param Rp: 代表日平均收益率
    :param Vp: 日波动率
    :param X: 置信水平
    :param N: 持有期
    :return:
    """
    z = abs(st.norm.ppf(q=1 - X))
    VaR_1day= Value * (z * Vp - Rp)
    VaR_Nday = VaR_1day * sqrt(N)
    return VaR_Nday

if __name__ == '__main__':
    sheet_name = 'DRESSTK_2021_'

    file_path = './股票数据/'
    file_names = os.listdir(file_path)
    plt.figure(figsize=(15, 10))
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
    R = R.dropna()
    R_mean = R.mean()  # 计算日收益率的均值
    R_vol = R.std() # 计算日收益率的波动率
    R_cov = R.cov() # 计算日收益率的协方差
    R_corr = R.corr() # 计算日收益率的相关系数

    W = np.array([0.15, 0.2, 0.5, 0.05, 0.1]) # 假设各资产配置权重
    Rp_daily = np.sum(W*R_mean) # 计算总资产日平均收益率
    Vp_daily = np.sqrt(np.dot(W, np.dot(R_cov, W.T)))
    value_port = 1e9
    D1 = 1
    D2 = 10
    X1 = 0.95
    X2 = 0.99

    VaR95_1day_VCM = VaR_VCM(Value=value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D1)
    VaR99_1day_VCM = VaR_VCM(Value=value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D1)
    print(VaR95_1day_VCM, VaR99_1day_VCM)
    VaR95_10day_VCM = VaR_VCM(Value=value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D2)
    VaR99_10day_VCM = VaR_VCM(Value=value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D2)
    print(VaR95_10day_VCM, VaR99_10day_VCM)

    value_past = value_port * W

    profit_past = np.dot(R, value_past)
    profit_past = pd.DataFrame(data=profit_past, index=R.INDEX, columns=['投资组合的模拟日收益'])
    profit_2021 = profit_past.loc['2021-01-01':'2021-12-31']
    profit_2022 = profit_past.loc['2022-01-01':'2022-12-31']
    profit_2023 = profit_past.loc['2023-01-01':'2023-12-31']

    tmp = [profit_2021, profit_2022, profit_2023]

    plt.figure(figsize=(9, 12))
    for count, profit in enumerate(tmp, start=1):
        VaR = -VaR95_1day_VCM * np.ones_like(profit)
        VaR = pd.DataFrame(data=VaR, index=profit.INDEX)
        plt.subplot(3, 1, count)
        plt.plot(profit, 'b-', label=f'{str(profit.INDEX[0])[:4]}年投资组合收益')
        plt.plot(VaR, 'r-', label=u'风险价值亏损', lw=2.0)
        plt.ylabel(u'收益')
        plt.legend(fontsize=12)
        plt.grid()
        days = len(profit)
        print(f'{str(profit.INDEX[0])[:4]}年交易天数', days)
        dayexcept = len(profit[profit['投资组合的模拟日收益'] < -VaR95_1day_VCM])
        print(f'{str(profit.INDEX[0])[:4]}年超过风险价值对应亏损天数', dayexcept)
        print(f'{str(profit.INDEX[0])[:4]}年比例', dayexcept / days)


    plt.show()

    price_stress = df
    R_stress = np.log(price_stress / price_stress.shift(1))
    R_stress = R_stress.dropna()

    profit_stress = np.dot(R_stress, value_past)
    profit_stress = pd.DataFrame(data=profit_stress, index=R_stress.INDEX, columns=['投资组合的模拟日收益'])

    profit_zero = np.zeros_like(profit_stress)
    profit_zero = pd.DataFrame(data=profit_zero, index=profit_stress.index)

    plt.figure(figsize=(9, 12))
    plt.plot(profit_stress, 'b-', label=u'压力期间投资组合的日收益')
    plt.plot(profit_zero, 'r-', label=u'收益为0', lw=2.5)
    plt.xlabel(u'日期', fontsize=13)
    plt.xticks(fontsize=13)
    plt.ylabel(u'收益',fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(u'压力期间投资组合收益情况', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    SVaR95_1day = np.abs(np.percentile(a=profit_stress, q=(1 - X1) * 100))
    SVaR99_1day = np.abs(np.percentile(a=profit_stress, q=(1 - X2) * 100))

    print(SVaR95_1day, SVaR99_1day)

    SVaR95_10day = np.sqrt(D2) * SVaR95_1day
    SVaR99_10day = np.sqrt(D2) * SVaR99_1day

    print(SVaR95_10day, SVaR99_10day)


