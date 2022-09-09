from preprocessing import preprocessing_data
from constrained_linear_regression import ConstrainedLinearRegression
from utility import chaboon, yeokchaboon
from scipy.stats import pearsonr   
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model
from scipy import ndimage


if __name__ == "__main__":
    data = preprocessing_data(dir='../aT_train_raw/pummok_*.csv', 
                              dir_outlier='./outliers')

    data.chaboon()
    data.moving_avg(30)
    data.chaboon(mvavg=1)

    corr = []
    corr_diff = []
    corr_mvavg = []
    corr_mvavg_diff = []

    for i in range(len(data)):
        splits = np.split(data.prices[i], [365, 365*2, 365*3, 365*4])
        splits.pop(-1)
        splits_diff = np.split(data.prices_diff[i], [365, 365*2, 365*3])
        splits_mvavg = np.split(data.prices_mvavg[i], [365, 365*2, 365*3, 365*4])
        splits_mvavg.pop(-1)
        splits_mvavg_diff = np.split(data.prices_mvavg_diff[i], [365, 365*2, 365*3])
        
        # 변수이름 거지같이 만듦
        corrs = []
        corrs_diff = []
        corrs_mvavg = []
        corrs_mvavg_diff = []

        for j in range(len(splits)):
            for k in range(j+1, len(splits)):
                corrs.append(pearsonr(splits[j], splits[k]).statistic)
                corrs_diff.append(pearsonr(splits_diff[j], splits_diff[k]).statistic)
                corrs_mvavg.append(pearsonr(splits_mvavg[j], splits_mvavg[k]).statistic)
                corrs_mvavg_diff.append(pearsonr(splits_mvavg_diff[j], splits_mvavg_diff[k]).statistic)

        # print(i, min(corrs), min(corrs_diff), min(corrs_mvavg), min(corrs_mvavg_diff))
        # 일단은 최소로 했는데, 재형이가 말했듯이 끝에서 두번째로 해도 될 듯? 아닌가? ㅁㄹ
        corr.append(min(corrs))
        corr_diff.append(min(corrs_diff))
        corr_mvavg.append(min(corrs_mvavg))
        corr_mvavg_diff.append(min(corrs_mvavg_diff))

    whatthefuck = list(enumerate(corr_mvavg))
    whatthefuck_diff = list(enumerate(corr_mvavg_diff))

    whatthefuck.sort(key=lambda x:x[1])
    whatthefuck_diff.sort(key=lambda x:x[1])

    # correlation이 높은 상위 5개의 품목은?! (각각 moving average, moving average의 차분으로 계산)
    # print(whatthefuck[-5:])
    # print(whatthefuck_diff[-5:])

    # 11번이 두 분야 모두에서 1등을 차지했네요. 

    # print(list(whatthefuck[-20:]))
    # print(whatthefuck_diff)

    # plt.plot(data.prices[0])
    # plt.plot(data.prices_mvavg_diff[11])
    # plt.show()

    # no_more_random = [31, 17, 23, 10, 15, 14, 20, 19, 27, 35, 9, 26, 0, 25, 34, 8, 16, 6, 18, 11] # whatthefuck_diff 상위 15개. 일단 귀찮아서 하드코딩함. 
    # no_more_random = [31, 17, 23, 10, 15, 14, 20, 19, 27, 35, 9, 26, 0, 25, 34, 8, 16, 6, 18, 11]
    no_more_random = list(range(37))

    for sep in range(10):
        for i in no_more_random:
            price = data.prices_test[sep][i]
            firstdate = int(data.date_test[sep][i])
            
            # startprice, price = chaboon(price)

            splits_mvavg = np.split(data.prices_mvavg[i], [365, 365*2, 365*3, 365*4])
            splits_mvavg.pop(-1)
            # 년도별로 노말라이즈를 하고 평균내야 할듯
            mvavg = np.sum(splits_mvavg, axis=0)/len(splits_mvavg)
            # 지금은 그냥 마지막해로 함 
            # mvavg = splits_mvavg[-1]

            # x = data.prices[i][firstdate:firstdate+14]
            x = mvavg[firstdate:firstdate+14]
            y = price
            y = ndimage.median_filter(price, size=5)

            if len(mvavg[firstdate:])<42:
                rate_future_avg = np.random.normal(0, 0.001, 28)
                price_future_avg = (1+rate_future_avg)*price[-1]
                plt.plot(np.arange(firstdate, firstdate+14), y)
                plt.plot(np.arange(firstdate+14, firstdate+42), price_future_avg)
                plt.title(f"pummok: {i}, test set: {sep}, random: yes")
                # plt.show()
                continue

            else: 
                
                # regr = linear_model.LinearRegression()
                regr = ConstrainedLinearRegression()
                regr.fit(x.reshape(-1, 1), y.reshape(-1, 1), max_coef=[1.2], min_coef=[0.4])
                price_future = regr.coef_[0]*data.prices[i][firstdate+14:firstdate+42]+regr.intercept_
                price_future_avg = regr.coef_[0]*mvavg[firstdate+14:firstdate+42]+regr.intercept_

                if ((price_future_avg[0] - y[-1]) * (price_future_avg[0] - mvavg[firstdate+13]))<0:
                    coeffi = 0.4
                else:
                    # print(price_future_avg[0], mvavg[firstdate+13], y[-1], "coeffi high")
                    coeffi = 0.1

                price_future_avg = price_future_avg - (price_future_avg[0] - price[-1])*coeffi
                
                plt.plot(np.arange(firstdate, firstdate+42), mvavg[firstdate:firstdate+42])
                plt.plot(np.arange(firstdate, firstdate+14), y)
                plt.plot(np.arange(firstdate+14, firstdate+42), price_future)
                plt.plot(np.arange(firstdate+14, firstdate+42), price_future_avg)
                plt.title(f"pummok: {i}, test set: {sep}, random: no")
                # if sep==0:
                #     plt.show()

                rate_future = (price_future - price[-1])/price[-1]
                rate_future_avg = (price_future_avg - price[-1])/price[-1]

                # print(rate_future_avg)

            os.makedirs(f"../model_output/set_{sep}", exist_ok = True)
            with open(f"../model_output/set_{sep}/predict_{i}.csv", "w") as answerfile:
                np.savetxt(answerfile, rate_future_avg)

        for i in range(37):
            if i not in no_more_random:
                random = np.random.normal(0, 0.001, 28)
                os.makedirs(f"../model_output/set_{sep}", exist_ok = True)
                with open(f"../model_output/set_{sep}/predict_{i}.csv", "w") as answerfile:
                    np.savetxt(answerfile, random)
