from preprocessing import preprocessing_data
from utility import time_window, normalize_xy
from glob import glob

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self):
        ## 훈련 데이터 전처리 및 저장 (중간저장 X, 최종저장 X) - train
        self.data = preprocessing_data('./aT_train_raw/*.csv')
        self.data.add_pummock()
        self.data.add_dosomae()
        self.data.add_dosomae(option=2)
        self.data.add_imexport()
        self.data.add_weather()
        self.data.add_categorical('train', data_type="train" ,check=0)

        ## 검증 데이터셋 전처리 및 저장 (중간저장 X, 최종저장 X) - test
        for i in range(10): #원래 10임
            self.data = preprocessing_data(f'./aT_test_raw/sep_{i}/*.csv')
            self.data.add_pummock()
            self.data.add_dosomae()
            self.data.add_dosomae(option=2)
            self.data.add_imexport()
            self.data.add_weather()
            self.data.add_categorical(f'set_{i}', data_type="test", check=0)


class TrainDataset():
    def __init__(self, tr_del_list, i):
        self.df_number = i.split("_")[-1].split(".")[0] #농산물 번호 (0~36 중 하나)
        df = pd.read_csv(i)

        for j in df.columns:
            df[j] = df[j].replace({' ': np.nan})

        # 사용할 열 선택 및 index 설정
        df.drop(tr_del_list, axis=1, inplace=True)
        df.set_index('datadate', drop=True, inplace=True)

        # nan 처리
        df = df.fillna(0)

        # 변수와 타겟 분리
        # x = df[[i for i in df.columns if i != '해당일자_전체평균가격(원)']]
        x = df[[i for i in df.columns]]
        y = df['해당일자_전체평균가격(원)']

        # 2주 입력을 통한 이후 4주 예측을 위해 y의 첫 14일을 제외
        y = y[14:]

        # time series window 생성
        data_x = time_window(x, 13, 1)
        data_y = time_window(y, 27, 1)

        # y의 길이와 같은 길이로 설정
        self.xdata = data_x[:len(data_y)]
        self.ydata = data_y

        mean = self.ydata[np.nonzero(self.ydata)].mean()
        self.ydata[self.ydata == 0] = mean 

        idx = df.columns.get_loc('해당일자_전체평균가격(원)')

        self.xdata, self.ydata = normalize_xy(xdata=self.xdata, ydata=self.ydata, idx=idx)

        # self.xdata = self.xdata[300:300]
        # self.ydata = self.ydata[300:300]

