from preprocessing import preprocessing_data
from utility import time_window
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
        for i in range(1): #원래 10임
            self.data = preprocessing_data(f'./aT_test_raw/sep_{i}/*.csv')
            self.data.add_pummock()
            self.data.add_dosomae()
            self.data.add_dosomae(option=2)
            self.data.add_imexport()
            self.data.add_weather()
            self.data.add_categorical(f'set_{i}', data_type="test", check=0)

        ## 데이터 불러오기 및 parameter 설정
        self.data_list = glob('./data/train/*.csv')
        self.tr_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 '] # train 에서 사용하지 않는 열
        self.ts_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '해당일자_전체평균가격(원)'] # test 에서 사용하지 않는 열
        self.check_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순','월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', 
                    '월구분_4월','월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월'] # 열 개수 맞추기

class TrainDataset():
    def __init__(self, dataset, i):
        self.df_number = i.split("_")[-1].split(".")[0] #농산물 번호 (0~36 중 하나)
        df = pd.read_csv(i)

        for j in df.columns:
            df[j] = df[j].replace({' ': np.nan})

        # 사용할 열 선택 및 index 설정
        df.drop(dataset.tr_del_list, axis=1, inplace=True)
        df.set_index('datadate', drop=True, inplace=True)

        # nan 처리
        df = df.fillna(0)

        # 변수와 타겟 분리
        x = df[[i for i in df.columns if i != '해당일자_전체평균가격(원)']]
        y = df['해당일자_전체평균가격(원)']

        # 2주 입력을 통한 이후 4주 예측을 위해 y의 첫 14일을 제외
        y = y[14:]

        # time series window 생성
        data_x = time_window(x, 13, 1)
        data_y = time_window(y, 27, 1)

        # y의 길이와 같은 길이로 설정
        self.xdata = data_x[:len(data_y)]
        self.ydata = data_y
