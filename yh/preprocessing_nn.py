from glob import glob
import pandas as pd
import numpy as np
import os


class preprocessing_data:
    def __init__(self, dir, istrain=True):
        self.train_data = []
        self.test_data = None
        self.ncolumn = 39
        self.tr_dir = dir

        self.remove_col = ['datadate', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '일자별_도매가격_최대', '일자별_도매가격_최소',
                           '일자별_소매가격_최대', '일자별_소매가격_최소', '주산지_0_초기온도', '주산지_0_평균온도',
                           '주산지_1_초기온도', '주산지_1_평균온도', '주산지_2_초기온도', '주산지_2_평균온도']  # 학습에서 제외할 변수
        self.scale_col = ['단가', '거래량', '거래대금', '해당일자_전체거래물량', '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량',
                          '상위가격 거래물량', '일자별_도매가격_평균', '일자별_소매가격_평균', '수출중량', '수출금액', '수입중량',
                          '수입금액', '무역수지', '주산지_0_강수량', '주산지_1_강수량', '주산지_2_강수량']  # 정규화할 변수

        self.scale_bias = np.zeros((37, self.ncolumn))  # 정규화 상수
        self.scale_mult = np.zeros((37, self.ncolumn))
        self.scale_temp = np.array([40, 0.01])  # T_norm = 0.01(T + 40)으로 정규화

        self.add_pummok(mdfilter=False)

    def add_pummok(self, save=1, mdfilter=True):
        tr_list = glob(self.tr_dir)
        tr_list.sort()

        for k in tr_list:
            df = pd.read_csv(k)
            npum = int(k.split('_')[-1].split('.')[0])  # 품목 번호

            # 항목 단위 날리기
            for col in df.columns:
                df.rename(columns={col: col.split('(')[0]}, inplace=True)
            df.drop(self.remove_col, axis=1, inplace=True)

            # 공백인데 nan이 아닌 칸을 nan으로 만들기
            df = df.mask(df == ' ').astype(float)
            if mdfilter:
                with open(f'../euijun/data/prices/{npum}.txt') as txtfile:
                    df['해당일자_전체평균가격'] = np.loadtxt(txtfile)  # medianfilter로 아웃라이어 제거 후 nan 채움
            else:
                price = df['해당일자_전체평균가격'].fillna(0).to_numpy()
                for i in range(1, len(price) - 1):
                    if price[i] == 0 and price[i - 1] * price[i + 1] != 0:
                        price[i] = (price[i - 1] + price[i + 1]) / 2
                nz = price.nonzero()[0]
                # add element to a list if nonzero otherwise get its closest nonzero element
                price = [x if x else price[nz[np.argmin(np.abs(i - nz))]] for i, x in enumerate(price)]
                df['해당일자_전체평균가격'] = price  # nan만 채움

            # 거래가 없었던 날 지우기
            dfnew = df.copy()
            dfnew.drop(['수출중량', '수출금액', '수입중량', '수입금액', '무역수지', '해당일자_전체평균가격',
                        '주산지_0_최대온도', '주산지_0_최저온도', '주산지_0_강수량', '주산지_0_습도', '주산지_1_최대온도', '주산지_1_최저온도',
                        '주산지_1_강수량', '주산지_1_습도', '주산지_2_최대온도', '주산지_2_최저온도', '주산지_2_강수량',
                        '주산지_2_습도', '일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월',
                        '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', '월구분_4월', '월구분_5월', '월구분_6월',
                        '월구분_7월', '월구분_8월', '월구분_9월'], axis=1, inplace=True)  # 거래 없는 날에도 채워져 있는 항목들
            dfnew = dfnew.isna().all(axis=1)
            df.drop([i for i in range(len(df)) if dfnew[i]], axis=0, inplace=True)
            df.index = list(range(len(df)))

            # 온도 평균내서 합치기 & 정규화
            highlow = ['최대', '최저']
            for txt in highlow:
                dfnew = df[[f'주산지_{i}_{txt}온도' for i in range(3)]]
                dfnew = dfnew.mean(axis=1, skipna=True)  # 세 주산지의 온도를 평균냄
                df[f'주산지_0_{txt}온도'] = (dfnew + self.scale_temp[0]) * self.scale_temp[1]  # T_norm = 0.01(T + 40)으로 정규화
            df.drop(['주산지_1_최대온도', '주산지_1_최저온도', '주산지_2_최대온도', '주산지_2_최저온도'], axis=1, inplace=True)

            # 습도 정규화
            for i in range(3):
                df[f'주산지_{i}_습도'] = df[f'주산지_{i}_습도'] / 100  # 습도 100%는 1, 0%는 0

            # 나머지 변수 정규화
            for i, col in enumerate(self.scale_col):  # 최솟값을 0, 최댓값을 1로 scaling
                self.scale_bias[npum][i] = df[col].min()
                self.scale_mult[npum][i] = df[col].max() - self.scale_bias[npum][i]
                df[col] = (df[col] - self.scale_bias[npum][i]) / self.scale_mult[npum][i]

            # 여기까지 '해당일자_전체평균가격' 제외하고 전부 normalize함. 가격은 geninput에서 날짜별로 변동률을 구해서 normalize함.
            # 데이터 저장
            self.train_data.append(df)
            if save:
                if not os.path.exists('../data/train_nn/'):
                    os.makedirs('../data/train_nn')
                df.to_csv(f'../data/train_nn/train_{npum}.csv')

    def add_pummok_test(self, sep, npum, save=0, mdfilter=False):
        ts_dir = f'../data/test/set_{sep}/test_{npum}.csv'
        df = pd.read_csv(ts_dir)

        # 항목 단위 날리기
        for col in df.columns:
            df.rename(columns={col: col.split('(')[0]}, inplace=True)
        df.drop(self.remove_col, axis=1, inplace=True)

        df = df.mask(df == ' ').astype(float)

        # testset에는 월 구분이 없는 게 있어서 추가하는 코드. 항목 순서가 일치해야 해서 제거하고 추가해야 한다
        dfnew = pd.DataFrame()
        add_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월',
                   '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', '월구분_4월', '월구분_5월', '월구분_6월',
                   '월구분_7월', '월구분_8월', '월구분_9월']
        for col in add_col:
            if col in df.columns:
                dfnew[col] = df[col]
                df.drop(col, axis=1, inplace=True)
            else:
                dfnew[col] = [0] * len(df)
        df = df.join(dfnew)

        # 거래가 없었던 날 지우기
        dfnew = df.copy()
        dfnew.drop(['수출중량', '수출금액', '수입중량', '수입금액', '무역수지', '해당일자_전체평균가격',
                    '주산지_0_최대온도', '주산지_0_최저온도', '주산지_0_강수량', '주산지_0_습도', '주산지_1_최대온도', '주산지_1_최저온도',
                    '주산지_1_강수량', '주산지_1_습도', '주산지_2_최대온도', '주산지_2_최저온도', '주산지_2_강수량',
                    '주산지_2_습도', '일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월',
                    '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', '월구분_4월', '월구분_5월', '월구분_6월',
                    '월구분_7월', '월구분_8월', '월구분_9월'], axis=1, inplace=True)  # 거래 없는 날에도 채워져 있는 항목들
        dfnew = dfnew.isna().all(axis=1)
        df.drop([i for i in range(len(df)) if dfnew[i]], axis=0, inplace=True)
        df.index = list(range(len(df)))

        # 온도 평균내서 합치기 & 정규화
        highlow = ['최대', '최저']
        for txt in highlow:
            dfnew = df[[f'주산지_{i}_{txt}온도' for i in range(3)]]
            dfnew = dfnew.mean(axis=1, skipna=True)  # 세 주산지의 온도를 평균냄
            df[f'주산지_0_{txt}온도'] = (dfnew + self.scale_temp[0]) * self.scale_temp[1]  # T_norm = 0.01(T + 40)으로 정규화
        df.drop(['주산지_1_최대온도', '주산지_1_최저온도', '주산지_2_최대온도', '주산지_2_최저온도'], axis=1, inplace=True)

        # 습도 정규화
        for i in range(3):
            df[f'주산지_{i}_습도'] = df[f'주산지_{i}_습도'] / 100  # 습도 100%는 1, 0%는 0

        # 나머지 변수 정규화 - train에서 구한 scaling 상수를 사용
        for i, col in enumerate(self.scale_col):
            df[col] = (df[col] - self.scale_bias[npum][i]) / self.scale_mult[npum][i]
        lastprice = df.loc[len(df)-1, '해당일자_전체평균가격']  # testset에서는 가격도 normalize해 준다
        df['해당일자_전체평균가격'] = 1 - df['해당일자_전체평균가격'] / lastprice

        # row 개수를 12로 고정
        if len(df) < 12:
            addrow = 12 - len(df)
            dfdummy = pd.DataFrame(np.zeros((addrow, self.ncolumn)), columns=df.columns)
            df = pd.concat([dfdummy, df])

        # 데이터 저장
        self.test_data = df.to_numpy().reshape((-1, 12, 39, 1))
        if save:
            if not os.path.exists('../data/test_nn/'):
                os.makedirs('../data/test_nn')
            df.to_csv(f'../data/test_nn/test_{sep}_{npum}.csv')

    def geninput(self, npum):
        sz = len(self.train_data[npum]) - 40
        x_data = np.zeros((sz, 12, 39))
        y_data = np.zeros((sz, 28))
        for i in range(sz):
            lastprice = self.train_data[npum].loc[i + 11, '해당일자_전체평균가격']
            xdt = self.train_data[npum][i:i + 12].copy()
            xdt['해당일자_전체평균가격'] = 1 - xdt['해당일자_전체평균가격'] / lastprice
            x_data[i] = xdt
            y_data[i] = self.train_data[npum].loc[i + 12:i + 39, '해당일자_전체평균가격'] / lastprice - 1
        return x_data, y_data
