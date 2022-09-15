from glob import glob
import pandas as pd
import numpy as np

npummok = 37
ncolumn = 39
tr_list = glob('../data/train/*.csv')
tr_list.sort()
ts_list = [glob(f'../data/test/set_{i}/*.csv') for i in range(ncolumn)]

remove_col = ['datadate', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '일자별_도매가격_최대', '일자별_도매가격_최소',
              '일자별_소매가격_최대', '일자별_소매가격_최소', '주산지_0_초기온도', '주산지_0_평균온도',
              '주산지_1_초기온도', '주산지_1_평균온도', '주산지_2_초기온도', '주산지_2_평균온도']
scaling_col = ['단가', '거래량', '거래대금', '해당일자_전체평균가격', '해당일자_전체거래물량',
               '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량', '상위가격 거래물량',
               '일자별_도매가격_평균', '일자별_소매가격_평균', '수출중량', '수출금액', '수입중량',
               '수입금액', '무역수지', '주산지_0_최대온도', '주산지_0_최저온도',
               '주산지_0_강수량', '주산지_0_습도', '주산지_1_강수량', '주산지_1_습도',
               '주산지_2_강수량', '주산지_2_습도']
scale_bias = np.zeros((npummok, ncolumn))  # 정규화할 때 쓸 값
scale_mult = np.zeros((npummok, ncolumn))

for k in tr_list:
    df = pd.read_csv(k)
    numpum = int(k.split('_')[-1].split('.')[0])  # 품목 번호

    # 항목 단위 날리기
    for col in df.columns:
        df.rename(columns={col: col.split('(')[0]}, inplace=True)
    df.drop(remove_col, axis=1, inplace=True)

    # 공백인데 nan이 아닌 칸을 nan으로 만들기
    df = df.mask(df == ' ')

    # 거래가 없었던 날 지우기
    dfnew = df.copy()
    dfnew.drop(['수출중량', '수출금액', '수입중량', '수입금액', '무역수지',
                '주산지_0_최대온도', '주산지_0_최저온도', '주산지_0_강수량',
                '주산지_0_습도', '주산지_1_최대온도', '주산지_1_최저온도',
                '주산지_1_강수량', '주산지_1_습도', '주산지_2_최대온도', '주산지_2_최저온도', '주산지_2_강수량',
                '주산지_2_습도', '일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월',
                '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', '월구분_4월', '월구분_5월', '월구분_6월',
                '월구분_7월', '월구분_8월', '월구분_9월'], axis=1, inplace=True)  # 거래 없는 날에도 채워져 있는 항목들
    dfnew = dfnew.isna().all(axis=1)
    df.drop([i for i in range(len(df)) if dfnew[i]], axis=0, inplace=True)

    # 온도 평균내서 합치기
    highlow = ['최대', '최저']
    for txt in highlow:
        dfnew = df[[f'주산지_{i}_{txt}온도' for i in range(3)]]
        df[f'주산지_0_{txt}온도'] = dfnew.mean(axis=1, skipna=True)
    df.drop(['주산지_1_최대온도', '주산지_1_최저온도', '주산지_2_최대온도', '주산지_2_최저온도'], axis=1, inplace=True)

    # normalize
    for i, col in enumerate(scaling_col):
        df[col] = df[col].astype(float)
        scale_bias[numpum][i] = df[col].min()
        scale_mult[numpum][i] = df[col].max() - scale_bias[numpum][i]
        df[col] = (df[col] - scale_bias[numpum][i]) / scale_mult[numpum][i]

    # 데이터 저장
    x_train = np.zeros((len(df), 14, 39))
    y_train = np.zeros((len(df), 28))
    for i in range(len(df)-42):
        x_train[i] = df.values[i:i+14, :]
        y_train[i] = df['해당일자_전체평균가격'].values[i+14:i+42]
    np.savetxt(f'../data/train_cnn/data_tr_{numpum}.csv', x_train.reshape(-1, 39), delimiter=",")
    np.savetxt(f'../data/train_cnn/label_tr_{numpum}.csv', y_train, delimiter=",")

np.savetxt('../data/train_cnn/scaler_bias.csv', scale_bias, delimiter=",")
np.savetxt('../data/train_cnn/scaler_mult.csv', scale_mult, delimiter=",")