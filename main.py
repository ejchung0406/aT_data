import warnings
import os, random, glob, tqdm
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import preprocessing_data
from utility import astype_data, time_window
from trainer import train

# 경고 끄기
warnings.filterwarnings(action='ignore')

# 시드고정
tf.random.set_seed(19970119)
random.seed(19970119)
np.random.seed(19970119)

## 훈련 데이터 전처리 및 저장 (중간저장 X, 최종저장 X) - train
data = preprocessing_data('./aT_train_raw/*.csv')
data.add_pummock()
data.add_dosomae()
data.add_dosomae(option=2)
data.add_imexport()
data.add_weather()
data.add_categorical('train', data_type="train" ,check=0)

## 검증 데이터셋 전처리 및 저장 (중간저장 X, 최종저장 X) - test
for i in range(10):
    data = preprocessing_data(f'./aT_test_raw/sep_{i}/*.csv')
    data.add_pummock()
    data.add_dosomae()
    data.add_dosomae(option=2)
    data.add_imexport()
    data.add_weather()
    data.add_categorical(f'set_{i}', data_type="test", check=0)

## 데이터 불러오기 및 parameter 설정
data_list = glob('./data/train/*.csv')
epoch = 1000
batch = 15
tr_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 '] # train 에서 사용하지 않는 열
ts_del_list = ['단가(원)', '거래량', '거래대금(원)', '경매건수', '도매시장코드', '도매법인코드', '산지코드 ', '해당일자_전체평균가격(원)'] # test 에서 사용하지 않는 열
check_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순','월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', 
             '월구분_4월','월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월'] # 열 개수 맞추기

## Train 과정
for i in tqdm(data_list):
    df_number = i.split("_")[-1].split(".")[0] #농산물 번호 (0~36 중 하나)
    df = pd.read_csv(i)

    for j in df.columns:
        df[j] = df[j].replace({' ': np.nan})

    # 사용할 열 선택 및 index 설정
    df.drop(tr_del_list, axis=1, inplace=True)
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
    xdata = data_x[:len(data_y)]
    ydata = data_y

    # train, validation 분리 (8 : 2)
    x_train, x_val, y_train, y_val = train_test_split(xdata, ydata, test_size=0.2, shuffle=False, random_state=119)

    # transformer 모델 훈련
    transformer = train(astype_data(x_train), y_train, astype_data(x_val), y_val, f'transformer-{df_number}', epoch,
                        batch)
    transformer.load_weights(f'./check/transformer-{df_number}-{epoch}-{batch}.h5')

    if os.path.exists(f'./model') == False:
        os.mkdir(f'./model')

    # 모델 저장
    transformer.save(f'./model/transformer-{df_number}-{epoch}-{batch}.h5')

# ## Test 과정
# zero_csv = [0 for i in range(14)]  # 시점이 비어있는 데이터 0으로 채우기 위한 변수

# for i in tqdm(range(10)):
#     data_list = glob(f'./data/test/set_{i}/*.csv')

#     for idx,j in enumerate(data_list):
#         df = pd.read_csv(j)

#         if len(df) == 0:
#             df['zero_non'] = zero_csv
#             df = df.fillna(0)
#             df.drop('zero_non', axis=1, inplace=True)


#         file_number = j.split('test_')[1].split('.')[0]

#         # 사용할 열 선택, index 설정
#         df.drop(ts_del_list, axis=1, inplace=True)
#         df.set_index('datadate', drop=True, inplace=True)

#         # train input 과 형상 맞추기
#         add_col = [i for i in check_col if i not in df.columns]

#         for a in add_col:
#             df[a] = 0

#         # ' ' -> nan 으로 변경
#         for a in df.columns:
#             df[a] = df[a].replace({' ': np.nan})

#         # nan 처리
#         df = df.fillna(0)

#         # x_test  생성
#         df_test = astype_data(df.values.reshape(1, df.values.shape[0], df.values.shape[1]))


#         # model test
#         if os.path.exists('./model_output') == False:
#             os.mkdir('./model_output')

#         if os.path.exists(f'./model_output/set_{i}') == False:
#             os.mkdir(f'./model_output/set_{i}')

#         # 해당하는 모델 불러오기
#         model_test = tf.keras.models.load_model(f'./model/transformer-{file_number}-{epoch}-{batch}.h5')
#         pred = transformer.predict(df_test)


#         # 결과 저장
#         save_df = pd.DataFrame(pred).T
#         save_df.to_csv(f'./model_output/set_{i}/predict_{file_number}.csv', index=False)

# ## 정답 제출 파일생성
# for k in tqdm(range(10)):

#   globals()[f'set_df_{k}'] = pd.DataFrame()
#   answer_df_list = glob(f'./model_output/set_{k}/*.csv') # 예측한 결과 불러오기
#   pum_list = glob(f'./aT_test_raw/sep_{k}/*.csv') # 기존 test input 불러오기
#   pummok = [a for a in pum_list if 'pummok' in a.split('/')[-1]]

#   for i in answer_df_list:
#     df = pd.read_csv(i)
#     number = i.split('_')[-1].split('.')[0]

#     base_number = 0
#     for p in pummok:
#       if number == p.split('_')[-1].split('.')[0]:
#         pum_df = pd.read_csv(p)

#         if len(pum_df) != 0:
#            base_number = pum_df.iloc[len(pum_df)-1]['해당일자_전체평균가격(원)']  # 기존 각 sep 마다 test input의 마지막 target 값 가져오기 (변동률 계산을 위해)
#         else:
#           base_number = np.nan

#     globals()[f'set_df_{k}'][f'품목{number}']  = [base_number] + list(df[df.columns[-1]].values) # 각 품목당 순서를 t, t+1 ... t+28 로 변경

#   globals()[f'set_df_{k}'] = globals()[f'set_df_{k}'][[f'품목{col}' for col in range(37)]] # 열 순서를 품목0 ~ 품목36 으로 변경

# """- 변동률 계산을 위한 t, t+1 ... t+28 설정"""

# # set_df_0

# """- 변동률 계산 """

# date = [f'd+{i}' for i in range(1,15)] + ['d+22 ~ 28 평균']


# for k in range(10):
#   globals()[f'answer_df_{k}'] = pd.DataFrame()
#   for c in globals()[f'set_df_{k}'].columns:
#     base_d = globals()[f'set_df_{k}'][c][0] # 변동률 기준 t 값

#     ans_1_14 = []
#     for i in range(14):
#       ans_1_14.append((globals()[f'set_df_{k}'][c].iloc[i+1]- base_d)/base_d)  # t+1 ~ t+14 까지는 (t+n - t)/t 로 계산

#     ans_22_28 = (globals()[f'set_df_{k}'][c][22:29].mean() - base_d)/base_d # t+22 ~ t+28은 np.mean(t+22 ~ t+28) - t / t

#     globals()[f'answer_df_{k}'][f'{c} 변동률'] = ans_1_14 + [ans_22_28]
  
#   globals()[f'answer_df_{k}']['Set'] = k # set 번호 설정
#   globals()[f'answer_df_{k}']['일자'] = date # 일자 설정

# """- sep 0  ~ sep 9 까지 합치기"""

# # 위에서 계산된 변동률 들을 합쳐주는 과정

# all_df =pd.DataFrame()
# for i in range(10):
#   if i== 0 :
#     all_df = pd.concat([all_df, globals()[f'answer_df_{i}']],axis=1)
#   else:
#     all_df = pd.concat([all_df, globals()[f'answer_df_{i}']])


# all_df = all_df[['Set','일자'] + list(all_df.columns[:-2])]
# all_df.reset_index(drop=True, inplace=True)

# """- 정답 양식으로 변경"""

# # set, 일자 기억하기위해 따로 저장

# re_set = list(all_df['Set'])
# re_date = list(all_df['일자'])


# # 정답 양식 불러오기
# out_ans = pd.read_csv('./answer_example.csv')

# # 두 dataframe 합치기 (nan + 숫자 = nan 이용)
# submit_df = all_df + out_ans

# submit_df['Set'] = re_set
# submit_df['일자'] = re_date


# # 최종 저장
# submit_df.to_csv('./submit.csv',index=False)

# """- 계산된 변동률 결과물"""

# all_df

# """- 제출 양식"""

# out_ans

# """- 제출 양식 반영한 최종 결과물 (**실 제출용**)"""

# submit_df