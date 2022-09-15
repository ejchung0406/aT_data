import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_nn import preprocessing_data
from utility_submit import ratelistToCsv
import os


data = preprocessing_data('../data/train/*.csv')

test_except = [(9, 7)]  # testdata가 없는 거
input_shape = (12, 39, 1)
batch_size = 8
epochs = 5

ansSheet = []

for pummok in range(37):

    # model = tf.keras.models.Sequential([  # CNN 모델
    #     tf.keras.layers.Conv2D(64, (1, 39), activation='relu', input_shape=input_shape),
    #     tf.keras.layers.Reshape((12, 64, 1)),
    #     tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    #     tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    #     tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    #     tf.keras.layers.Flatten(),
    #     # tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     # tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(28)
    # ])

    model = tf.keras.models.Sequential([  # MLP
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(28)
    ])

    model.compile(optimizer='adam', loss='mean_absolute_error')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'./cnn/cnn_train_{pummok}.ckpt',
                                                     save_weights_only=True, verbose=1)

    x_train, y_train = data.geninput(pummok)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_split=0.1, callbacks=[cp_callback], shuffle=True)

    anspum = []
    for sep in range(10):
        if (sep, pummok) in test_except:
            print(f'WWWW({sep}, {pummok}) skipped')
            anspum += [0] * 15
            continue
        data.add_pummok_test(sep, pummok)
        x_test = data.test_data
        y_test = model(x_test).numpy().reshape(-1)
        avg2228 = np.mean(y_test[21:])
        y_test = np.append(y_test[:14], avg2228)
        anspum += y_test.tolist()
    ansSheet.append(anspum)

# print(ansSheet)

# 자동으로 존재하지 않는 파일명 생성해서 덮어씌우기 방지
dirnum = 0
if not os.path.exists('./answers'):
    os.mkdir('./answers')
while os.path.exists(f'./answers/answer_{dirnum}.csv'):
    dirnum += 1
ansDir = f'./answers/answer_{dirnum}.csv'

# 정답 입력
ratelistToCsv(ansSheet, csvDir=ansDir)
