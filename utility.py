from tqdm import tqdm

import re
import numpy as np
import tensorflow as tf

## 입력 shape 및 형태 정의 함수
def make_Tensor(array):
    return tf.convert_to_tensor(array, dtype=tf.float32)

def astype_data(data, normalize=False):
    df = data.astype(np.float32)
    if normalize:
        df, _ = normalize_xy(xdata=df)
    return make_Tensor(df)

## 시점 윈도우 생성 함수
def time_window(df, t, t_sep):
    seq_len = t
    seqence_length = seq_len + t_sep

    result = []
    for index in tqdm(range(len(df) - seqence_length)):
        result.append(df[index: index + seqence_length].values)

    return np.array(result, dtype=np.float32)
        
def normalize_xy(xdata, ydata=[], idx=0):
    idx_to_remove = []
    for i in range(len(xdata)):
        for j in range(np.shape(xdata[i])[1]):
            p = xdata[i, :, j]
            if len(p[p>0]) != 0:
                last = p[p>0][-1]
                xdata[i, :, j] = (xdata[i, :, j] - last)/last
                if j == idx and len(ydata)!=0:
                    ydata[i] = (ydata[i] - last)/last
            else:
                if j == idx:
                    idx_to_remove.append(i)

        if len(ydata)!=0 and len(ydata[ydata>0])==0:
            print("asdfasdfadsfasdfasdfasdf")
 
    for i in sorted(idx_to_remove, reverse=True):
        np.delete(xdata, i, axis=0)
        if len(ydata)!=0:
            np.delete(ydata, i, axis=0)
        print(f"deleted {i}")

    return xdata, ydata

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text