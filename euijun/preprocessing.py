from glob import glob
from json import load
from tqdm import tqdm
from pandasql import sqldf
from utility import natural_keys
from scipy import ndimage

import pandas as pd
import numpy as np
import os


## 전처리 Class
class preprocessing_data(object):
    def __init__(self, dir, dir_outlier):
        self.dir = dir
        self.pummok_list = glob(self.dir)
        self.pummok_list.sort(key=natural_keys)
        self.pummok_list_test = []
        self.dir_outlier = dir_outlier

        self.prices = []
        self.volumes = []
        self.date = []
        self.add_pummock()

        self.prices_test = [[] for _ in range(10)]
        self.volumes_test = [[] for _ in range(10)]
        self.date_test = [[] for _ in range(10)]
        for i in range(10):
            self.add_pummock_test(i)
            

    def add_pummock(self, save=1):
        for idx, pummok in enumerate(tqdm(self.pummok_list)):
            if os.path.exists(f"./data/prices/{idx}.txt"):
                with open(f"./data/prices/{idx}.txt", "r") as pricefile:
                    price = np.loadtxt(pricefile)
                with open(f"./data/volumes/{idx}.txt", "r") as volumefile:
                    volume = np.loadtxt(volumefile)
                with open(f"./data/date.txt", "r") as datefile:
                    if len(self.date) == 0:
                        self.date = np.loadtxt(datefile).astype('int')
                self.prices.append(price)
                self.volumes.append(volume)

            else:
                df = pd.read_csv(pummok, low_memory=False)  # pummock의 csv 읽어오기
                georaeryang = sqldf(f"select sum(거래량) as '해당일자_전체거래물량(kg)' from df group by datadate")

                df = df[["datadate", "해당일자_전체평균가격(원)"]].drop_duplicates()

                self.date = df["datadate"].to_numpy()
                price = df["해당일자_전체평균가격(원)"].fillna(0).to_numpy()
                volume = georaeryang.fillna(0).to_numpy()
                
                
                # remove outliers explicitly
                with open(os.path.join(self.dir_outlier, f"{idx}.txt"), "r") as file:
                    lines = file.readlines()
                    outliers = [int(line.rstrip()) for line in lines]

                for outlier in outliers:
                    outlier_index = np.where(self.date == outlier)[0]
                    price[outlier_index] = 0
                    volume[outlier_index] = 0

                indices = price.nonzero()
                price[indices] = ndimage.median_filter(price[indices], size=3)

                for i in range(1, len(price)-1):
                    if price[i] == 0 and price[i-1] * price[i+1] != 0:
                        price[i] = (price[i-1] + price[i+1])/2

                nz = price.nonzero()[0]
                # add element to a list if nonzero otherwise get its closest nonzero element
                price = [x if x else price[nz[np.argmin(np.abs(i-nz))]] for i, x in enumerate(price)]

                self.prices.append(price)
                self.volumes.append(volume)

                if save==1:
                    os.makedirs("./data/prices/", exist_ok = True)
                    with open(f"./data/prices/{idx}.txt", "w") as pricefile:
                        np.savetxt(pricefile, price)
                    os.makedirs("./data/volumes/", exist_ok = True)
                    with open(f"./data/volumes/{idx}.txt", "w") as volumefile:
                        np.savetxt(volumefile, volume)
                    with open(f"./data/date.txt", "w") as datefile:
                        np.savetxt(datefile, self.date)

    def add_pummock_test(self, sep, save=1):
        test_dir = os.path.join(os.path.dirname(os.path.dirname(self.dir)), "aT_test_raw", f"sep_{sep}", f"pummok_*.csv")
        self.pummok_list_test.append(glob(test_dir))
        self.pummok_list_test[sep].sort(key=natural_keys)
        for idx, pummok in enumerate(tqdm(self.pummok_list_test[sep])):
            if os.path.exists(f"./test/{sep}/prices/{idx}.txt"):
                with open(f"./test/{sep}/prices/{idx}.txt", "r") as pricefile:
                    price = np.loadtxt(pricefile)
                with open(f"./test/{sep}/volumes/{idx}.txt", "r") as volumefile:
                    volume = np.loadtxt(volumefile)
                with open(f"./test/{sep}/date.txt", "r") as datefile:
                    date_test = datefile.read()
                self.prices_test[sep].append(price)
                self.volumes_test[sep].append(volume)
                self.date_test[sep].append(date_test)

            else:
                df = pd.read_csv(pummok, low_memory=False)  # pummock의 csv 읽어오기
                georaeryang = sqldf(f"select sum(거래량) as '해당일자_전체거래물량(kg)' from df group by datadate")

                df = df[["datadate", "해당일자_전체평균가격(원)"]].drop_duplicates()

                test_date = df["datadate"].to_numpy()[0]
                price = df["해당일자_전체평균가격(원)"].fillna(0).to_numpy()
                volume = georaeryang.fillna(0).to_numpy()

                indices = price.nonzero()
                price[indices] = ndimage.median_filter(price[indices], size=3)

                for i in range(1, len(price)-1):
                    if price[i] == 0 and price[i-1] * price[i+1] != 0:
                        price[i] = (price[i-1] + price[i+1])/2

                nz = price.nonzero()[0]
                # add element to a list if nonzero otherwise get its closest nonzero element
                price = [x if x else price[nz[np.argmin(np.abs(i-nz))]] for i, x in enumerate(price)]

                self.prices_test[sep].append(price)
                self.volumes_test[sep].append(volume)
                self.date_test[sep].append(self.date2idx(test_date))

                if save==1:
                    os.makedirs(f"./test/{sep}/prices/", exist_ok = True)
                    with open(f"./test/{sep}/prices/{idx}.txt", "w") as pricefile:
                        np.savetxt(pricefile, price)
                    os.makedirs(f"./test/{sep}/volumes/", exist_ok = True)
                    with open(f"./test/{sep}/volumes/{idx}.txt", "w") as volumefile:
                        np.savetxt(volumefile, volume)
                    with open(f"./test/{sep}/date.txt", "w") as datefile:
                        datefile.write(str(self.date2idx(test_date)))
    
    def chaboon(self, mvavg=0):
        if mvavg==0:
            price = self.prices
            self.prices_diff = []
            prices_diff = self.prices_diff
        else:
            price = self.prices_mvavg
            self.prices_mvavg_diff = []
            prices_diff = self.prices_mvavg_diff
        
        for i in range(len(self.pummok_list)):
            price_diff = np.zeros_like(price[i])[:-1]
            for j in range(len(price[i])-1):
                price_diff[j]=(price[i][j+1]-price[i][j])/price[i][j]
            prices_diff.append(price_diff)
        return
            

    def moving_avg(self, size=10):
        self.prices_mvavg = []
        for i in range(len(self.pummok_list)):
            price = self.prices[i]
            price = ndimage.uniform_filter(price, size)
            self.prices_mvavg.append(price)
        return

    def date2idx(self, date):
        return np.where(self.date%10000 == date%10000)[0][0]

    def __len__(self):
        return len(self.pummok_list)