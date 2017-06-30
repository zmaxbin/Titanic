# -*- coding:utf-8 -*-

import pandas as pd

Demo_data  = pd.read_csv("../data/Demo.csv")

X = Demo_data.iloc[:,:-1].values