

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

datafile = 'train.txt'
datatset = pd.read_csv(datafile,sep=',',header=0)

print(len(dataset))
print(dataset.shape)
dataset.head()