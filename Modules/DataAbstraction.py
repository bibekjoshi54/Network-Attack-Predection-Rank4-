import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Visualization:

    def distView(data,size=(9,8),bin=100):
        plt.figure(figsize=(9,8))
        return sns.distplot(data, bins=bin, hist_kws={'alpha': 0.4})

class Data(Visualization):
    def __init__(self,train,test):
        self.test = test
        self.train = train
        print('train : ' + str(self.train.shape))
        print('test: ' + str(self.test.shape))
        
    def describe(self,test=False):
        if test:
            return self.train.describe(), self.test.describe()
        return self.train.describe()
    
    def columns(self):
        return [i for i in self.test.columns if i not in ['connection_id','target']]
    
    def getCustom(self,substring):
        return [i for i in self.columns() if substring in i]

    def drop(self,colName):
        self.train = self.train.drop(colName,axis=1)
        self.test = self.test.drop(colName,axis=1)

    def scaler(self,colName):
        from sklearn.preprocessing import StandardScaler
        std_scale = StandardScaler().fit(self.test[colName])
        self.test[colName] = std_scale.transform(self.test[colName])
        self.train[colName] = std_scale.transform(self.train[colName])

    def decomposition(self):
        classSet = list(set(self.train['target']))
        self.classDict = { i: [1 if j==i else 0 for j in self.train['target']] for i in classSet}
# class Stacker(object):

#     def d