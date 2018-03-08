import pandas as pd

test = pd.read_csv('../Data/test_data.csv')
train = pd.read_csv('../Data/train_data.csv')

class FeatureWrapper(object):
    def __init__(self,test,train):
        self.test = test
        self.train = train
        print('Test : ' + str(self.test.shape))
        print('Train: ' + str(self.train.shape))
        
    def describe(self,test=False):
        if test:
            return self.train.describe(), self.test.describe()
        return self.train.describe()
    
    def columns(self):
        return [i for i in self.test.columns if i not in ['connection_id','target']]
    
    def getCustom(self,substring):
        return [i for i in self.columns() if substring in i]
    
#     def oneHET(self,substring):
#         from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#         custLabel = self.getCustom(substring)
#         for i in custLabel:
#             label = LabelEncoder()
#             self.train[i] = 
            
    
    def SS(self,substring):
        from sklearn.preprocessing import StandardScaler as SC
        sc = SS()
        self.train[self.getCustom(substring)] = sc.fit_transform(self.train[self.getCustom(substring)])
        self.test[self.getCustom(substring)] = sc.transform(self.test[self.getCustom(substring)])
        

fw = FeatureWrapper(train=train,test=test)
print(fw.columns())
