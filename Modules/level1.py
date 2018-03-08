from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pickle
import os
import time
import numpy as np
import random

class Level1:
    def __init__(self, dataWrapper,Name=str(int(time.time())),directory='Models'):
        self.dataWrapper = dataWrapper
        self.name = "".join(Name.split(' ')) + str(int(time.time()))
        self.directory = directory
        self.clasifier = {}
        self.trainpred = {}
        self.testpred = {}
        self.counter = 0
        self.folds = 1
        try:
            os.mkdir('Models')
            print('Seems Like new Data!! YUM YUM')
        except:
            print('Lets Learn :D')
        self.path = self.directory + '/' + self.name +'level1' + str(self.counter) +'.lvl1'

    def crossValidation(self,train,target,train_size=0.7):
        return train_test_split(train, target, train_size = train_size, stratify = target, random_state = 2017)

    def multAcc(self, pred, dtrain):
        label = dtrain.get_label()
        acc = accuracy_score(label, pred)
        return 'maccuracy', acc    

    def xgboostclf(self, X_train, y_train,X_valid=None,y_valid=None,partial=False):
        params = {}
        params['objective'] = 'multi:softmax'
        params['eta'] = 0.02
        params['silent'] = True
        params['max_depth'] = 6
        params['subsample'] = 0.9
        params['colsample_bytree'] = 0.9
        params['nthread']=4
        params['num_class'] = 2

        
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        if partial:
            dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
            watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
            clf = xgb.train(params, dtrain, 1000, watchlist, maximize=True, verbose_eval=25, early_stopping_rounds=400, feval=self.multAcc)
            return clf
        else:
            clf = xgb.train(params, dtrain, 2000)
            return clf
             
             

    def fit_Partial(self):
        for i in self.dataWrapper.classDict:
          X_train, X_valid, y_train, y_valid = self.crossValidation(self.dataWrapper.train[self.dataWrapper.columns()],self.dataWrapper.classDict[i],train_size=0.2)
          self.clasifier[i] = self.xgboostclf(X_train,y_train,X_valid,y_valid,partial=True)
          self.counter += 1
        return self.clasifier
    
    def fit(self):
        for i in self.dataWrapper.classDict:
          print('Trainning for ' + str(i))
          self.clasifier[i] = self.xgboostclf(self.dataWrapper.train[self.dataWrapper.columns()],self.dataWrapper.classDict[i])
        pickle.dump(self.clasifier,open(self.path,'wb'))
        self.counter += 1
        return self.clasifier
    
    def generate(self):
        for i in self.clasifier:
            self.trainpred[i] = self.clasifier[i].predict(xgb.DMatrix(self.dataWrapper.train[self.dataWrapper.columns()]))
            self.testpred[i] = self.clasifier[i].predict(xgb.DMatrix(self.dataWrapper.test[self.dataWrapper.columns()]))
        return self.trainpred, self.testpred
            
    # def split(self):
    #     if self.folds:
    #         pass
    #     else:
    #         queue = [i + 1 for i in range(self.folds)]
    #         train_index , test_index = self.getsplitIndex()
    #         processed = []
    #         for _ in queue:
    #             a = random.choice(queue)
    #             queue.pop(queue.index(a))
    #             test = 1
    #             train = 1
                
    def getsplitIndex(self):
        markerTrain = self.dataWrapper.train.shape[0] / self.folds # Single Fold size for train
        markerTest = self.dataWrapper.test.shape[0] /self.folds    # Single Fold Size for test
        train_index = []
        test_index = []
        for i in self.kfolds:
            if i != self.kfolds-1:
                test_index.append(range(markerTest*(i),markerTest*(i+1)))
                train_index.append(range(markerTrain*(i),markerTrain*(i+1)))
            else:
                test_index.append(range(markerTest*(i),self.dataWrapper.test.shape[0]))
                train_index.append(range(markerTrain*(i),self.dataWrapper.train.shape[0]))
        return train_index,test_index

# def predict(X_Test,classfierSet):
#     for i in classfierSet:
