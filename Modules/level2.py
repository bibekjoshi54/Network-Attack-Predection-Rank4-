from sklearn.linear_model import LogisticRegressionCV
class level2:
    def __init__(self,train,label,test):
        self.train = train
        self.test = test
        self.label = label
        self.clf = LogisticRegressionCV(Cs=100,cv=5,n_jobs=4)

    # def crossValidation(self,train,target):
    #     return train_test_split(train, target, train_size = 0.2, stratify = target, random_state = 2017)

    def fit_Partial(self):
        pass
    
    def fit(self):
        return self.clf.fit(self.train,self.label)
    
    def pred(self):
        return self.clf.predict(self.test)

    def pred_proba(self):

        return self.clf.predict_proba(self.test)