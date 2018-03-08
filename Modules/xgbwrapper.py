import xgboost as xgb


class CustomXGboost:

    # def __init__(self,):
    #     """
    #     dtrain=None,epochs,watchlist,maximize, verbose_eval,earlyStopingRounds,feval Order of kwargs
    #     """
    #     self.clf = xgb.train(params,dtrain, num_boost_round=10, evals=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None)

    def fit(self,params,dtrain, num_boost_round=10, evals=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None):
        self.clf = xgb.train(params,dtrain=dtrain, num_boost_round=num_boost_round, evals=evals, obj=obj, feval=feval, maximize=maximize, early_stopping_rounds=early_stopping_rounds, evals_result=evals_result, verbose_eval=verbose_eval, xgb_model=xgb_model, callbacks=callbacks, learning_rates=learning_rates)
        return self.clf
    
    def predict(self,dtest):
        return self.clf.predict(dtest)