# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:34:32 2020

@author: Masih
classes for machine learning analysis
?? ?? ?????? ????? ?????, ???? ??? ?? ????? ???
"""
import xgboost as xgb
from sklearn.metrics import confusion_matrix, explained_variance_score, roc_auc_score,r2_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp
from pylab import plot, show, savefig, xlim, figure,  ylim, legend, boxplot, setp, axes
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


#function to estimate 95% confidence interval 
import scipy.stats
import numpy as np
import pandas as pd

###helper functions##################################
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return (round(m,3), round(m-h,3), round(m+h,3))

def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
#####################################################

class param_optimizer:
    """ optimization of hyperparameters 

    Parameters
    ----------
    df : pandas dataframe, shape = df[samples,features]
      pandas dataframe or numpy array representing the sample,feature matrix
      
     target : array-like, shape = [tg1,tg2,..tgn]
     target feature to be predicted

    """
    def __init__(self,df,target):
        self.df=df
        self.dfv=df.values
        self.target=target
        
    def cv_folds(self,test_siz=0.2,typ='sscv',folds=20,random_state=123):
            """ 
            Folds of cross-validation to be used
            Parameters
            ----------
            test_size : float
              test size for the folds
              
            typ: 'str',
            type of cross-validation, one of '{sscv:startifiedShuffled kfold Cross-validation,
                                                scv: stratified kfold cross-validation,
                                                kfold: kfold cross validation}'
            
            folds:int,
            folds of cross-validation
             
            random_state : int
            random state of cross validation for results to be replicable
            """
            if typ == 'sscv':   
                cvs=StratifiedShuffleSplit(n_splits=folds, test_size=test_siz,random_stat=random_state)
            return cvs
    
    def hyperopt_xgb(self,xgbpar):
        xgb_params = xgbpar
        xgb_params['max_depth'] = int(xgb_params['max_depth']) 
        mdl=xgb.XGBClassifier(
            max_depth=xgb_params['max_depth'],
            colsample_bytree=xgb_params['colsample_bytree'],
            eta=xgb_params['eta'],
            min_child_weight=xgb_params['min_child_weight'],
            n_estimators=int(xgb_params['n_estimators']),
            subsample=xgb_params['subsample'],
            nthread=-1)
        
        auc=cross_val_score(mdl,self.dfv,self.target,cv=self.xgb_cv,scoring='roc_auc').mean()
        if (auc) > self.xgb_best_score:
            self.xgb_best_score = (auc)
            print ('new best:', auc, xgb_params)
        loss = 1-auc
        return {'loss': loss, 'status': STATUS_OK}
    
    def optimize_xgb(self,evals, optimizer=tpe.suggest, nthread=-1, random_state=123):
        #global spacexgb
        spacexgb = {
            'n_estimators': hp.quniform('n_estimators', 200, 1000, 50),
            'eta': hp.quniform('eta', 0.025, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
            'max_depth': hp.quniform('max_depth', 2, 15, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
            'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
            #'gamma': hp.quniform('gamma', 0.1, 1, 0.1),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
            #'alpha' :  hp.quniform('alpha', 0, 10, 1),
            #'lambda': hp.quniform('lambda', 1, 2, 0.1),
            'nthread': nthread,
            #'objective': 'multi:softprob',
            #'num_class':3,
            #'booster': 'gbtree',
            #'seed': random_state,
            #'silent':1,
            #'tree_method': 'gpu_hist', #To speed-up the process by using gpu
             #'predictor': 'cpu_predictor', #to solve the memory error
            }
        ssf=self.cv_folds(random_state=random_state)
        self.xgb_cv=ssf
        self.xgb_best_score=0
        trials = Trials()
        best_param = fmin(self.hyperopt_xgb, spacexgb, algo=tpe.suggest, max_evals=evals, trials = trials)
        best_param=space_eval(spacexgb,best_param)
        best_param['max_depth'] = int(best_param['max_depth'])
        print('Best XGBClassfier ROCAUC score is equal to: {0:.3f}'.format(self.xgb_best_score))
        return best_param
        
        
class rocauc_estimate:
    """ estimating and plotting rocauc score for different models 

    Parameters
    ----------
    df : pandas dataframe, shape = df[samples,features]
      pandas dataframe or numpy array representing the sample,feature matrix
      
     target : array-like, shape = [tg1,tg2,..tgn]
     target feature to be predicted

    """
    def __init__(self,df,target):
        self.df=df
        self.dfv=df.values
        self.target=target
    def cv_folds(self,test_size=0.2,cv_type='sscv',folds=20,random_state=123):
        if cv_type == 'sscv':   
            cvs=StratifiedShuffleSplit(n_splits=folds, test_size=test_size,random_state=random_state)
        return cvs

    #############################RandomForest################################
    def hyperopt_rf (self,rfpar):
          rf_params = rfpar
          mdl=RandomForestClassifier(
              max_features=rf_params['max_features'],
              max_depth=rf_params['max_depth'],
              n_estimators=rf_params['n_estimators'], 
              criterion=rf_params['criterion'],
              n_jobs=-1,
              )
          auc=cross_val_score(mdl,self.x_train,self.y_train,cv=self.rf_cv,scoring='roc_auc',n_jobs=-1).mean()
          if (auc) > self.rf_best_score:
              self.rf_best_score = auc
              self.rfmod=mdl
              print ('new best:', auc, rf_params)
          loss = 1-auc
          return {'loss': loss, 'status': STATUS_OK}

    def optimize_rf (self,evals, cv_fold=5,optimizer=tpe.suggest, nthread=-1, random_state=123):
          spacerf = {
              'max_features' : hp.choice('max_features',["auto","sqrt", "log2", 0.5]),
              'max_depth': hp.choice('max_depth', range(1,20)),
              'n_estimators': hp.choice('n_estimators', [100,300,500,1000]),
              'criterion': hp.choice('criterion', ["gini", "entropy"])
              }
          ssf=self.cv_folds(folds=cv_fold,random_state=random_state)
          self.rf_cv=ssf
          self.rf_best_score=0
          trials = Trials()
          best_param = fmin(self.hyperopt_rf, spacerf, algo=tpe.suggest, max_evals=evals, trials = trials)
          best_param=space_eval(spacerf,best_param)
          print('Best RandomForest ROCAUC score is equal to: {0:.3f}'.format(self.rf_best_score))
          return (best_param)

        #############################SVM################################
    def hyperopt_svc (self,svcpar):
          svc_params = svcpar
          mdl=LinearSVC(
              C=svc_params['C'],
              penalty=svc_params['penalty'],
              dual=False, 
              max_iter=2000,
              #kernel=svc_params['kernel'], #hp.choice(['rbf', 'poly', "sigmoid"]) when svc is used instead of linearSVC
              )
          auc=cross_val_score(mdl,self.x_train,self.y_train,cv=self.svc_cv,scoring='roc_auc',n_jobs=-1).mean()
          if (auc) > self.svc_best_score:
              self.svc_best_score = auc
              self.svcmod=mdl
              print ('new best:', auc, svc_params)
          loss = 1-auc
          return {'loss': loss, 'status': STATUS_OK}

    def optimize_svc (self,evals, cv_fold=5,optimizer=tpe.suggest, nthread=-1, random_state=123):
          spacesvc = {
              'C' : hp.loguniform('C', low=-2*np.log(10), high=2*np.log(10)), #[0.001,0.01,0.1,1,10,100]
              'penalty':hp.choice('penalty',['l1','l2']),
              }
          ssf=self.cv_folds(folds=cv_fold,random_state=random_state)
          self.svc_cv=ssf
          self.svc_best_score=0
          trials = Trials()
          best_param = fmin(self.hyperopt_svc, spacesvc, algo=tpe.suggest, max_evals=evals, trials = trials)
          best_param=space_eval(spacesvc,best_param)
          print('Best linearSVC ROCAUC score is equal to: {0:.3f}'.format(self.svc_best_score))
          return (best_param)

    #############################logisticRegression################################
    def hyperopt_lr (self,lrpar):
          lr_params = lrpar
          mdl=LogisticRegression(
              C=lr_params['C'],
              penalty=lr_params['penalty'],
              solver='liblinear', # default 'lbfgs' does not support l1 and liblinear is better fo small dataset
              n_jobs=1)
          auc=cross_val_score(mdl,self.x_train,self.y_train,cv=self.lr_cv,scoring='roc_auc',n_jobs=-1).mean()
          if (auc) > self.lr_best_score:
              self.lr_best_score = auc
              self.lrmod=mdl
              print ('new best:', auc, lr_params)
          loss = 1-auc
          return {'loss': loss, 'status': STATUS_OK}

    def optimize_lr (self,evals, cv_fold=5,optimizer=tpe.suggest, nthread=-1, random_state=123):
          spacelr = {
              'C' : hp.loguniform('C', low=-2*np.log(10), high=2*np.log(10)), #[0.001,0.01,0.1,1,10,100]
              'penalty':hp.choice('penalty',['l1','l2']),
              'n_jobs': nthread,
              }
          ssf=self.cv_folds(folds=cv_fold,random_state=random_state)
          self.lr_cv=ssf
          self.lr_best_score=0
          trials = Trials()
          best_param = fmin(self.hyperopt_lr, spacelr, algo=tpe.suggest, max_evals=evals, trials = trials)
          best_param=space_eval(spacelr,best_param)
          print('Best logisticRegression ROCAUC score is equal to: {0:.3f}'.format(self.lr_best_score))
          return (best_param)
    
    ##############################xgboost##########################################    
    def hyperopt_xgb(self,xgbpar):
        xgb_params = xgbpar
        xgb_params['max_depth'] = int(xgb_params['max_depth']) 
        mdl=xgb.XGBClassifier(
            max_depth=xgb_params['max_depth'],
            colsample_bytree=xgb_params['colsample_bytree'],
            eta=xgb_params['eta'],
            min_child_weight=xgb_params['min_child_weight'],
            n_estimators=int(xgb_params['n_estimators']),
            subsample=xgb_params['subsample'],
            nthread=-1)
        
        auc=cross_val_score(mdl,self.x_train,self.y_train,cv=self.xgb_cv,scoring='roc_auc',n_jobs=-1).mean()
        if (auc) > self.xgb_best_score:
            self.xgb_best_score = auc
            self.xgbmod=mdl
            print ('new best:', auc, xgb_params)
        loss = 1-auc
        return {'loss': loss, 'status': STATUS_OK}
    
    def optimize_xgb(self,evals, cv_fold=5,optimizer=tpe.suggest, nthread=-1, random_state=123):
        spacexgb = {
            'n_estimators': hp.quniform('n_estimators', 200, 1000, 50),
            'eta': hp.quniform('eta', 0.025, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
            'max_depth': hp.quniform('max_depth', 2, 15, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
            'subsample': hp.quniform('subsample', 0.6, 1, 0.05),
            #'gamma': hp.quniform('gamma', 0.1, 1, 0.1),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
            #'alpha' :  hp.quniform('alpha', 0, 10, 1),
            #'lambda': hp.quniform('lambda', 1, 2, 0.1),
            'nthread': nthread,
            #'objective': 'multi:softprob',
            #'num_class':3,
            #'booster': 'gbtree',
            #'seed': random_state,
            #'silent':1,
            #'tree_method': 'gpu_hist', #To speed-up the process by using gpu
             #'predictor': 'cpu_predictor', #to solve the memory error
            }
        ssf=self.cv_folds(folds=cv_fold,random_state=random_state)
        self.xgb_cv=ssf
        self.xgb_best_score=0
        trials = Trials()
        best_param = fmin(self.hyperopt_xgb, spacexgb, algo=tpe.suggest, max_evals=evals, trials = trials)
        best_param=space_eval(spacexgb,best_param)
        best_param['max_depth'] = int(best_param['max_depth'])
        print('Best XGBClassfier ROCAUC score is equal to: {0:.3f}'.format(self.xgb_best_score))
        return best_param
    #######################################################################################
        
    def rocauc_estimator(self,out,n_splits=100,test_size=0.2,figsize=(8,8),
                         hp_tune=False,hp_try=20,hp_cv_fold=3,which_hp={'lr':False,'rf':False,'xgb':False,'svc':True}):
        ###@1-11) comparing the performance of different models
        plt.style.use('seaborn')
        sns.set_context("paper" )
        import numpy as np
        cv=self.cv_folds(cv_type='sscv',folds=n_splits,test_size=test_size,random_state=123)
        tprs_lgr, aucs_lgr,f1_lgr,rec_lgr,pre_lgr,bacc_lgr,acc_lgr = [],[],[],[],[],[],[]
        tprs_xgb, aucs_xgb,f1_xgb,rec_xgb,pre_xgb,bacc_xgb,acc_xgb  = [],[],[],[],[],[],[]
        tprs_rf, aucs_rf,f1_rf,rec_rf,pre_rf,bacc_rf,acc_rf  = [],[],[],[],[],[],[]
        tprs_svc, aucs_svc,f1_svc,rec_svc,pre_svc,bacc_svc,acc_svc  = [],[],[],[],[],[],[]
        mean_fpr = np.linspace(0, 1, 100)
        X=self.dfv
        Y=self.target
        for train_index,test_index in cv.split(X,Y):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=Y[train_index],Y[test_index]
            
            ###set of models
            #1)logistic regression
            if hp_tune and which_hp['lr']:
                self.x_train=X_train
                self.y_train=y_train
                best_param = self.optimize_lr(evals = hp_try,
                                          optimizer=tpe.suggest,cv_fold=hp_cv_fold)
                lgr=self.lrmod
            else:
                lgr=LogisticRegression()
            lgr.fit(X_train,y_train)
            y_pred_lgr = lgr.predict_proba(X_test)[:, 1]
            fpr_lgr, tpr_lgr, thresholds_lgr = roc_curve(y_test, y_pred_lgr)
            auc_lgr = auc(fpr_lgr, tpr_lgr)
            interp_lgr_tpr = interp(mean_fpr, fpr_lgr, tpr_lgr)
            interp_lgr_tpr[0] = 0.0
            tprs_lgr.append(interp_lgr_tpr)
            aucs_lgr.append(auc_lgr)
            #estimate performance metrics
            y_pred_lgr=lgr.predict(X_test)
            f1_lgr.append(f1_score(y_test, y_pred_lgr, average='weighted'))
            pre_lgr.append(precision_score(y_test, y_pred_lgr, average='weighted'))
            rec_lgr.append(recall_score(y_test, y_pred_lgr, average='weighted'))
            bacc_lgr.append(balanced_accuracy_score(y_test, y_pred_lgr))
            acc_lgr.append(accuracy_score(y_test, y_pred_lgr))

            #2)Random Forests
            if hp_tune and which_hp['rf']:
                self.x_train=X_train
                self.y_train=y_train
                best_param = self.optimize_rf(evals = hp_try,
                                          optimizer=tpe.suggest,cv_fold=hp_cv_fold)
                rf=self.rfmod
            else:
                rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict_proba(X_test)[:, 1]
            fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
            auc_rf = auc(fpr_rf, tpr_rf)
            interp_rf_tpr = interp(mean_fpr, fpr_rf, tpr_rf)
            interp_rf_tpr[0] = 0.0
            tprs_rf.append(interp_rf_tpr)
            aucs_rf.append(auc_rf)
            #estimate performance metrics
            y_pred_rf=rf.predict(X_test)
            f1_rf.append(f1_score(y_test, y_pred_rf, average='weighted'))
            pre_rf.append(precision_score(y_test, y_pred_rf, average='weighted'))
            rec_rf.append(recall_score(y_test, y_pred_rf, average='weighted'))
            bacc_rf.append(balanced_accuracy_score(y_test, y_pred_rf))
            acc_rf.append(accuracy_score(y_test, y_pred_rf))
            
            #gardient boosting
            if hp_tune and which_hp['xgb']:
                self.x_train=X_train
                self.y_train=y_train
                best_param = self.optimize_xgb(evals = hp_try,
                                          optimizer=tpe.suggest,cv_fold=hp_cv_fold)
                xgmod=self.xgbmod
            else:
                xgmod= xgb.XGBClassifier()
            xgmod.fit(X_train, y_train)
            y_pred_xgb = xgmod.predict_proba(X_test)[:, 1]
            fpr_xgb, tpr_xgb, thresholds_xg = roc_curve(y_test, y_pred_xgb)
            auc_xgb= auc(fpr_xgb, tpr_xgb)
            interp_xgb_tpr = interp(mean_fpr, fpr_xgb, tpr_xgb)
            interp_xgb_tpr[0] = 0.0
            tprs_xgb.append(interp_xgb_tpr)
            aucs_xgb.append(auc_xgb)
            #estimate performance metrics
            y_pred_xgb=xgmod.predict(X_test)
            f1_xgb.append(f1_score(y_test, y_pred_xgb, average='weighted'))
            pre_xgb.append(precision_score(y_test, y_pred_xgb, average='weighted'))
            rec_xgb.append(recall_score(y_test, y_pred_xgb, average='weighted'))
            bacc_xgb.append(balanced_accuracy_score(y_test, y_pred_xgb))
            acc_xgb.append(accuracy_score(y_test, y_pred_xgb))
            
            #SVM (for the case of linearSVC and svm.SVC different methods should be used to get probabilites)
            #svcmod= svm.SVC(probability=True)
            #svcmod=CalibratedClassifierCV(base_estimator=LinearSVC(C=15.0, dual=False, loss="squared_hinge", 
            #                                                       penalty="l1", tol=0.1))
            if hp_tune and which_hp['svc']:
                self.x_train=X_train
                self.y_train=y_train
                best_param = self.optimize_svc(evals = hp_try,
                                          optimizer=tpe.suggest,cv_fold=hp_cv_fold)
                svc=CalibratedClassifierCV(base_estimator=self.svcmod)
            else:
                svc=CalibratedClassifierCV(base_estimator=LinearSVC())
            svc.fit(X_train, y_train)
            y_pred_svc = svc.predict_proba(X_test)[:, 1]
            fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_pred_svc)
            auc_svc= auc(fpr_svc, tpr_svc)
            interp_svc_tpr = interp(mean_fpr, fpr_svc, tpr_svc)
            interp_svc_tpr[0] = 0.0
            tprs_svc.append(interp_svc_tpr)
            aucs_svc.append(auc_svc)
            #estimate performance metrics
            y_pred_svc=svc.predict(X_test)
            f1_svc.append(f1_score(y_test, y_pred_svc, average='weighted'))
            pre_svc.append(precision_score(y_test, y_pred_svc, average='weighted'))
            rec_svc.append(recall_score(y_test, y_pred_svc, average='weighted'))
            bacc_svc.append(balanced_accuracy_score(y_test, y_pred_svc))
            acc_svc.append(accuracy_score(y_test, y_pred_svc))
        #writing model performances across different metrics in the output
        aucs_lgr_m,f1_lgr_m,rec_lgr_m,pre_lgr_m,bacc_lgr_m,acc_lgr_m = ( 
        mean_confidence_interval(aucs_lgr),mean_confidence_interval(f1_lgr),
        mean_confidence_interval(rec_lgr),mean_confidence_interval(pre_lgr),
        mean_confidence_interval(bacc_lgr),mean_confidence_interval(acc_lgr)
        )
        aucs_xgb_m,f1_xgb_m,rec_xgb_m,pre_xgb_m,bacc_xgb_m,acc_xgb_m  = ( 
        mean_confidence_interval(aucs_xgb),mean_confidence_interval(f1_xgb),
        mean_confidence_interval(rec_xgb),mean_confidence_interval(pre_xgb),
        mean_confidence_interval(bacc_xgb),mean_confidence_interval(acc_xgb)
        )
        aucs_rf_m,f1_rf_m,rec_rf_m,pre_rf_m,bacc_rf_m,acc_rf_m  = (
        mean_confidence_interval(aucs_rf),mean_confidence_interval(f1_rf),
        mean_confidence_interval(rec_rf),mean_confidence_interval(pre_rf),
        mean_confidence_interval(bacc_rf),mean_confidence_interval(acc_rf)
        )
        aucs_svc_m,f1_svc_m,rec_svc_m,pre_svc_m,bacc_svc_m,acc_svc_m  = (
        mean_confidence_interval(aucs_svc),mean_confidence_interval(f1_svc),
        mean_confidence_interval(rec_svc),mean_confidence_interval(pre_svc),
        mean_confidence_interval(bacc_svc),mean_confidence_interval(acc_svc)
        )
        #writing the evaluation in a dataframe
        df_lr=pd.DataFrame({'AUROC':aucs_lgr_m,'bACC':bacc_lgr_m,'Accuracy':acc_lgr_m,'F1':f1_lgr_m,'Precision':pre_lgr_m,'Recall':rec_lgr_m},index=pd.Series(['Logistric_regression','lrCI-','lrCI+'], name='Tag'))
        df_xgb=pd.DataFrame({'AUROC':aucs_xgb_m,'bACC':bacc_xgb_m,'Accuracy':acc_xgb_m,'F1':f1_xgb_m,'Precision':pre_xgb_m,'Recall':rec_xgb_m},index=pd.Series(['XGBoost','xgbCI-','xgbCI+'], name='Tag'))
        df_rf=pd.DataFrame({'AUROC':aucs_rf_m,'bACC':bacc_rf_m,'Accuracy':acc_rf_m,'F1':f1_rf_m,'Precision':pre_rf_m,'Recall':rec_rf_m},index=pd.Series(['Random_forest','rfCI-','rfCI+'], name='Tag'))
        df_svc=pd.DataFrame({'AUROC':aucs_svc_m,'bACC':bacc_svc_m,'Accuracy':acc_svc_m,'F1':f1_svc_m,'Precision':pre_svc_m,'Recall':rec_svc_m},index=pd.Series(['SVM','svmCI-','svmCI+'], name='Tag'))
        frames = [df_lr, df_xgb, df_rf, df_svc]
        result = pd.concat(frames).T
        result.to_csv(out+'.csv')
        #writing rocauc scores to estimate statistical significance between them
        df=pd.DataFrame({'lr_auc':aucs_lgr,'lr_bacc':bacc_lgr,'lr_acc':acc_lgr,'lr_f1':f1_lgr,'lr_pre':pre_lgr,'lr_rec':rec_lgr,
                         'svc_auc':aucs_svc,'svc_bacc':bacc_svc,'svc_acc':acc_svc,'svc_f1':f1_svc,'svc_pre':pre_svc,'svc_rec':rec_svc,
                         'rf_auc':aucs_rf,'rf_bacc':bacc_rf,'rf_acc':acc_rf,'rf_f1':f1_rf,'rf_pre':pre_rf,'rf_rec':rec_rf,
                         'xgb_auc':aucs_xgb,'xgb_bacc':bacc_xgb,'xgb_acc':acc_xgb,'xgb_f1':f1_xgb,'xgb_pre':pre_xgb,'xgb_rec':rec_xgb})
        df.to_csv(out+'_rocaucs.csv')
        #plotting roc_auc curve for the different models
        fig, ax = plt.subplots(figsize=cm2inch(figsize))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        ###plotting the results
        #1) plotting lgr results
        mean_tpr_lgr = np.mean(tprs_lgr, axis=0)
        mean_tpr_lgr[-1] = 1.0
        mean_auc_lgr= auc(mean_fpr, mean_tpr_lgr)
        std_auc_lgr = np.std(aucs_lgr)
        ax.plot(mean_fpr, mean_tpr_lgr, color='b',
                label=r'Logistic regression (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_lgr, std_auc_lgr),
                lw=2, alpha=.8)
        std_tpr_lgr = np.std(tprs_lgr, axis=0)
        tprs_lgr_upper = np.minimum(mean_tpr_lgr + std_tpr_lgr, 1)
        tprs_lgr_lower = np.maximum(mean_tpr_lgr - std_tpr_lgr, 0)
        ax.fill_between(mean_fpr, tprs_lgr_lower, tprs_lgr_upper, color='grey', alpha=.2,
                        #label=r'$\pm$ 1 std. dev.'
                        )
        #2) plotting xgb results
        mean_tpr_xgb = np.mean(tprs_xgb, axis=0)
        mean_tpr_xgb[-1] = 1.0
        mean_auc_xgb = auc(mean_fpr, mean_tpr_xgb)
        std_auc_xgb = np.std(aucs_xgb)
        ax.plot(mean_fpr, mean_tpr_xgb, color='r',
                label=r'XGBoost (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_xgb, std_auc_xgb),
                lw=2, alpha=.8)
        std_tpr_xgb = np.std(tprs_xgb, axis=0)
        tprs_xgb_upper = np.minimum(mean_tpr_xgb + std_tpr_xgb, 1)
        tprs_xgb_lower = np.maximum(mean_tpr_xgb - std_tpr_xgb, 0)
        ax.fill_between(mean_fpr, tprs_xgb_lower, tprs_xgb_upper, color='pink', alpha=.2,
                        #label=r'$\pm$ 1 std. dev.'
                        )
        #3) plotting rf results
        mean_tpr_rf = np.mean(tprs_rf, axis=0)
        mean_tpr_rf[-1] = 1.0
        mean_auc_rf = auc(mean_fpr, mean_tpr_rf)
        std_auc_rf = np.std(aucs_rf)
        ax.plot(mean_fpr, mean_tpr_rf, color='g',
                label=r'RandomForest (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_rf, std_auc_rf),
                lw=2, alpha=.8)
        std_tpr_rf = np.std(tprs_rf, axis=0)
        tprs_rf_upper = np.minimum(mean_tpr_rf + std_tpr_rf, 1)
        tprs_rf_lower = np.maximum(mean_tpr_rf - std_tpr_rf, 0)
        ax.fill_between(mean_fpr, tprs_rf_lower, tprs_rf_upper, color='yellow', alpha=.2,
                        #label=r'$\pm$ 1 std. dev.'
                        )
        
        #4) plotting svc results
        mean_tpr_svc = np.mean(tprs_svc, axis=0)
        mean_tpr_svc[-1] = 1.0
        mean_auc_svc = auc(mean_fpr, mean_tpr_svc)
        std_auc_svc = np.std(aucs_svc)
        ax.plot(mean_fpr, mean_tpr_svc, color='c',
                label=r'SVM (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_svc, std_auc_svc),
                lw=2, alpha=.8)
        std_tpr_svc = np.std(tprs_svc, axis=0)
        tprs_svc_upper = np.minimum(mean_tpr_svc + std_tpr_svc, 1)
        tprs_svc_lower = np.maximum(mean_tpr_svc - std_tpr_svc, 0)
        ax.fill_between(mean_fpr, tprs_svc_lower, tprs_svc_upper, color='yellow', alpha=.2,
                        #label=r'$\pm$ 1 std. dev.'
                        )
        #setting x and y axis boundary    
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               #title="Receiver operating characteristic averaged over 50 cross-validations"
               )
        ax.legend(loc="lower right")
        plt.setp(ax.get_legend().get_texts(), fontsize=9)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.xticks(size=9)
        plt.yticks(size=9)
        plt.tight_layout()
        plt.savefig(out,dpi=300)
        
    def permutation_score(self,mdl,out,scoring='roc_auc',
                          permutation_number=1000,figsize=(8.8),fnt_size=9,
                          n_jobs=-1):
        """estimate permutation score and plot the results"""
        from sklearn.model_selection import permutation_test_score
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('seaborn')
        sns.set_context("paper" )
        X=self.dfv
        Y=self.target
        rfm = mdl
        score, permutation_scores, pvalue = permutation_test_score(
            rfm, X, Y, scoring=scoring, cv=self.cv_folds, n_permutations=permutation_number, n_jobs=n_jobs)
        print("Classification score %s (pvalue : %s)" % (score, pvalue))
        
        # #############################################################################
        # View histogram of permutation scores
        fig=plt.figure(figsize=cm2inch(figsize))
        
        plt.hist(permutation_scores, 20, label='Permutation score =%s'% round(np.mean(permutation_scores),3),
                 edgecolor='black')
        fig.text(0.6,0.80,'pvalue= %s' % round(pvalue,3),size=fnt_size)
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=2,
                 label='Classification score =%s'% round(score,3))
        plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3) #, label='Luck'
        
        plt.ylim(ylim)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  fancybox=True, shadow=True, ncol=1,prop={'size': fnt_size})
        

        plt.xticks(size=fnt_size)
        plt.yticks(size=fnt_size)
        plt.tight_layout()
        plt.savefig(out,dpi=300)