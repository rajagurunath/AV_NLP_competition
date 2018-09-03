# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:32:10 2018

@author: Gurunath
"""

import lightgbm as lgb
import pandas as pd
from tqdm import tqdm
import numpy as np
import spacy
model_path=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model'
custom_nlp=spacy.load(model_path)
def spacy_vectors(tweets):
    res=list()
    for tw in tqdm(tweets):
        res.append(custom_nlp(tw).vector)
    return res



tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')

train_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_train.npy','rb'))
test_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_test.npy','rb'))


train_data = lgb.Dataset(train_mat, label=tweet_df['label'])

test_data = train_data.create_valid(train_mat)

param = {'num_leaves':31, 'num_trees':500, 'objective':'binary'}

param['metric'] = 'f1-score'

num_round = 100

bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

pred=(bst.predict(test_mat)>0.5)*1

ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=pred
ensemble_out.to_csv('lgbm_try.csv')

param = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
    }

def fit_lgbm(X,y,param=param,num_round=10):
    
    evals_result = {}
    train_data = lgb.Dataset(X, label=y)
    test_data = train_data.create_valid(train_mat)
    lgb_model = lgb.train(param, 
                      train_data, 
                      num_round, 
                      valid_sets = [test_data], 
                      feval = evalerror,
                      evals_result = evals_result)

    return lgb_model

from sklearn.metrics import f1_score

def evalerror(preds, dtrain):
    
    labels = dtrain.get_label()
#    print(preds.shape,labels.shape)
    preds = preds #.reshape(len(np.unique(labels)), -1)

#    preds = preds.reshape(-1, 5)
    preds = (preds>0.5)*1#.argmax(axis = 0)
    f_score = f1_score(preds, labels, average = 'weighted')
    return 'f1_score', f_score, True

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(len(np.unique(y_true)), -1)

#    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

evals_result = {}

clf = lgb.train(param, train_data, valid_sets=[test_data],
                valid_names=['val'], 
                feval=lgb_f1_score, 
                evals_result=evals_result)

lgb.plot_metric(evals_result, metric='f1_score')

lgb_model = lgb.train(param, 
                      train_data, 
                      num_round, 
                      valid_sets = [test_data], 
                      feval = evalerror,
                      evals_result = evals_result)



pred=(lgb_model.predict(test_mat)>0.5)*1

ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=pred
ensemble_out.to_csv('spacy_univlog.csv')





ensemble_train_df=pd.read_csv('ensemble_train_df.csv')
ensemble_test_df=pd.read_csv('ensemble_test_df.csv')





train_data = lgb.Dataset(ensemble_train_df, label=tweet_df['label'])

test_data = train_data.create_valid(ensemble_train_df)


train_vect=spacy_vectors(tweet_df['tweet'])
test_vect=spacy_vectors(test_df['tweet'])

train_vect=np.array(train_vect)
test_vect=np.array(test_vect)
np.save(open(r'spacy_train50.npy','wb'),train_vect)
np.save(open(r'spacy_test50.npy','wb'),test_vect)


spacy_train_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_train.npy','rb'))
spacy_test_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_test.npy','rb'))


from sklearn.linear_model import LogisticRegression
logclf=LogisticRegression()

logclf.fit(train_vect,tweet_df['label'])
pred=logclf.predict(test_vect)



spacy_univ_train=np.hstack([train_mat,spacy_train_mat])
spacy_univ_test=np.hstack([test_mat,spacy_test_mat])

logclf=LogisticRegression()

logclf.fit(spacy_univ_train,tweet_df['label'])
pred=logclf.predict(spacy_univ_test)

param = {'num_leaves':31, 'num_trees':500, 'tree_learner':'voting','objective':'binary','metric':'binary_error'}

labels=tweet_df['label']
lgbmodel=fit_lgbm(spacy_univ_train,labels,param=param,num_round=100)


pred=(lgbmodel.predict(spacy_univ_test)>0.5)*1

ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=pred
ensemble_out.to_csv('spacy_lgbm.csv')




























