# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 21:11:29 2018

@author: Gurunath
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,make_scorer
from sklearn.decomposition import PCA
from yellowbrick.classifier import ROCAUC ,ClassificationReport,ConfusionMatrix

color = sns.color_palette()

##plt.style.use('fivethirtyeight')
#
#tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
#
#tweet_df.head()
#
#"""
#Out[12]: Index(['id', 'label', 'tweet'], dtype='object')
#"""
#
#label_counts=tweet_df['label'].value_counts()
#plt.bar(label_counts.index,label_counts.values)
#plt.title('Label distribuition')
#plt.xticks([0,1])
#plt.ylabel('counts')
#plt.xlabel('sentiment')
#plt.show()
#
#tweet_df['no_of_words']=tweet_df['tweet'].apply(lambda x :len(x.split(' ')))
#
#cnt_words=tweet_df['no_of_words'].value_counts()
#
#plt.figure(figsize=(12,6))
#sns.barplot(cnt_words.index, cnt_words.values, alpha=0.8, color=color[0])
#plt.ylabel('Number of Occurrences', fontsize=12)
#plt.xlabel('Number of words in the tweet', fontsize=12)
#plt.xticks(rotation='vertical')
#plt.show()

score_fn=make_scorer(f1_score,average='weighted')

def dimesionality_reduction(vect):
    pca=PCA(n_components=2)
    d2_array=pca.fit_transform(vect)
    return pca,d2_array

def plot_reduced_dimension(d2_array,label):
    plt.scatter(d2_array[:,0],d2_array[:,1],c=label,cmap='viridis')
#    plt
    plt.colorbar()
    plt.title('2 D array')
    plt.show()

def cross_validation(model,x,y,cv=3):
    cv_res=cross_validate(model,x,y,return_train_score=True,scoring=score_fn,cv=cv)
    return cv_res
def plot_cv_res(cv_res):
    plt.plot(cv_res['test_score'])
    plt.title('Test score')
    plt.show()
    
    plt.plot(cv_res['train_score'])
    plt.title('Train score')
    plt.show()
    
    
    return

def baseline_model(df):
    tfidf=TfidfVectorizer(stop_words='english')
    vect=tfidf.fit_transform(df['tweet'])
    with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\baseline_logistics.pkl','wb') as f:
        pickle.dump(tfidf,f)
    logistics=LogisticRegression()
#    logistics.fit(vect,df['label'])
    cv_res=cross_validation(logistics,vect,tweet_df['label'])
    plot_cv_res(cv_res)
    return logistics.fit(vect,tweet_df['label']),vect


#log,vect=baseline_model(tweet_df)

def predict_test_values(test_df,model,transformer):
    vect=transformer.transform(test_df['tweet'])
    predictions=model.predict(vect)
    df=pd.DataFrame()
    df['id']=test_df['id'].values
    df['label']=predictions
    return  df          #pd.DataFrame(data=[test_df['id'].values,predictions],columns=['id,label'],index=False)




#test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
#test_df.columns
#"""
#Out[58]: Index(['id', 'tweet'], dtype='object')
#"""
#
#
#
#
#with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\baseline_logistics.pkl','rb') as f:
#        tfidf=pickle.load(f)
#
#pred_df=predict_test_values(test_df,log,tfidf)
#
#pred_df.to_csv('baseline_predictions.csv',index=False)
#
#pca,arr=dimesionality_reduction(vect.toarray())
#plot_reduced_dimension(arr,tweet_df['label'])
#
#visualizer = ROCAUC(log)
#visualizer.score(vect,tweet_df['label'])
#visualizer.poof()
#
##visualizer = ClassificationReport(log)
##
##visualizer.fit(X_train, y_train)
##visualizer.score(X_test, y_test)
##visualizer.poof()
#
#
#feature_series=pd.Series(index=tfidf.get_feature_names(),data=tfidf.idf_)



#from yellowbrick.features import JointPlotVisualizer
#
#visualizer = JointPlotVisualizer(feature='pca 1', target='pca 2')
#visualizer.fit(arr[:,0], arr[:,1])
#visualizer.poof()



if __name__=='__main__':
    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
    tweet_df.head()
    label_counts=tweet_df['label'].value_counts()
    plt.bar(label_counts.index,label_counts.values)
    plt.title('Label distribuition')
    plt.xticks([0,1])
    plt.ylabel('counts')
    plt.xlabel('sentiment')
    plt.show()
    tweet_df['no_of_words']=tweet_df['tweet'].apply(lambda x :len(x.split(' ')))
    cnt_words=tweet_df['no_of_words'].value_counts()
    plt.figure(figsize=(12,6))
    sns.barplot(cnt_words.index, cnt_words.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Number of words in the tweet', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()
    log,vect=baseline_model(tweet_df)    
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
    with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\baseline_logistics.pkl','rb') as f:
            tfidf=pickle.load(f)
    pred_df=predict_test_values(test_df,log,tfidf)
    pred_df.to_csv('baseline_predictions.csv',index=False)
    pca,arr=dimesionality_reduction(vect.toarray())
    plot_reduced_dimension(arr,tweet_df['label'])
    visualizer = ROCAUC(log)
    visualizer.score(vect,tweet_df['label'])
    visualizer.poof()
    feature_series=pd.Series(index=tfidf.get_feature_names(),data=tfidf.idf_)
    
    

























