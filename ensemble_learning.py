# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 19:33:35 2018

@author: Gurunath
"""

from feature_eng_cleaned import *
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import spacy
import xgboost
def model_fit(model,X,y):
    model.fit(X,y)
    return model

def model_predict(model,X):
    return model.predict(X)


def prepare_datasets(tweet_df,off_words_cv,url_cv):
    tfidf,only_tfidf=fit_tfidf(tweet_df['tweet'],'ensemble')
    tfidf_with_features=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv)
#    tfidf_with_features.to_csv('tfidf_with_features.csv')
    tfidf_with_cleanedfeatures=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv,do_text_clean=True)
#    tfidf_with_cleanedfeatures.to_csv('tfidf_with_cleanedfeatures.csv')
    only_features=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv,need_tfidf=False)
    print('datasets prepared')
    
    return only_tfidf,tfidf_with_features,tfidf_with_cleanedfeatures,only_features

def prepare_test_datasets(tweet_df,off_words_cv,url_cv):
    name='ensemble'
    with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\{}.pkl'.format(name),'rb') as f:
        tfidf=pickle.load(f)
    only_tfidf=tfidf.transform(tweet_df['tweet']).toarray()
#    tfidf,only_tfidf=fit_tfidf(tweet_df['tweet'],'ensemble')
    tfidf_with_features=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv)
#    tfidf_with_features.to_csv('tfidf_with_features.csv')
    tfidf_with_cleanedfeatures=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv,do_text_clean=True)
#    tfidf_with_cleanedfeatures.to_csv('tfidf_with_cleanedfeatures.csv')
    only_features=create_features(tweet_df['tweet'],tfidf,off_words_cv,url_cv,need_tfidf=False)
    print('datasets prepared')
    
    return only_tfidf,tfidf_with_features,tfidf_with_cleanedfeatures,only_features


#r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model'
def spacy_load_and_predict(tweets,model_path):
    custom_nlp=spacy.load(model_path)
    predictions=[]
    for tw in tweets:
        test=custom_nlp(tw)
    #    print(tw[:10], test.cats)
        predictions.append(test.cats)
    df=pd.DataFrame(predictions)
#    df.to_csv('spacy_proba.csv')

#    spacy_pred=pd.DataFrame()
#    spacy_pred['id']=test_df['id']
#    
#    spacy_pred['label']=(df['POSITIVE']>0.9) *1
#    spacy_pred.to_csv('spacy_pred.csv')
    return df['POSITIVE'].values
#def test_spacy():
#    model_path=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_modelv1'
#    custom_nlp2=spacy.load(model_path)
#    for _ in range(10):
#        ridx=np.random.randint(0,7919,size=(1,))[0]
#        print('text: ',tweet_df['tweet'][ridx])
#        print('groud_truth: ',tweet_df['label'][ridx])
#        print(custom_nlp2(tweet_df['tweet'][ridx]).cats['POSITIVE'])
#        print(custom_nlp(tweet_df['tweet'][ridx]).cats['POSITIVE'])
#        
#predv1=spacy_load_and_predict(test_df.tweet,model_path) 
#spacy_pred=pd.DataFrame()
#spacy_pred['id']=test_df['id']
#
#spacy_pred['label']=(predv1>0.8) *1
#spacy_pred.to_csv('spacy_predv1.csv')

#tfidf_with_features=create_features()
#tfidf,_=fit_tfidf(tweet_df['tweet'],'complete_text')
    
if __name__=='__main__':
    """
    Forming three datasets
    """
    save_dir=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\ensemble_models'
    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')

    off_words_cv=CountVectorizer()
    off_words_cv.fit(list(kp.get_all_keywords().keys()))
    url_cv=CountVectorizer()
    url_cv.fit(['fb','instagr','google','bit','tmblr','youtu','goo','ebay'])
    
    
    """
    Model-Training
    
    """
    logclf=LogisticRegression()
    rf=RandomForestClassifier()
    gbc=GradientBoostingClassifier()
    
    models=[logclf,rf,gbc]
    model_name=['logclf','rf','gbc']
    
#    datasets=[only_tfidf,tfidf_with_features,
#              tfidf_with_cleanedfeatures,only_features]
    train_datasets=list(prepare_datasets(tweet_df,off_words_cv,url_cv))
#    dataset_name=['only_tfidf','tfidf_with_features',
#                  'tfidf_with_cleanedfeatures','only_features']
    ensemble_train_df=pd.DataFrame()
    for X,dat_name in zip(train_datasets,dataset_name):
        for model,mod_name in zip(models,model_name):
            fitted_model=model_fit(model,X,tweet_df['label'])
            pickle.dump(fitted_model,open(save_dir+'\{}_{}.pkl'.format(mod_name,dat_name),'wb'))
            ensemble_train_df['{}_{}'.format(mod_name,dat_name)]=\
            model_predict(fitted_model,X)
    model_path1=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model'
    model_path2=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_modelv1'
    ensemble_train_df['spacy_v0']=spacy_load_and_predict(tweet_df['tweet'],model_path1)
    ensemble_train_df['spacy_v1']=spacy_load_and_predict(tweet_df['tweet'],model_path2)
    ensemble_train_df.to_csv('ensemble_train_df.csv',index=False)
    """
    Model-predict
    """
    test_datasets=list(prepare_test_datasets(test_df,off_words_cv,url_cv))
    ensemble_test_df=pd.DataFrame()
    for X,dat_name in zip(test_datasets,dataset_name):
        for model,mod_name in zip(models,model_name):
#            fitted_model=model_fit(model,X,tweet_df['label'])
            fitted_model=pickle.load(open(save_dir+'\{}_{}.pkl'.format(mod_name,dat_name),'rb'))
            ensemble_test_df['{}_{}'.format(mod_name,dat_name)]=\
            model_predict(fitted_model,X)
    ensemble_test_df['spacy_v0']=spacy_load_and_predict(test_df['tweet'],model_path1)
    ensemble_test_df['spacy_v1']=spacy_load_and_predict(test_df['tweet'],model_path2)
    ensemble_test_df.to_csv('ensemble_test_df.csv',index=False)
    
    
    """
    Ensemble-model
    (meta learner)
    """
    ensemble_train_df=pd.read_csv('ensemble_train_df.csv')
    ensemble_test_df=pd.read_csv('ensemble_test_df.csv')
#    logclf.fit(ensemble_train_df,tweet_df['label'])
    gbc.fit(ensemble_train_df,tweet_df['label'])
    
    xgb=xgboost.XGBClassifier(max_depth=3,n_estimators=500)
    xgb.fit(ensemble_train_df,tweet_df['label'])
    ensemble_out=pd.DataFrame()
    ensemble_out['id']=test_df['id']
    ensemble_out['label']=xgb.predict(ensemble_test_df)
    ensemble_out.to_csv('ensemble_out.csv')

    
#pred_filenames=['baseline_predictions','baseline2','baseline2_without_tweet',\
#                'gbc_baseline','gbc_baseline_withou_tweet',\
#                'rf_baseline','rf_baseline_without_tweet','spacy_pred','spacy_proba']
#
#'features_without_tweet'




#csv_dir=r'F:\E\Learning_DL_fastai\competition\NLP_data'
#for file in pred_filenames:
#    print(file)
#    tmp_df=pd.read_csv(csv_dir+'\{}.csv'.format(file))
#    ensemble_df[file]=tmp_df['label']


#mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_train.npy','rb'))
#logclf=LogisticRegression()
#logclf.fit(mat,tweet_df['label'])
#test_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_test.npy','rb'))
#log_pred=logclf.predict(test_mat)
#
#xgb=xgboost.XGBClassifier(max_depth=3,n_estimators=500)
#
#xgb.fit(mat,tweet_df['label'])
#pred=xgb.predict(test_mat)
#
#
#
#ensemble_out=pd.DataFrame()
#ensemble_out['id']=test_df['id']
#ensemble_out['label']=pred
#ensemble_out.to_csv('universal_xgb.csv')















