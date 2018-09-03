# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:09:12 2018

@author: Gurunath
"""

import flashtext
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import pickle

kp=flashtext.KeywordProcessor()
kp.add_keyword('$&@*#')
kp.add_keyword('fuck')
kp.add_keyword('fucking')
kp.add_keyword('sucks')
kp.add_keyword('ass')
kp.get_all_keywords()
def extract_urls(x):
    try:
        res=re.search("(?P<url>https?://[^\s]+)",x).group("url")
    except:
        pass
        return ''
    return res
def UrlPresence(X):
    if X!=None:
        return 1
    else:
        return 0
def offensive_words_presence(X):
    if len(X)!=0:
        return 1
    else:
        return 0
def encode_pos_neg(x):
    if x.classification=='pos':
        return 0
    else:
        return 1
def clean_tweet(x):
#    w_list=word_tokenize(x)
#    table = str.maketrans(" ",string.punctuation)

    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', x)
#    text=text.translate(table, string.punctuation)
    text=''.join(ch for ch in text if ch not in string.punctuation)
    text=''.join(ch for ch in text if not ch.isdigit())
#    pattern = re.compile(r'\b(' + r'|'+string.punctuation + r')\b\s*')
#    text = pattern.sub('', str(x))
    return text
def textblob_sentiment(x):
    sent=TextBlob(x)
    return sent.sentiment #[sent.subjectivity,sent.polarity]


def create_features(tweets,off_words_cv,url_cv):
    
    
    df=pd.DataFrame()
    df['tweet']=tweets
    df['no_of_words']=df['tweet'].apply(lambda x :len(x.split(' ')))

    df['offensive_words']=df.tweet.apply(lambda x :kp.extract_keywords(x))
    df['offensive_words']=df['offensive_words'].apply(lambda x :','.join(x))
    df['offwords_presence']=df['offensive_words']\
    .apply(offensive_words_presence)
    off_words_mat=off_words_cv.transform(df['offensive_words']).toarray()
#    print(off_words_mat)
    off_words_df=pd.DataFrame(data=off_words_mat,columns=off_words_cv.get_feature_names())

    df['urls']=df['tweet'].apply(extract_urls)
    df['url_presence']=df['urls'].apply(UrlPresence)
    
    url_words_mat=url_cv.transform(df['urls']).toarray()
    url_words_df=pd.DataFrame(data=url_words_mat,columns=url_cv.get_feature_names())
#    print(url_words_df)
#    df.drop(['urls','offensive_words'],axis=1,inplace=True)
#    print(df.columns)
    sentiment=df['tweet'].apply(textblob_sentiment)
    blob_df=pd.DataFrame(columns=['subjectivity','polarity'])
    blob_df['subjectivity']=sentiment.apply(lambda x:x.subjectivity)
    blob_df['polarity']=sentiment.apply(lambda x:x.polarity)
#    sentiment=df['tweet'].apply(textblob_sentiment_with_analyzer)
#    blob_df['class']=sentiment.apply(encode_pos_neg)
#    blob_df['p_neg']=sentiment.apply(lambda x:x.p_neg)
#    blob_df['p_pos']=sentiment.apply(lambda x:x.p_pos)
#    if do_text_clean:
#        tweets=df['tweet'].apply(clean_tweet)
#    else:
#        tweets=df['tweet']
#    if need_tfidf:
#        tfidf_df=pd.DataFrame(tfidf.transform(tweets).toarray())
#        res=pd.concat([off_words_df,url_words_df,blob_df,tfidf_df],axis=1)
#    else:
    res=pd.concat([off_words_df,url_words_df,blob_df],axis=1)
    return res


if __name__=='__main__':
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')

    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
#    tfidf,_=fit_tfidf(tweet_df['tweet'],'complete_text')
    off_words_cv=CountVectorizer()
    off_words_cv.fit(list(kp.get_all_keywords().keys()))
    url_cv=CountVectorizer()
    url_cv.fit(['fb','instagr','google','bit','tmblr','youtu','goo','ebay'])
    train=create_features(tweet_df['tweet'],off_words_cv,url_cv)
    train.to_csv('train_features.csv')
    test=create_features(test_df['tweet'],off_words_cv,url_cv)
    test.to_csv('test_features.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
