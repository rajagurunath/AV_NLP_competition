# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 08:52:32 2018

@author: Gurunath
"""
from data_analysis import tweet_df
import flashtext
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

kp=flashtext.KeywordProcessor()
kp.add_keyword('$&@*#')
kp.add_keyword('fuck')
kp.add_keyword('fucking')
kp.add_keyword('sucks')
kp.add_keyword('ass')
kp.get_all_keywords()
tweet_df.tweet.apply(lambda x :kp.extract_keywords(x))

tweet_df['offensive_words']=tweet_df.tweet.apply(lambda x :kp.extract_keywords(x))



tweet_df['offensive_words']=tweet_df['offensive_words'].apply(lambda x :','.join(x))


print(re.search("(?P<url>https?://[^\s]+)", myString).group("url"))

#sextract_urls=lambda x: re.search("(?P<url>https?://[^\s]+)",x).group("url")

def extract_urls(x):
    try:
        res=re.search("(?P<url>https?://[^\s]+)",x).group("url")
    except:
        pass
        return ''
    return res

#import contextlib
#contextlib.suppress('AttributeError')
tweet_df['urls']=tweet_df['tweet'].apply(extract_urls)

nlp=spacy.load(r'F:\anaconda\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.0.0')

tweet_df['spacy_sent']=tweet_df['tweet'].apply(lambda x :nlp(x).sentiment)
tweet_df.to_csv('features.csv',index=False)

#tweet_df[]

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
    
    
tweet_df['url_presence']=tweet_df['urls'].apply(UrlPresence)
"""
np.mean(tweet_df['url_presence']==tweet_df['label'])
Out[68]: 0.24545454545454545
"""
tweet_df['offwords_presence']=tweet_df['offensive_words'].apply(offensive_words_presence)

"""
np.mean(tweet_df['offwords_presence']==tweet_df['label'])
Out[72]: 0.7659090909090909

After adding sucks
Out[81]: 0.768560606060606
"""
tweet_df['offensive_words'].str.split(',')
s=pd.get_dummies(tweet_df['offensive_words'])


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

off_words_mat=cv.fit_transform(tweet_df['offensive_words']).toarray()
off_words_df=pd.DataFrame(data=off_words_mat,columns=cv.get_feature_names())

tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\features.csv')

tweet_df=pd.concat([tweet_df,off_words_df],axis=1)

tweet_df['urls_split']=tweet_df['urls'].str.split('/')

tokenizer=lambda x:x.split('//')
cv1=CountVectorizer(tokenizer=tokenizer,stop_words='english')

url_words_mat=cv1.fit_transform(tweet_df['urls']).toarray()
url_words_df=pd.DataFrame(data=url_words_mat,columns=cv1.get_feature_names())

cv2=CountVectorizer()
cv2.fit(['fb','instagr','google','bit','tmblr','youtu','goo','ebay'])

url_words_mat1=cv2.transform(tweet_df['urls']).toarray()
url_words_df1=pd.DataFrame(data=url_words_mat1,columns=cv2.get_feature_names())

s=pd.Series(cv1.get_feature_names()).value_counts()


tweet_df=pd.concat([tweet_df,url_words_df1],axis=1)



training_features_df=tweet_df.copy()

training_features_df.drop(['label','offensive_words','urls','spacy_sent',''])

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
blob = TextBlob("I love this library",analyzer=NaiveBayesAnalyzer())
#blob.sentiment

def create_features(tweets,tfidf,off_words_cv,url_cv):
    
    
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
    print(url_words_df)
    df.drop(['urls','offensive_words'],axis=1,inplace=True)
    print(df.columns)
    sentiment=df['tweet'].apply(textblob_sentiment)
    blob_df=pd.DataFrame(columns=['subjectivity','polarity'])
    blob_df['subjectivity']=sentiment.apply(lambda x:x.subjectivity)
    blob_df['polarity']=sentiment.apply(lambda x:x.polarity)
#    sentiment=df['tweet'].apply(textblob_sentiment_with_analyzer)
#    blob_df['class']=sentiment.apply(encode_pos_neg)
#    blob_df['p_neg']=sentiment.apply(lambda x:x.p_neg)
#    blob_df['p_pos']=sentiment.apply(lambda x:x.p_pos)
    
    tweets=df['tweet'].apply(clean_tweet)
    tfidf_df=pd.DataFrame(tfidf.transform(tweets).toarray())
    return pd.concat([off_words_df,url_words_df,blob_df,tfidf_df],axis=1)
    
def encode_pos_neg(x):
    if x.classification=='pos':
        return 0
    else:
        return 1
    
    
off_words_cv=CountVectorizer()
off_words_cv.fit(tweet_df['offensive_words'])
url_cv=CountVectorizer()
url_cv.fit(['fb','instagr','google','bit','tmblr','youtu','goo','ebay'])


from nltk import word_tokenize
from nltk.corpus import stopwords
import string

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

tfidf=TfidfVectorizer(stop_words='english')
tweets=tweet_df['tweet'].apply(clean_tweet)
tfidf.fit(tweets)
imp_words=pd.Series(index=tfidf.get_feature_names(),data=tfidf.idf_)

sam_df1=create_features(tweet_df['tweet'].values,tfidf,off_words_cv,url_cv)

sam_df1.to_csv('features_without_tweet.csv')

with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\baseline_tfidf_cleaned_text.pkl','wb') as f:
        pickle.dump(tfidf,f)

logistics=LogisticRegression()
#    logistics.fit(vect,df['label'])
cv_res=cross_validation(logistics,sam_df1,tweet_df['label'],cv=10)
plot_cv_res(cv_res)
logistics.fit(sam_df1,tweet_df['label'])
test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
df=pd.DataFrame()
df['id']=test_df['id'].values

test_feat=create_features(test_df['tweet'].values,tfidf,off_words_cv,url_cv)
df['label']=logistics.predict(test_feat)

df.to_csv('baseline2_without_tweet.csv')

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier



rf=RandomForestClassifier()
cv_res=cross_validation(rf,sam_df1,tweet_df['label'],cv=10)
plot_cv_res(cv_res)
rf.fit(sam_df1,tweet_df['label'])
df=pd.DataFrame()
df['id']=test_df['id'].values

#test_feat=create_features(test_df['tweet'].values,tfidf,off_words_cv,url_cv)
df['label']=rf.predict(test_feat)
df.to_csv('rf_baseline_without_tweet.csv')


gbc=GradientBoostingClassifier()
cv_res=cross_validation(gbc,sam_df1,tweet_df['label'],cv=10)
plot_cv_res(cv_res)
gbc.fit(sam_df1,tweet_df['label'])

df=pd.DataFrame()
df['id']=test_df['id'].values

#test_feat=create_features(test_df['tweet'].values,tfidf,off_words_cv,url_cv)
df['label']=gbc.predict(test_feat)
df.to_csv('gbc_baseline_withou_tweet.csv')


#textblob_sentiment=lambda x :TextBlob(x).sentiment
def textblob_sentiment(x):
    sent=TextBlob(x)
    return sent.sentiment #[sent.subjectivity,sent.polarity]

def textblob_sentiment_with_analyzer(x):
    sent=TextBlob("I love this library",analyzer=NaiveBayesAnalyzer())

    return sent.sentiment #[sent.subjectivity,sent.polarity]

a=tweet_df['tweet'].apply(textblob_sentiment)



import spacy
custom_nlp=spacy.load(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model')


predictions=[]
for tw in test_df['tweet']:
    test=custom_nlp(tw)
#    print(tw[:10], test.cats)
    predictions.append(test.cats)
spacy_pred=pd.DataFrame()
spacy_pred['id']=test_df['id']

spacy_pred['label']=(df['POSITIVE']>0.9) *1
spacy_pred.to_csv('spacy_pred.csv')


df=pd.DataFrame(predictions)
df.to_csv('spacy_proba.csv')
pred_filenames=['baseline_predictions','baseline2','baseline2_without_tweet',\
                'gbc_baseline','gbc_baseline_withou_tweet',\
                'rf_baseline','rf_baseline_without_tweet','spacy_pred','spacy_proba']

'features_without_tweet'

ensemble_df=pd.DataFrame()
csv_dir=r'F:\E\Learning_DL_fastai\competition\NLP_data'
for file in pred_filenames:
    print(file)
    tmp_df=pd.read_csv(csv_dir+'\{}.csv'.format(file))
    ensemble_df[file]=tmp_df['label']


















