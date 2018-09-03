import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier

import flashtext
import re
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


class textMLP(nn.Module):
    def __init__(self,array_size=910,n_classes=2):
        super(textMLP,self).__init__()

#        self.X=array
        self.lin1=nn.Linear(array_size,1000)
        self.lin2=nn.Linear(1000,500)
        self.lin3=nn.Linear(500,100)
        self.lin4=nn.Linear(100,n_classes)
    def forward(self,x):
        x=F.relu(self.lin1(x))
        x=F.dropout(x,0.5)
        x=F.relu(self.lin2(x))
        x=F.dropout(x,0.3)
        x=F.relu(self.lin3(x))
        x=F.dropout(x,0.2)
        x=F.softmax(self.lin4(x),dim=1)
        return x


def universal_sent_embeddings(text_series):
    """
    
    Taking too much time to process but produces good
    quality word embeddings using universal sentance embeddings
    """
    
    test_decriptions_list=text_series.tolist()
    
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      test_descriptions_embeddings = session.run(embed(test_decriptions_list))
    return test_descriptions_embeddings

def spacy_vectors(tweets,custom_nlp):
    res=list()
    for tw in tqdm(tweets):
        res.append(custom_nlp(tw).vector)
    return res



if __name__=='__main__':
    model_path=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model'
    custom_nlp=spacy.load(model_path)



    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
    
    train_mat=universal_sent_embeddings(tweet_df['tweet'])
    np.save(open(r'universal_train.npy','wb'),train_mat)
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
    
    test_mat=universal_sent_embeddings(tweet_df['tweet'])
    np.save(open(r'universal_test.npy','wb'),test_mat)
    train_vect=spacy_vectors(tweet_df['tweet'])
    test_vect=spacy_vectors(test_df['tweet'])

    train_vect=np.array(train_vect)
    test_vect=np.array(test_vect)
    np.save(open(r'spacy_train50.npy','wb'),train_vect)
    np.save(open(r'spacy_test50.npy','wb'),test_vect)
    
    
    spacy_train_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_train.npy','rb'))
    spacy_test_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_test.npy','rb'))
    
    
    spacy_univ_train=np.hstack([train_mat,spacy_train_mat])
    spacy_univ_test=np.hstack([test_mat,spacy_test_mat])




    net = NeuralNetClassifier(
        textMLP,
        max_epochs=200,
        lr=0.00025,
        optimizer=optim.Adam,
    #    criterion=nn.BCELoss,
    )
    off_words_cv=CountVectorizer()
    off_words_cv.fit(list(kp.get_all_keywords().keys()))
    url_cv=CountVectorizer()
    url_cv.fit(['fb','instagr','google','bit','tmblr','youtu','goo','ebay'])
    train=create_features(tweet_df['tweet'],off_words_cv,url_cv)
    train.to_csv('train_features.csv')
    test=create_features(test_df['tweet'],off_words_cv,url_cv)
    test.to_csv('test_features.csv')
    
    train=pd.read_csv('train_features.csv')
    test=pd.read_csv('test_features.csv')
    train_comb=np.hstack([spacy_univ_train,train.as_matrix().astype('float32')])
    test_comb=np.hstack([spacy_univ_test,test.as_matrix().astype('float32')])
    
    
    net.fit(train_comb,tweet_df['label'])
    submission=pd.DataFrame()
    submission['id']=test_df['id']
    submission['label']=net.predict(test_comb)
    submission.to_csv('submission.csv')












