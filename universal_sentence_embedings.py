# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:08:58 2018

@author: Gurunath
"""

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

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


if __name__=='__main__':
    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
    
    train_mat=universal_sent_embeddings(tweet_df['tweet'])
    np.save(open(r'universal_train.npy','wb'),train_mat)
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
    
    test_mat=universal_sent_embeddings(tweet_df['tweet'])
    np.save(open(r'universal_test.npy','wb'),test_mat)




















