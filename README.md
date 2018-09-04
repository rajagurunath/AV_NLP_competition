# AV_NLP_competition

I am quite new to the AV competition ,this above code managed to get Rank 30 in the Public leaderboard and Rank 63 in private leaderboard  
I enjoyed the whole process of the competition !!!! 

## Approach Taken:

1. Tfidf + Logistic regression
2. Extarcted features with flashtext ,countvectorizer and used as features along with tfidf +
    logistic regression +RandomForest +Gradient boosting
3. Used an Universal sentence embedding  from tensorflow_hub via tensorflow as features +
    logistic regression+RandomForest +Gradient Boosting +lightgbm +xgbost
    
4. Ensembeling all the above models result and trained a Logistic regression on the training data
  and extracted the predictions for test data for all the above models and gave final predictions
  
5. Using spacy CNN clssifier for text classification : (got highscore among other models -due to
                                                        the usage of pretrained models in the spacy)
6. Extracting the wordvec vectors for each word and summing those words for a given sentences used as 
    features for above specified Algorithms(models)

7.Final score model : -using Spacy sentence vectors (mean embedings of word2vec) along with universal embedings 
  from Tensorflow +handcoded features with flashtext + countvectorizer (such as using separated columns for 
   vulgar words used in the tweets etc) totally got nearly 918 features approx.
8. For above built features applied a Multi-layer Perceptron from Torch along with SKORCH for easily fiting the 
    classifier .
    
## Future Plain (tried to implement) :
    * ULMFIT from Fastai for text classification 
    * ELMO 
    



