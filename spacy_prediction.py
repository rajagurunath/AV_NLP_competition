# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:51:32 2018

@author: Gurunath
"""
import spacy
import pandas as pd
from pathlib import Path
import thinc.extra.datasets
#import spacy
from spacy.util import minibatch, compounding
from spacy.util import decaying
dropout = decaying(0.6, 0.2, 1e-4)
from tqdm import tqdm


def get_batches(train_data, model_type):
    max_batch_sizes = {'tagger': 32, 'parser': 16, 'ner': 16, 'textcat': 64}
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, 1.001)
    batches = minibatch(train_data, size=batch_size)
    return batches


def train_CNN(model=None, output_dir=None, n_iter=20, n_texts=2000):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('POSITIVE')

    # load the IMDB dataset
    print("Loading IMDB data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
#            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            batches=get_batches(train_data,'textcat')
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=next(dropout),
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test_text = "This movie sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def load_data(limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
#    train_data, _ = thinc.extra.datasets.imdb()
    train_data=tweet_df['tweet'].values
#    random.shuffle(train_data)
#    train_data = train_data[-limit:]
    texts, labels = tweet_df['tweet'].values,tweet_df['label'].values
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

def spacy_vectors(tweets,custom_nlp):
    res=list()
    for tw in tqdm(tweets):
        res.append(custom_nlp(tw).vector)
    return res
def spacy_load_and_predict(tweets,model_path):
    custom_nlp=spacy.load(model_path)
    predictions=[]
    for tw in tqdm(tweets):
        test=custom_nlp(tw)
    #    print(tw[:10], test.cats)
        predictions.append(test.cats)
    df=pd.DataFrame(predictions)
    return df['POSITIVE'].values

def spacy_predictions(tweets,custom_nlp):
    res=list()
    for tw in tqdm(tweets):
        res.append(custom_nlp(tw).cats)
    return res


if __name__=='__main__':
    
    model_path=r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\spacy_trained_model'
    tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
    test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')
    train=False
    
    """
    Train
    score :0.886151405449651
    """
    
    if train:
        print('Training starts')
        print('='*50)
        model_dir=r'F:\anaconda\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.0.0'
        out_dir=model_path
        train_CNN(model=model_dir,output_dir=out_dir,n_iter=50)
        print('Training finished')
        

    pred=spacy_load_and_predict(test_df['tweet'],model_path)
    ensemble_out=pd.DataFrame()
    ensemble_out['id']=test_df['id']
    ensemble_out['label']=(pred>0.7)*1
    ensemble_out.to_csv('spacy_50_epoch.csv')




















