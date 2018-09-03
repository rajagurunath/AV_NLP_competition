# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:35:12 2018

@author: Gurunath
"""

from torch import nn
from torchtext import data,datasets
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
import torch.optim as optim

class TweetDataset(Dataset):
    """Loads tweet dataset"""

    def __init__(self, csv_file,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional):transform to be applied
                on a sample.
        """
        self.tweet_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.tweet_df)

    def __getitem__(self, idx):
        instance = self.tweet_df.iloc[idx,2]
        if self.transform:
            instance = self.transform.fit(instance)
        return instance

class textMLP(nn.Module):
    def __init__(self,array_size=512,n_classes=2):
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
        x=F.softmax(self.lin4(x))
        return x
    

tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')

name='ensemble'
with open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\{}.pkl'.format(name),'rb') as f:
    tfidf=pickle.load(f)
only_tfidf=tfidf.transform(tweet_df['tweet']).toarray()
text_clf=textMLP(only_tfidf.shape)


adam_optimizer=optim.Adam(text_clf.parameters())
criterian=nn.BCELoss()



train_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_train.npy','rb'))
test_mat=np.load(open(r'F:\E\Learning_DL_fastai\competition\NLP_data\model_files\universal_test.npy','rb'))
tweet_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\train_2kmZucJ.csv')
test_df=pd.read_csv(r'F:\E\Learning_DL_fastai\competition\NLP_data\test_oJQbWVk.csv')

ensemble_train_df=pd.read_csv('ensemble_train_df.csv')
ensemble_test_df=pd.read_csv('ensemble_test_df.csv')

from skorch import NeuralNetClassifier

net = NeuralNetClassifier(
    textMLP,
    max_epochs=200,
    lr=0.00025,
    optimizer=optim.Adam,
#    criterion=nn.BCELoss,
)

net.fit(train_mat,tweet_df['label'])
ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=net.predict(test_mat)
ensemble_out.to_csv('universal_mlp_torch.csv')



net.fit(ensemble_train_df.values.astype('float32'),tweet_df['label'])
ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=net.predict(ensemble_test_df.values.astype('float32'))
ensemble_out.to_csv('universal_mlp_torch.csv')

#for _ in range(10):

class textMLP(nn.Module):
    def __init__(self,array_size=526,n_classes=2):
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
        x=F.softmax(self.lin4(x))
        return x
    
class textMLP(nn.Module):
    def __init__(self,array_size=896,n_classes=2):
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



univ_feat_train=np.hstack([train_mat,ensemble_train_df.values.astype('float32')])
univ_feat_test=np.hstack([test_mat,ensemble_test_df.values.astype('float32')])



net.fit(spacy_univ_train,tweet_df['label'])
ensemble_out=pd.DataFrame()
ensemble_out['id']=test_df['id']
ensemble_out['label']=net.predict(spacy_univ_test)
ensemble_out.to_csv('universal_mlp_torch.csv')
























