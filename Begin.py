import pandas as pd
import numpy as np
import os
import pickle


train_path='./train.csv'
dataframe=pd.read_csv(train_path)

X_train=dataframe['comment_text'].fillna('<UNK>').values

tokenizer=load_create_tokenizer(X_train,None,True)
X_train=load_create_padded_data(X_train=X_train,savetokenizer=False,isPaddingDone=False,maxlen=maxlen,tokenizer_path='./New_Tokenizer.tkn')
word_index=tokenizer.word_index
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,300,'./../glove.840B.300d.txt',False,True,'./Emb_Mat.mat')