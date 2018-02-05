import pandas as pd
import numpy as np
import os
import pickle


def load_jigsaw():
	f=open('./Emb_Mat.mat','r')
	embedding_matrix=pickle.load(f)
	f.close()
	f=open('./New_Tokenizer.tkn','r')
	tokenizer=pickle.load(f)
	f.close()
	train_path='./train.csv'
	dataframe=pd.read_csv(train_path)
	X_train=dataframe['comment_text'].fillna('<UNK>').values
	labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
	Y_train=dataframe[labels].values
	maxlen=100
	word_index=tokenizer.word_index
	X_train=tokenizer.texts_to_sequences(X_train)
	from keras.preprocessing.sequence import pad_sequences
	X_train=pad_sequences(X_train,maxlen=maxlen)
	we_dim=300
	NUM_CLASSES=Y_train.shape[1]
	destination=raw_input('Plz Enter Model Name : ')
	destination = destination + '_jigsaw'
	if os.path.isdir(destination):
		shutil.rmtree(destination, ignore_errors=True)
	os.mkdir(destination)
	return embedding_matrix,tokenizer,dataframe,X_train,labels,Y_train,maxlen,word_index,we_dim,NUM_CLASSES,destination