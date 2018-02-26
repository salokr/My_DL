from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from My_TP import *

dataframe = pd.read_csv('train.csv')

X_train=dataframe['comment_text'].fillna('<UNK>').values

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

word_index_1 = getTopK_words(tokenizer, 3000, True)


my_dict={}
for index, word in enumerate(word_index_1.keys()):
	my_dict[word] = index + 1

#Add a unknown token
my_dict['<UNK>'] = len(my_dict) + 1


#change the X_train also
for i, sentences in enumerate(X_train):
	in_tokens = keras.preprocessing.text.text_to_word_sequence(sentences)
	for j, tokens in enumerate(in_tokens):
		if my_dict.get(tokens) is None:#We have an unknown word
			in_tokens[j] = '<UNK>'
	new_sen = ' '.join(in_tokens)
	X_train[i] = new_sen

#Assume now that we have fresh X_train and do the rest of things from here :D


new_tokenizer = Tokenizer()
new_tokenizer.fit_on_texts(X_train)
X_train = new_tokenizer.texts_to_sequences(X_train)
word_index = new_tokenizer.word_index

maxlen = 80
X_train = pad_sequences(X_train, maxlen=maxlen)



X_train=load_create_padded_data(X_train=X_train,savetokenizer=False,isPaddingDone=False,maxlen=maxlen,tokenizer_path='./New_Tokenizer_3000.tkn')
word_index=tokenizer.word_index
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,300,'./../glove.840B.300d.txt',False,True,'./Emb_Mat_3000.mat')

#Embedding matrix will have zero index for padding, the unknown will get something form mor,mal distribution of the glove and remaining less frequent will get <UNK> embedding from 1st index
#embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,300,'./../glove.840B.300d.txt',False,True,'./Emb_Mat_3000.mat')

