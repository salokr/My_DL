from Load_Jigsaw import *
import numpy as np
import os
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.layers.recurrent import *
from keras.layers.pooling import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.callbacks import *
from keras.utils import *
import keras


embedding_matrix,tokenizer,dataframe,X_train,labels,Y_train,maxlen,word_index,we_dim,NUM_CLASSES,destination,checkpoint,earlystopping=load_jigsaw()
input_layer=Input(shape=(maxlen,),dtype='int32')
embedding_layer=Embedding(len(word_index)+1,we_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False)
emb_seq=embedding_layer(input_layer)

#An implementation of Hierarchical ConvNet

x=Conv1D(128,2,strides=1)(emb_seq)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
first_pool=GlobalMaxPooling1D()(x)
x=MaxPool1D(2)(x)
x=Conv1D(128,3,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
sec_pool=GlobalMaxPooling1D()(x)
x=MaxPool1D(2)(x)
x=Conv1D(128,4,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
third_pool=GlobalMaxPooling1D()(x)
x=MaxPool1D(2)(x)
x=Conv1D(128,4,strides=1)(x)
x=LeakyReLU()(x)
x=BatchNormalization()(x)
fourth_pool=GlobalMaxPooling1D()(x)

u=keras.layers.concatenate([first_pool,sec_pool,third_pool,fourth_pool],axis=-1)
x=Dense(64)(u)
x=LeakyReLU()(x)
x=Dense(NUM_CLASSES,activation='sigmoid')(x)

model=Model(inputs=input_layer,outputs=x)

from keras.utils import plot_model
plot_model(model,'Hierarchical_Conv_Net.py')

model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
model.fit(X_train,Y_train,epochs=51,verbose=1,callbacks=[earlystopping,checkpoint],batch_size=500,validation_split=.10)
