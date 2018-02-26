from New_Utils import *
import pickle

#Loadd padded data
f=open('New_X_train_3000.trn','r')
X_train=pickle.load(f)
f.close()
#Load Tokenizer
f=open('New_Tokenizer_3000.tkn','r')
tokenizer=pickle.load(f)
f.close()
#Load Embedding Matrix
f=open('Emb_Mat_3000.mat','r')
embedding_matrix=pickle.load(f)
f.close()
#Load word_index
word_index=tokenizer.word_index
maxlen = 80

def create_batched_y(batch_size,sentences,maxlen,word_index):
	output_vector=np.zeros((batch_size, maxlen, len(word_index) + 1))# +1 beacause indexing begins with 1
	for i,sentence in enumerate(sentences):
		for j, word in enumerate(sentence):
			output_vector[i, j, word] = 1
	return output_vector


def my_gen(X_train, batch_size, maxlen, word_index):
	while 1:
		total_length = len(X_train)
		last_index=-1
		for i in range(total_length//batch_size + 1):
			start_index = last_index + 1
			if start_index >= total_length:
				break
			remaining_data = total_length - last_index
			last_index = start_index + batch_size-1 #determine correct length
			if remaining_data <= (batch_size-1):
				last_index = total_length - 1
			current_batch_x = X_train[start_index:last_index+1]
			current_batch_y = create_batched_y(len(current_batch_x), current_batch_x, maxlen, word_index)
			#Stop Printing, annoying keras calls
			yield (current_batch_x, current_batch_y)



from keras.layers import *
from keras.layers.recurrent import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.pooling import *
from keras.models import *

model=Sequential()
model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_shape = (maxlen,),trainable = True))
model.add(Dropout(0.4))
#Create a summary vector
model.add(LSTM(512, name = 'Encoder', dropout=0.3, recurrent_dropout=0.3))
model.add(LeakyReLU())
#repeat this thought vector
model.add(RepeatVector(maxlen))
model.add(LSTM(512, return_sequences = True, name='Decoder',recurrent_dropout=0.3,dropout=0.3))
model.add(LeakyReLU())
model.add(Dense(len(word_index) + 1, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics =['acc'])

#model.fit_generator(my_gen(X_train, maxlen, word_index), steps_per_epoch = len(X_train)//64)
checkpoint=ModelCheckpoint(filepath='Mine_AE_'+os.sep+"weights.AE.h5",monitor='loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto',period=1)
earlystopping=EarlyStopping(monitor='loss',patience=10)



model.fit_generator(my_gen(X_train, 32, 80, word_index), steps_per_epoch = 100, epochs = 100, callbacks = [earlystopping, checkpoint])