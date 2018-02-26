from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def custom_sort(a,b):
	#assume first index is word and second is frequencyz,in descending order
	if a[1] <= b[1]:
		return 1;
	return -1


'''
Given a word_index dictionary, extract words, with fequency >= K, or extract top K words
return the dictionaries, one like word_index and one like  word_counts
'''
def getTopK_words(tokenizer,k,topk=False):
	word_index=tokenizer.word_index
	word_counts=tokenizer.word_counts
	word_freq_list=zip(word_counts.keys(),word_counts.values())
	word_freq_list.sort(custom_sort)
	dictionary={}
	unk_tok = -1
	if not topk:#Extract words with frequency >= K
		for word,frequency in word_freq_list:
			if frequency >= k:
				dictionary[word]=frequency
			else:
				if unk_tok == -1:
					unk_tok = word_index[word]
	else:
		while(k>0):#Extract top K words
			word,frequency=word_freq_list[k]
			dictionary[word]=frequency
			k=k-1
		#get a raandom unknown word
		for word in word_index.keys():
			if word not in dictionary.keys():
				unk_tok = word_index[word]
				break
	return dictionary,unk_tok





'''
for i,sentences in enumerate(X_train):
    for j,indexes in enumerate(sentences):
    	if indexes not in word_index_1.values():
            X_train[i,j] = len(word_index_1)+1
'''

