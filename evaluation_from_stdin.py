import gensim
import keras
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle


def word2vec(model, text_list):
    word2vec_dims = 300
    vector_data = np.zeros((1, word2vec_dims))

    for word in text_list:
        try:
            element_vector = model.wv[word]
        except Exception as e:
            print("{} is undefined key.".format(word))
            element_vector = np.zeros((1, word2vec_dims))

        vector_data = np.vstack([vector_data, element_vector])

    vector_data = np.delete(vector_data, 0, 0)

    return vector_data

print("Loading Word2Vec model...(This take a while)")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

json_file = open('./model.json', 'r')
json_str = json_file.read()

model = keras.models.model_from_json(json_str)
model.load_weights('./param.hdf5')


text = ""
print("type 'quit' to exit.")

while True:
	text = raw_input()
	if text == "quit":
		break

	text = text.lower()
	text = re.sub('[^a-zA-z0-9\s]', '', text)

        text_list = text.split()

        text_vector = word2vec(word2vec_model, text_list)
        text_vector = text_vector.reshape(1, -1, 300)

	pad_text = pad_sequences(text_vector, maxlen=140, dtype='float32')

	predict_result = model.predict_classes(pad_text)
        print(predict_result)

	if predict_result[0] == 1:
		print("Positive")
	else:
		print("Negative")




