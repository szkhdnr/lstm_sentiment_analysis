import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

import pandas as pd
import numpy as np
import gensim
import re 
import pickle


dataset = pd.read_csv('../dataset.csv')
dataset = dataset[['text', 'sentiment']]

# teacher
y = np.array([])

for sentiment in dataset['sentiment']:
    if sentiment == "Positive":
        y = np.append(y, 1)
    else:
        y = np.append(y, 0)

# data
x = pickle.load(open('./text_vector.pkl', 'rb'))

# padding
# note: should set dtype. otherwise, all values become zero.
x = pad_sequences(x, maxlen=140, dtype='float32')

model = Sequential()
model.add(LSTM(1200, input_shape=(140, 300), dropout_U=0.2, dropout_W=0.2, return_sequences=True))
model.add(LSTM(196, input_shape=(140, 300), dropout_U=0.2, dropout_W=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=15, batch_size=32, verbose=2)


json_string = model.to_json()
json_file = open('model.json', 'w')
json_file.write(json_string)
json_file.close()

model.save_weights('param.hdf5')
