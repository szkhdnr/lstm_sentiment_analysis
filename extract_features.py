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
import gensim
import sys

def word2vec(model, text_list):
    word2vec_dims = 300
    vector_data = np.zeros((1, word2vec_dims))

    for word in text_list:
        try:
            element_vector = model.wv[word]
        except Exception as e:
            #print("{} is undefined key.".format(word))
            element_vector = np.zeros((1, word2vec_dims))

        vector_data = np.vstack([vector_data, element_vector])

    vector_data = np.delete(vector_data, 0, 0)

    return vector_data

target_file_list = sys.argv[1:]
print("target_file_list={}".format(target_file_list))

json_file = open('./model.json', 'r')
json_str = json_file.read()

model = keras.models.model_from_json(json_str)
model.load_weights('./param.hdf5')

print("Original network architecture")
model.summary()

out_model = keras.models.Model(input=model.input, output=model.get_layer('lstm_2').output)

print("Loading Word2Vec model...(This may take a while)")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
#word2vec_model = pickle.load(open('./word2vec_model.pkl', 'rb'))

for target_file in target_file_list:
    print("processing {}".format(target_file))
    extracted_data = np.empty((1, int(196) + 2))

    data =  pd.read_csv(target_file, header=None)
    total_num = data[data[0] != 2][0].size
    check_nums = 0

    for index, row in data.iterrows():
            text = row[5]
            gland_truth = row[0]
    
            # convert integer
            if gland_truth == 4:
                gland_truth = 1 # -> Pos
            
            if gland_truth == 2:
                continue # -> Neutral, dose not use.
    
            if gland_truth == 0:
                gland_truth = 0 # -> Neg
    
            check_nums += 1
            
            text = text.lower()
            text = re.sub('[^a-zA-z0-9\s]', '', text)
    
            text_list = text.split()
    
            text_vector = word2vec(word2vec_model, text_list)
            text_vector = text_vector.reshape(1, -1, 300)
    
            pad_text = pad_sequences(text_vector, maxlen=140, dtype='float32')
    
            extract_vector = out_model.predict(pad_text)
            
            tag = np.array([index])
            sentiment = np.array([gland_truth])
    
            extract_vector = np.concatenate((sentiment, extract_vector[0]))
            extract_vector = np.concatenate((tag, extract_vector))
    
            extracted_data = np.vstack([extracted_data, extract_vector])
    
    print("total_num={}".format(total_num))
    print("check_nums={}".format(check_nums))
    extracted_data = np.delete(extracted_data, 0, 0)
    np.save("{}_feature.npy".format(target_file), extracted_data)
    


