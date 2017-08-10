import pandas as pd
import numpy as np
import gensim
import re 
import pickle

def word2vec(model, text_list):
    word2vec_dims = 300
    vector_data = np.zeros((1, word2vec_dims)) # stupid

    for word in text_list:
        try:
            element_vector = model.wv[word]
        except Exception as e:
            print("{} is undefined key.".format(word))
            element_vector = np.zeros((1, word2vec_dims))

        vector_data = np.vstack([vector_data, element_vector])

    vector_data = np.delete(vector_data, 0, 0)

    return vector_data


# load dataset
data = pd.read_csv('./dataset.csv')
data = data[['text', 'sentiment']]

# exclude Neutral text
data = data[data.sentiment != "Neutral"]

# normalization
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print("Loading Word2Vec model...(This take a while)")
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
# others: http://qiita.com/Hironsan/items/8f7d35f0a36e0f99752c

print("Converting text to vector...")
text_vector = []
for text in data['text']:
    text_list = text.split()
    vector = word2vec(word2vec_model, text_list)
    text_vector.append(vector)

print("Dumping text vector as pickle")
pickle.dump(text_vector, open("./text_vector.pkl", "wb"), -1)


