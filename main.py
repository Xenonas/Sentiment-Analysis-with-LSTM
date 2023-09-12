import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import random
from basic_func import *
from models import *


documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

print("Got the training data...")

perc_tr = 0.70
perc_vl = 0.85
lim = int(perc_tr*len(documents))
lim_v = int(perc_vl*len(documents))
train = documents[:lim]
val = documents[lim: lim_v]
test = documents[lim_v:]

print("Started preprocessing...")
clean_data(train), clean_data(val), clean_data(test)
x, y, word_tokenizer, tag_tokenizer = preprocess_data(train)
x_val, y_val, word_tokenizer, tag_tokenizer = preprocess_data(val, word_tokenizer, tag_tokenizer)
x_test, y_test, word_tokenizer, tag_tokenizer = preprocess_data(test, word_tokenizer, tag_tokenizer)
inp = len(word_tokenizer.word_index) + 1
# its possible to use pretrained word embeddings as initialized weights in the model
"""
model_word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word_id = word_tokenizer.word_index
embedding_weights = np.zeros((inp, 300))
for word, i in word_id.items():
    if i >= 2000:
        continue
    if word in list(model_word2vec.index_to_key):
        embedding_weights[i] = model_word2vec.word_vec(word)
"""
print("Finished preprocessing!")

lstm_model(inp, x, y, x_val, y_val, x_test, y_test)