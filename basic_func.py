from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from nltk.stem import *


def clean_data(dataset):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                  'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                  "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                  'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                  'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                  'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                  'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
                  "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
                  "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                  'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
                  'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
                  "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    stm = PorterStemmer()
    for i in range(len(dataset)):
        temp = []
        for j in range(len(dataset[i][0])):
            if dataset[i][0][j] in stop_words or len(dataset[i][0][j])==1:
                continue
            temp.append(stm.stem(dataset[i][0][j]))
        dataset[i] = (temp, dataset[i][1])
    
def split_data(dataset):

    # we are gonna be splitting the dataset into the input sentences and the output corresponding tags
    x_data = []
    y_data = []
    for sentence in dataset:
        temp_x = []
        for word in sentence[0]:
            temp_x.append(word)
        x_data.append(temp_x)
        y_data.append(sentence[1])

    return x_data, y_data


def preprocess_data(data, word_tokenizer=None, tag_tokenizer=None):

    x_data, y_data = split_data(data)
    if word_tokenizer is None:
        word_tokenizer = Tokenizer(oov_token=True)
        word_tokenizer.fit_on_texts(x_data)
    x = word_tokenizer.texts_to_sequences(x_data)

    if tag_tokenizer is None:
        tag_tokenizer = Tokenizer()
        tag_tokenizer.fit_on_texts(y_data)
    y = tag_tokenizer.texts_to_sequences(y_data)

    x = pad_sequences(x, maxlen=2000, padding="pre", truncating="post")

    y = to_categorical(y)

    return x, y, word_tokenizer, tag_tokenizer