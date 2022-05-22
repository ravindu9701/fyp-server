from os.path import dirname, join, realpath
import gensim
import pickle
import string
import tensorflow as tf
import re
import numpy as np
from keras.models import load_model
import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from sinling.sinhala.tokenizer import SinhalaTweetTokenizer


lda_model = gensim.models.ldamodel.LdaModel.load('./utilities/irrelevant/lda/lda_model')
id2word = gensim.corpora.Dictionary.load('./utilities/irrelevant/lda/lda_id2word')


def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = load_model('./utilities/irrelevant/model/weights', custom_objects={"f1": f1}, compile=False)

with open('./utilities/irrelevant/stop_words', 'rb') as fp:   # Unpickling
    remove_words = pickle.load(fp)

with open('./utilities/irrelevant/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

topics_per_doc = 20
topics = []


def replace_url(text: str) -> str:
    return re.sub(r'(http://www\.|https://www\.|http://|https://)[a-z0-9]+([\-.]{1}[a-z0-9A-Z/]+)*', '', text)


def stem_word(word: str) -> str:

    if len(word) < 4:
        return word

    # remove 'ට'
    if word[-1] == 'ට':
        return word[:-1]

    # remove 'ද'
    if word[-1] == 'ද':
        return word[:-1]

    # remove 'ටත්'
    if word[-3:] == 'ටත්':
        return word[:-3]

    # remove 'එක්'
    if word[-3:] == 'ෙක්':
        return word[:-3]

    # remove 'එ'
    if word[-1:] == 'ෙ':
        return word[:-1]

    # remove 'ක්'
    if word[-2:] == 'ක්':
        return word[:-2]

    # remove 'ගෙ' (instead of ගේ because this step comes after simplifying text)
    if word[-2:] == 'ගෙ':
        return word[:-2]

    # else
    return word


puncDict = {"‘": ' ', "’": ' ', "‚": ' ', "‛": ' ', "“": ' ', "”": ' ', "„": ' ', "‟": ' ',
            "!": ' ', "(": ' ', ")": ' ', "-": ' ', "[": ' ', "]": ' ', "{": ' ', "}": ' ',
            ";": ' ', ":": ' ', "'": ' ', "\"": ' ', "\\": ' ', "<": ' ', ">": ' ', "/": ' ',
            "@": ' ', "#": ' ', "$": ' ', "%": ' ', "^": ' ', "&": ' ', "*": ' ', "_": ' ',
            "~": ' '}


def get_topic_num(row, num_topics):
    prob_dict = {}

    for i in range(0, num_topics):
        prob_dict[i] = 0.0000000

    for j, (topic_num, topic_prob) in enumerate(row):
        prob_dict[topic_num] = round(topic_prob*100)
        #prob_dict[topic_num] = round(topic_prob, 8)
    return prob_dict


def generate_post_vector(topic_prob):
    post_vector = [[]]
    for i in range(0, topics_per_doc):
        for j in range(0, 1):
            post_vector[0].append(topic_prob[i])
    return tf.convert_to_tensor(post_vector, dtype=tf.float64)


def get_post_vector(post: str):

    sinhala_tokenizer = SinhalaTweetTokenizer()

    char_simplified_post = str(replace_url(post)).translate(
        str.maketrans(puncDict))  # removed simplifying characters
    preprocessed_post = str(char_simplified_post).translate(
        str.maketrans('', '', string.punctuation))

    tokenized_post = []

    for sent in sinhala_tokenizer.split_sentences(preprocessed_post):
        tokens = sinhala_tokenizer.tokenize(sent)
        [tokenized_post.append(stem_word(token)) for token in tokens if not (re.search(
            r'[a-zA-Z0-9]+', token) or (len(token) < 2) or (token in remove_words))]

    topic_prob = get_topic_num(
        lda_model[id2word.doc2bow(tokenized_post)], topics_per_doc)
    return generate_post_vector(topic_prob)


def tokenize_comment(comment: str):
    preprocessed_comment = str(replace_url(comment)).translate(
        str.maketrans(puncDict))  # removed simplifying characters

    list_of_tokens = []

    # for sent in tokenizer.split_sentences(preprocessed_comment):
    sinhala_tokenizer = SinhalaTweetTokenizer()
    tokens = sinhala_tokenizer.tokenize(preprocessed_comment)
    [list_of_tokens.append(stem_word(token)) for token in tokens if not (
        re.search(r'[a-zA-Z0-9]+', token))]

    return list_of_tokens

def predict_comment_relevance(post: str, comment: str):

    post_vector = np.array(get_post_vector(post))

    cleaned_comment = tokenize_comment(comment)
    encoded_comment = tokenizer.texts_to_sequences([cleaned_comment])

    padded_comment = np.array(pad_sequences(
        encoded_comment, maxlen=142, padding="post"))

    prediction = model.predict([padded_comment, post_vector])

    print(prediction[0])
    # show results
    if(prediction[0] > 0.5):
        print(prediction[0])
        return "This comment is relevant"
    else:
        print(prediction[0])
        return "This comment is irrelevant"
    
    return "message: Error Occured"
