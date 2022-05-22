from json import load
from os.path import dirname, join, realpath
import pickle
import string
import tensorflow as tf
import re
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

model = load_model('./utilities/code_mixed/weights/model')
encoder_model = load_model('./utilities/code_mixed/weights/encoder')
decoder_model = load_model('./utilities/code_mixed/weights/decoder')

with open('./utilities/code_mixed/weights/tokenizer/singlish_tokenizer.pickle', 'rb') as handle:
    singlish_tokenizer = pickle.load(handle)

with open('./utilities/code_mixed/weights/tokenizer/sinhala_tokenizer.pickle', 'rb') as handle:
    sinhala_tokenizer = pickle.load(handle)

def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    # states_value = encoder_model.predict(input_seq)
    enc_output, enc_h, enc_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = sinhala_tokenizer.word_index['start']
    
    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, enc_h, enc_c ])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index==0:
          break
        else:   
         # convert max index number to marathi word
         sampled_char = sinhala_tokenizer.index_word[sampled_token_index]
        # aapend it ti decoded sent
        decoded_sentence += ' '+sampled_char
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'end' or len(decoded_sentence) >= 34):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        # states_value = [h, c]
        enc_h, enc_c = h, c
    
    return decoded_sentence

def clean_text(text):
    return re.sub('[^A-Za-z]+', '', text)

def transliterate(sentence: str):
    # sentence = clean_text(sentence)
    print(sentence)
    sentence_encoded = singlish_tokenizer.texts_to_sequences([sentence])
    sentence_encoded_padded = pad_sequences(sentence_encoded, maxlen=43, padding='post')
    sinhala_sentence = get_predicted_sentence(sentence_encoded_padded.reshape(1,43))[:-4]
    print(sinhala_sentence)
    return sinhala_sentence

        
