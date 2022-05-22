from fastapi import FastAPI
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import pickle
from keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel

max_length = 20
trunc_type='post'
padding_type='post'

model = load_model('./utilities/hate_speech/final_model (2).h5')

with open('./utilities/hate_speech/final_model.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def hate_speech_filtering(post: str):
    value = [post]
    post_sequence = tokenizer.texts_to_sequences(value)
    padded_post_sequence = pad_sequences(post_sequence,
                                         maxlen=max_length, padding=padding_type,
                                         truncating=trunc_type)
    post_prediction = model.predict(padded_post_sequence)
    print(post_prediction)
    label = post_prediction.round().item()
    if label == 0:
        print('called 0')
        return "This comment is NOT Hate speech"
    elif label == 1:
        print('called 1')
        return "This comment is Hate speech"

    return "Error Occured"

