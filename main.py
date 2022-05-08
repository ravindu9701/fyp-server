from fastapi import FastAPI
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import pickle
from keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel

origins = ["*"]

app = FastAPI(
    title="FYP-Ether_IT-19",
    description="Sinhala Hate Speech Detection",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


max_length = 20
trunc_type='post'
padding_type='post'


model = load_model('./final_model (2).h5')
model.summary()

with open('final_model.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


class Item(BaseModel):
    post: str


@app.post("/hate-speech")
async def hate_speech_filtering(post: Item):
    value = [post.post]
    print(post)
    post_sequence = tokenizer.texts_to_sequences(value)
    padded_post_sequence = pad_sequences(post_sequence,
                                         maxlen=max_length, padding=padding_type,
                                         truncating=trunc_type)
    post_prediction = model.predict(padded_post_sequence)
    print(post_prediction)
    label = post_prediction.round().item()
    if label == 0:
        print('called 0')
        return {"message": "This comment is NOT Hate speech"}
    elif label == 1:
        print('called 1')
        return {"message": "This comment is Hate speech"}

    return {"message": "Error Occured"}

