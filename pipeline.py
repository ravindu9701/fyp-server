from os.path import dirname, join, realpath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#import nest_asyncio
#from pyngrok import ngrok
from fastapi import FastAPI

from code_mixed import transliterate
from hate_speech import hate_speech_filtering
from irrelevant import predict_comment_relevance

app = FastAPI(
    title="FYP-Ether_IT-19",
    description="Hate and irrelevant comment detection from Sinhala code mixed data ",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class PipelineRequest(BaseModel):
    post: str
    singlish_comment: str
    sinhala_comment: str

class TransRequest(BaseModel):
    sentence: str

class HateRequest(BaseModel):
    post: str

class IrrelevantRequest(BaseModel):
    post: str
    comment: str

@app.post("/predict-output")
async def predict(request: PipelineRequest):
    post = request.post
    singlish_comment = request.singlish_comment
    sinhala_comment = request.sinhala_comment

    transliteration_output = transliterate(singlish_comment)
    hate_output = hate_speech_filtering(sinhala_comment)
    irrelevant_output = predict_comment_relevance(post, sinhala_comment)

    result = {  "model_1": transliteration_output,
                "model_2": hate_output,
                "model_3": irrelevant_output,
             }

    return result

@app.post("/transliterate")
async def transliterate_ep(sentence : TransRequest):
    singlish_comment = sentence.sentence
    transliteration_output = transliterate(singlish_comment)

    return {"Transliterated_sentence": transliteration_output}

@app.post("/hate-speech")
async def hate_speech_filtering_ep(post: HateRequest):
    sinhala_comment = post.post
    print(sinhala_comment)
    hate_output = hate_speech_filtering(sinhala_comment)

    return {"Hate_prediction": hate_output }

@app.post("/relevance")
async def predict_relevance_ep(input: IrrelevantRequest):
    post = input.post
    sinhala_comment = input.comment
    irrelevant_output = predict_comment_relevance(post, sinhala_comment)

    return { "Relevance_prediction": irrelevant_output }
