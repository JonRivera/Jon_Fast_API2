import logging

import pickle
import basilica
import numpy as np
import pandas as pd
from fastapi import APIRouter

from pydantic import BaseModel, Field, validator
from decouple import config

import os

from dotenv import load_dotenv

load_dotenv()

BASILICA_KEY = os.getenv('BASILICA_KEY')
log = logging.getLogger(__name__)
router = APIRouter()

reg_file3 = open('app/api/logistic.model3', 'rb')
# trained with 10,000 subreddit posts
# represent the model and can take a post and out a prediction

log_reg3 = pickle.load(reg_file3)

BASILICA = basilica.Connection(BASILICA_KEY)


# update

class Subreddit_Post(BaseModel):
    """Use this data model to parse the request body JSON."""
    title: str = Field(...,
                       example='This is a Random Title')
    reddit_post: str = Field(...,
                             example="Made this meme for the upcoming tsm match. Don't have a good"
                                     " editing software but I hope you enjoy! #C9WIN")

    @validator('title')
    def check_title(cls, value):
        """Validate that title is a string."""
        assert type(value) == str, f'Title == {value}, must be a string'
        return value

    @validator('reddit_post')
    def check_body(cls, value):
        """Validate that title is a string."""
        assert type(value) == str, f'Body == {value}, must be a string'
        return value


@router.post('/predict')
async def predict(item: Subreddit_Post):
    """
    Make predictions for classification problem 🔮

    ### Request Body
    - `title`: string format
    - `reddit_post`: string format

    ### Response
    - `prediction`: predicted subreddit category
    - `predict_proba`: gives the probability for post of interest

    Replace the placeholder docstring and fake predictions with your own model.
    """
    post = item.reddit_post
    reddit_embedding = BASILICA.embed_sentence(post, model='reddit')
    y_pred = log_reg3.predict(np.array(reddit_embedding).reshape(1, -1))
    y_proba = log_reg3.predict_proba(np.array(reddit_embedding).reshape(1, -1))

    return {
        'Reditt_Post': item.title,
        'Predicted_Subreddit': str(y_pred[0]),
        'Probabillity': str(y_proba[0][0]),

    }
