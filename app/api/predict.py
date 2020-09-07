import logging

import pickle
import basilica
import numpy as np
import pandas as pd
from fastapi import APIRouter

from pydantic import BaseModel, Field, validator
import os

log = logging.getLogger(__name__)
router = APIRouter()
BASILICA_KEY ="79a607aa-3e02-eb67-96cb-a7b21ff06e79"
#BASILICA_KEY = os.getenv('BASILICA_KEY')

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
    n: int = Field(..., example=1)

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

    @validator('n')
    def check_n(cls, value):
        """Validate that title is a string."""
        assert type(value) == int, f'value == {value}, must be an integer'
        assert value >= 1, f'n == {value} must be greater or to 1'
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
    prediction = log_reg3.predict(np.array(reddit_embedding).reshape(1, -1))
    probability = log_reg3.predict_proba(np.array(reddit_embedding).reshape(1, -1))[0]



    # 'Predicted_Subreddit': str(prediction[0]),
    # 'Probability': str(probability[0][0]),
    subreddits_probs = list(zip(log_reg3.classes_, probability))
    subreddits_probs_sorted = sorted(subreddits_probs, key=lambda x:x[1], reverse=True)
    return {'Post Title': item.title,
        'subreddits_predicted':subreddits_probs_sorted[:item.n]}
