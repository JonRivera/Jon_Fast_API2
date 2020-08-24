import logging

import pickle
import basilica
import numpy as np
import pandas as pd
from fastapi import APIRouter

from pydantic import BaseModel, Field, validator
from decouple import config

log = logging.getLogger(__name__)
router = APIRouter()

reg_file3 = open('app/api/logistic.model3', 'rb')
# trained with 10,000 subreddit posts
# represent the model and can take a post and out a prediction

log_reg3 = pickle.load(reg_file3)

BASILICA = basilica.Connection(config('BASILICA_KEY'))


# update

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""
    title: str = Field(...,
                       example='This is a Random Title')
    reddit_post: str = Field(...,
                             example="Made this meme for the upcoming tsm match. Don't have a good"
                                     " editing software but I hope you enjoy! #C9WIN")


@router.post('/predict')
async def predict(item: Item):
    """
    Make predictions for classification problem 🔮

    ### Request Body
    - `title`: string format
    - `reddit_post`: string format

    ### Response
    - `prediction`: predicted subreddit category
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    # X_new = item.to_df()
    # log.info(X_new)
    # change
    post = item.reddit_post
    post2 = str(post)
    reddit_embedding = BASILICA.embed_sentence(post2)
    y_pred = log_reg3.predict(np.array(reddit_embedding).reshape(1, -1))
    prediction = str(y_pred[0])
    return {
        'Reditt_Post': item.title,
        'Predicted_Subreddit': prediction,

    }
