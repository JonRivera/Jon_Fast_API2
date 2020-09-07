from fastapi import APIRouter, HTTPException
import pandas as pd
from .predict import Subreddit_Post, log_reg3, BASILICA
import numpy as np
import plotly.express as px
router = APIRouter()


@router.post('/viz/{subreddits_visuals}')
async def viz(item:Subreddit_Post):
    """
    ### Request Body
    - `title`: string the title of the post
    - `body`: string the meat of the post
    - `n`: int number of subreddits you want back

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/)
    """

    data = item.reddit_post
    # transform into embeds
    embed = BASILICA.embed_sentence(data)
    # reshape for usability
    embed = np.array(embed).reshape(1, -1)
    subreddits_probs = list(zip(log_reg3.classes_, log_reg3.predict_proba(embed)[0]))
    subreddits_probs_sorted = sorted(subreddits_probs, key=lambda x: x[1], reverse=True)
    # Make Plotly figure
    names = [subreddits_probs_sorted[i][0] for i in range(item.n)]
    values = [subreddits_probs_sorted[i][1] for i in range(item.n)]
    # this just adds a 'other' category bc pie chart == 1
    if sum(values) < 1:
        names.append(f'{1000 - len(names)} other sub-reddits')
        values.append(1 - sum(values))
    fig = px.pie(values=values, names=names, title='Subreddits To Post To')
    fig.show()
    # Return Plotly figure as JSON
    return fig.to_json()

