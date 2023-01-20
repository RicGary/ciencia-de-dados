import plotly.express as px
import pandas as pd


def heatmap(dataframe: pd.DataFrame):
    fig = px.imshow(dataframe, text_auto=True)
    return fig