import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def topic_barcharts(
    pipeline,
    topic_names: list[str],
    top_n: int = 15,
    n_columns: int = 6,
):
    _, vectorizer = pipeline.steps[0]
    _, topic_model = pipeline.steps[-1]
    components = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    n_topics = components.shape[0]
    n_rows = (n_topics // n_columns) + 1
    fig = make_subplots(n_rows, n_columns, subplot_titles=topic_names)
    for topic_id, component in enumerate(components):
        highest = np.argpartition(-component, top_n)[:top_n]
        words = vocab[highest]
        importances = component[highest]
        order = np.argsort(importances)
        row, column = (topic_id // n_columns) + 1, (topic_id % n_columns) + 1
        fig.add_trace(
            go.Bar(
                y=words[order],
                x=importances[order],
                orientation="h",
                showlegend=False,
            ),
            row=row,
            col=column,
        )
    fig.update_layout(
        margin=dict(l=0, r=0, b=5, pad=2), template="plotly_white"
    )
    return fig
